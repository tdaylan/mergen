# https://github.com/crmaximo/VAEGAN/blob/master/VAEGAN.py

from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *
import os
from keras import metrics, backend as K
from PIL import Image
import numpy as np
import tensorflow as tf
# tf.enable_eager_execution()

data_dir='/Users/studentadmin/Dropbox/TESS_UROP/data/'
lib_dir = '../main/'
output_dir = '../../plots/vae_gan/'
sector=2
custom_mask=[]

import sys
sys.path.insert(0, lib_dir)
import data_functions as df

flux, x, ticid, target_info = \
    df.load_data_from_metafiles(data_dir, sector, DEBUG=False,
                                output_dir=output_dir, nan_mask_check=True,
                                custom_mask=custom_mask)
dataset = df.standardize(flux)

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same',
                    kernel_initializer='random_normal', activation='relu',
                    kernel_regularizer='l2', activity_regularizer='l2',
                    bias_regularizer='l2'):
    """Conv1DTranpose has not been implemented in Keras, so I convert the 1D
    tensor to a 2D tensor with an image width of 1, then apply
    Conv2Dtranspose.
    
    This code is inspired by this:
    https://stackoverflow.com/questions/44061208/how-to-implement-the-
    conv1dtranspose-in-keras
        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
    """
    
    # >> save shape of input tensor
    dim = input_tensor.get_shape().as_list()[1]

    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1),
                        strides=(strides, 1), padding=padding,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        activity_regularizer=activity_regularizer)(x)
    
    # >> explicitly set the shape, because TensorFlow hates me >:(  
    x = Lambda(lambda x: tf.ensure_shape(x, (None, strides*dim, 1, filters)))(x)
    
     
    # x.set_shape((None, strides*dim, 1, filters))
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    # x = x[:,:,0] # >> convert to 1D    
    
    x = Activation(activation)(x)
    return x

def sampling(args):
    mean, logsigma = args
    epsilon = K.random_normal(shape=(K.shape(mean)[0], 512), mean=0., stddev=1.0)
    return mean + K.exp(logsigma / 2) * epsilon

def encoder(kernel, filters, rows, columns):
    X = Input(shape=(rows,))
    
    model = Reshape(target_shape=(rows, columns))(X)
    model = Conv1D(filters=filters, kernel_size=kernel, strides=2, padding='same')(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    model = Conv1D(filters=filters*2, kernel_size=kernel, strides=2, padding='same')(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    model = Conv1D(filters=filters*4, kernel_size=kernel, strides=2, padding='same')(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    model = Conv1D(filters=filters*8, kernel_size=kernel, strides=2, padding='same')(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    model = Flatten()(model)

    mean = Dense(512)(model)
    logsigma = Dense(512, activation='tanh')(model)
    latent = Lambda(sampling, output_shape=(512,))([mean, logsigma])
    meansigma = Model([X], [mean, logsigma, latent])
    return meansigma


def decgen(kernel, filters, rows, columns):
    X = Input(shape=(512,))

    model = Dense(filters*8*rows*columns)(X)
    model = Reshape(target_shape=(rows, filters * 8))(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = Activation('relu')(model)

    model = Conv1DTranspose(model, filters=filters*4, kernel_size=kernel, strides=2, padding='same')
    model = BatchNormalization(epsilon=1e-5)(model)
    model = Activation('relu')(model)

    model = Conv1DTranspose(model, filters=filters*2, kernel_size=kernel, strides=2, padding='same')
    model = BatchNormalization(epsilon=1e-5)(model)
    model = Activation('relu')(model)

    model = Conv1DTranspose(model, filters=filters, kernel_size=kernel, strides=2, padding='same')
    model = BatchNormalization(epsilon=1e-5)(model)
    model = Activation('relu')(model)

    model = Conv1DTranspose(model, filters=1, kernel_size=kernel, strides=2, padding='same')
    model = Activation('tanh')(model)

    model = Model(X, model)
    return model


def discriminator(kernel, filters, rows, columns):
    X = Input(shape=(rows, columns))

    model = Conv1D(filters=filters*2, kernel_size=kernel, strides=2, padding='same')(X)
    model = LeakyReLU(alpha=0.2)(model)

    model = Conv1D(filters=filters*4, kernel_size=kernel, strides=2, padding='same')(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    model = Conv1D(filters=filters*8, kernel_size=kernel, strides=2, padding='same')(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    model = Conv1D(filters=filters*8, kernel_size=kernel, strides=2, padding='same')(model)


    dec = BatchNormalization(epsilon=1e-5)(model)
    dec = LeakyReLU(alpha=0.2)(dec)
    dec = Flatten()(dec)
    dec = Dense(1, activation='sigmoid')(dec)

    output = Model([X], [dec, model])
    return output


batch_size = 64
rows = np.shape(dataset)[1]
columns = 1
filters = 4 # 32
epochs = 10
datasize = len(dataset)
# noise = np.random.normal(0, 1, (batch_size, 256))
noise = np.random.normal(0, 1, (batch_size, 512))

# optimizers
SGDop = SGD(lr=0.0003)
ADAMop = Adam(lr=0.0002)
# encoder
E = encoder(5, filters, rows, columns)
E.compile(optimizer=SGDop, loss='mse')
E.summary()
# generator/decoder
G = decgen(5, filters, rows, columns)
G.compile(optimizer=SGDop, loss='mse')
G.summary()
# discriminator
D = discriminator(5, filters, rows, columns)
D.compile(optimizer=SGDop, loss='mse')
D.summary()
D_fixed = discriminator(5, filters, rows, columns)
D_fixed.compile(optimizer=SGDop, loss='mse')
# VAE
X = Input(shape=(rows, columns))
# latent_rep = E(X)[0]
# output = G(latent_rep)
E_mean, E_logsigma, Z = E(X)

# Z = Input(shape=(512,))
# Z2 = Input(shape=(batch_size, 512))

output = G(Z)
G_dec = G(E_mean + E_logsigma)
D_fake, F_fake = D(output)
D_fromGen, F_fromGen = D(G_dec)
D_true, F_true = D(X)

VAE = Model(X, output)
kl = - 0.5 * K.sum(1 + E_logsigma - K.square(E_mean) - K.exp(E_logsigma), axis=-1)
crossent = 64 * metrics.mse(K.flatten(X), K.flatten(output))
VAEloss = K.mean(crossent + kl)
VAE.add_loss(VAEloss)
VAE.compile(optimizer=SGDop)
VAE.summary()

for epoch in range(epochs):
    print('Getting latent vector')
    latent_vect = E.predict(dataset)[0]
    print('Decoding image')
    encImg = G.predict(latent_vect)
    print('Generating image')
    fakeImg = G.predict(noise)

    print('Training D_true')
    DlossTrue = D_true.train_on_batch(dataset, np.ones((batch_size, 1)))
    print('Training DlossEnc')
    DlossEnc = D_fromGen.train_on_batch(encImg, np.ones((batch_size, 1)))
    print('Training DlossFake')
    DlossFake = D_fake.train_on_batch(fakeImg, np.zeros((batch_size, 1)))

    cnt = epoch
    while cnt > 3:
        cnt = cnt - 4
    print('cnt = '+str(cnt))

    if cnt == 0:
        print('Training GlossEnc')
        GlossEnc = G.train_on_batch(latent_vect, np.ones((batch_size, 1)))
        print('Training GlossGen')
        GlossGen = G.train_on_batch(noise, np.ones((batch_size, 1)))
        print('Training VAE')
        Eloss = VAE.train_on_batch(dataset, None)

    chk = epoch

    while chk > 50:
        chk = chk - 51

    if chk == 0:
        print('Saving weights')
        D.save_weights('discriminator.h5')
        G.save_weights('generator.h5')
        E.save_weights('encoder.h5')

    print("epoch number", epoch + 1)
    print("loss:")
    print("D:", DlossTrue, DlossEnc, DlossFake)
    print("G:", GlossEnc, GlossGen)
    print("VAE:", Eloss)

print('Training done,saving weights')
D.save_weights('discriminator.h5')
G.save_weights('generator.h5')
E.save_weights('encoder.h5')
print('end')