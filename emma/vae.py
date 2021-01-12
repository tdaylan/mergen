# https://github.com/crmaximo/VAEGAN/blob/master/VAEGAN.py

from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *
import os
from keras.losses import mean_squared_error
from keras import metrics, backend as K
from PIL import Image
import numpy as np
import tensorflow as tf
# tf.enable_eager_execution()

data_dir='/Users/studentadmin/Dropbox/TESS_UROP/data/'
lib_dir = '../main/'
output_dir = '../../plots/vae/'
database_dir= '/Users/studentadmin/Dropbox/TESS_UROP/data/databases/'
mom_dump = '../../Table_of_momentum_dumps.csv'
simbad_database_dir = ''
single_file = False
sectors=[2]
cams = [1,2,3,4]
ccds =  [1,2,3,4]
norm_type = 'standardization'
train_test_ratio = 0.9
validation_targets = []
p = {'fully_conv': False, 'num_conv_layers':2, 'concat_ext_feats': False,
     'latent_dim': 35, 'pool_size': 1, 'strides': 2, 'batch_size': 32,
     'epochs': 2, 'num_consecutive': 2}
input_rms = False
input_psd = False
load_psd = False
n_pgram = 50
split = False
use_tess_features = False
use_tls_features = False

import sys
sys.path.insert(0, lib_dir)
import data_functions as df
import model as ml


if sectors[0] == 1:
    custom_mask = list(range(800)) + list(range(15800, 17400)) + list(range(19576, 20075))
elif 4 in sectors:
    custom_mask = list(range(7424, 9078))
else:
    custom_mask = []
    
    
if os.path.isdir(output_dir) == False: # >> check if dir already exists
    os.mkdir(output_dir)
    
if len(sectors) > 1:
    flux, x, ticid, target_info = \
        df.combine_sectors_by_lc(sectors, data_dir, custom_mask=custom_mask,
                                 output_dir=output_dir)
else:
    # >> currently only handles one sector
    flux, x, ticid, target_info = \
        df.load_data_from_metafiles(data_dir, sectors[0], cams=cams, ccds=ccds,
                                    DEBUG=True,
                                    output_dir=output_dir, nan_mask_check=True,
                                    custom_mask=custom_mask)
        
x_train, x_test, y_train, y_test, ticid_train, ticid_test, target_info_train, \
    target_info_test, rms_train, rms_test, x = \
    ml.autoencoder_preprocessing(flux, x, p, ticid, target_info,
                                 mock_data=False,
                                 sector=sectors[0],
                                 validation_targets=validation_targets,
                                 norm_type=norm_type,
                                 input_rms=input_rms, input_psd=input_psd,
                                 load_psd=load_psd, n_pgram=n_pgram,
                                 train_test_ratio=train_test_ratio,
                                 split=split,
                                 output_dir=output_dir, 
                                 data_dir=data_dir,
                                 use_tess_features=use_tess_features,
                                 use_tls_features=use_tls_features)
    
input_dim = np.shape(x_train)[1]
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
    
# class Sampling(Layer):
#     """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

#     def call(self, inputs):
#         z_mean, z_log_var = inputs
#         batch = tf.shape(z_mean)[0]
#         dim = tf.shape(z_mean)[1]
#         epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon    
# def sampling(args):
#     z_mean, z_log_sigma = args
#     epsilon = K.random_normal(shape=(K.shape(z_mean)[0], p['latent_dim']),
#                               mean=0., stddev=0.1)
#     return z_mean + K.exp(z_log_sigma) * epsilon


def sampling(args):
    """
    Adapted from https://github.com/piyush-kgp/VAE-MNIST-Keras
    Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1] # Returns the shape of tensor or variable as a tuple of int or None entries.
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
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
    
    out = Activation(activation)(x)
    return out

# original_dim = 13444
# intermediate_dim = 1000
# latent_dim = 35

# import keras
# from keras import layers

# inputs = keras.Input(shape=(original_dim,))
# h = layers.Dense(intermediate_dim, activation='relu')(inputs)
# z_mean = layers.Dense(latent_dim)(h)
# z_log_sigma = layers.Dense(latent_dim)(h)
# def sampling(args):
#     z_mean, z_log_sigma = args
#     epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
#                               mean=0., stddev=0.1)
#     return z_mean + K.exp(z_log_sigma) * epsilon

# z = layers.Lambda(sampling)([z_mean, z_log_sigma])

# encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

# # Create decoder
# latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
# x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
# outputs = layers.Dense(original_dim, activation='sigmoid')(x)
# decoder = keras.Model(latent_inputs, outputs, name='decoder')

# # instantiate VAE model
# outputs = decoder(encoder(inputs)[2])
# vae = keras.Model(inputs, outputs, name='vae_mlp')

# reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
# reconstruction_loss *= original_dim
# kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
# kl_loss = K.sum(kl_loss, axis=-1)
# kl_loss *= -0.5
# vae_loss = K.mean(reconstruction_loss + kl_loss)
# vae.add_loss(vae_loss)
# vae.compile(optimizer='adam')

# inputs = Input(shape=(input_dim, 1))
# x = Conv1D(32, 3, activation="relu", strides=2, padding="same")(inputs)
# x = Conv1D(64, 3, activation="relu", strides=2, padding="same")(x)
# x = Flatten()(x)
# z_mean = Dense(p['latent_dim'], name="z_mean")(x)
# z_log_var = Dense(p['latent_dim'], name="z_log_var")(x)
# # z = Sampling()([z_mean, z_log_var])
# z = Lambda(sampling, output_shape=(p['latent_dim'],))([z_mean, z_log_var])
# encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

# layer_inputs = Input(shape=(p['latent_dim'],), name='z_sampling')
# num_iter = int(p['num_conv_layers']/2)
# reduction_factor = p['pool_size'] * p['strides']**p['num_consecutive']
# tot_reduction_factor = reduction_factor**num_iter
# x = Dense(int(input_dim/tot_reduction_factor * 64), activation="relu")(layer_inputs)
# x = Reshape((int(input_dim/tot_reduction_factor), 64))(x)
# x = Conv1DTranspose(x, 64, 3, activation="relu", strides=2, padding="same")
# x = Conv1DTranspose(x, 32, 3, activation="relu", strides=2, padding="same")
# outputs = Conv1DTranspose(x, 1, 3, activation="sigmoid", padding="same",
#                                   strides=1)
# decoder = Model(latent_inputs, outputs, name='decoder')

# outputs = decoder(encoder(inputs)[2])
# vae= Model(inputs, outputs, name='vae')
# reconstruction_loss = mean_squared_error(inputs, outputs)
# reconstruction_loss *= input_dim
# kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
# kl_loss = K.sum(kl_loss, axis=-1)
# kl_loss *= -0.5
# vae_loss = K.mean(reconstruction_loss + kl_loss)
# vae.add_loss(vae_loss)
# vae.compile(optimizer='adam')
# vae.summary()
# ml.model_summary_txt(output_dir, vae)

# vae.fit(x_train,
#         epochs=p['epochs'],
#         batch_size=p['batch_size'])

# https://blog.paperspace.com/how-to-build-variational-autoencoder-keras/

inputs = Input(shape=(input_dim, 1))
x = Conv1D(32, 3, activation="relu", strides=2, padding="same")(inputs)
x = Conv1D(64, 3, activation="relu", strides=2, padding="same")(x)
x = Flatten()(x)
x = Dense(16, activation="relu")(x)
z_mean = Dense(p['latent_dim'], name="z_mean")(x)
z_log_var = Dense(p['latent_dim'], name="z_log_var")(x)
z = Lambda(sampling, output_shape=(p['latent_dim'],))([z_mean, z_log_var])
# z = Sampling()([z_mean, z_log_var])
encoder = Model(inputs, z, name="encoder")
encoder.summary()

latent_inputs = Input(shape=(p['latent_dim'],))
num_iter = int(p['num_conv_layers']/2)
reduction_factor = p['pool_size'] * p['strides']**p['num_consecutive']
tot_reduction_factor = reduction_factor**num_iter
x = Dense(int(input_dim/tot_reduction_factor * 64), activation="relu")(latent_inputs)
x = Reshape((int(input_dim/tot_reduction_factor), 64))(x)
x = Conv1DTranspose(x, 64, 3, activation="relu", strides=2, padding="same")
x = Conv1DTranspose(x, 32, 3, activation="relu", strides=2, padding="same")
decoder_outputs = Conv1DTranspose(x, 1, 3, activation="sigmoid", padding="same",
                                  strides=1)
decoder = Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()


vae_input = Input(shape=(input_dim, 1))
vae_encoder_output = encoder(vae_input)
vae_decoder_output = decoder(vae_encoder_output)
vae = Model(vae_input, vae_decoder_output, name="VAE")


def loss_func(z_mean, z_log_var):
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss_factor = 1000
        reconstruction_loss = K.mean(K.square(y_true-y_predict), axis=[1, 2])
        return reconstruction_loss_factor * reconstruction_loss

    def vae_kl_loss(z_mean, z_log_var):
        kl_loss = -0.5 * K.sum(1.0 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
        return kl_loss

    def vae_kl_loss_metric(y_true, y_predict):
        kl_loss = -0.5 * K.sum(1.0 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
        return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)

        loss = reconstruction_loss + kl_loss
        return loss

    return vae_loss

vae.compile(optimizer=Adam(lr=0.0005), loss=loss_func(z_mean, z_log_var))
vae.fit(x_train, x_train, epochs=p['epochs'], batch_size=p['batch_size'],
        shuffle=True)


# # class VAE(Model):
# #     def __init__(self, encoder, decoder, **kwargs):
# #         super(VAE, self).__init__(**kwargs)
# #         self.encoder = encoder
# #         self.decoder = decoder

# #     def train_step(self, data):
# #         if isinstance(data, tuple):
# #             data = data[0]
# #         with tf.GradientTape() as tape:
# #             z_mean, z_log_var, z = encoder(data)
# #             reconstruction = decoder(z)
# #             reconstruction_loss = tf.reduce_mean(
# #                 keras.losses.binary_crossentropy(data, reconstruction)
# #             )
# #             reconstruction_loss *= 28 * 28
# #             kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
# #             kl_loss = tf.reduce_mean(kl_loss)
# #             kl_loss *= -0.5
# #             total_loss = reconstruction_loss + kl_loss
# #         grads = tape.gradient(total_loss, self.trainable_weights)
# #         self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
# #         return {
# #             "loss": total_loss,
# #             "reconstruction_loss": reconstruction_loss,
# #             "kl_loss": kl_loss,
# #         }
    
# # vae = VAE(encoder, decoder)
# # vae.compile(optimizer=keras.optimizers.Adam())
# # vae.fit(x_train, epochs=p['epochs'], batch_size=p['batch_size'])

# # outputs = decoder(encoder(inputs)[2])
# outputs = decoder_outputs
# vae = Model(inputs, outputs, name='vae_mlp')

# reconstruction_loss = mean_squared_error(inputs, outputs)
# reconstruction_loss *= input_dim
# kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
# kl_loss = K.sum(kl_loss, axis=-1)
# kl_loss *= -0.5
# vae_loss = K.mean(reconstruction_loss + kl_loss)
# vae.add_loss(vae_loss)
# vae.compile(optimizer='adam')
# vae.summary()
# ml.model_summary_txt(output_dir, vae)

# vae.fit(x_train,
#         epochs=p['epochs'],
#         batch_size=p['batch_size'])