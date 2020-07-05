# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# 
# 2020-05-26 - modellibrary.py
# Keras novelty detection in TESS dataset
# / Emma Chickles
# 
# TODO: finish these function summaries
# Model
# * run_model
# * model_summary_txt
# * param_summary_txt
# * conv_autoencoder
# * cnn
# * cnn_mock
# * mlp
# * simple autoencoder: fully-connected layers
# * compile_model
# 
# Helper functions
# * ticid_label
#
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 

import os
import pdb
import matplotlib.pyplot as plt
import numpy as np
import plotting_functions as pl
import data_functions as df

def run_model(x_train, y_train, x_test, y_test, p, supervised=False,
              mock_data=False):
    '''Runs autoencoder or CNN.'''
    if not supervised:
        history, model = conv_autoencoder(x_train, x_train, x_test,
                                             x_test, p)
    if supervised:
        if mock_data:
            history, model = cnn_mock(x_train, y_train, x_test, y_test, p)
        else:
            history, model = cnn(x_train, y_train, x_test, y_test, p)
        
    x_predict = model.predict(x_test)
    return history, model, x_predict
    
def param_summary(history, x_test, x_predict, p, output_dir, param_set_num,
                  title, supervised=False, y_test=False):
    from sklearn.metrics import confusion_matrix
    with open(output_dir + 'param_summary.txt', 'a') as f:
        f.write('parameter set ' + str(param_set_num) + ' - ' + title +'\n')
        f.write(str(p.items()) + '\n')
        if supervised:
            label_list = ['loss', 'accuracy', 'precision', 'recall']
            key_list =['loss', 'accuracy', list(history.history.keys())[-2],
                    list(history.history.keys())[-1]]
        else:
            label_list = ['loss']
            key_list = ['loss']

        for j in range(len(label_list)):
            f.write(label_list[j]+' '+str(history.history[key_list[j]][-1])+\
                    '\n')
        if supervised:
            y_predict = np.argmax(x_predict, axis=-1)
            y_true = np.argmax(y_test, axis=-1)
            cm = confusion_matrix(y_predict, y_true)
            f.write('confusion matrix\n')
            f.write(str(cm))
            f.write('\ny_true\n')
            f.write(str(y_true)+'\n')
            f.write('y_predict\n')
            f.write(str(y_predict)+'\n')
        else:
            # >> assuming uncertainty of 0.02
            chi_2 = np.average((x_predict-x_test)**2 / 0.02)
            # f.write('chi_squared ' + str(chi_2) + '\n')
            mse = np.average((x_predict - x_test)**2)
            # f.write('mse '+ str(mse) + '\n')
        f.write('\n')
    
        
def model_summary_txt(output_dir, model):
    with open(output_dir + 'model_summary.txt', 'a') as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

# :: autoencoder ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def conv_autoencoder(x_train, y_train, x_test, y_test, params):
    from keras.models import Model

    # -- encoding -------------------------------------------------------------
    encoded = encoder(x_train, params)

    # -- decoding -------------------------------------------------------------
    decoded = decoder(x_train, encoded.output, params)
    model = Model(encoded.input, decoded)
    print(model.summary())
    
    # -- compile model --------------------------------------------------------
    compile_model(model, params)

    # -- train model ----------------------------------------------------------
    history = model.fit(x_train, x_train, epochs=params['epochs'],
                        batch_size=params['batch_size'], shuffle=True,
                        validation_data=(x_test, x_test))
        
    return history, model

def cnn(x_train, y_train, x_test, y_test, params, num_classes=4):
    from keras.models import Model
    from keras.layers import Dense

    # -- encoding -------------------------------------------------------------
    encoded = encoder(x_train, params)
    
    # -- supervised mode: softmax ---------------------------------------------
    x = Dense(int(num_classes),
          activation='softmax')(encoded.output)
    model = Model(encoded.input, x)
    model.summary()
        
    # -- compile model --------------------------------------------------------
    compile_model(model, params)

    # -- train model ----------------------------------------------------------
    history = model.fit(x_train, y_train, epochs=params['epochs'],
                        batch_size=params['batch_size'], shuffle=True,
                        validation_data=(x_test,y_test))
    
    return history, model

def cnn_mock(x_train, y_train, x_test, y_test, params, num_classes = 2):
    from keras.models import Model
    from keras.layers import Dense

    # -- encoding -------------------------------------------------------------
    encoded = encoder(x_train, params)
    
    # -- supervised mode: softmax ---------------------------------------------
    x = Dense(int(num_classes),
          activation='softmax')(encoded.output)
    model = Model(encoded.input, x)
    model.summary()
        
    # -- compile model --------------------------------------------------------
    compile_model(model, params)

    # -- train model ----------------------------------------------------------
    history = model.fit(x_train, y_train, epochs=params['epochs'],
                        batch_size=params['batch_size'], shuffle=True,
                        validation_data=(x_test,y_test))
    
    return history, model


def mlp(x_train, y_train, x_test, y_test, params, resize=True):
    '''a simple classifier based on a fully-connected layer'''
    from keras.models import Model
    from keras.layers import Input, Dense, Flatten

    num_classes = np.shape(y_train)[1]
    input_dim = np.shape(x_train)[1]
    if resize:
        input_img = Input(shape = (input_dim,1))
        x = Flatten()(input_img)
    else:
        input_img = Input(shape = (input_dim,))
        x = input_img
    for i in range(len(params['hidden_units'])):
        x = Dense(params['hidden_units'][i],activation=params['activation'])(x)
    x = Dense(num_classes, activation='softmax')(x)
        
    model = Model(input_img, x)
    model.summary()
    compile_model(model, params, mlp=True)

    history = model.fit(x_train, y_train, epochs=params['epochs'],
                            batch_size=params['batch_size'], shuffle=True,
                            validation_data=(x_test, y_test))
        
    return history, model

def simple_autoencoder(x_train, y_train, x_test, y_test, params, resize = False,
                       batch_norm=False):
    '''a simple autoencoder based on a fully-connected layer'''
    from keras.models import Model
    from keras.layers import Input, Dense, Flatten, Reshape, BatchNormalization

    num_classes = np.shape(y_train)[1]
    input_dim = np.shape(x_train)[1]
    if resize:
        input_img = Input(shape = (input_dim,1))
        x = Flatten()(input_img)
    else:
        input_img = Input(shape = (input_dim,))
        x = input_img
    for i in range(len(params['hidden_units'])):
        if batch_norm: x = BatchNormalization()(x)
        x = Dense(params['hidden_units'][i], activation=params['activation'],
                  kernel_initializer=params['initializer'])(x)
    if batch_norm: x = BatchNormalization()(x)
    x = Dense(params['latent_dim'], activation=params['activation'],
              kernel_initializer=params['initializer'])(x)
    for i in np.arange(len(params['hidden_units'])-1, -1, -1):
        if batch_norm: x = BatchNormalization()(x)        
        x = Dense(params['hidden_units'][i], activation=params['activation'],
                  kernel_initializer=params['initializer'])(x)

    if batch_norm: x = BatchNormalization()(x)    
    x = Dense(input_dim, activation=params['last_activation'],
              kernel_initializer=params['initializer'])(x)
    if resize:
        x = Reshape((input_dim, 1))(x)
        
    model = Model(input_img, x)
    model.summary()
    compile_model(model, params)

    history = model.fit(x_train, x_train, epochs=params['epochs'],
                        batch_size=params['batch_size'], shuffle=True,
                        validation_data=(x_test, x_test))
        
    return history, model

def compile_model(model, params, mlp=False):
    from keras import optimizers
    import keras.metrics
    if params['optimizer'] == 'adam':
        opt = optimizers.adam(lr = params['lr'], 
                              decay=params['lr']/params['epochs'])
    elif params['optimizer'] == 'adadelta':
        opt = optimizers.adadelta(lr = params['lr'])
        
    # model.compile(optimizer=opt, loss=params['losses'],
    #               metrics=['accuracy'])
    if mlp:
        import tensorflow as tf
        model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics = ['accuracy', keras.metrics.Precision(),
                  keras.metrics.Recall()])
    else:
        model.compile(optimizer=opt, loss=params['losses'],
                      metrics=['accuracy', keras.metrics.Precision(),
                      keras.metrics.Recall()])

# def encoder1(x_train):
    

def encoder(x_train, params):
    '''https://towardsdatascience.com/autoencoders-in-keras-c1f57b9a2fd7'''
    from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten
    from keras.layers import Dense, AveragePooling1D, BatchNormalization
    from keras.models import Model
    
    input_dim = np.shape(x_train)[1]
    num_iter = int(params['num_conv_layers']/2)
    # num_iter = int(len(params['num_filters'])/2)
    
    input_img = Input(shape = (input_dim, 1))
    for i in range(num_iter):
        if i == 0:
            x = Conv1D(params['num_filters'], int(params['kernel_size']),
                    activation=params['activation'], padding='same',
                    kernel_initializer=params['initializer'])(input_img)
            x = BatchNormalization()(x)
            # x = Conv1D(params['num_filters'], int(params['kernel_size']),
            #             activation=params['activation'], padding='same')(x)
        else:
            x = Conv1D(params['num_filters'], int(params['kernel_size']),
                        activation=params['activation'], padding='same',
                        kernel_initializer=params['initializer'])(x)
            x = BatchNormalization()(x)
            # x = Conv1D(params['num_filters'], int(params['kernel_size']),
            #             activation=params['activation'], padding='same')(x)
        x = MaxPooling1D(2, padding='same')(x)

        x = Dropout(params['dropout'])(x)
    
    x = Flatten()(x)
    encoded = Dense(params['latent_dim'], activation=params['activation'],
                    kernel_initializer=params['initializer'])(x)
    
    encoder = Model(input_img, encoded)

    return encoder

def decoder(x_train, bottleneck, params):
    from keras.layers import Dense, Reshape, Conv1D, UpSampling1D, Dropout
    from keras.layers import Lambda, BatchNormalization
    input_dim = np.shape(x_train)[1]
    # num_iter = int(len(params['num_filters'])/2)
    num_iter = int(params['num_conv_layers']/2)
    
    def repeat_elts(x):
        '''helper function for lambda layer'''
        import tensorflow as tf
        return tf.keras.backend.repeat_elements(x,params['num_filters'],2)
        # return tf.keras.backend.repeat_elements(x,params['num_filters'][num_iter+i],2)
    
    # x = Dense(int(input_dim/(2**(num_iter))))(bottleneck)
    # x = Reshape((int(input_dim/(2**(num_iter))), 1))(x)
    # x = Lambda(repeat_elts)(x)
    x = Dense(int(input_dim*params['num_filters']/(2**(num_iter))),
              kernel_initializer=params['initializer'])(bottleneck)
    x = Reshape((int(input_dim/(2**(num_iter))), params['num_filters']))(x)
    for i in range(num_iter):
        # x = Lambda(repeat_elts)(x)
        x = Dropout(params['dropout'])(x)
        x = UpSampling1D(2)(x)
        if i == num_iter-1:
            x = BatchNormalization()(x)
            decoded = Conv1D(1, int(params['kernel_size']),
                              activation=params['last_activation'],
                              padding='same',
                              kernel_initializer=params['initializer'])(x)            
            # decoded = Conv1D(1, int(params['kernel_size'][num_iter]),
            #                  activation=params['last_activation'],
            #                  padding='same')(x)
        else:
            x = BatchNormalization()(x)
            x = Conv1D(params['num_filters'], int(params['kernel_size']),
                       activation=params['activation'], padding='same',
                       kernel_initializer=params['initializer'])(x)            
            # x = Conv1D(1, int(params['kernel_size']),
            #            activation=params['activation'], padding='same')(x)
            # x = Conv1D(1, int(params['kernel_size'][num_iter+i]),
            #            activation=params['activation'], padding='same')(x)
    return decoded


def encoder_split(x, params):
    from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten
    from keras.layers import Dense, concatenate
    from keras.models import Model
    
    num_iter = int((params['num_conv_layers'])/2)
    
    input_imgs = [Input(shape=(np.shape(a)[1], 1)) for a in x]

    for i in range(num_iter):
        conv_1 = Conv1D(params['num_filters'][i], int(params['kernel_size'][i]),
             activation=params['activation'], padding='same')
        x = [conv_1(a) for a in input_imgs]
        maxpool_1 = MaxPooling1D(2, padding='same')
        x = [maxpool_1(a) for a in x]
        dropout_1 = Dropout(params['dropout'])
        x = [dropout_1(a) for a in x]
        maxchannel_1 = MaxPooling1D([params['num_filters'][i]],
                                    data_format='channels_first')
        x = [maxchannel_1(a) for a in x]

    flatten_1 = Flatten()
    x = [flatten_1(a) for a in x]
    dense_1 = Dense(params['latent_dim'], activation=params['activation'])
    x = [dense_1(a) for a in x]
    encoded = concatenate(x)
    encoder = Model(inputs=input_imgs, outputs=encoded)
    return encoder

def decoder_split(x_train, bottleneck, params):
    from keras.layers import Dense, Reshape, Conv1D, UpSampling1D, Dropout
    from keras.layers import Lambda, concatenate
    from keras import backend as K
    
    input_dim = np.shape(x_train)[1]
    num_iter = int((params['num_conv_layers'])/2)
    
    dense_1 = Dense(int(input_dim/(2**(num_iter))))
    x = [dense_1(bottleneck), dense_1(bottleneck)]
    reshape_1 = Reshape((int(input_dim/(2**(num_iter))), 1))
    x = [reshape_1(a) for a in x]
    for i in range(num_iter):
        upsampling_channels = Lambda(lambda x: \
                    K.repeat_elements(x,params['num_filters'][num_iter+i],2))
        x = [upsampling_channels(a) for a in x]
        dropout_1 = Dropout(params['dropout'])(x)
        x = [dropout_1(a) for a in x]
        upsampling_1 = UpSampling1D(2)(x)
        x = [upsampling_1(a) for a in x]
        if i == num_iter-1:
            conv_2 = Conv1D(1, params['kernel_size'][num_iter+1],
                              activation=params['last_activation'],
                              padding='same')
            x = [conv_2(a) for a in x]
            decoded = concatenate(x)
        else:
            conv_1 = Conv1D(1, params['kernel_size'][num_iter+i],
                        activation=params['activation'], padding='same')
            x = [conv_1(a) for a in x]
    return decoded

def create_mlp(input_dim):
    '''Build multi-layer perceptron neural network model for numerical data
    (rms)'''
    from keras.models import Model
    from keras.layers import Dense, Input
    input_img = Input(shape = (input_dim,))
    x = Dense(8, activation='relu')(input_img)
    x = Dense(4, activation='relu')(x)
    x = Dense(1, activation='linear')(x)
    
    model = Model(input_img, x)
    return model
    
# :: preprocessing data :::::::::::::::::::::::::::::::::::::::::::::::::::::::

def split_data_features(flux, features, time, ticid, classes, p,
                        train_test_ratio = 0.9,
                        cutoff=16336, supervised=False, interpolate=False,
                        resize_arr=True, truncate=True):

    # >> truncate (must be a multiple of 2**num_conv_layers)
    if truncate:
        new_length = int(np.shape(flux)[1] / \
                     (2**(np.max(p['num_conv_layers'])/2)))*\
                     int((2**(np.max(p['num_conv_layers'])/2)))
        flux=np.delete(flux,np.arange(new_length,np.shape(flux)[1]),1)
        time = time[:new_length]

    # >> split test and train data
    if supervised:
        train_inds = []
        test_inds = []
        class_types, counts = np.unique(classes, return_counts=True)
        num_classes = len(class_types)
        #  = min(counts)
        y_train = []
        y_test = []
        for i in range(len(class_types)):
            inds = np.nonzero(classes==i)[0]
            num_train = int(len(inds)*train_test_ratio)
            train_inds.extend(inds[:num_train])
            test_inds.extend(inds[num_train:])
            labels = np.zeros((len(inds), num_classes))
            labels[:,i] = 1.
            y_train.extend(labels[:num_train])
            y_test.extend(labels[num_train:])

        y_train = np.array(y_train)
        y_test = np.array(y_test)
        x_train = np.copy(features[train_inds])
        x_test = np.copy(features[test_inds])
        flux_train = np.copy(flux[train_inds])
        flux_test = np.copy(flux[test_inds])
        ticid_train = np.copy(ticid[train_inds])
        ticid_test = np.copy(ticid[test_inds])
    else:
        split_ind = int(train_test_ratio*np.shape(flux)[0])
        x_train = np.copy(features[:split_ind])
        x_test = np.copy(features[split_ind:])
        flux_train = np.copy(flux[:split_ind])
        flux_test = np.copy(flux[split_ind:])
        ticid_train = np.copy(ticid[:split_ind])
        ticid_test = np.copy(ticid[split_ind:])
        y_test, y_train = [False, False]
        
        
    if resize_arr:
        x_train =  np.resize(x_train, (np.shape(x_train)[0],
                                       np.shape(x_train)[1], 1))
        x_test =  np.resize(x_test, (np.shape(x_test)[0],
                                       np.shape(x_test)[1], 1))
    return x_train, x_test, y_train, y_test, flux_train, flux_test,\
        ticid_train, ticid_test, time

def split_data(flux, time, p, train_test_ratio = 0.9, cutoff=16336,
               supervised=False, classes=False, interpolate=False,
               resize_arr=True, truncate=True):
    '''need to update, might not work'''

    # >> truncate (must be a multiple of 2**num_conv_layers)
    if truncate:
        new_length = int(np.shape(flux)[1] / \
                     (2**(np.max(p['num_conv_layers'])/2)))*\
                     int((2**(np.max(p['num_conv_layers'])/2)))
        flux=np.delete(flux,np.arange(new_length,np.shape(flux)[1]),1)
        time = time[:new_length]

    # >> split test and train data
    if supervised:
        train_inds = []
        test_inds = []
        class_types, counts = np.unique(classes, return_counts=True)
        num_classes = len(class_types)
        #  = min(counts)
        y_train = []
        y_test = []
        for i in range(len(class_types)):
            inds = np.nonzero(classes==i)[0]
            num_train = int(len(inds)*train_test_ratio)
            train_inds.extend(inds[:num_train])
            test_inds.extend(inds[num_train:])
            labels = np.zeros((len(inds), num_classes))
            labels[:,i] = 1.
            y_train.extend(labels[:num_train])
            y_test.extend(labels[num_train:])

        y_train = np.array(y_train)
        y_test - np.array(y_test)
        x_train = np.copy(flux[train_inds])
        x_test = np.copy(flux[test_inds])
    else:
        split_ind = int(train_test_ratio*np.shape(flux)[0])
        x_train = np.copy(flux[:split_ind])
        x_test = np.copy(flux[split_ind:])
        y_test, y_train = [False, False]
        
    # if interpolate:
    #     train_data, time = interpolate_lc(np.concatenate([x_train, x_test]), time)
    #     x_train = train_data[:len(x_train)]
    #     x_test = train_data[len(x_train):]
        
    if resize_arr:
        x_train =  np.resize(x_train, (np.shape(x_train)[0],
                                       np.shape(x_train)[1], 1))
        x_test =  np.resize(x_test, (np.shape(x_test)[0],
                                       np.shape(x_test)[1], 1))
    return x_train, x_test, y_train, y_test, time
    

def get_activations(model, x_test, input_rms = False, rms_test = False):
    '''Returns intermediate activations.'''
    from keras.models import Model
    layer_outputs = [layer.output for layer in model.layers][1:]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    if input_rms:
        activations = activation_model.predict([x_test, rms_test])
    else:
        activations = activation_model.predict(x_test)        
    return activations

def get_bottleneck(model, x_test, input_features=False, features=False,
                   input_rms=False, rms=False):
    from keras.models import Model
    # >> first find all Dense layers
    inds = np.nonzero(['dense' in x.name for x in model.layers])[0]
    
    # >> bottleneck layer is the first Dense layer
    bottleneck_ind = inds[0]
    activation_model = Model(inputs=model.input,
                             outputs=model.layers[bottleneck_ind].output)
    bottleneck = activation_model.predict(x_test)    
    if input_features: # >> concatenate features to bottleneck
        bottleneck = np.concatenate([bottleneck, features], axis=1)
    if input_rms:
        bottleneck = np.concatenate([bottleneck,
                                      np.reshape(rms, (np.shape(rms)[0],1))],
                                        axis=1)    
    
    bottleneck = df.standardize(bottleneck, ax=0)
    return bottleneck

def get_bottleneck_from_activations(model, activations, p, input_features=False, 
                   features=False, input_rms=False, rms=False):
    '''[deprecated 070320]
    Get bottleneck layer, with shape (num light curves, latent dimension)
    Parameters:
        * model : Keras Model()
        * activations : from get_activations()
        * p : parameter set, with p['latent_dim'] = dimension of latent space
        * input_features : bool
        * features : array of features to concatenate with bottleneck, must be
                     given if input_features=True
        * rms : list of RMS must be given if input_rms=True
    '''

    # >> first find all Dense layers
    inds = np.nonzero(['dense' in x.name for x in model.layers])[0]
    
    # >> now check which Dense layers has number of units = latent_dim
    for ind in inds:
        ind = ind - 1 # >> len(activations) = len(model.layers) - 1, since
                      #    activations doesn't include the Input layer
        num_units = np.shape(activations[ind])[1]
        if num_units == p['latent_dim']:
            bottleneck_ind = ind
    
    bottleneck = activations[bottleneck_ind]
    
    if input_features: # >> concatenate features to bottleneck
        bottleneck = np.concatenate([bottleneck, input_features], axis=1)
    if input_rms:
        bottleneck = np.concatenate([bottleneck,
                                      np.reshape(rms, (np.shape(rms)[0],1))],
                                    axis=1)
        
    return bottleneck

# :: mock data ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def gaussian(x, a, b, c):
    '''a = height, b = position of center, c = stdev'''
    import numpy as np
    return a * np.exp(-(x-b)**2 / (2*c**2))

def signal_data(training_size = 10000, test_size = 100, input_dim = 100,
                 time_max = 30., noise_level = 0.0, height = 1., center = 15.,
                 stdev = 0.8, h_factor = 0.2, center_factor = 5.,
                 reshape=True):
    '''Generate training data set with flat light curves and gaussian light
    curves, with variable height, center, noise level as a fraction of gaussian
    height)
    '''

    x = np.empty((training_size + test_size, input_dim))
    # y = np.empty((training_size + test_size))
    y = np.zeros((training_size + test_size, 2))
    l = int(np.shape(x)[0]/2)
    
    # >> no peak data
    x[:l] = np.zeros((l, input_dim))
    # y[:l] = 0.
    y[:l, 0] = 1.
    

    # >> with peak data
    time = np.linspace(0, time_max, input_dim)
    for i in range(l):
        a = height + h_factor*np.random.normal()
        b = center + center_factor*np.random.normal()
        x[l+i] = gaussian(time, a = a, b = b, c = stdev)
    # y[l:] = 1.
    y[l:, 1] = 1.

    # >> add noise
    x += np.random.normal(scale = noise_level, size = np.shape(x))
    
    # >> normalize
    # x = x / np.median(x, axis = 1, keepdims=True) - 1.

    # >> partition training and test datasets
    x_train = np.concatenate((x[:int(training_size/2)], 
                              x[l:-int(test_size/2)]))
    y_train = np.concatenate((y[:int(training_size/2)], 
                              y[l:-int(test_size/2)]))
    x_test = np.concatenate((x[int(training_size/2):l], 
                             x[-int(test_size/2):]))
    y_test = np.concatenate((y[int(training_size/2):l], 
                             y[-int(test_size/2):]))

    if reshape:
        x_train = np.reshape(x_train, (np.shape(x_train)[0],
                                       np.shape(x_train)[1], 1))
        x_test = np.reshape(x_test, (np.shape(x_test)[0],
                                     np.shape(x_test)[1], 1))

    return time, x_train, y_train, x_test, y_test

def no_signal_data(training_size = 10000, test_size = 100, input_dim = 100,
                   noise_level = 0., min0max1=True, reshape=False):
    import numpy as np

    x = np.empty((training_size + test_size, input_dim))
    y = np.empty((training_size + test_size))
    l = int(np.shape(x)[0]/2)
    
    # >> no peak data
    if min0max1:
        x = np.zeros(np.shape(x))
    else:
        x = np.ones(np.shape(x))
    y = 0.

    # >> add noise
    x += np.random.normal(scale = noise_level, size = np.shape(x))

    # >> partition training and test datasets
    x_train = np.concatenate((x[:int(training_size/2)], 
                              x[l:-int(test_size/2)]))
    y_train = np.concatenate((y[:int(training_size/2)], 
                              y[l:-int(test_size/2)]))
    x_test = np.concatenate((x[int(training_size/2):l], 
                             x[-int(test_size/2):]))
    y_test = np.concatenate((y[int(training_size/2):l], 
                             y[-int(test_size/2):]))

    if reshape:
        x_train = np.reshape(x_train, (np.shape(x_train)[0],
                                       np.shape(x_train)[1], 1))
        x_test = np.reshape(x_test, (np.shape(x_test)[0],
                                     np.shape(x_test)[1], 1))
    
    return x_train, y_train, x_test, y_test
        
# :: pull files with astroquery :::::::::::::::::::::::::::::::::::::::::::::::
# adapted from pipeline.py

def get_lc(ticid, out='./', DEBUG_INTERP=False, download_fits=True,
           prefix=''):
    '''input a ticid, returns light curve'''
    from astroquery.mast import Observations
    from astropy.io import fits
    import fnmatch
    
    # >> download fits file
    targ = 'TIC ' + str(int(ticid))
    
    if download_fits:
        try: 
            obs_table = Observations.query_object(targ, radius=".02 deg")
            
            # >> find all data products for ticid
            data_products_by_obs = Observations.get_product_list(obs_table[0:2])
            
            filter_products = \
                Observations.filter_products(data_products_by_obs,
                                             dataproduct_type = 'timeseries',
                                             description = 'Light curves',
                                             extension='fits')
                
            # >> download fits file
            manifest = \
                Observations.download_products(filter_products,
                                               download_dir = out)
        except (ConnectionError, OSError, TimeoutError):
            print(targ + "could not be accessed due to an error")
        
    # >> find fits file
    fnames_all = os.listdir(out)
    fname = fnmatch.filter(fnames_all, '*'+str(int(ticid))+'*fits*')[0]
    
    # >> read fits file
    f = fits.open(out+fname)
    time = f[1].data['TIME']
    flux = f[1].data['PDCSAP_FLUX']
    # time = fits.getdata(out+fname, 1)['TIME']
    # flux = fits.getdata(out+fname, 1)['PDCSAP_FLUX']
    # flux = interpolate_lc(flux, time, DEBUG_INTERP=DEBUG_INTERP,
    #                       output_dir=out, prefix=prefix)
    return time, flux
    

def get_fits_files(mypath, target_list):
    '''target_list from tess.txt generated in get_target_list'''
    from astroquery.mast import Observations
    for ticid in target_list:
        targ = 'TIC ' + str(int(ticid))
        try: 
            obs_table = Observations.query_object(targ, radius=".02 deg")
            data_products_by_obs = Observations.get_product_list(obs_table[0:2])
            
            filter_products = \
                Observations.filter_products(data_products_by_obs,
                                             dataproduct_type = 'timeseries',
                                             description = 'Light curves',
                                             extension='fits')
            manifest = \
                Observations.download_products(filter_products,
                                               download_dir = mypath)
        except (ConnectionError, OSError, TimeoutError):
            print(targ + "could not be accessed due to an error")
            
def get_target_list(sector_num, output_dir='./'):
    '''Get TICID from sector_num (given as int)'''
    from astroquery.mast import Observations
    from astroquery.mast import Tesscut
    obs_table = Observations.query_criteria(obs_collection='TESS',
                                            dataproduct_type='TIMESERIES',
                                            sequence_number=sector_num)
    
    print(obs_table)
    target_list = np.copy(obs_table['target_name'])
    
    cam_list = []
    ccd_list = []
    for target in target_list:
        obj_name = 'TIC ' + target
        try:
            obj_table = Tesscut.get_sectors(obj_name)
            ind = np.nonzero(obj_table['sector']==sector_num)
            cam_list.append(obj_table['camera'][ind][0])
            ccd_list.append(obj_table['ccd'][ind][0])
            
            with open(output_dir+'tess-s00'+str(sector_num)+'.txt', 'a') as f:
                f.write(obj_name + ' {} {} {}\n'.format(obj_table['sector'][ind][0],
                                                        obj_table['camera'][ind][0],
                                                        obj_table['ccd'][ind][0]))
        except:
            print('failed! '+target)
            with open(output_dir+'tess-s00'+str(sector_num)+'skip.txt', 'a') as f:
                f.write(obj_name+'\n')
                        
    # >> also save .txt files for each camera and ccd
    cam_list = np.array(cam_list)
    ccd_list = np.array(ccd_list)
    print(np.unique(cam_list))
    print(np.unique(ccd_list))
    for cam in range(4):
        for ccd in range(4):
            inds = np.nonzero( (cam_list==cam) * (cam_list==ccd) )[0]
            with open(output_dir+'tess-s00'+str(sector_num)+'-'+str(cam)+'-'+\
                      str(ccd)+'.txt', 'a') as f:
                for i in inds:
                    f.write(target_list[i]+'\n')
        
    return target_list
    
            

# move all fits files into one folder?

            

# def get_lc_file_and_data(temp_path, target_list):
#     from astroquery.mast import Observations
#     prefix = 'tess2019357164649-s0020-'
#     suffix = '-0165-s_lc.fits'
#     for ticid in target_list:
#         try:
#             targ = ticid.zfill(16)
#             fname = prefix+ targ + suffix
#             os.system('curl -C - -L -o ' + temp_path + fname +\
#                       ' https://mast.stsci.edu/api/v0.1/Download/file/'+\
#                           '?uri=mast:TESS/product/' + fname) 

                  
#         #then delete all downloads in the folder, no matter what type

#         except (ConnectionError, OSError, TimeoutError):
#             print(targ + "could not be accessed due to an error")
#             i1 = 0
#             time1 = 0
    
#     return time1, i1



# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
# def autoencoder_dual_input1(x_train, x_test, rms_train, rms_test, params,
#                             supervised=False, y_train = False, y_test=False,
#                             num_classes = False):
#     '''Adapted from: https://www.pyimagesearch.com/2019/02/04/keras-multiple-
#     inputs-and-mixed-data/'''
#     from keras.layers import concatenate
#     from keras.layers import Conv1D, Dense, Reshape
#     from keras.models import Model
#     from keras import optimizers
#     import keras.metrics
#     input_dim = np.shape(x_train)[1]
#     # >> create the MLP and autoencoder models
#     mlp = create_mlp(np.shape(rms_train)[1])
#     encoded = encoder1(x_train, params)
#     # autoencoder = create_conv_layers(x_train, params)

#     # x = Reshape((1,1))(mlp.output)
#     # x = concatenate([autoencoder.output, Reshape((1,1))(mlp.output)], axis = 1)
#     # x = Reshape((input_dim,))(autoencoder.output)
#     x = concatenate([mlp.output,
#                      Reshape((input_dim,))(encoded.output)], axis = 1)
#     # x = Reshape((input_dim+1,))(x)
    
    
#     # x = Dense(4, activation='relu')(combinedInput)
#     # x = Dense(1, activation='linear')(x)
#     x = Dense(input_dim, activation='relu')(x)
#     x = Reshape((input_dim,1))(x)
#     # x = Conv1D(1, params['kernel_size'], activation=params['last_activation'],
#     #            padding='same')(combinedInput)
#     if supervised:
#         x = Dense(num_classes,activation='softmax')(encoded.output)
#         model = Model(inputs=[encoded.input, mlp.input], output=x)
#         print(model.summary())
#     else:
#         model = Model(inputs=[encoded.input, mlp.input], outputs=x)
#         print(model.summary())
    
#     # !! find a better way to do this
#     if params['optimizer'] == 'adam':
#         opt = optimizers.adam(lr = params['lr'], 
#                               decay=params['lr']/params['epochs'])
#     elif params['optimizer'] == 'adadelta':
#         opt = optimizers.adadelta(lr = params['lr'])
        
#     model.compile(optimizer=opt, loss=params['losses'],
#                   metrics=['accuracy', keras.metrics.Precision(),
#                            keras.metrics.Recall()])
#     history = model.fit([x_train, rms_train], x_train, epochs=params['epochs'],
#                         batch_size=params['batch_size'], shuffle=True,
#                         validation_data=([x_test, rms_test], x_test))
#     return history, model

# def autoencoder_dual_input2(x_train, x_test, rms_train, rms_test, params,
#                             supervised=False, y_train=False, y_test=False,
#                             num_classes=False):
#     '''Adapted from: https://stackoverflow.com/questions/52435274/how-to-use-
#     keras-merge-layer-for-autoencoder-with-two-ouput'''
#     from keras.layers import concatenate
#     from keras.layers import Dense
#     from keras import optimizers
#     import keras.metrics
#     from keras.models import Model
    
#     # >> create the MLP and encoder models
#     mlp = create_mlp(np.shape(rms_train)[1])
#     encoded = encoder(x_train, params)

#     # >> shared representation layer
#     shared_input = concatenate([mlp.output,encoded.output])
#     shared_output = Dense(params['latent_dim'], activation='relu')(shared_input)
#     if supervised:
#         x = Dense(num_classes,activation='softmax')(shared_output)
#         model = Model(inputs=[encoded.input, mlp.input], output=x)
#         print(model.summary())
#     else:
#         decoded = decoder(x_train, shared_output, params)
#         model = Model(inputs=[encoded.input, mlp.input], outputs=decoded)
    
#     # >> get model
#     # model = Model(inputs=[encoded.input, mlp.input], outputs=decoded.output)
    
    
#     # !! find a better way to do this
#     if params['optimizer'] == 'adam':
#         opt = optimizers.adam(lr = params['lr'], 
#                               decay=params['lr']/params['epochs'])
#     elif params['optimizer'] == 'adadelta':
#         opt = optimizers.adadelta(lr = params['lr'])
        
#     model.compile(optimizer=opt, loss=params['losses'],
#                   metrics=['accuracy', keras.metrics.Precision(),
#                            keras.metrics.Recall()])
#     history = model.fit([x_train, rms_train], x_train, epochs=params['epochs'],
#                         batch_size=params['batch_size'], shuffle=True,
#                         validation_data=([x_test, rms_test], x_test))
#     return history, model
    
    
    
    # def encoder2(x_train, params):
#     '''https://towardsdatascience.com/applied-deep-learning-part-4
#     -convolutional-neural-networks-584bc134c1e2
#     stacked'''
#     from keras.layers import Input,Conv1D,MaxPooling1D,Dropout,Flatten,Dense
#     from keras.models import Model
    
#     input_dim = np.shape(x_train)[1]
#     num_iter = int((params['num_conv_layers'] - 1)/2)
    
#     input_img = Input(shape = (input_dim, 1))
#     x = Conv1D(params['num_filters'][0], params['kernel_size'],
#                activation=params['activation'], padding='same')(input_img)
#     for i in range(num_iter):
#         x = MaxPooling1D(2, padding='same')(x)
#         x = Dropout(params['dropout'])(x)
#         x = Conv1D(1, 1, activation='relu')(x)
#         # x = MaxPooling1D([params['num_filters'][i]],
#         #                  data_format='channels_first')(x)
#         x = Conv1D(params['num_filters'][1+i], params['kernel_size'],
#                    activation=params['activation'], padding='same')(x)
#         x = Conv1D(params['num_filters'][1+i], params['kernel_size'],
#                    activation=params['activation'], padding='same')(x)
#     x = MaxPooling1D([params['num_filters'][i]], 
#                      data_format='channels_first')(x)
#     x = Flatten()(x)
#     encoded = Dense(params['latent_dim'], activation=params['activation'])(x)
#     # return encoded
#     encoder = Model(input_img, encoded)
#     return encoder
    
    
    # def create_conv_layers(x_train, params, supervised = False):
#     from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
#     from keras.layers import Reshape, Dense, Flatten, Dropout
#     from keras.models import Model

#     input_dim = np.shape(x_train)[1]
#     num_iter = int((params['num_conv_layers'] - 1)/2)
    
#     input_img = Input(shape = (input_dim, 1))
#     x = Conv1D(params['num_filters'][0], params['kernel_size'],
#                 activation=params['activation'], padding='same')(input_img)
#     for i in range(num_iter):
#         x = MaxPooling1D(2, padding='same')(x)
#         x = Dropout(params['dropout'])(x)
#         x = MaxPooling1D([params['num_filters'][i]],
#                           data_format='channels_first')(x)
#         x = Conv1D(params['num_filters'][1+i], params['kernel_size'],
#                     activation=params['activation'], padding='same')(x)
#     x = MaxPooling1D([params['num_filters'][i]], 
#                       data_format='channels_first')(x)
#     x = Flatten()(x)
#     encoded = Dense(params['latent_dim'], activation=params['activation'])(x)

#     x = Dense(int(input_dim/(2**(i+1))))(encoded)
#     x = Reshape((int(input_dim/(2**(i+1))), 1))(x)
#     for i in range(num_iter):
#         x = Conv1D(params['num_filters'][num_iter+1], params['kernel_size'],
#                     activation=params['activation'], padding='same')(x)
#         x = UpSampling1D(2)(x)
#         x = Dropout(params['dropout'])(x)
#         x = MaxPooling1D([params['num_filters'][num_iter+1]],
#                           data_format='channels_first')(x)
#     decoded = Conv1D(1, params['kernel_size'],
#                       activation=params['last_activation'], padding='same')(x)
    
#     if supervised:
#         model = Model(input_img, encoded)
#         print(model.summary())
          
#     else:
#         model = Model(input_img, decoded)
#         print(model.summary())
    
#     return model
    
    # if inputs_before_after:
    #     orbit_gap_start = np.nonzero(time < orbit_gap[0])[0][-1]
    #     orbit_gap_end = np.nonzero(time > orbit_gap[1])[0][0]
    #     x_train_0 = x_train[:,:orbit_gap_start]
    #     x_train_1 = x_train[:,orbit_gap_end:]
    #     x_test_0 = x_test[:,:orbit_gap_start]
    #     x_test_1 = x_test[:,orbit_gap_end:]
    #     y_train_0, y_train_1 = [np.copy(y_train), np.copy(y_train)]
    #     y_test_0, y_test_1 = [np.copy(y_test), np.copy(y_test)]
    #     # x_train_0 = x_train[]
    #     return x_train_0, x_train_1, x_test_0, x_test_1, y_train_0, y_train_1,\
    #         y_test_0, y_test_1
    # else:
    
# def decoder(x_train, bottleneck, params):
#     '''042820
#     https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_conv_autoencoder_mnist.ipynb
#     '''
#     from keras.layers import Dense  ,Reshape, Conv1D, UpSampling1D, Dropout
#     from keras.layers import MaxPooling1D
#     input_dim = np.shape(x_train)[1]
#     num_iter = int(params['num_conv_layers']/2)
    
#     # x = Dense(int(input_dim/(2**(num_iter))))(bottleneck)
#     x = Dense(int(input_dim/(2**num_iter) * \
#                   params['num_filters'][num_iter-1]))(bottleneck)
#     x = Reshape((int(input_dim/(2**(num_iter))),
#                  params['num_filters'][num_iter-1]))(x)
#     for i in range(num_iter):
#         x = UpSampling1D(2)(x)
#         # !!
#         x = Conv1D(params['num_filters'][num_iter+i],
#                    params['kernel_size'][num_iter+i],
#                    activation=params['activation'], padding='same')(x)
#     decoded = Conv1D(1, params['kernel_size'][num_iter+i],
#                      activation=params['last_activation'], padding='same')(x)
#     return decoded
    
    
# def encoder(x_train,params):
#     '''https://github.com/gabrieleilertsen/hdrcnn/blob/master/network.py'''
#     from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten
#     from keras.layers import Dense
#     from keras.models import Model
    
#     input_dim = np.shape(x_train)[1]
#     # num_iter = int((params['num_conv_layers'] - 1)/2)
#     # num_iter = int(params['num_conv_layers']/2)
#     num_iter = int(params['num_conv_layers']/2)
    
#     input_img = Input(shape = (input_dim, 1))
#     for i in range(num_iter):
#         if i == 0:
#             x = Conv1D(params['num_filters'][i], params['kernel_size'][i],
#                    activation=params['activation'], padding='same')(input_img)
#         else:
#             x = Conv1D(params['num_filters'][i], params['kernel_size'][i],
#                        activation=params['activation'], padding='same')(x)
#         x = MaxPooling1D(2, padding='same')(x)
#     x = Flatten()(x)
#     # x = Dense(int(input_dim/(2**num_iter) * params['num_filters'][i]),
#     #           activation=params['activation'])(x)
#     encoded = Dense(params['latent_dim'], activation=params['activation'])(x)
#     encoder = Model(input_img, encoded)

#     return encoder

# def encoder(x_train,params):
#     '''https://github.com/gabrieleilertsen/hdrcnn/blob/master/network.py'''
#     from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten
#     from keras.layers import Dense
#     from keras.models import Model
    
#     input_dim = np.shape(x_train)[1]
#     # num_iter = int((params['num_conv_layers'] - 1)/2)
#     # num_iter = int(params['num_conv_layers']/2)
#     num_iter = int(params['num_conv_layers']/2)
    
#     input_img = Input(shape = (input_dim, 1))
#     for i in range(num_iter):
#         if i == 0:
#             x = Conv1D(params['num_filters'][i], params['kernel_size'][i],
#                    activation=params['activation'], padding='same')(input_img)
#         else:
#             x = Conv1D(params['num_filters'][i], params['kernel_size'][i],
#                        activation=params['activation'], padding='same')(x)
#         x = Conv1D(params['num_filters'][i], params['kernel_size'][i],
#                    activation=params['activation'], padding='same')(x)
#         x = MaxPooling1D(2, padding='same')(x)
#     x = Flatten()(x)
#     # x = Dense(int(input_dim/(2**num_iter) * params['num_filters'][i]),
#     #           activation=params['activation'])(x)
#     encoded = Dense(params['latent_dim'], activation=params['activation'])(x)
#     encoder = Model(input_img, encoded)

#     return encoder


# def encoder(x_train, params):
#     from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten
#     from keras.layers import Dense
#     from keras.models import Model
    
#     input_dim = np.shape(x_train)[1]
#     # num_iter = int((params['num_conv_layers'] - 1)/2)
#     # num_iter = int(params['num_conv_layers']/2)
#     num_iter = int(params['num_conv_layers']/2)
    
#     input_img = Input(shape = (input_dim, 1))
#     # x = Conv1D(params['num_filters'][0], params['kernel_size'][0],
#     #            activation=params['activation'], padding='same')(input_img)
#     for i in range(num_iter):
#         if i == 0:
#             x = Conv1D(params['num_filters'][i], params['kernel_size'][i],
#                    activation=params['activation'], padding='same')(input_img)
#         else:
#             x = Conv1D(params['num_filters'][i], params['kernel_size'][i],
#                        activation=params['activation'], padding='same')(x)
#         x = MaxPooling1D(2, padding='same')(x)
#     x = Flatten()(x)
#     # x = Dense(int(input_dim/(2**num_iter) * params['num_filters'][i]),
#     #           activation=params['activation'])(x)
#     encoded = Dense(params['latent_dim'], activation=params['activation'])(x)
#     encoder = Model(input_img, encoded)

#     return encoder  
    
# def decoder(x_train, bottleneck, params):
#     from keras.layers import Dense, Reshape, Conv1D, UpSampling1D, Dropout
#     from keras.layers import MaxPooling1D, Lambda
#     from keras import backend as K
#     input_dim = np.shape(x_train)[1]
#     num_iter = int(params['num_conv_layers']/2)
    
#     x = Dense(int(input_dim/(2**(num_iter))))(bottleneck)
#     x = Reshape((int(input_dim/(2**(num_iter))), 1))(x)
#     for i in range(num_iter):
#         x = Conv1D(params['num_filters'][num_iter+i],
#                     params['kernel_size'][num_iter+i],
#                     activation=params['activation'], padding='same')(x)
#         x = UpSampling1D(2)(x)
#         x = Dropout(params['dropout'])(x)
#         x = MaxPooling1D([params['num_filters'][num_iter+i]],
#                           data_format='channels_first')(x)


#     decoded = Conv1D(1, params['kernel_size'][num_iter+1],
#                       activation=params['last_activation'], padding='same')(x)
#     return decoded
    
    
# def encoder1(x_train, params):
#     '''https://machinelearningmastery.com/introduction-to-1x1-convolutions-to
#     -reduce-the-complexity-of-convolutional-neural-networks/
#     Using convolutions over channels to downsample feature maps'''
#     from keras.layers import Input,Conv1D,MaxPooling1D,Dropout,Flatten,Dense
#     from keras.models import Model
    
#     input_dim = np.shape(x_train)[1]
#     num_iter = int((params['num_conv_layers'] - 1)/2)
    
#     input_img = Input(shape = (input_dim, 1))
#     x = Conv1D(params['num_filters'][0], params['kernel_size'],
#                activation=params['activation'], padding='same')(input_img)
#     for i in range(num_iter):
#         x = MaxPooling1D(2, padding='same')(x)
#         x = Dropout(params['dropout'])(x)
#         x = Conv1D(1, 1, activation='relu')(x)
#         # x = MaxPooling1D([params['num_filters'][i]],
#         #                  data_format='channels_first')(x)
#         x = Conv1D(params['num_filters'][1+i], params['kernel_size'],
#                    activation=params['activation'], padding='same')(x)
#     x = MaxPooling1D([params['num_filters'][i]], 
#                      data_format='channels_first')(x)
#     x = Flatten()(x)
#     encoded = Dense(params['latent_dim'], activation=params['activation'])(x)
#     # return encoded
#     encoder = Model(input_img, encoded)
#     return encoder
    
    # def normalize1(x):
#     xmin = np.min(x, axis=1, keepdims=True)
#     x = x - xmin
#     xmax = np.max(x, axis=1, keepdims=True)
#     x = x * 2 / xmax
#     x = x - 1.
#     # scale = 2/(xmax-xmin)
#     # offset = (xmin - xmax)/(xmax-xmin)
#     # x = x*scale + offset
#     return x
    
    
# def conv_autoencoder(x_train, y_train, x_test, y_test, params, input_rms=False,
#                 rms_train=False,
#                 rms_test=False, supervised = False, num_classes=False, 
#                 split_lc=False,
#                 orbit_gap=[8794, 8795]):
#     '''If supervised = True, must provide y_train, y_test, num_classes'''
#     from keras.models import Model
#     from keras.layers import Dense, concatenate

#     # -- encoding -------------------------------------------------------------
#     if split_lc:
#         x_train_0,x_train_1 = x_train[:,:orbit_gap[0]],x_train[:,orbit_gap[1]:]
#         x_test_0,x_test_1 = x_test[:,:orbit_gap[0]],x_test[:,orbit_gap[1]:]
#         encoded = encoder_split([x_train_0, x_train_1], params)
#     else:
#         encoded = encoder(x_train, params)
    
#     if input_rms:
#         mlp = create_mlp(np.shape(rms_train)[1])
#         shared_input = concatenate([mlp.output,encoded.output])
#         shared_output = Dense(params['latent_dim'],
#                               activation='relu')(shared_input)

#     # -- supervised mode: softmax --------------------------------------------------
#     if supervised:
#         if input_rms:
#             x = Dense(num_classes, activation='softmax')(shared_output)
#             model = Model(inputs=[encoded.input,mlp.input], outputs=x)
#         else:
#             x = Dense(int(num_classes),
#                   activation='softmax')(encoded.output)
#             model = Model(encoded.input, x)
#         model.summary()
        
        
#     else: # -- decoding -------------------------------------------------------
#         if split_lc:
#             decoded = decoder_split(x_train, encoded.output, params)
#         else:
#             decoded = decoder(x_train, encoded.output, params)
#         model = Model(encoded.input, decoded)
#         print(model.summary())
        
#     # -- compile model --------------------------------------------------------
#     compile_model(model, params)

#     # -- train model ----------------------------------------------------------
    
#     if supervised and input_rms:
#         history = model.fit([x_train, rms_train], y_train,
#                             epochs=params['epochs'],
#                             batch_size=params['batch_size'], shuffle=True)
#     elif supervised and not input_rms:
#         history = model.fit(x_train, y_train, epochs=params['epochs'],
#                             batch_size=params['batch_size'], shuffle=True)
#     elif input_rms and not supervised:
#         history = model.fit([x_train, rms_train], x_train,
#                             epochs=params['epochs'],
#                             batch_size=params['batch_size'], shuffle=True)
#     else:
#         history = model.fit(x_train, x_train, epochs=params['epochs'],
#                             batch_size=params['batch_size'], shuffle=True,
#                             validation_data=(x_test, x_test))
        
#     return history, model

# def autoencoder1(x_train, y_train, x_test, y_test, params):
#     from keras.models import Model
#     encoded = encoder(x_train, params)
#     decoded = decoder(x_train, encoded.output, params)
#     model = Model(encoded.input, decoded)
#     model.summary()
    
#     compile_model(model, params)
    
#     history = model.fit(x_train, x_train, epochs=params['epochs'],
#                         batch_size=params['batch_size'], shuffle=True,
#                         validation_data=(x_test,x_test))
#     return history, model
    
# def simple_encoder(x_train, params):
#     from keras.layers import Input, Dense, Flatten
#     from keras.models import Model
#     input_dim = np.shape(x_train)[1]
#     input_img = Input(shape = (input_dim,1))
#     x = Flatten()(input_img)
#     encoded = Dense(params['latent_dim'], activation=params['activation'])(x)
#     encoder = Model(input_img, encoded)
#     return encoder

# def simple_decoder(x_train, bottleneck, params):
#     from keras.layers import Dense, Reshape
#     input_dim = np.shape(x_train)[1]
#     x = Dense(input_dim, activation='sigmoid')(bottleneck)
#     decoded = Reshape((input_dim, 1))(x)
#     return decoded
    # def normalize(flux, ax=1):
# #     medians = np.median(flux, axis = ax, keepdims=True)
# #     flux = flux / medians - 1.
# #     return flux

        
# def interpolate_all(flux, time, flux_err=False, interp_tol=20./(24*60),
#                     num_sigma=10, DEBUG_INTERP=False, output_dir='./',
#                     prefix=''):
#     '''Interpolates each light curves in flux array.'''
    
#     flux_interp = []
#     for i in flux:
#         i_interp = interpolate_lc(i, time, flux_err=flux_err,
#                                   interp_tol=interp_tol,
#                                   num_sigma=num_sigma,
#                                   DEBUG_INTERP=DEBUG_INTERP,
#                                   output_dir=output_dir, prefix=prefix)
#         flux_interp.append(i_interp)
    
#     return np.array(flux_interp)

# def interpolate_lc(i, time, flux_err=False, interp_tol=20./(24*60),
#                    num_sigma=10, DEBUG_INTERP=False,
#                    output_dir='./', prefix=''):
#     '''Interpolation for one light curve. Linearly interpolates nan gaps less
#     than 20 minutes long. Spline interpolates nan gaps more than 20 minutes
#     long (and shorter than orbit gap)'''
#     from astropy.stats import SigmaClip
#     from scipy import interpolate
    
#     # >> plot original light curve
#     if DEBUG_INTERP:
#         fig, ax = plt.subplots(5, 1, figsize=(8, 3*5))
#         ax[0].plot(time, i, '.k', markersize=2)
#         ax[0].set_title('original')
    
#     # -- sigma clip ----------------------------------------------------------
#     sigclip = SigmaClip(sigma=num_sigma, maxiters=None, cenfunc='median')
#     clipped_inds = np.nonzero(np.ma.getmask(sigclip(i, masked=True)))
#     if np.shape(clipped_inds)[1] != 0:
#         i[clipped_inds] = np.nan
#     if DEBUG_INTERP:
#         ax[1].plot(time, i, '.k', markersize=2)
#         ax[1].set_title('clipped')
    
#     # >> find nan windows
#     n = np.shape(i)[0]
#     loc_run_start = np.empty(n, dtype=bool)
#     loc_run_start[0] = True
#     np.not_equal(np.isnan(i)[:-1], np.isnan(i)[1:], out=loc_run_start[1:])
#     run_starts = np.nonzero(loc_run_start)[0]

#     # >> find nan window lengths
#     run_lengths = np.diff(np.append(run_starts, n))
#     tdim = time[1]-time[0]
    
#     # -- interpolate small nan gaps ------------------------------------------
#     interp_gaps = np.nonzero((run_lengths * tdim <= interp_tol) * \
#                              np.isnan(i[run_starts]))
#     interp_inds = run_starts[interp_gaps]
#     interp_lens = run_lengths[interp_gaps]

#     i_interp = np.copy(i)
#     for a in range(np.shape(interp_inds)[0]):
#         start_ind = interp_inds[a]
#         end_ind = interp_inds[a] + interp_lens[a]
#         i_interp[start_ind:end_ind] = np.interp(time[start_ind:end_ind],
#                                                 time[np.nonzero(~np.isnan(i))],
#                                                 i[np.nonzero(~np.isnan(i))])
#     i = i_interp
#     if DEBUG_INTERP:
#         ax[2].plot(time, i, '.k', markersize=2)
#         ax[2].set_title('interpolated')
    
#     # -- spline interpolate large nan gaps -----------------------------------
#     # >> fit spline to non-nan points
#     num_inds = np.nonzero( (~np.isnan(i)) * (~np.isnan(time)) )[0]
#     ius = interpolate.InterpolatedUnivariateSpline(time[num_inds], i[num_inds])
    
#     # >> new time array (take out orbit gap)
#     orbit_gap_start = num_inds[ np.argmax(np.diff(time[num_inds])) ]
#     orbit_gap_end = num_inds[ orbit_gap_start+1 ]
#     orbit_gap_len = orbit_gap_end - orbit_gap_start
#     # orbit_gap_len = (time[num_inds][orbit_gap_ind]-\
#     #                  time[num_inds][orbit_gap_ind+1]) * tdim
#     t_spl = np.copy(time)
#     t_spl = np.delete(t_spl, range(num_inds[-1], len(t_spl)))
#     t_spl = np.delete(t_spl, range(orbit_gap_start, orbit_gap_end))
#     t_spl = np.delete(t_spl, range(num_inds[0]))
#     # t_spl = np.copy(time[ np.nonzero(~np.isnan(time)) ])
#     # orbit_gap_inds = np.nonzero((t_spl > time[num_inds][orbit_gap_ind]) *\
#     #                             (t_spl < time[num_inds][orbit_gap_ind+1]))
#     # t_spl = np.delete(t_spl, orbit_gap_inds)
#     # t_spl = np.delete(t_spl, range(-num_inds[0][-1], ))
#     # t_spl = np.delete(t_spl, range(num_inds[0][0]))
    
    
#     # t1 = np.concatenate([np.linspace(np.nanmin(time[num_inds]),
#     #                                  time[num_inds][orbit_gap_ind],
#     #                                  num_inds[0][orbit_gap_ind]),
#     #                      np.linspace(time[num_inds][orbit_gap_ind+1],
#     #                                  np.nanmax(time[num_inds]),
#     #                                  len(time)-num_inds[0][orbit_gap_ind+1])],
#     #                     axis=0)
    
#     # >> spline fit for new time array
#     i_spl = ius(t_spl)
    
#     if DEBUG_INTERP:
#         ax[3].plot(t_spl, i_spl, '.')
#         ax[3].set_title('spline') 
    
#     # >> find nan gaps to spline interpolate over
#     interp_gaps = np.nonzero((run_lengths * tdim > interp_tol) * \
#                               np.isnan(i[run_starts]) * \
#                               (((run_starts > orbit_gap_start) * \
#                                 (run_starts < orbit_gap_end)) == False))       
#     interp_inds = run_starts[interp_gaps]
#     interp_lens = run_lengths[interp_gaps]  
    
#     # >> spline interpolate nan gaps
#     i_interp = np.copy(i)
#     for a in range(np.shape(interp_inds)[0]):
#         start_ind = interp_inds[a]
#         end_ind   = interp_inds[a] + interp_lens[a] - 1

#         if not np.isnan(time[start_ind]):
#             start_ind_spl = np.argmin(np.abs(t_spl - time[start_ind]))
#             end_ind_spl = start_ind_spl + (end_ind-start_ind)
#         else:
#             end_ind_spl = np.argmin(np.abs(t_spl - time[end_ind]))
#             start_ind_spl = end_ind_spl - (end_ind-start_ind)
#         i_interp[start_ind:end_ind] = i_spl[start_ind_spl:end_ind_spl]
        
#     if DEBUG_INTERP:
#         ax[4].plot(time, i_interp, '.k', markersize=2)
#         ax[4].set_title('spline interpolate')
#         fig.tight_layout()
#         fig.savefig(output_dir + prefix + 'interpolate_debug.png',
#                     bbox_inches='tight')
#         plt.close(fig)
        
#     return i_interp
    
# def nan_mask(flux, time, flux_err=False, DEBUG=False, debug_ind=1042,
#              ticid=False, output_dir='./', prefix='', tol1=0.05, tol2=0.1):
#     '''Apply nan mask to flux and time array.
#     Returns masked, homogenous flux and time array.
#     If there are only a few (less than tol1 % light curves) light curves that
#     contribute (more than tol2 % data points are NaNs) to NaN mask, then will
#     remove those light curves.'''

#     mask = np.nonzero(np.prod(~np.isnan(flux), axis = 0) == False)
    
#     # >> plot histogram of number of data points thrown out
#     num_masked = []
#     num_nan = []
#     for lc in flux:
#         num_inds = np.nonzero( ~np.isnan(lc) )
#         num_masked.append( len( np.intersect1d(num_inds, mask) ) )
#         num_nan.append( len(num_inds) )
#     plt.figure()
#     plt.hist(num_masked, bins=50)
#     plt.ylabel('number of light curves')
#     plt.xlabel('number of data points masked')
#     plt.savefig(output_dir + 'nan_mask.png')
#     plt.close()
    
#     # >> debugging plots
#     if DEBUG:
#         fig, ax = plt.subplots()
#         ax.plot(time, flux[debug_ind], '.k', markersize=2)
#         ax.set_title('removed orbit gap')
#         fig.tight_layout()
#         fig.savefig(output_dir + prefix + 'nanmask_debug.png',
#                     bbox_inches='tight')
#         plt.close(fig) 
        
#         # >> plot nan-y light curves
#         sorted_inds = np.argsort(num_masked)
#         for k in range(2): # >> plot top and lowest
#             fig, ax = plt.subplots(nrows=10, figsize=(8, 3*10))
#             for i in range(10):
#                 if k == 0:
#                     ind = sorted_inds[i]
#                 else:
#                     ind = sorted_inds[-i-1]
#                 ax[i].plot(time, flux[ind], '.k', markersize=2)
#                 pl.ticid_label(ax[i], ticid[ind], title=True)
#                 num_nans = np.count_nonzero(np.isnan(flux[ind]))
#                 ax[i].text(0.98, 0.98, 'Num NaNs: '+str(num_nans)+\
#                            '\nNum masked: '+str(num_masked[ind]),
#                            transform=ax[i].transAxes,
#                            horizontalalignment='right',
#                            verticalalignment='top', fontsize='xx-small')
#             if k == 0:
#                 fig.savefig(output_dir + prefix + 'nanmask_top.png',
#                             bbox_inches='tight')
#             else:
#                 fig.savefig(output_dir + prefix + 'nanmask_low.png',
#                             bbox_inches='tight')
       
#     # >> check if only a few light curves contribute to NaN mask
#     num_nan = np.array(num_nan)
#     worst_inds = np.nonzero( num_nan > tol2 )
#     if len(worst_inds[0]) < tol1 * len(flux): # >> only a few bad light curves
#         np.delete(flux, worst_inds, 0)
        
#         # >> and calculate new mask
#         mask = np.nonzero(np.prod(~np.isnan(flux), axis = 0) == False)    
        
#         # >> make new histogram
#         num_masked = []
#         for lc in flux:
#             num_inds = np.nonzero( ~np.isnan(lc) )
#             num_masked.append( len( np.intersect1d(num_inds, mask) ) )
#         plt.figure()
#         plt.hist(num_masked, bins=50)
#         plt.ylabel('number of light curves')
#         plt.xlabel('number of data points masked')
#         plt.savefig(output_dir + 'nan_mask_new.png')
#         plt.close()
        
#     # >> apply NaN mask
#     time = np.delete(time, mask)
#     flux = np.delete(flux, mask, 1)
    
#     if type(flux_err) != bool:
#         flux_err = np.delete(flux_err, mask, 1)
#         return flux, time, flux_err
#     else:
#         return flux, time
