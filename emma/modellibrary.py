# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# 
# 2020-05-26 - modellibrary.py
# Keras novelty detection in TESS dataset and data visualization
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
# Data visualization
# * diagnostic_plots : runs all plots
# 
# Helper functions
# * ticid_label
#
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 

import os
import pdb
import matplotlib.pyplot as plt
import numpy as np

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

def diagnostic_plots(history, model, p, output_dir, 
                     x, x_train, x_test, x_predict, sharey=False, prefix='',
                     mock_data=False, ticid_train=False, ticid_test=False,
                     supervised=False, y_true=False, y_predict=False,
                     y_train=False, y_test=False,
                     flux_test=False, flux_train=False, time=False,
                     rms_train=False, rms_test = False, input_rms = False,
                     inds = [0,1,2,3,4,5,6,7,-1,-2,-3,-4,-5,-6,-7],
                     intermed_inds = [6,0],
                     input_bottle_inds = [0,1,2,-6,-7],
                     addend = 1., feature_vector=False, percentage=False,
                     plot_epoch = True,
                     plot_in_out = True,
                     plot_in_bottle_out=True,
                     plot_latent_test = False,
                     plot_latent_train = False,
                     plot_kernel=False,
                     plot_intermed_act=False,
                     plot_clustering=False,
                     make_movie = False,
                     plot_lof_test=True,
                     plot_lof_train=True):
    '''Produces all plots.
    Parameters:
        * history : Keras model.history
        * model : Keras Model()
        * p : parameter set given as a dictionary, e.g. {'latent_dim': 21, ...}
        * outout_dir : directory to save plots in
        * x : time array
        * x_train : 
        '''

    # !! TODO: change supervised inputs to just y_train, y_test
    plt.rcParams.update(plt.rcParamsDefault)
    activations = get_activations(model, x_test, rms_test = rms_test,
                                  input_rms=input_rms)
    
    # >> plot loss, accuracy, precision, recall vs. epochs
    if plot_epoch:
        epoch_plots(history, p, output_dir+prefix+'epoch-',
                    supervised=supervised)   

    # -- unsupervised ---------------------------------------------------------
    # >> plot some decoded light curves
    if plot_in_out and not supervised:
        fig, axes = input_output_plot(x, x_test, x_predict,
                                      output_dir+prefix+'input_output.png',
                                      ticid_test=ticid_test,
                                      inds=inds,
                                      addend=addend, sharey=sharey,
                                      mock_data=mock_data,
                                      feature_vector=feature_vector,
                                      percentage=percentage)
        
    # >> plot input, bottleneck, output
    if plot_in_bottle_out and not supervised:
        input_bottleneck_output_plot(x, x_test, x_predict,
                                     activations, model, ticid_test,
                                     output_dir+prefix+\
                                     'input_bottleneck_output.png',
                                     addend=addend, inds = input_bottle_inds,
                                     sharey=False, mock_data=mock_data,
                                     feature_vector=feature_vector)
            
    # -- supervised -----------------------------------------------------------
    if supervised:
        y_train_classes = np.argmax(y_train, axis = 1)
        num_classes = len(np.unique(y_train_classes))
        training_test_plot(x,x_train,x_test,
                              y_train_classes,y_true,y_predict,num_classes,
                              output_dir+prefix+'lc-', ticid_train, ticid_test,
                              mock_data=mock_data)
        
    # -- latent space visualization -------------------------------------------
    if plot_latent_test:
        fig, axes = latent_space_plot(model, activations, p,
                                      output_dir+prefix+'latent_space.png')

        
    if plot_latent_train:
        activations_train = get_activations(model, x_train, rms_test=rms_train,
                                            input_rms=input_rms)
        fig, axes = latent_space_plot(model, activations_train, p,
                                      output_dir+prefix+\
                                          'latent_space-x_train.png')
            
    if plot_lof_test:
        bottleneck = get_bottleneck(model, activations, p)
        for n in [20]: # [20, 50, 100]:
            if type(flux_test) != bool:
                plot_lof(time, flux_test, ticid_test, bottleneck, 20,
                         output_dir, prefix='test-', n_neighbors=n,
                         mock_data=mock_data, feature_vector=feature_vector)
            else:
                plot_lof(x, x_test, ticid_test, bottleneck, 20, output_dir,
                         prefix = 'test-', n_neighbors=n, mock_data=mock_data,
                         feature_vector=feature_vector)

            
    if plot_lof_train:
        activations_train = get_activations(model, x_train, rms_test=rms_train,
                                            input_rms=input_rms)
        bottleneck_train = get_bottleneck(model, activations_train, p)
        for n in [20]: # [20, 50, 100]:
            if type(flux_train) != bool:
                plot_lof(time, flux_train, ticid_train, bottleneck_train, 20,
                         output_dir, prefix='train-', n_neighbors=n,
                         mock_data=mock_data, feature_vector=feature_vector)
            else:
                plot_lof(x, x_train, ticid_train, bottleneck_train, 20,
                         output_dir, prefix = 'train-', n_neighbors=n,
                         mock_data=mock_data, feature_vector=feature_vector)            
        

    # >> plot kernel vs. filter
    if plot_kernel:
        kernel_filter_plot(model, output_dir+prefix+'kernel-')
        

    # if plot_clustering:
    #     bottleneck_ind = np.nonzero(['dense' in x.name for x in \
    #                                  model.layers])[0][0]
    #     bottleneck = activations[bottleneck_ind - 1]        
    #     latent_space_clustering(bottleneck, x_test, x, ticid_test,
    #                             out=output_dir+prefix+\
    #                                 'clustering-x_test-', addend=addend)
        

    # -- intermediate activations visualization -------------------------------
    if plot_intermed_act:
        intermed_act_plot(x, model, activations, x_test,
                          output_dir+prefix+'intermed_act-', addend=addend,
                          inds=intermed_inds, feature_vector=feature_vector)
    
    if make_movie:
        movie(x, model, activations, x_test, p, output_dir+prefix+'movie-',
              ticid_test, addend=addend, inds=intermed_inds)
        
    
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
    from keras.layers import Dense, AveragePooling1D
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
            # x = Conv1D(params['num_filters'], int(params['kernel_size']),
            #             activation=params['activation'], padding='same')(x)
        else:
            x = Conv1D(params['num_filters'], int(params['kernel_size']),
                        activation=params['activation'], padding='same',
                        kernel_initializer=params['initializer'])(x)
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
    from keras.layers import Lambda
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
            decoded = Conv1D(1, int(params['kernel_size']),
                              activation=params['last_activation'],
                              padding='same',
                              kernel_initializer=params['initializer'])(x)            
            # decoded = Conv1D(1, int(params['kernel_size'][num_iter]),
            #                  activation=params['last_activation'],
            #                  padding='same')(x)
        else:
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

def split_data_features(flux, features, time, ticid, classes, p, train_test_ratio = 0.9,
                        cutoff=16336,
               supervised=False, interpolate=False,
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
    

def rms(x):
    rms = np.sqrt(np.mean(x**2, axis = 1))
    return rms

def standardize(x, ax=1):
    means = np.mean(x, axis = ax, keepdims=True) # >> subtract mean
    x = x - means
    stdevs = np.std(x, axis = ax, keepdims=True) # >> divide by standard dev
    x = x / stdevs   
    return x
    
def normalize(flux):
    medians = np.median(flux, axis = 1, keepdims=True)
    flux = flux / medians - 1.
    return flux

def interpolate_lc(flux, time, flux_err=False, interp_tol=20./(24*60),
                   num_sigma=5, orbit_gap_len = 3, DEBUG=False,
                   output_dir='./', prefix='',
                   spline_interpolate=True):
    '''Interpolates nan gaps less than 20 minutes long.'''
    from astropy.stats import SigmaClip
    from scipy import interpolate
    flux_interp = []
    for j in range(len(flux)):
        i = flux[j]
        if DEBUG and j == 1042:
            fig, ax = plt.subplots(6, 1, figsize=(8, 3*6))
            ax[0].plot(time, i, '.k', markersize=2)
            ax[0].set_title('original')
        # >> sigma clip
        sigclip = SigmaClip(sigma=num_sigma, maxiters=None, cenfunc='median')
        clipped_inds = np.nonzero(np.ma.getmask(sigclip(i, masked=True)))
        i[clipped_inds] = np.nan
        if DEBUG and j == 1042:
            ax[1].plot(time, i, '.k', markersize=2)
            ax[1].set_title('clipped')
        
        # >> find nan windows
        n = np.shape(i)[0]
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(np.isnan(i)[:-1], np.isnan(i)[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]
    
        # >> find nan window lengths
        run_lengths = np.diff(np.append(run_starts, n))
        tdim = time[1]-time[0]
        
        # -- interpolate small nan gaps ---------------------------------------
        interp_gaps = np.nonzero((run_lengths * tdim <= interp_tol) * \
                                            np.isnan(i[run_starts]))
        interp_inds = run_starts[interp_gaps]
        interp_lens = run_lengths[interp_gaps]


        i_interp = np.copy(i)
        for a in range(np.shape(interp_inds)[0]):
            start_ind = interp_inds[a]
            end_ind = interp_inds[a] + interp_lens[a]
            i_interp[start_ind:end_ind] = np.interp(time[start_ind:end_ind],
                                                    time[np.nonzero(~np.isnan(i))],
                                                    i[np.nonzero(~np.isnan(i))])
        i = i_interp
        if DEBUG and j == 1042:
            ax[2].plot(time, i, '.k', markersize=2)
            ax[2].set_title('interpolated')
        
        # -- spline interpolate large nan gaps --------------------------------
        if spline_interpolate:
            orbit_gap_len = np.count_nonzero(np.isnan(time))*tdim
            interp_gaps = np.nonzero((run_lengths * tdim > interp_tol) * \
                                     (run_lengths*tdim < 0.9*orbit_gap_len) * \
                                     np.isnan(i[run_starts]))
            interp_inds = run_starts[interp_gaps]
            interp_lens = run_lengths[interp_gaps]
            
            # >> fit spline to non-nan points
            num_inds = np.nonzero(~np.isnan(i))
            use_splrep = False
            if use_splrep:
                tck = interpolate.splrep(time[num_inds], i[num_inds], k=3)
            else:
                cs= interpolate.CubicSpline(time[num_inds], i[num_inds])
                # i_cs = cs(time[num_inds])
                # num_inds_time = np.nonzero(~np.isnan(time))
                # i_cs = cs(time[num_inds_time])
                t1 = np.linspace(np.min(time[num_inds]), np.max(time[num_inds]),
                                 len(time))
                t1 = np.delete(t1, np.nonzero(np.isnan(time)))
                i_cs = cs(t1)
            if DEBUG and j==1042:
                if use_splrep:
                    i_plot = interpolate.splev(time[num_inds],tck)
                    ax[3].plot(time[num_inds], i_plot, '-')
                else:
                    i_plot = i_cs
                    ax[3].plot(t1, i_cs, '-')
                
                ax[3].set_title('spline') 
            
            i_interp = np.copy(i)
            for a in range(np.shape(interp_inds)[0]):
                start_ind = interp_inds[a]
                end_ind   = interp_inds[a] + interp_lens[a] - 1
                # >> pad [etc 060720]
                # start_ind = max(0, start_ind-10)
                # end_ind = min(len(time)-1, end_ind + 10)
                # t_new = np.linspace(time[start_ind-1]+tdim, time[end_ind-1]+tdim,
                #                     end_ind-start_ind)

                if use_splrep:
                    t_new = np.linspace(time[start_ind], time[end_ind],
                                    end_ind-start_ind)     
                    i_interp[start_ind:end_ind]=interpolate.splev(t_new, tck)
                else:
                    # start_ind_cs = num_inds_time[0].tolist().index(start_ind)
                    # end_ind_cs = num_inds_time[0].tolist().index(end_ind)
                    # start_ind_cs = num_inds[0].tolist().index(start_ind)
                    # end_ind_cs = num_inds[0].tolist().index(end_ind)                    
                    # i_interp[start_ind:end_ind]=i_cs[start_ind_cs:end_ind_cs]
                    # t_new = np.linspace(time[start_ind], time[end_ind],
                    #                     end_ind-start_ind)
                    # i_interp[start_ind:end_ind]=cs(t_new)
                    if not np.isnan(time[start_ind]):
                        start_ind_cs = np.argmin(np.abs(t1 - time[start_ind]))
                        end_ind_cs = start_ind_cs + (end_ind-start_ind)
                    else:
                        end_ind_cs = np.argmin(np.abs(t1 - time[end_ind]))
                        start_ind_cs = end_ind_cs - (end_ind-start_ind)
                    i_interp[start_ind:end_ind] = i_cs[start_ind_cs:end_ind_cs]
            flux_interp.append(i_interp)
            if DEBUG and j==1042:
                ax[4].plot(time, i_interp, '.k', markersize=2)
                ax[4].set_title('spline interpolate')
        else:
            flux_interp.append(i)
        
    # -- remove orbit nan gap -------------------------------------------------
    flux = np.array(flux_interp)
    nan_inds = np.nonzero(np.prod(~np.isnan(flux), 
                                  axis = 0) == False)
    time = np.delete(time, nan_inds)
    flux = np.delete(flux, nan_inds, 1)
    if DEBUG:
        ax[5].plot(time, flux[1042], '.k', markersize=2)
        ax[5].set_title('removed orbit gap')
        fig.tight_layout()

        # for a in ax.flatten():
        #     format_axes(a, xlabel=True, ylabel=True)
        fig.savefig(output_dir + prefix + 'interpolate_debug.png',
                    bbox_inches='tight')
        plt.close(fig) 
    
    if type(flux_err) != bool:
        flux_err = np.delete(flux_err, nan_inds, 1)
        return flux, time, flux_err
    else:
        return flux, time

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

# :: plotting :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def ticid_label(ax, ticid, title=False):
    '''Query catalog data and add text to axis
    https://arxiv.org/pdf/1905.10694.pdf'''
    from astroquery.mast import Catalogs

    target = 'TIC '+str(int(ticid))
    catalog_data = Catalogs.query_object(target, radius=0.02, catalog='TIC')
    Teff = catalog_data[0]["Teff"]
    if np.isnan(Teff):
        Teff = 'nan'
    else: Teff = '%.4d'%Teff
    rad = catalog_data[0]["rad"]
    mass = catalog_data[0]["mass"]
    GAIAmag = catalog_data[0]["GAIAmag"]
    d = catalog_data[0]["d"]
    # Bmag = catalog_data[0]["Bmag"]
    # Vmag = catalog_data[0]["Vmag"]
    objType = catalog_data[0]["objType"]
    # Tmag = catalog_data[0]["Tmag"]
    # lum = catalog_data[0]["lum"]

    info = target+'\nTeff {}\nrad {}\nmass {}\nG {}\nd {}\nO {}'
    info1 = target+', Teff {}, rad {}, mass {},\nG {}, d {}, O {}'
    
    if title:
        ax.set_title(info1.format(Teff, '%.2g'%rad, '%.2g'%mass, 
                                  '%.3g'%GAIAmag, '%.3g'%d, objType),
                     fontsize='xx-small')
    else:
        ax.text(0.98, 0.98, info.format(Teff, '%.2g'%rad, '%.2g'%mass, 
                                        '%.3g'%GAIAmag, '%.3g'%d, objType),
                  transform=ax.transAxes, horizontalalignment='right',
                  verticalalignment='top', fontsize='xx-small')
    
def format_axes(ax, xlabel=False, ylabel=False):
    '''Helper function to plot TESS light curves. Aspect ratio is 3/8.
    Parameters:
        * ax : matplotlib axis'''
    # >> force aspect = 3/8
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.set_aspect(abs((xlim[1]-xlim[0])/(ylim[1]-ylim[0])*(3./8.)))
    # ax.set_aspect(3./8., adjustable='box')
    
    if list(ax.get_xticklabels()) == []:
        ax.tick_params('x', bottom=False) # >> remove ticks if no label
    else:
        ax.tick_params('x', labelsize='small')
    ax.tick_params('y', labelsize='small')
    ax.ticklabel_format(useOffset=False)
    if xlabel:
        ax.set_xlabel('Time [BJD - 2457000]')
    if ylabel:
        ax.set_ylabel('Relative flux')
    
def get_bottleneck(model, activations, p):
    '''Get bottleneck layer, with shape (num light curves, latent dimension)
    Parameters:
        * model : Keras Model()
        * activations : from get_activations()
        * p : parameter set, with p['latent_dim'] = dimension of latent space
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
    
    return bottleneck

def corner_plot(activation, p, n_bins = 50, log = True):
    '''Creates corner plot.'''
    from matplotlib.colors import LogNorm
    
    latentDim = p['latent_dim']

    fig, axes = plt.subplots(nrows = latentDim, ncols = latentDim,
                             figsize = (10, 10))

    # >> deal with 1 latent dimension case
    if latentDim == 1:
        axes.hist(np.reshape(activation, np.shape(activation)[0]), n_bins,
                  log=log)
        axes.set_ylabel('\u03C61')
        axes.set_ylabel('frequency')
    else:
        # >> row 1 column 1 is first latent dimension (phi1)
        for i in range(latentDim):
            axes[i,i].hist(activation[:,i], n_bins, log=log)
            axes[i,i].set_aspect(aspect=1)
            for j in range(i):
                if log:
                    norm = LogNorm()
                axes[i,j].hist2d(activation[:,j], activation[:,i],
                                 bins=n_bins, norm=norm)
                # >> remove axis frame of empty plots
                axes[latentDim-1-i, latentDim-1-j].axis('off')

            # >> x and y labels
            axes[i,0].set_ylabel('\u03C6' + str(i))
            axes[latentDim-1,i].set_xlabel('\u03C6' + str(i))

        # >> removing axis
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        plt.subplots_adjust(hspace=0, wspace=0)

    return fig, axes

def input_output_plot(x, x_test, x_predict, out, ticid_test=False,
                      inds = [0, -14, -10, 1, 2], addend = 0., sharey=False,
                      mock_data=False, feature_vector=False,
                      percentage=False):
    '''Plots input light curve, output light curve and the residual.
    !! Can only handle len(inds) divisible by 3 or 5
    Parameters:
        * x : time array
        * x_test
        * x_predict : output of model.predict(x_test)
        * out : output directory'''

    if len(inds) % 5 == 0:
        ncols = 5
    elif len(inds) % 3 == 0:
        ncols = 3
    ngroups = int(len(inds)/ncols)
    nrows = int(3*ngroups)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15,12), sharey=sharey,
                             sharex=True)
    plt.subplots_adjust(hspace=0)
    for i in range(ncols):
        for ngroup in range(ngroups):
            ind = int(ngroup*ncols + i)
            if not mock_data:
                ticid_label(axes[ngroup*3,i], ticid_test[inds[ind]],title=True)
            axes[ngroup*3,i].plot(x,x_test[inds[ind]]+addend,'.k',markersize=2)
            axes[ngroup*3+1,i].plot(x,x_predict[inds[ind]]+addend,'.k',
                                    markersize=2)
            # >> residual
            residual = (x_test[inds[ind]] - x_predict[inds[ind]])
            if percentage:
                residual = residual / x_test[inds[ind]]
            axes[ngroup*3+2, i].plot(x, residual, '.k', markersize=2)
            for j in range(3):
                format_axes(axes[ngroup*3+j,i])
        if feature_vector:
            axes[-1, i].set_xlabel('\u03C8', fontsize='small')
        else:
            axes[-1, i].set_xlabel('time [BJD - 2457000]', fontsize='small')
    for i in range(ngroups):
        axes[3*i,   0].set_ylabel('input\nrelative flux',  fontsize='small')
        axes[3*i+1, 0].set_ylabel('output\nrelative flux', fontsize='small')
        axes[3*i+2, 0].set_ylabel('residual', fontsize='small') 
    fig.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    return fig, axes
    
def get_activations(model, x_test, input_rms = False, rms_test = False):
    from keras.models import Model
    layer_outputs = [layer.output for layer in model.layers][1:]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    if input_rms:
        activations = activation_model.predict([x_test, rms_test])
    else:
        activations = activation_model.predict(x_test)
    return activations

def latent_space_plot(model, activations, params, out):
    # >> get ind for plotting latent space
    dense_inds = np.nonzero(['dense' in x.name for x in \
                                 model.layers])[0]
    # dense_inds = np.nonzero(['flatten' in x.name for x in model.layers])[0]
    for ind in dense_inds:
        if np.shape(activations[ind-1])[1] == params['latent_dim']:
            bottleneck_ind = ind - 1
    fig, axes = corner_plot(activations[bottleneck_ind-1], params)
    plt.savefig(out)
    plt.close(fig)
    return fig, axes

def kernel_filter_plot(model, out_dir):
    # >> get inds for plotting kernel and filters
    layer_inds = np.nonzero(['conv' in x.name for x in model.layers])[0]
    for a in layer_inds: # >> loop through conv layers
        filters, biases = model.layers[a].get_weights()
        fig, ax = plt.subplots()
        ax.imshow(np.reshape(filters, (np.shape(filters)[0],
                                       np.shape(filters)[2])))
        ax.set_xlabel('filter')
        ax.set_ylabel('kernel')
        plt.savefig(out_dir + 'layer' + str(a) + '.png')
        plt.close(fig)

def intermed_act_plot(x, model, activations, x_test, out_dir, addend=0.5,
                      inds = [0, -1], movie = True, feature_vector=False):
    '''Visualizing intermediate activations.
    Parameters:
        * x: time array
        * model: Keras Model()
        * activations: output from get_activations()
        * x_test
        * out_dir
        * append : 
    Note that activation.shape = (test_size, input_dim, num_filters)'''
    # >> get inds for plotting intermediate activations
    act_inds = np.nonzero(['conv' in x.name or \
                           'max_pool' in x.name or \
                           'dropout' in x.name or \
                               'dense' in x.name or \
                           'reshape' in x.name for x in \
                           model.layers])[0]
    act_inds = np.array(act_inds) -1

    for c in range(len(inds)): # >> loop through light curves
        fig, axes = plt.subplots(figsize=(8,3))
        addend = 1. - np.median(x_test[inds[c]])
        axes.plot(np.linspace(np.min(x), np.max(x), np.shape(x_test)[1]),
                x_test[inds[c]] + addend, '.k', markersize=2)
        if feature_vector:
            axes.set_xlabel('\u03C8')
        else:
            axes.set_xlabel('time [BJD - 2457000]')
        axes.set_ylabel('relative flux')
        plt.tight_layout()
        fig.savefig(out_dir+str(c)+'ind-0input.png')
        plt.close(fig)
        for a in act_inds: # >> loop through layers
            activation = activations[a]
            
            if len(np.shape(activation)) == 2:
                ncols, nrows = 1, 1
                num_filters=1
                
            else:   
                if np.shape(activation)[2] == 1:
                    nrows = 1
                    ncols = 1
                    num_filters=1
                else:
                    num_filters = np.shape(activation)[2]
                    ncols = 4
                    nrows = int(num_filters/ncols)
                    
            fig, axes = plt.subplots(nrows,ncols,figsize=(8*ncols,3*nrows))                    
            # fig, axes = plt.subplots(nrows,ncols,figsize=(8*ncols*0.5,3*nrows))
            for b in range(num_filters): # >> loop through filters
                if ncols == 1:
                    ax = axes
                else:
                    ax = axes.flatten()[b]
                x1 = np.linspace(np.min(x), np.max(x), np.shape(activation)[1])
                if num_filters > 1:
                    ax.plot(x1, activation[inds[c]][:,b]+addend,'.k',markersize=2)
                else:
                    ax.plot(x1, activation[inds[c]]+addend, '.k', markersize=2)
                    
                
            if nrows == 1:
                if feature_vector:
                    axes.set_xlabel('\u03C8')
                else:
                    axes.set_xlabel('time [BJD - 2457000]')        
                axes.set_ylabel('relative flux')
            else:
                for i in range(nrows):
                    axes[i,0].set_ylabel('relative\nflux')
                for j in range(ncols):
                    if feature_vector:
                        axes[-1,j].set_xlabel('\u03C8')
                    else:
                        axes[-1,j].set_xlabel('time [BJD - 2457000]')
            fig.tight_layout()
            fig.savefig(out_dir+str(c)+'ind-'+str(a+1)+model.layers[a+1].name\
                        +'.png')
            plt.close(fig)


def epoch_plots(history, p, out_dir, supervised):
    if supervised:
        label_list = [['loss', 'accuracy'], ['precision', 'recall']]
        key_list = [['loss', 'accuracy'], [list(history.history.keys())[-2],
                                           list(history.history.keys())[-1]]]

        for i in range(len(key_list)):
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(history.history[key_list[i][0]], label=label_list[i][0])
            ax1.set_ylabel(label_list[i][0])
            ax2.plot(history.history[key_list[i][1]], '--', label=label_list[i][1])
            ax2.set_ylabel(label_list[i][1])
            ax1.set_xlabel('epoch')
            ax1.set_xticks(np.arange(0, int(p['epochs']),
                                     max(int(p['epochs']/10),1)))
            ax1.tick_params('both', labelsize='x-small')
            ax2.tick_params('both', labelsize='x-small')
            fig.tight_layout()
            if i == 0:
                plt.savefig(out_dir + 'acc_loss.png')
            else:
                plt.savefig(out_dir + 'prec_recall.png')
            plt.close(fig)
            
    else:
        fig, ax1 = plt.subplots()
        ax1.plot(history.history['loss'], label='loss')
        ax1.set_ylabel('loss')
        ax1.set_xlabel('epoch')
        ax1.set_xticks(np.arange(0, int(p['epochs']),
                                 max(int(p['epochs']/10),1)))
        ax1.tick_params('both', labelsize='x-small')
        fig.tight_layout()
        plt.savefig(out_dir + 'loss.png')
            
    
def input_bottleneck_output_plot(x, x_test, x_predict, activations, model,
                                 ticid_test, out, inds=[0,1,-1,-2,-3],
                                 addend = 1., sharey=False, mock_data=False,
                                 feature_vector=False):
    '''Can only handle len(inds) divisible by 3 or 5'''
    bottleneck_ind = np.nonzero(['dense' in x.name for x in \
                                 model.layers])[0][0]
    bottleneck = activations[bottleneck_ind - 1]
    if len(inds) % 5 == 0:
        ncols = 5
    elif len(inds) % 3 == 0:
        ncols = 3
    ngroups = int(len(inds)/ncols)
    nrows = int(3*ngroups)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15,5), sharey=sharey,
                             sharex=True)
    plt.subplots_adjust(hspace=0)
    for i in range(ncols):
        for ngroup in range(ngroups):
            ind = int(ngroup*ncols + i)
            axes[ngroup*3,i].plot(x,x_test[inds[ind]]+addend,'.k',markersize=2)
            axes[ngroup*3+1,i].plot(np.linspace(np.min(x),np.max(x),
                                              len(bottleneck[inds[ind]])),
                                              bottleneck[inds[ind]], '.k',
                                              markersize=2)
            axes[ngroup*3+2,i].plot(x,x_predict[inds[ind]]+addend,'.k',
                                    markersize=2)
            if not mock_data:
                ticid_label(axes[ngroup*3,i],ticid_test[inds[ind]], title=True)
            for j in range(3):
                format_axes(axes[ngroup*3+j,i])
        axes[1, i].set_xlabel('\u03C6', fontsize='small')
        if feature_vector:
            axes[0, i].set_xlabel('\u03C8', fontsize='small')            
            axes[-1, i].set_xlabel('\u03C8', fontsize='small') 
        else:
            axes[0, i].set_xlabel('time [BJD - 2457000]', fontsize='small')        
            axes[-1, i].set_xlabel('time [BJD - 2457000]', fontsize='small')
    for i in range(ngroups):
        axes[3*i,   0].set_ylabel('input\nrelative flux',  fontsize='small')
        axes[3*i+1, 0].set_ylabel('bottleneck', fontsize='small')
        axes[3*i+2, 0].set_ylabel('output\nrelative flux', fontsize='small')
    fig.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    return fig, axes
    

def movie(x, model, activations, x_test, p, out_dir, ticid_test, inds = [0, -1],
          addend=0.5):
    '''Make a .mp4 file of intermediate activations.
    Parameters:
        * x : time array
        * model : Keras Model()
        * activations : output from get_activations()
        * x_test
        * p : parameter set
        * out_dir : output directory
        * inds : light curve indices in x_test'''
    for c in range(len(inds)):
        fig, axes = plt.subplots(figsize=(8,3*1.5))
        ymin = []
        ymax = []
        for activation in activations:
            # if np.shape(activation)[1] == p['latent_dim']:
            ymin.append(min(np.min(activation[inds[c]]),
                            np.min(x_test[inds[c]])))
                # ymax.append(max(activation[inds[c]]))
            ymax.append(max(np.max(activation[inds[c]]),
                            np.max(x_test[inds[c]])))
            # elif len(np.shape(activation)) > 2:
                # if np.shape(activation)[2] == 1:
                    # ymin.append(min(activation[inds[c]]))
                    # ymax.append(max(activation[inds[c]]))
        ymin = np.min(ymin) + addend + 0.3*np.median(x_test[inds[c]])
        ymax = np.max(ymax) + addend - 0.3*np.median(x_test[inds[c]])
        addend = 1. - np.median(x_test[inds[c]])

        # >> plot input
        axes.plot(np.linspace(np.min(x), np.max(x), np.shape(x_test)[1]),
                  x_test[inds[c]] + addend, '.k', markersize=2)
        axes.set_xlabel('time [BJD - 2457000]')
        axes.set_ylabel('relative flux')
        axes.set_ylim(ymin=ymin, ymax=ymax)
        # fig.tight_layout()
        fig.savefig('./image-000.png')

        # >> plot intermediate activations
        n=1
        for a in range(len(activations)):
            activation = activations[a]
            if np.shape(activation)[1] == p['latent_dim']:
                length = p['latent_dim']
                axes.cla()
                axes.plot(np.linspace(np.min(x), np.max(x), length),
                          activation[inds[c]] + addend, '.k', markersize=2)
                axes.set_xlabel('time [BJD - 2457000]')
                axes.set_ylabel('relative flux')
                # format_axes(axes, xlabel=True, ylabel=True)
                ticid_label(axes, ticid_test[inds[c]])
                axes.set_ylim(ymin=ymin, ymax =ymax)
                # fig.tight_layout()
                fig.savefig('./image-' + f'{n:03}.png')
                n += 1
            elif len(np.shape(activation)) > 2:
                # >> don't plot activations with multiple filters
                if np.shape(activation)[2] == 1:
                    length = np.shape(activation)[1]
                    y = np.reshape(activation[inds[c]], (length))
                    axes.cla()
                    axes.plot(np.linspace(np.min(x), np.max(x), length),
                              y + addend, '.k', markersize=2)
                    axes.set_xlabel('time [BJD - 2457000]')
                    axes.set_ylabel('relative flux')
                    # format_axes(axes, xlabel=True, ylabel=True)
                    ticid_label(axes, ticid_test[inds[c]])
                    axes.set_ylim(ymin = ymin, ymax = ymax)
                    # fig.tight_layout()
                    fig.savefig('./image-' + f'{n:03}.png')
                    n += 1
        os.system('ffmpeg -framerate 2 -i ./image-%03d.png -pix_fmt yuv420p '+\
                  out_dir+str(c)+'ind-movie.mp4')

# def latent_space_clustering(activation, x_test, x, ticid, out = './', 
#                             n_bins = 50, addend=1., scatter = True):
#     '''[deprecated 052620] Clustering latent space. deprecated.
#     '''
#     from matplotlib.colors import LogNorm
#     # from sklearn.cluster import DBSCAN
#     from sklearn.neighbors import LocalOutlierFactor
#     latentDim = np.shape(activation)[1]

#     # -- calculate lof --------------------------------------------------------           
#     clf = LocalOutlierFactor()
#     clf.fit_predict(activation)
#     lof = -1 * clf.negative_outlier_factor_
#     inds = np.argsort(lof)[-20:] # >> outliers
#     inds2 = np.argsort(lof)[:20] # >> inliers
    
#     # >> deal with 1 latent dimension case
#     if latentDim == 1: # TODO !!
#         fig, axes = plt.subplots(figsize = (15,15))
#         axes.hist(np.reshape(activation, np.shape(activation)[0]), n_bins,
#                   log=True)
#         axes.set_ylabel('\u03C61')
#         axes.set_ylabel('frequency')
#     else:


#         # >> row 1 column 1 is first latent dimension (phi1)
#         for i in range(latentDim):
#             for j in range(i):
#                 z1, z2 = activation[:,j], activation[:,i]
#                 # X = np.array((z1, z2)).T                
#                 # clf = LocalOutlierFactor()
#                 # clf.fit_predict(X)
#                 # lof = -1 * clf.negative_outlier_factor_

                
#                 # -- plot latent space w/ inset plots -------------------------
#                 fig, ax = plt.subplots(figsize = (15,15))
                
#                 if scatter:
#                     ax.plot(z1, z2, '.')
#                 else:
#                     ax.hist2d(z1, z2, bins=n_bins, norm=LogNorm())
                
#                 plt.xticks(fontsize='xx-large')
#                 plt.yticks(fontsize='xx-large')
                
#                 h = 0.047
#                 x0 = 0.85
#                 y0 = 0.9
#                 xstep = h*8/3 + 0.025
#                 ystep = h + 0.025
                
#                 # >> sort to clean up plot
#                 inds0 = inds[:10]
#                 inds0 = sorted(inds, key=lambda z: ((z1[z]-np.max(z1))+\
#                                                     (z2[z]-np.min(z2)))**2)
                
#                 for k in range(10):
#                     # >> make inset axes
#                     if k < 5:
#                         axins = ax.inset_axes([x0 - k*xstep, y0, h*8/3, h])
#                     else:
#                         axins = ax.inset_axes([x0, y0 - (k-4)*ystep, h*8/3, h])
#                     xp, yp = z1[inds0[k]], z2[inds0[k]]
            
#                     xextent = ax.get_xlim()[1] - ax.get_xlim()[0]
#                     yextent = ax.get_ylim()[1] - ax.get_ylim()[0]
#                     x1, x2 = xp-0.01*xextent, xp+0.01*xextent
#                     y1, y2 = yp-0.01*yextent, yp+0.01*yextent
#                     axins.set_xlim(x1, x2)
#                     axins.set_ylim(y1, y2)
#                     ax.indicate_inset_zoom(axins)
                    
#                     # >> plot light curves
#                     axins.set_xlim(min(x), max(x))
#                     axins.set_ylim(min(x_test[inds0[k]]),
#                                    max(x_test[inds0[k]]))
#                     axins.plot(x, x_test[inds0[k]] + addend, '.k',
#                                markersize=2)
#                     axins.set_xticklabels('')
#                     axins.set_yticklabels('')
#                     axins.patch.set_alpha(0.5)

#                 # >> x and y labels
#                 ax.set_ylabel('\u03C61' + str(i), fontsize='xx-large')
#                 ax.set_xlabel('\u03C61' + str(j), fontsize='xx-large')
#                 fig.savefig(out + 'phi' + str(j) + 'phi' + str(i) + '.png')
                
#                 # -- plot 20 light curves -------------------------------------
#                 # >> plot light curves with lof label
#                 fig1, ax1 = plt.subplots(20, figsize = (7,28))
#                 fig1.subplots_adjust(hspace=0)
#                 fig2, ax2 = plt.subplots(20, figsize = (7,28))
#                 fig2.subplots_adjust(hspace=0)
#                 fig3, ax3 = plt.subplots(20, figsize = (7,28))
#                 fig3.subplots_adjust(hspace=0)
#                 for k in range(20):
#                     # >> outlier plot
#                     ax1[k].plot(x,x_test[inds[19-k]]+addend,'.k',markersize=2)
#                     ax1[k].set_xticks([])
#                     ax1[k].set_ylabel('relative\nflux')
#                     ax1[k].text(0.8, 0.65,
#                                 'LOF {}\nTIC {}'.format(str(lof[inds[19-k]])[:9],
#                                                         str(int(ticid[inds[19-k]]))),
#                                 transform = ax1[k].transAxes)
                    
#                     # >> inlier plot
#                     ax2[k].plot(x, x_test[inds2[k]]+addend, '.k', markersize=2)
#                     ax2[k].set_xticks([])
#                     ax2[k].set_ylabel('rellative\nflux')
#                     ax2[k].text(0.8, 0.65,
#                                 'LOF {}\nTIC {}'.format(str(lof[inds2[k]])[:9],
#                                                         str(int(ticid[inds2[k]]))),
#                                 transform = ax2[k].transAxes)
                    
#                     # >> random lof plot
#                     ind = np.random.choice(range(len(lof)-1))
#                     ax3[k].plot(x, x_test[ind] + addend, '.k', markersize=2)
#                     ax3[k].set_xticks([])
#                     ax3[k].set_ylabel('relative\nflux')
#                     ax3[k].text(0.8, 0.65,
#                                 'LOF {}\nTIC {}'.format(str(lof[ind])[:9],
#                                                         str(int(ticid[ind]))),
#                                 transform = ax3[k].transAxes)
                
#                 ax1[-1].set_xlabel('time [BJD - 2457000]')
#                 ax2[-1].set_xlabel('time [BJD - 2457000]')
#                 ax3[-1].set_xlabel('time [BJD - 2457000]')
#                 fig1.savefig(out + 'phi' + str(j) + 'phi' + str(i) + \
#                             '-outliers.png')
#                 fig2.savefig(out + 'phi' + str(j) + 'phi' + str(i) + \
#                              '-inliers.png')
#                 fig3.savefig(out + 'phi' + str(j) + 'phi'  + str(i) + \
#                              '-randomlof.png')
                

#         # >> removing axis
#         # for ax in axes.flatten():
#         #     ax.set_xticks([])
#         #     ax.set_yticks([])
#         # plt.subplots_adjust(hspace=0, wspace=0)

#     return fig, ax

def training_test_plot(x, x_train, x_test, y_train_classes, y_test_classes,
                       y_predict, num_classes, out, ticid_train, ticid_test,
                       mock_data=False):
    # !! add more rows
    colors = ['r', 'g', 'b', 'm'] # !! add more colors
    # >> training data set
    fig, ax = plt.subplots(nrows = 7, ncols = num_classes, figsize=(15,10),
                           sharex=True)
    plt.subplots_adjust(hspace=0)
    # >> test data set
    fig1, ax1 = plt.subplots(nrows = 7, ncols = num_classes, figsize=(15,10),
                             sharex=True)
    plt.subplots_adjust(hspace=0)
    for i in range(num_classes): # >> loop through classes
        inds = np.nonzero(y_train_classes == i)[0]
        inds1 = np.nonzero(y_test_classes == i)[0]
        for j in range(min(7, len(inds))): # >> loop through rows
            ax[j,i].plot(x, x_train[inds[j]], '.'+colors[i], markersize=2)
            if not mock_data:
                ticid_label(ax[j,i], ticid_train[inds[j]])
        for j in range(min(7, len(inds1))):
            ax1[j,i].plot(x, x_test[inds1[j]], '.'+colors[y_predict[inds1[j]]],
                          markersize=2)
            if not mock_data:
                ticid_label(ax1[j,i], ticid_test[inds1[j]])    
            ax1[j,i].text(0.98, 0.02, 'True: '+str(i)+'\nPredicted: '+\
                          str(y_predict[inds1[j]]),
                          transform=ax1[j,i].transAxes, fontsize='xx-small',
                          horizontalalignment='right',
                          verticalalignment='bottom')
    for i in range(num_classes):
        ax[0,i].set_title('True class '+str(i), color=colors[i])
        ax1[0,i].set_title('True class '+str(i), color=colors[i])
        
        for axis in [ax[-1,i], ax1[-1,i]]:
            axis.set_xlabel('time [BJD - 2457000]', fontsize='small')
    for j in range(7):
        for axis in [ax[j,0],ax1[j,0]]:
            axis.set_ylabel('relative\nflux', fontsize='small')
            
    for axis in  ax.flatten():
        format_axes(axis)
    for axis in ax1.flatten():
        format_axes(axis)
    # fig.tight_layout()
    # fig1.tight_layout()
    fig.savefig(out+'train.png')
    fig1.savefig(out+'test.png')

def plot_lof(time, intensity, targets, features, n, path,
             momentum_dump_csv = './Table_of_momentum_dumps.csv',
             n_neighbors=20,
             prefix='', mock_data=False, addend=1., feature_vector=False):
    """ Adapted from Lindsey Gordon's feature_functions.py
    Plots the 20 most and least interesting light curves based on LOF.
    Parameters:
        * time array
        * intensity
        * targets : list of TICIDs
        * feature vector
        * n : number of curves to plot
        * path : output directory
    """
    from sklearn.neighbors import LocalOutlierFactor

    # >> calculate LOF
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)
    fit_predictor = clf.fit_predict(features)
    negative_factor = clf.negative_outlier_factor_
    
    lof = -1 * negative_factor
    ranked = np.argsort(lof)
    largest_indices = ranked[::-1][:n] # >> outliers
    smallest_indices = ranked[:n] # >> inliers
    
    # >> get momentum dump times
    with open(momentum_dump_csv, 'r') as f:
        lines = f.readlines()
        mom_dumps = [ float(line.split()[3][:-1]) for line in lines[6:] ]
        inds = np.nonzero((mom_dumps >= np.min(time)) * \
                          (mom_dumps <= np.max(time)))
        mom_dumps = np.array(mom_dumps)[inds]

    # [etc 052620] merged for loops, now reports random lof
    for i in range(3): # >> loop through smallest, largest, random LOF plots
        # fig, ax = plt.subplots(n, 1, sharex=True)
        fig, ax = plt.subplots(n, 1, sharex=True, figsize = (8, 3*n))
        # fig.subplots_adjust(hspace=0)
        
        for k in range(n): # >> loop through each row
            if i == 0: ind = largest_indices[k]
            elif i == 1: ind = smallest_indices[k]
            elif i == 2: ind = np.random.choice(range(len(lof)-1))
            
            
            # >> plot light curve
            ax[k].plot(time, intensity[ind] + addend, '.k', markersize=2)
            ax[k].text(0.98, 0.02, '%.3g'%lof[ind], transform=ax[k].transAxes,
                       horizontalalignment='right', verticalalignment='bottom',
                       fontsize='xx-small')
            format_axes(ax[k], ylabel=True)
            if not mock_data:
                ticid_label(ax[k], targets[ind], title=True)

            # >> plot momentum dumps
            for t in mom_dumps:
                ax[k].plot([t,t], [0, 1], '--g', alpha=0.5,
                           transform=ax[k].transAxes)
        if feature_vector:
            ax[n-1].set_xlabel('\u03C8')
        else:
            ax[n-1].set_xlabel('time [BJD - 2457000]')
        if i == 0:
            fig.suptitle(str(n) + ' largest LOF targets', fontsize=16, y=0.9)
            fig.savefig(path + 'lof-' + prefix + 'kneigh' + str(n_neighbors) \
                        + "-largest.png", bbox_inches='tight')
        elif i == 1:
            fig.suptitle(str(n) + ' smallest LOF targets', fontsize=16, y=0.9)
            fig.savefig(path + 'lof-' + prefix + 'kneigh' + str(n_neighbors)  \
                        + "-smallest.png", bbox_inches='tight')
        elif i == 2:
            fig.suptitle(str(n) + ' random LOF targets', fontsize=16, y=0.9)
            fig.savefig(path + 'lof-' + prefix + 'kneigh' + str(n_neighbors) \
                        + "-random.png", bbox_inches='tight')
    
def hyperparam_opt_diagnosis(analyze_object, output_dir, supervised=False):
    import pandas as pd
    import matplotlib.pyplot as plt
    # analyze_object = talos.Analyze('talos_experiment.csv')
    
    print(analyze_object.data)
    print(analyze_object.low('val_loss'))
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    df = analyze_object.data
    print(df.iloc[[np.argmin(df['val_loss'])]])
    
    with open(output_dir + 'best_params.txt', 'a') as f: 
        best_param_ind = np.argmin(df['val_loss'])
        f.write(str(df.iloc[best_param_ind]) + '\n')
    
    if supervised:
        label_list = ['val_loss', 'val_acc', 'val_precision',
                      'val_recall']
        key_list = ['val_loss', 'val_accuracy', 'val_precision_1',
                    'val_recall_1']
    else:
        label_list = ['val_loss']
        key_list = ['val_loss']
        
    for i in range(len(label_list)):
        analyze_object.plot_line(key_list[i])
        plt.xlabel('round')
        plt.ylabel(label_list[i])
        plt.savefig(output_dir + label_list[i] + '_plot.png')
    
    # >> kernel density estimation
    analyze_object.plot_kde('val_loss')
    plt.xlabel('val_loss')
    plt.ylabel('kernel density\nestimation')
    plt.savefig(output_dir + 'kde.png')
    
    analyze_object.plot_hist('val_loss', bins=50)
    plt.xlabel('val_loss')
    plt.ylabel('num observations')
    plt.tight_layout()
    plt.savefig(output_dir + 'hist_val_loss.png')
    
    # >> heat map correlation
    analyze_object.plot_corr('val_loss', ['acc', 'loss', 'val_acc'])
    plt.tight_layout()
    plt.savefig(output_dir + 'correlation_heatmap.png')
    
    # >> get best parameter set
    hyperparameters = list(analyze_object.data.columns)
    for col in ['round_epochs', 'val_loss', 'val_accuracy', 'val_precision_1',
            'val_recall_1', 'loss', 'accuracy', 'precision_1', 'recall_1']:
        hyperparameters.remove(col)
        
    p = {}
    for key in hyperparameters:
        p[key] = df.iloc[best_param_ind][key]
    
    return df, best_param_ind, p

def plot_reconstruction_error(time, intensity, x_test, x_predict, ticid_test,
                              output_dir='./', addend=1., mock_data=False,
                              feature_vector=False):
    '''For autoencoder, intensity = x_test'''
    # >> calculate reconstruction error (mean squared error)
    err = (x_test - x_predict)**2
    err = np.mean(err, axis=1)
    err = err.reshape(np.shape(err)[0])
    
    # >> get top 20
    n=20
    ranked = np.argsort(err)
    largest_inds = ranked[::-1][:n]
    smallest_inds = ranked[:n]
    for i in range(2):
        fig, ax = plt.subplots(n, 1, sharex=True, figsize = (8, 3*n))
        for k in range(n): # >> loop through each row
            if i == 0: ind = largest_inds[k]
            else: ind = smallest_inds[k]
            
            # >> plot light curve
            ax[k].plot(time, intensity[ind]+addend, '.k', markersize=2)
            if not feature_vector:
                ax[k].plot(time, x_predict[ind]+addend, '-')
            ax[k].text(0.98, 0.02, 'mse: ' +str(err[ind]),
                       transform=ax[k].transAxes, horizontalalignment='right',
                       verticalalignment='bottom', fontsize='xx-small')
            format_axes(ax[k], ylabel=True)
            if not mock_data:
                ticid_label(ax[k], ticid_test[ind], title=True)
                
        if feature_vector:
            ax[n-1].set_xlabel('\u03C8')
        else:
            ax[n-1].set_xlabel('time [BJD - 2457000]')
        if i == 0:
            fig.suptitle('largest reconstruction error', fontsize=16, y=0.9)
            fig.savefig(output_dir + 'reconstruction_error-largest.png',
                        bbox_inches='tight')
        else:
            fig.suptitle('smallest reconstruction error', fontsize=16, y=0.9)
            fig.savefig(output_dir + 'reconstruction_error-smallest.png',
                        bbox_inches='tight')            
    
def plot_classification(time, intensity, targets, labels, path,
             momentum_dump_csv = './Table_of_momentum_dumps.csv',
             n=20,
             prefix='', mock_data=False, addend=1., feature_vector=False):
    """ 
    """

    classes, counts = np.unique(labels, return_counts=True)
    
    # >> get momentum dump times
    with open(momentum_dump_csv, 'r') as f:
        lines = f.readlines()
        mom_dumps = [ float(line.split()[3][:-1]) for line in lines[6:] ]
        inds = np.nonzero((mom_dumps >= np.min(time)) * \
                          (mom_dumps <= np.max(time)))
        mom_dumps = np.array(mom_dumps)[inds]
        
    for i in range(len(classes)): # >> loop through each class
        fig, ax = plt.subplots(n, 1, sharex=True, figsize = (8, 3*n))
        class_inds = np.nonzero(labels == classes[i])[0]
        if classes[i] == 0:
            color = 'red'
        elif classes[i] == -1:
            color = 'black'
        elif classes[i] == 1:
            color = 'blue'
        elif classes[i] == 2:
            color = 'green'
        else:
            color = 'purple'
        
        for k in range(min(n, counts[i])): # >> loop through each row
            ind = class_inds[k]
            
            # >> plot momentum dumps
            for t in mom_dumps:
                ax[k].plot([t,t], [0, 1], '--g', alpha=0.5,
                           transform=ax[k].transAxes)            
            
            # >> plot light curve
            ax[k].plot(time, intensity[ind] + addend, '.k', markersize=2)
            ax[k].text(0.98, 0.02, str(labels[ind]), transform=ax[k].transAxes,
                       horizontalalignment='right', verticalalignment='bottom',
                       fontsize='xx-small')
            format_axes(ax[k], ylabel=True)
            if not mock_data:
                ticid_label(ax[k], targets[ind], title=True)

        if feature_vector:
            ax[n-1].set_xlabel('\u03C8')
        else:
            ax[n-1].set_xlabel('time [BJD - 2457000]')
    
        if classes[i] == -1:
            fig.suptitle('Class -1 (outliers)', fontsize=16, y=0.9,
                         color=color)
        else:
            fig.suptitle('Class ' + str(classes[i]), fontsize=16, y=0.9,
                         color=color)
        fig.savefig(path + prefix + '-class' + str(classes[i]) + '.png',
                    bbox_inches='tight')
        plt.close(fig)
        
# :: pull files with astroquery :::::::::::::::::::::::::::::::::::::::::::::::
# adapted from pipeline.py
    

def get_fits_files(mypath, target_list):
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
