# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# 
# keras model library and artificial training data
# emma feb 2020
# 
# Includes:
# * simple CNN (1D and 2D)
# * autoencoder
# 
# :: models ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

import pdb

def simplecnn(input_dim = 100, num_classes = 1):
    '''
    Adapted from:
    * https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-
      networks-in-keras-for-time-sequences-3a7ff801a2cf
    * https://keras.io/getting-started/sequential-model-guide/
      * CNN works well for identifying simple patterns within data
      * effective when interesting features are shorter(fixed-length) sigments, 
        location within the segment is not very relevant
    '''
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Reshape, Activation
    from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
    
    model = Sequential() # >> linear stack of layers
    model.add(Reshape((input_dim, 1), input_shape=(input_dim,)))

    # >> first 1d cnn layer: defines 100 filters w/ kernel_size = 10
    #    >> trains 100 different features on the first layer of the network
    # model.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
    model.add(Conv1D(100, 10, activation='relu', input_shape=(input_dim, 1)))

    # >> second 1d cnn layer: result from 1st cnn fed into 2nd cnn
    # >> define 100 different filters to be trained
    model.add(Conv1D(100, 10, activation='relu'))

    # >> max pooling layer: used to reduce the complexity of the output and prevent
    #    overfitting the data
    #    >> size = 3 (size of output matrix of this layer is only a third of the
    #       input matrix
    model.add(MaxPooling1D(3))

    # >> third and fourth 1d cnn layers: learn higher level features
    model.add(Conv1D(160, 10, activation='relu'))
    model.add(Conv1D(160, 10, activation='relu'))

    # >> average pooling layer: to avoid overfitting (average instead of max value)
    model.add(GlobalAveragePooling1D())

    # >> dropout layer: randomly assign 0 weights to the neurons (network becomes
    #    less sensitive to smaller variations in the data)
    #    >> rate = 0.5 (50% of neurons will receive zero weight)
    model.add(Dropout(0.5))

    # >> fully connected layer w/ softmax activation: reduction to a matrix with
    #    height 2 (done by matrix multiplication)
    # >> output value will represent probability for each class
    if num_classes == 2:
        model.add(Dense(num_classes, activation='softmax'))
    else:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

    print(model.summary())

    # >> 'categorical_crossentropy': expects binary matrices(1s and 0s) of shape
    #    (samples, classes)
    if num_classes == 2:
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
    else:
        model.compile(loss='mse', optimizer='rmsprop',
                      metrics=['accuracy', keras.metrics.Precision(),
                               keras.metrics.Recall()])

    return model

def autoencoder(input_dim = 18954, kernel_size = 3):
    '''
    Adapted from: 
    https://towardsdatascience.com/autoencoders-for-the-compression-of-stock-
    market-data-28e8c1a2da3e
    '''
    # from keras.layers import Reshape
    from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
    from keras.models import Model
    import keras.metrics
    import pdb

    # fig, axes = plt.subplots(nrows = 4, ncols=1)

    # !! how many filters should be in each conv layer? 
    # input_window = Reshape((input_dim, 1), input_shape=(input_dim,))
    input_window = Input(shape=(input_dim, 1))
    
    # input_dim (x 16) dims
    conv_1 = Conv1D(16, kernel_size, activation='relu', padding="same")(input_window)
    
    # input_dim/2 dims
    x = MaxPooling1D(2, padding="same")(conv_1)

    # input_dim/2 dims --> filter = 1?
    conv_2 = Conv1D(1, kernel_size, activation='relu', padding="same")(x)
    
    encoded = MaxPooling1D(2, padding="same")(conv_2) # input_dim/4 dims
    encoder = Model(input_window, encoded)

    # input_dim/4 dims --> filter = 1?
    conv_3 = Conv1D(1, kernel_size, activation='relu', padding="same")(encoded)
    x = UpSampling1D(2)(conv_3) # input_dim/2 dims

    # !! kernel size change in example?
    if input_dim % 4 == 0:
        filter_num = 1
    else:
        filter_num = 2
    conv_4 = Conv1D(16, filter_num, activation='relu')(x) # input_dim/2 dims
    x = UpSampling1D(2)(conv_4) # input_dim dims

    # input_dim dims --> filter = 1?
    decoded = Conv1D(1, kernel_size, activation='sigmoid', padding="same")(x)

    autoencoder = Model(input_window, decoded)
    
    print(autoencoder.summary())

    # !! optimizer adadelta ?
    autoencoder.compile(optimizer = 'adam', loss='binary_crossentropy',
                        metrics=['accuracy', keras.metrics.Precision(),
                                 keras.metrics.Recall()])
    
    return autoencoder, encoder

def simpleautoencoder(input_dim = 100, encoding_dim=3):
    from keras.layers import Input, Dense
    from keras.models import Model
    input_window = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation="relu")(input_window)
    decoded = Dense(input_dim, activation = 'sigmoid')(encoded)
    autoencoder = Model(input_window, decoded)
    #encoder = Model(input_window, encoded)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

def autoencoder2(input_dim = 18954, kernel_size = 3):
    '''
    Adapted from: 
    https://blog.keras.io/building-autoencoders-in-keras.html
    '''
    # from keras.layers import Reshape
    from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
    from keras.models import Model
    import keras.metrics
    import pdb

    input_window = Input(shape=(input_dim, 1))
    
    x = Conv1D(16, kernel_size, activation='relu', padding="same")(input_window)
    x = MaxPooling1D(2, padding="same")(x)
    x = Conv1D(8, kernel_size, activation='relu', padding="same")(x)
    x = MaxPooling1D(2, padding="same")(x)
    x = Conv1D(8, kernel_size, activation='relu', padding="same")(x)
    encoded = MaxPooling1D(2, padding="same")(x)

    x = Conv1D(8, kernel_size, activation='relu', padding="same")(encoded)
    x = UpSampling1D(2)(x)
    x = Conv1D(8, 2, activation='relu')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, kernel_size, activation='relu', padding="same")(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(1, kernel_size, activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_window, decoded)
    
    print(autoencoder.summary())

    # !! optimizer adadelta ?
    autoencoder.compile(optimizer = 'adadelta', loss='binary_crossentropy',
                        metrics=['accuracy', keras.metrics.Precision(),
                                 keras.metrics.Recall()])
    
    return autoencoder

def autoencoder3(input_dim = 18954, kernel_size = 3, latentDim = 1):
    '''
    Adapted from: 
    https://blog.keras.io/building-autoencoders-in-keras.html
    https://www.pyimagesearch.com/2020/02/17/autoencoders-with-keras-tensorflow-
    and-deep-learning/
    '''
    # from keras.layers import Reshape
    from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
    from keras.layers import Flatten, Reshape
    from keras.models import Model
    import keras.metrics
    import pdb
    import numpy as np

    k1 = int(np.ceil(input_dim/2))
    k2 = int(np.ceil(input_dim/(2**2)))
    k3 = int(np.ceil(input_dim/(2**3)))
    
    input_window = Input(shape=(input_dim, 1))
    
    x = Conv1D(16, kernel_size, activation='relu', padding="same")(input_window)
    x = MaxPooling1D(2, padding="same")(x)
    x = Conv1D(8, kernel_size, activation='relu', padding="same")(x)
    x = MaxPooling1D(2, padding="same")(x)
    x = Conv1D(8, kernel_size, activation='relu', padding="same")(x)
    x = MaxPooling1D(2, padding="same")(x)
    x = Flatten()(x)
    encoded = Dense(latentDim)(x)

    # k = int(np.ceil(input_dim/(2**3)))
    x = Dense(k3*8)(encoded)
    x = Reshape((k3, 8))(x)
    x = Conv1D(8, kernel_size, activation='relu', padding="same")(x)
    x = UpSampling1D(2)(x)
    pdb.set_trace()
    x = Conv1D(8, kernel_size, activation='relu')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, kernel_size, activation='relu', padding="same")(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(1, kernel_size, activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_window, decoded)
    
    print(autoencoder.summary())

    # !! optimizer adadelta ?
    autoencoder.compile(optimizer = 'adadelta', loss='binary_crossentropy',
                        metrics=['accuracy', keras.metrics.Precision(),
                                 keras.metrics.Recall()])
    
    return autoencoder

def autoencoder4(input_dim = 100, kernel_size = 3, latentDim = 1):
    '''
    Adapted from: 
    https://blog.keras.io/building-autoencoders-in-keras.html
    https://www.pyimagesearch.com/2020/02/17/autoencoders-with-keras-tensorflow-
    and-deep-learning/
    https://stackoverflow.com/questions/45245396/can-i-share-weights-between
    -keras-layers-but-have-other-parameters-differ
    '''
    # from keras.layers import Reshape
    from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
    from keras.layers import Flatten, Reshape, Lambda
    from keras.models import Model
    import keras.metrics
    import pdb
    import numpy as np

    import tensorflow as tf

    k1 = int(np.ceil(input_dim/2))
    k2 = int(np.ceil(input_dim/(2**2)))
    k3 = int(np.ceil(input_dim/(2**3)))
    
    input_window = Input(shape=(input_dim, 1))
    
    x = Conv1D(16, kernel_size, activation='relu', padding="same")(input_window)
    x = MaxPooling1D(2, padding="same")(x)
    # pdb.set_trace()

    # def custom_layer(tensor):
    #     return tensor + 2
    # x = Lambda(custom_layer)(x)
    x = Lambda(lambda x: tf.math.reduce_mean(x, axis=2, keepdims = True),
               output_shape=(k1, 1))(x)
    x = Conv1D(8, kernel_size, activation='relu', padding="same")(x)
    x = MaxPooling1D(2, padding="same")(x)
    x = Lambda(lambda x: tf.math.reduce_mean(x, axis=2, keepdims = True),
               output_shape=(k2, 1))(x)
    x = Conv1D(8, kernel_size, activation='relu', padding="same")(x)
    x = MaxPooling1D(2, padding="same")(x)
    x = Flatten()(x)
    encoded = Dense(latentDim)(x)

    # k = int(np.ceil(input_dim/(2**3)))
    x = Dense(k3*8)(encoded)
    x = Reshape((k3, 8))(x)
    # pdb.set_trace()
    # x = tf.math.reduce_mean(x,axis=2,keepdims=True)
    x = Lambda(lambda x: tf.math.reduce_mean(x, axis=2, keepdims = True),
               output_shape=(k3, 1))(x)
    x = Conv1D(8, kernel_size, activation='relu', padding="same")(x)
    x = UpSampling1D(2)(x)
    # x = tf.math.reduce_mean(x,axis=2,keepdims=True)
    x = Lambda(lambda x: tf.math.reduce_mean(x, axis=2, keepdims = True),
               output_shape=(k2, 1))(x)
    x = Conv1D(8, 2, activation='relu')(x)
    x = UpSampling1D(2)(x)
    # x = tf.math.reduce_mean(x,axis=2,keepdims=True)
    x = Lambda(lambda x: tf.math.reduce_mean(x, axis=2, keepdims = True),
               output_shape=(k1, 1))(x)
    x = Conv1D(16, kernel_size, activation='relu', padding="same")(x)
    x = UpSampling1D(2)(x)
    # x = tf.math.reduce_mean(x,axis=2,keepdims=True)
    x = Lambda(lambda x: tf.math.reduce_mean(x, axis=2, keepdims = True),
               output_shape=(input_dim, 1))(x)
    decoded = Conv1D(1, kernel_size, activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_window, decoded)
    
    print(autoencoder.summary())

    # !! optimizer adadelta ?
    autoencoder.compile(optimizer = 'adadelta', loss='binary_crossentropy',
                        metrics=['accuracy', keras.metrics.Precision(),
                                 keras.metrics.Recall()])
    
    return autoencoder

def autoencoder5(input_dim = 100, kernel_size = 3, latentDim = 1,strides = 1,
                 filter_num = [16,8,8,8,8,16]):
    '''
    Adapted from: 
    https://blog.keras.io/building-autoencoders-in-keras.html
    https://www.pyimagesearch.com/2020/02/17/autoencoders-with-keras-tensorflow-
    and-deep-learning/
    https://stackoverflow.com/questions/45245396/can-i-share-weights-between
    -keras-layers-but-have-other-parameters-differ
    '''
    # from keras.layers import Reshape
    from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
    from keras.layers import Flatten, Reshape, Lambda, BatchNormalization
    from keras.models import Model
    import keras.metrics
    import pdb
    import numpy as np

    import tensorflow as tf

    k1 = int(np.ceil(input_dim/2))
    k2 = int(np.ceil(input_dim/(2**2)))
    k3 = int(np.ceil(input_dim/(2**3)))

    # if normalize: input_window = BatchNormalization(input_shape=(input_dim,1))
    input_window = Input(shape=(input_dim, 1))
    x = Conv1D(filter_num[0], kernel_size, activation='relu', padding="same",
               strides=strides)(input_window)
    x = MaxPooling1D(2, padding="same")(x)
    x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims = True),
               output_shape=(k1, 1))(x)
    x = Conv1D(filter_num[1], kernel_size, activation='relu', padding="same",
               strides=strides)(x)
    x = MaxPooling1D(2, padding="same")(x)
    x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims = True),
               output_shape=(k2, 1))(x)
    x = Conv1D(filter_num[2], kernel_size, activation='relu', padding="same",
               strides=strides)(x)
    x = MaxPooling1D(2, padding="same")(x)
    x = Flatten()(x)
    encoded = Dense(latentDim)(x)

    # k = int(np.ceil(input_dim/(2**3)))
    x = Dense(k3*8)(encoded)
    x = Reshape((k3, 8))(x)
    # pdb.set_trace()
    # x = tf.math.reduce_mean(x,axis=2,keepdims=True)
    x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims = True),
               output_shape=(k3, 1))(x)
    x = Conv1D(filter_num[3], kernel_size, activation='relu', padding="same",
               strides=strides)(x)
    x = UpSampling1D(2)(x)
    # x = tf.math.reduce_mean(x,axis=2,keepdims=True)
    x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims = True),
               output_shape=(k2, 1))(x)
    x = Conv1D(filter_num[4], kernel_size, activation='relu', padding='same',
               strides=strides)(x)
    x = UpSampling1D(2)(x)
    # x = tf.math.reduce_mean(x,axis=2,keepdims=True)
    x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims = True),
               output_shape=(k1, 1))(x)
    x = Conv1D(filter_num[5], kernel_size, activation='relu', padding="same",
               strides=strides)(x)
    x = UpSampling1D(2)(x)
    # x = tf.math.reduce_mean(x,axis=2,keepdims=True)
    x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims = True),
               output_shape=(input_dim, 1))(x)
    decoded = Conv1D(1, kernel_size, activation='sigmoid', padding='same',
                     strides=strides)(x)
    
    autoencoder = Model(input_window, decoded)
    
    print(autoencoder.summary())

    # !! optimizer adadelta ?
    autoencoder.compile(optimizer = 'adadelta', loss='binary_crossentropy',
                        metrics=['accuracy', keras.metrics.Precision(),
                                 keras.metrics.Recall()])
    
    return autoencoder

def autoencoder6(input_dim=100, kernel_size=3, latentDim=1, strides=1,
                 filter_num = [16,8,8,8,8,16]):
    '''now using lambda.layers.maxium'''
    from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
    from keras.layers import Flatten, Reshape, Maximum
    from keras.models import Model
    import keras.metrics
    import pdb
    import numpy as np
    import tensorflow as tf

    k1 = int(np.ceil(input_dim/2))
    k2 = int(np.ceil(input_dim/(2**2)))
    k3 = int(np.ceil(input_dim/(2**3)))

    input_window = Input(shape=(input_dim,1))
    x = Conv1D(filter_num[0], kernel_size, activation='relu', padding='same',
               strides=strides)(input_window)
    x = MaxPooling1D(2, padding='same')(x)
    pdb.set_trace()
    x = Maximum()(tf.split(x, np.ones(filter_num[0], dtype=np.int32), 2))
    pdb.set_trace()
    x = Conv1D(filter_num[1], kernel_size, activation='relu', padding='same',
               strides=strides)(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Maximum()(tf.split(x, np.ones(filter_num[1], dtype=np.int32), 2))
    x = Conv1D(filter_num[2], kernel_size, activation='relu', padding='same',
              strides=strides)(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Flatten()(x)
    encoded = Dense(latentDim)(x)

    x = Dense(k3*8)(encoded)
    x = Reshape((k3,8))(x)
    x = Maximum()(tf.split(x, np.ones(filter_num[2], dtype=np.int32), 2))
    x = Conv1D(filter_num[3], kernel_size, activation='relu', padding='same',
               strides=strides)(x)
    x = UpSampling1D(2)(x)
    x = Maximum()(tf.split(x, np.ones(filter_num[3], dtype=np.int32), 2))
    x = Conv1D(filter_num[4], 2, activation='relu', strides=strides)(x) #!!
    x = UpSampling1D(2)(x)
    x = Maximum()(tf.split(x, np.ones(filter_num[4], dtype=np.int32), 2))
    x = Conv1D(filter_num[5], kernel_size, activation='relu', padding='same',
               strides=strides)(x)
    x = UpSampling1D(2)(x)
    x = Maximum()(tf.split(x, np.ones(filter_num[5], dtype=np.int32), 2))
    decoded = Conv1D(1, kernel_size, activation='sigmoid', padding='same',
                     strides=strides)(x)

    pdb.set_trace()

    autoencoder = Model(input_window, decoded)
    print(autoencoder.summary())
    autoencoder.compile(optimizer = 'adadelta', loss='binary_crossentropy',
                        metrics=['accuracy', keras.metrics.Precision(),
                                 keras.metrics.Recall()])
    return autoencoder
    

# :: artificial data :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def gaussian(x, a, b, c):
    '''a = height, b = position of center, c = stdev'''
    import numpy as np
    return a * np.exp(-(x-b)**2 / 2*c**2)

def signal_data(training_size = 10000, test_size = 100, input_dim = 100,
                time_max = 30., noise_level = 0.0, height = 20., center = 15.,
                stdev = 10., reshape=False):
    '''Generate training data set with flat light curves and gaussian light
    curves.
    * training_size
    * test_size
    * input_dim
    * time_max = 30. days
    * noise >= 0. (noise level as a fraction of gaussian height)
    '''
    import numpy as np

    x = np.empty((training_size + test_size, input_dim))
    y = np.copy(x)
    l = int(np.shape(x)[0]/2)
    
    # >> no peak data
    x[:l] = np.ones((l, input_dim))
    y[:l] = np.zeros((l, 1))

    # >> with peak data
    time = np.linspace(0, time_max, input_dim)
    for i in range(l):
        x[l+i] = gaussian(time, a = height, b = center, c = stdev)
    x[l:] = x[l:]/np.amax(x[l:]) + 1. # >> normalize
    y[l:] = np.ones((l, 1))

    # >> add noise
    x += np.random.normal(scale = noise_level, size = np.shape(x))

    # >> partition training and test datasets
    x_train = np.concatenate((x[:int(training_size/2)], x[l:-int(test_size/2)]))
    y_train = np.concatenate((y[:int(training_size/2)], y[l:-int(test_size/2)]))
    x_test = np.concatenate((x[int(training_size/2):l], x[-int(test_size/2):]))
    y_test = np.concatenate((y[int(training_size/2):l], y[-int(test_size/2):]))

    if reshape:
        x_train = np.reshape(x_train, (np.shape(x_train)[0], np.shape(x_train)[1], 1))
        x_test = np.reshape(x_test, (np.shape(x_test)[0], np.shape(x_test)[1], 1))
    
    return x_train, y_train, x_test, y_test

def no_signal_data(training_size = 10000, test_size = 100, input_dim = 100,
                   noise_level = 0.):
    import numpy as np

    x = np.empty((training_size + test_size, input_dim))
    y = np.copy(x)
    l = int(np.shape(x)[0]/2)
    
    # >> no peak data
    x = np.ones(np.shape(x))
    y = np.zeros((np.shape(x)[0], 1))

    # >> add noise
    x += np.random.normal(scale = noise_level, size = np.shape(x))

    # >> partition training and test datasets
    x_train = np.concatenate((x[:int(training_size/2)], x[l:-int(test_size/2)]))
    y_train = np.concatenate((y[:int(training_size/2)], y[l:-int(test_size/2)]))
    x_test = np.concatenate((x[int(training_size/2):l], x[-int(test_size/2):]))
    y_test = np.concatenate((y[int(training_size/2):l], y[-int(test_size/2):]))
    
    return x_train, y_train, x_test, y_test

                      
                      


