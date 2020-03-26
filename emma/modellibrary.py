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
import matplotlib.pyplot as plt
import numpy as np

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
    
    return autoencoder

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
                 filter_num = [16,8,8,8,8,16], lr = 1.):
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
    from keras import optimizers
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
    adad = optimizers.adadelta(lr = lr)
    autoencoder.compile(optimizer = adad, loss='binary_crossentropy',
                        metrics=['accuracy', keras.metrics.Precision(),
                                 keras.metrics.Recall()])
    
    return autoencoder

def autoencoder6(input_dim=100, kernel_size=3, latentDim=1, strides=1,
                 filter_num = [16,8,8,8,8,16]):
    '''now using lambda.layers.maxium'''
    from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
    from keras.layers import Flatten, Reshape, Maximum, Lambda
    from keras.models import Model
    import keras.metrics
    import pdb
    import numpy as np
    # import tensorflow as tf
    import keras.backend as K

    k1 = int(np.ceil(input_dim/2))
    k2 = int(np.ceil(input_dim/(2**2)))
    k3 = int(np.ceil(input_dim/(2**3)))

    def channelPool(x):
        return K.max(x, axis=-1)

    input_window = Input(shape=(input_dim,1))
    x = Conv1D(filter_num[0], kernel_size, activation='relu', padding='same',
               strides=strides)(input_window)
    x = MaxPooling1D(2, padding='same')(x)
    x = Lambda(channelPool)(x)
    x = Reshape((int(input_dim/2), 1))(x)
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
    

def autoencoder7(x_train, x_val, y_val, params):
    '''
    Adapted from autoencoder5, able to run with talos.
    '''
    # from keras.layers import Reshape
    from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
    from keras.layers import Flatten, Reshape, Lambda, BatchNormalization
    from keras.models import Model
    import keras.metrics
    import pdb
    import numpy as np

    import tensorflow as tf

    input_dim = np.shape(x_train)[1]
    k1 = int(np.ceil(input_dim/2))
    k2 = int(np.ceil(input_dim/(2**2)))
    k3 = int(np.ceil(input_dim/(2**3)))

    # if normalize: input_window = BatchNormalization(input_shape=(input_dim,1))
    input_window = Input(shape=(input_dim, 1))
    x = Conv1D(params['filter_num'][0], kernel_size, activation=params['activation'],
               padding="same",
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

def autoencoder9(input_dim = 104, encoding_dim = 32):
    from keras.layers import Input, Dense
    from keras.models import Model

    input_img = Input(shape = (input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(input_img, decoded)
    print(autoencoder.summary())

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder

# 032520

def autoencoder10(input_dim = 160, encoding_dim = 40):
    '''https://towardsdatascience.com/
    autoencoder-on-dimension-reduction-100f2c98608c
    This works with:
    x_train, y_train, x_test, y_test = ml.signal_data(training_size=45000, 
    input_dim=160)
    model.fit(x_train, x_train, epochs = 150, batch_size=256, shuffle=True,
    validation_data=(x_test, x_test))

    Doesn't do as well when encoding_dim = 1, but still works.
    '''
    from keras.layers import Input, Dense
    from keras.models import Model

    input_df = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_df)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(input_df, decoded)

    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    return autoencoder

def autoencoder11(input_dim = 160, kernel_size=3):
    '''Adapted from autoencoder10 (now using conv layers). Requires
    input_dim divisible by 4
    Works with:
    x_train, y_train, x_test, y_test = ml.signal_data(training_size=45000,
    input_dim=160, reshape = True)
    model.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True,
    validation_data=(x_test, x_test))'''
    from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
    from keras.models import Model
    from keras import optimizers
    import keras.metrics
    input_img = Input(shape = (input_dim, 1))
    x = Conv1D(16, kernel_size, activation='relu', padding='same')(input_img)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(32, kernel_size, activation='relu', padding='same')(x)
    encoded = MaxPooling1D(2, padding='same')(x)

    x = UpSampling1D(2)(encoded)
    x = Conv1D(32, kernel_size, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(1, kernel_size, activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    print(autoencoder.summary())
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    return autoencoder

def autoencoder12(input_dim = 160, kernel_size=3):
    '''Adapted from autoencoder11 (now adding lambda layers). Requires
    input_dim divisible by 4
    Works with:
    x_train, y_train, x_test, y_test = ml.signal_data(training_size=45000,
    input_dim=160, reshape=True)
    model.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True,
    validation_data=(x_test, x_test))
    '''
    from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Lambda
    from keras.models import Model
    from keras import optimizers
    import tensorflow as tf
    import keras.metrics
    input_img = Input(shape = (input_dim, 1))
    x = Conv1D(16, kernel_size, activation='relu', padding='same')(input_img)
    x = MaxPooling1D(2, padding='same')(x)
    x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims=True),
               output_shape=(int(input_dim/2), 1))(x)
    x = Conv1D(32, kernel_size, activation='relu', padding='same')(x)
    encoded = MaxPooling1D(2, padding='same')(x)

    x = UpSampling1D(2)(encoded)
    x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims=True),
               output_shape=(int(input_dim/2), 1))(x)
    x = Conv1D(32, kernel_size, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims=True),
               output_shape=(input_dim, 1))(x)
    decoded = Conv1D(1, kernel_size, activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    print(autoencoder.summary())
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    return autoencoder

def autoencoder13(input_dim = 160, kernel_size = 3):
    '''Adapted from autoencoder12 (now with dense layer to get latent_dim =1). Requires input_dim to
    be divisible by 8.
    Doesn't work.'''
    from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Lambda, Dense, Reshape, Flatten
    from keras.models import Model
    from keras import optimizers
    import tensorflow as tf
    import keras.metrics
    input_img = Input(shape = (input_dim, 1))
    x = Conv1D(16, kernel_size, activation='relu', padding='same')(input_img)
    x = MaxPooling1D(2, padding='same')(x)
    x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims=True),
               output_shape=(int(input_dim/2), 1))(x)
    x = Conv1D(32, kernel_size, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims=True),
               output_shape=(int(input_dim/4), 1))(x)
    x = Conv1D(32, kernel_size, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims=True),
               output_shape=(int(input_dim/8), 1))(x)
    x = Flatten()(x)
    encoded = Dense(1, activation='relu')(x)

    x = Dense(int(input_dim/8), activation='relu')(encoded)
    x = Reshape((int(input_dim/8), 1))(x)
    x = Conv1D(32, kernel_size, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims=True),
               output_shape=(int(input_dim/4), 1))(x)
    x = Conv1D(32, kernel_size, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims=True),
               output_shape=(input_dim/2, 1))(x)
    x = Conv1D(16, kernel_size, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Lambda(lambda x:tf.math.reduce_max(x, axis=2, keepdims=True),
               output_shape=(input_dim, 1))(x)
    decoded = Conv1D(1, kernel_size, activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    print(autoencoder.summary())
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    return autoencoder

def autoencoder14(input_dim = 256, kernel_size=3):
    '''Adapted from autoencoder12 (now using max_size = 4 to get latent_dim=1). Requires input_dim to be divisible by 4.
    Does not work.
    '''
    from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Lambda
    from keras.models import Model
    from keras import optimizers
    import tensorflow as tf
    import keras.metrics
    input_img = Input(shape = (input_dim, 1))
    x = Conv1D(16, kernel_size, activation='relu', padding='same')(input_img)
    num_iter = int(np.log(input_dim) / np.log(4)) # >> log base 4 input_dim
    for i in range(num_iter):
        x = MaxPooling1D(4, padding='same')(x)
        x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims=True),
                   output_shape=(int(input_dim/(4**(i+1))), 1))(x)
        if i != (num_iter - 1):
            x = Conv1D(32, kernel_size, activation='relu', padding='same')(x)

    for i in range(num_iter):
        x = UpSampling1D(4)(x)
        x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims=True),
                   output_shape=(int(input_dim/(4**(num_iter-i-1))), 1))(x)
        if i == (num_iter - 1):
            decoded = Conv1D(1, kernel_size, activation='sigmoid',
                             padding='same')(x)
        else:
            x = Conv1D(32, kernel_size, activation='relu',
                             padding='same')(x)
            
    autoencoder = Model(input_img, decoded)
    print(autoencoder.summary())
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    return autoencoder

def autoencoder15(input_dim = 512, kernel_size=3):
    '''Adapted from autoencoder12 (now using max_size = 8). Requires
    input_dim to be divisible by 8.
    Doesn't work. Same number of parameters as autoencoder12. But more
    information is lost, I'm guessing.
    '''
    from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Lambda
    from keras.models import Model
    from keras import optimizers
    import tensorflow as tf
    import keras.metrics

    input_img = Input(shape = (input_dim, 1))
    x = Conv1D(16, kernel_size, activation='relu', padding='same')(input_img)
    x = MaxPooling1D(8, padding='same')(x)
    x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims=True),
               output_shape=(int(input_dim/2), 1))(x)
    x = Conv1D(32, kernel_size, activation='relu', padding='same')(x)
    encoded = MaxPooling1D(8, padding='same')(x)

    x = UpSampling1D(8)(encoded)
    x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims=True),
               output_shape=(int(input_dim/2), 1))(x)
    x = Conv1D(32, kernel_size, activation='relu', padding='same')(x)
    x = UpSampling1D(8)(x)
    x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims=True),
               output_shape=(input_dim, 1))(x)
    decoded = Conv1D(1, kernel_size, activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    print(autoencoder.summary())
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    return autoencoder

def autoencoder16(input_dim = 128, kernel_size=3):
    '''Adapted from autoencoder12 (now using a bunch of max pool layers with
    max_size still at 2 until latentdim=1.
    Does not work.
    '''
    from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Lambda
    from keras.models import Model
    from keras import optimizers
    import tensorflow as tf
    import keras.metrics
    input_img = Input(shape = (input_dim, 1))
    x = Conv1D(16, kernel_size, activation='relu', padding='same')(input_img)
    num_iter = int(np.log2(input_dim)) # >> log base 4 input_dim
    for i in range(num_iter):
        x = MaxPooling1D(2, padding='same')(x)
        x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims=True),
                   output_shape=(int(input_dim/(2**(i+1))), 1))(x)
        if i != (num_iter - 1):
            x = Conv1D(32, kernel_size, activation='relu', padding='same')(x)

    for i in range(num_iter):
        x = UpSampling1D(2)(x)
        x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims=True),
                   output_shape=(int(input_dim/(2**(num_iter-i-1))), 1))(x)
        if i == (num_iter - 1):
            decoded = Conv1D(1, kernel_size, activation='sigmoid',
                             padding='same')(x)
        else:
            x = Conv1D(32, kernel_size, activation='relu',
                             padding='same')(x)
            
    autoencoder = Model(input_img, decoded)
    print(autoencoder.summary())
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    return autoencoder

def autoencoder17(input_dim = 160, kernel_size=3):
    '''Adapted from autoencoder12 (now with dense layer).
    Works with
    x_train, y_train, x_test, y_test = ml.signal_data(training_size=45000, input_dim=128, reshape=True)
    model = ml.autoencoder17(input_dim=128, kernel_size=21)
    model.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

    '''
    from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Lambda, Reshape, Dense, Flatten
    from keras.models import Model
    from keras import optimizers
    import tensorflow as tf
    import keras.metrics
    input_img = Input(shape = (input_dim, 1))
    x = Conv1D(16, kernel_size, activation='relu', padding='same')(input_img)
    x = MaxPooling1D(2, padding='same')(x)
    x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims=True),
               output_shape=(int(input_dim/2), 1))(x)
    x = Conv1D(32, kernel_size, activation='relu', padding='same')(x)
    x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims=True),
               output_shape=(int(input_dim/2), 1))(x)
    x = Flatten()(x)
    encoded = Dense(1, activation='relu')(x)

    x = Dense(int(input_dim/2))(encoded)
    x = Reshape((int(input_dim/2), 1))(x)
    x = Conv1D(32, kernel_size, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims=True),
               output_shape=(input_dim, 1))(x)
    decoded = Conv1D(1, kernel_size, activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    print(autoencoder.summary())
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    return autoencoder

def autoencoder18(input_dim = 160, kernel_size=3):
    '''Adapted from autoencoder17 (now with dropout layers)
    https://arxiv.org/pdf/1207.0580.pdf
    use dropout after pooling layers
    
    tried using 'relu' for the last CNN layer, didn't work as well
    https://stackoverflow.com/questions/53191408/always-same-output-for-tensorflow-autoencoder

    tried without dropout layer between dense layers, worked better sometimes?

    works better with low dropout = 0.1

    works sometimes?
    x_train, y_train, x_test, y_test = ml.signal_data(input_dim=128, training_size=45000, reshape = True)
    model = ml.autoencoder18(input_dim = 128, kernel_size=21)
    model.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test, x_test))
    https://stats.stackexchange.com/questions/352036/what-should-i-do-when-my-neural-network-doesnt-learn
    
    '''
    from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Lambda, Reshape, Dense, Flatten, Dropout
    from keras.models import Model
    from keras import optimizers
    import tensorflow as tf
    import keras.metrics
    input_img = Input(shape = (input_dim, 1))
    x = Conv1D(16, kernel_size, activation='relu', padding='same')(input_img)
    x = MaxPooling1D(2, padding='same')(x)
    x = Dropout(0.1)(x)
    x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims=True),
               output_shape=(int(input_dim/2), 1))(x)
    x = Conv1D(32, kernel_size, activation='relu', padding='same')(x)
    x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims=True),
               output_shape=(int(input_dim/2), 1))(x)
    x = Flatten()(x)
    encoded = Dense(1, activation='relu')(x)

    #x = Dropout(0.1)(encoded)
    x = Dense(int(input_dim/2))(encoded)
    x = Reshape((int(input_dim/2), 1))(x)
    x = Conv1D(32, kernel_size, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Dropout(0.1)(x)
    x = Lambda(lambda x: tf.math.reduce_max(x, axis=2, keepdims=True),
               output_shape=(input_dim, 1))(x)
    decoded = Conv1D(1, kernel_size, activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    print(autoencoder.summary())
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    return autoencoder

# num_conv_layers must be an odd number
p = {'kernel_size': (21),
     'latent_dim': (1),
     'strides': (1),
     'epochs': (10),
     'dropout': (0.1),
     'num_conv_layers': (3),
     'num_filters': [[16, 32, 32]],
     'batch_size': (256),
     'activation': ['relu'],
     'optimizer': ['adadelta'],
     'last_activation': ['sigmoid'],
     'losses': ['mean_squared_error', 'binary_crossentropy']}

def autoencoder19(x_train, x_test, params):
    '''Adapted from autoencoder18 (now with variable number of conv layers)
    replaced tensor flow max function
    '''
    from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Lambda
    from keras.layers import Reshape, Dense, Flatten, Dropout
    from keras.models import Model
    from keras import optimizers
    import keras.metrics
    import keras.backend as K

    def channelPool(x):
        return K.max(x, axis=-1)

    input_dim = np.shape(x_train)[1]
    num_iter = int((params['num_conv_layers'] - 1)/2)
    
    input_img = Input(shape = (input_dim, 1))
    x = Conv1D(params['num_filters'][0], params['kernel_size'],
               activation=params['activation'], padding='same')(input_img)
    for i in range(num_iter):
        x = MaxPooling1D(2, padding='same')(x)
        x = Dropout(params['dropout'])(x)
        x = Lambda(channelPool)(x)
        x = Reshape((int(input_dim/(2**(i+1))), 1))(x)
        x = Conv1D(params['num_filters'][1+i], params['kernel_size'],
                   activation=params['activation'], padding='same')(x)
    x = Lambda(channelPool)(x)
    x = Reshape((int(input_dim/(2**(i+1))), 1))(x)
    x = Flatten()(x)
    encoded = Dense(params['latent_dim'], activation=params['activation'])(x)

    x = Dense(int(input_dim/(2**(i+1))))(encoded)
    x = Reshape((int(input_dim/(2**(i+1))), 1))(x)
    for i in range(num_iter):
        x = Conv1D(params['num_filters'][num_iter+1], params['kernel_size'],
                   activation=params['activation'], padding='same')(x)
        x = UpSampling1D(2)(x)
        x = Dropout(0.1)(x)
        x = Lambda(channelPool)(x)
        x = Reshape((int(input_dim/(2**(num_iter-i-1))), 1))(x)
    decoded = Conv1D(1, params['kernel_size'],
                     activation=params['last_activation'], padding='same')(x)

    autoencoder = Model(input_img, decoded)
    print(autoencoder.summary())
    
    autoencoder.compile(optimizer=params['optimizer'], loss=params['losses'],
                        metrics=['accuracy', keras.metrics.Precision(),
                                 keras.metrics.Recall()])

    
    history = autoencoder.fit(x_train, x_train, epochs=params['epochs'],
                              batch_size=params['batch_size'], shuffle=True,
                              validation_data=(x_test, x_test))
    
    return history, autoencoder

def autoencoder20(hp):
    '''adapted to use kerastuner
    '''
    from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Lambda
    from keras.layers import Reshape, Dense, Flatten, Dropout
    from keras.models import Model
    from keras import optimizers
    import keras.metrics
    import keras.backend as K

    def channelPool(x):
        return K.max(x, axis=-1)

    input_dim = 128
    num_filt = hp.Int('num_filters', 16, 32, step=16)
    num_iter = int((hp.Int('conv_blocks', 3, 5, default=3) - 1)/2)
    dropout = hp.Float('dropout', 0, 0.5, step=0.1, default=0.1)
    kernel_size = hp.Int('kernel_size', 21, 23, step=2)
    latentDim = hp.Int('latent_dim', 1, 3, default=1)
    
    input_img = Input(shape = (input_dim, 1))
    x = Conv1D(num_filt, kernel_size, activation='relu', padding='same')(input_img)
    for i in range(num_iter):
        x = MaxPooling1D(2, padding='same')(x)
        x = Dropout(dropout)(x)
        x = Lambda(channelPool)(x)
        x = Reshape((int(input_dim/(2**(i+1))), 1))(x)
        x = Conv1D(num_filt, kernel_size, activation='relu', padding='same')(x)
    x = Lambda(channelPool)(x)
    x = Reshape((int(input_dim/(2**(i+1))), 1))(x)
    x = Flatten()(x)
    encoded = Dense(latentDim, activation='relu')(x)

    x = Dense(int(input_dim/(2**(i+1))))(encoded)
    x = Reshape((int(input_dim/(2**(i+1))), 1))(x)
    for i in range(num_iter):
        x = Conv1D(num_filt, kernel_size, activation='relu', padding='same')(x)
        x = UpSampling1D(2)(x)
        x = Dropout(dropout)(x)
        x = Lambda(channelPool)(x)
        x = Reshape((int(input_dim/(2**(num_iter-i-1))), 1))(x)
    decoded = Conv1D(1, kernel_size, activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    print(autoencoder.summary())
    
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error',
                        metrics=['accuracy'])
    
    return autoencoder


# :: artificial data :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def gaussian(x, a, b, c):
    '''a = height, b = position of center, c = stdev'''
    import numpy as np
    return a * np.exp(-(x-b)**2 / (2*c**2))

def signal_data(training_size = 10000, test_size = 100, input_dim = 100,
                time_max = 30., noise_level = 0.0, height = 1., center = 15.,
                stdev = 0.5, min0max1 = True, reshape=False):
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
    y = np.empty((training_size + test_size))
    l = int(np.shape(x)[0]/2)
    
    # >> no peak data
    if min0max1:
        x[:l] = np.zeros((l, input_dim))
    else:
        x[:l] = np.ones((l, input_dim))
    y[:l] = 0.

    # >> with peak data
    time = np.linspace(0, time_max, input_dim)
    for i in range(l):
        x[l+i] = gaussian(time, a = height, b = center, c = stdev)

    # >> normalize
    if min0max1:
        x[l:] = x[l:] / np.amax(x[l:]) + 1.
        x[l:] = x[l:] - 1.
    else:
        x[l:] = x[l:]/np.amax(x[l:]) + 1.
    y[l:] = 1.

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
    y = np.empty((training_size + test_size))
    l = int(np.shape(x)[0]/2)
    
    # >> no peak data
    x = np.ones(np.shape(x))
    y = 0.

    # >> add noise
    x += np.random.normal(scale = noise_level, size = np.shape(x))

    # >> partition training and test datasets
    x_train = np.concatenate((x[:int(training_size/2)], x[l:-int(test_size/2)]))
    y_train = np.concatenate((y[:int(training_size/2)], y[l:-int(test_size/2)]))
    x_test = np.concatenate((x[int(training_size/2):l], x[-int(test_size/2):]))
    y_test = np.concatenate((y[int(training_size/2):l], y[-int(test_size/2):]))
    
    return x_train, y_train, x_test, y_test


# :: plotting :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 

# def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
#     print('----- activations -----')
#     activations = []
#     inp = model.input

#     model_multi_inputs_cond = True
#     if not isinstance(inp, list):
#         # only one input! let's wrap it in a list.
#         inp = [inp]
#         model_multi_inputs_cond = False

#     outputs = [layer.output for layer in model.layers if
#                layer.name == layer_name or layer_name is None]  # all layer outputs

#     funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

#     if model_multi_inputs_cond:
#         list_inputs = []
#         list_inputs.extend(model_inputs)
#         list_inputs.append(0.)
#     else:
#         list_inputs = [model_inputs, 0.]


#     print(list_inputs)
#     layer_outputs = [func(list_inputs)[0] for func in funcs]
#     for layer_activations in layer_outputs:
#         activations.append(layer_activations)
#         if print_shape_only:
#             print(layer_activations.shape)
#         else:
#             print(layer_activations)
#     return activations



def corner_plot(activation, n_bins = 40):
    '''Creates corner plot for intermediate activation with shape 
    (test_size, latentDim).
    '''
    latentDim = np.shape(activation)[1]

    fig, axes = plt.subplots(nrows = latentDim, ncols = latentDim,
                             figsize = (14, 10))

    # >> deal with 1 latent dimension case
    if latentDim == 1:
        axes.hist(np.reshape(activation, np.shape(activation)[0]), n_bins)
        axes.set_ylabel('phi1')
        axes.set_ylabel('frequency')
    else:
        # >> row 1 column 1 is first latent dimension (phi1)
        for i in range(latentDim):
            axes[i,i].hist(activation[:,i], n_bins)
            for j in range(i):
                H, xedges, yedges = np.histogram2d(activation[:,j],
                                                   activation[:,i], n_bins)
                axes[i, j].imshow(H)

            # >> x and y labels
            axes[i,0].set_ylabel('phi' + str(i))
            axes[latentDim-1,i].set_xlabel('phi' + str(i))
    return fig, axes

def kernel_filter_plot(model, layer_index):
    filters, biases = model.layers[layer_index].get_weights()
    fig, axes = plt.subplots()

def split_data(fname, train_test_ratio = 0.9, cutoff=16336, normalize=True):
    intensity = np.loadtxt(open(fname, 'rb'), delimiter=',')

    # >> truncate
    intensity = np.delete(intensity, np.arange(cutoff,np.shape(intensity)[1]),1)

    if normalize:
        # >> divide by maximum range so max spread is 1
        maxs = np.max(intensity, axis=-1)
        mins = np.min(intensity, axis=-1)
        maxrange = np.max(maxs-mins)
        intensity = intensity / maxrange
        
        # >> divide by median so centered around 1
        medians = np.median(intensity, axis = 1)
        medians = np.reshape(medians, (np.shape(medians)[0], 1))
        medians = np.repeat(medians, 16336, axis = 1)
        # medians = np.resize(medians, (np.shape(medians)[0], 1))
        # medians = np.repeat(medians, np.shape(intensity)[1], axis = 1)
        # intensity = np.divide(intensity, medians)
        intensity = intensity - medians + 0.5

        # # >> subtract 0.5 to center around 0.5
        # intensity = intensity - 0.5
    # >> reshape data
    intensity = np.resize(intensity, (np.shape(intensity)[0],
                                      np.shape(intensity)[1], 1))

    # >> split test and train data
    split_ind = int(train_test_ratio*np.shape(intensity)[0])
    x_train = np.copy(intensity[:split_ind])
    x_test = np.copy(intensity[split_ind:])

    return x_train, x_test

# --
plot = False
if plot:
    x_predict = model.predict(x_test)
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure()
    plt.plot(np.linspace(0, 30, 128), x_predict[0])
    plt.plot(np.linspace(0, 30, 128), x_predict[-1])
    plt.show()


def input_output_plot(x, x_test, x_predict, inds = [0, -14, -10]):
    fig, axes = plt.subplots(nrows=2, ncols=len(inds), figsize=(8*1.5,3*1.5),
                             sharey = True)
    for i in range(len(inds)):
        axes[0, i].plot(x, x_test[inds[i]][:,0], '.', label='input')
        axes[1, i].plot(x, x_predict[inds[i]][:,0], '.', label='output')
        axes[0, i].set_xlabel('time [days]')
        axes[1, i].set_xlabel('time [days]')
    axes[0, 0].set_ylabel('relative flux')
    axes[1, 0].set_ylabel('relative flux')
    axes[0, 1].set_title('input', fontsize=16)
    axes[1, 1].set_title('output', fontsize=16)
    return fig, axes
    
def get_activations(model):
    # >> get ind for plotting latent space
    bottleneck_ind = np.nonzero(['dense' in x.name for x in model.layers])[0][0] 
    # >> get inds for plotting intermediate activations
    act_inds = np.nonzero(['conv' in x.name or 'lambda' in x.name for x in \
                           model.layers])[0]
    # >> get inds for plotting kernel and filters
    layer_inds = np.nonzero(['conv' in x.name for x in model.layers])[0]
    
    # >> subtract one from all indices
    #    (activation_model doens't include input layer)
    act_index = np.array(act_index) - 1
    layer_index = np.array(layer_index) - 1
    
    
