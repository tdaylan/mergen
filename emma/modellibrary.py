# keras model library
# Feb. 2020

# Includes:
# * simple CNN (1D and 2D)
# * autoencoder

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

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

def autoencoder(input_dim = 18954, kernel_size = 100):
    from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
    from keras.models import Model
    # input_img = Reshape((input_dim, 1), input_shape=(input_dim,))
    input_img = Input(shape=(input_dim, 1))

    # output shape 18954 x 64
    x = Conv1D(64, kernel_size, activation='relu')(input_img)

    # output shape 9477 x 64
    x = MaxPooling1D(3)(x)

    x = Conv1D(64, kernel_size, activation='relu')(x)
    x = MaxPooling1D(3)(x)
    x = Conv1D(64, kernel_size, activation='relu')(x)
    x = UpSampling1D(3)(x)
    x = Conv1D(64, kernel_size, activation='relu')(x)
    x = UpSampling1D(3)(x)
    x = Conv1D(64, kernel_size, activation='relu')(x)

    autoencoder = Model(input_img, x)
    autoencoder.compile(optimizer = 'adadelta', loss='binary_crossentropy')
    return autoencoder
