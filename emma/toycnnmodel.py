# simple CNN (classify flares)
# feb. 2020
# adapted from https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
# and https://keras.io/getting-started/sequential-model-guide/
# * CNN works well for identifying simple patterns within data
# * effective when interesting features are shorter(fixed-length) sigments,
#   location within the segment is not very relevant
#
# TO DO (+done, -notdone)
# - add colors to histogram
# + try with more noise (4 different noise levels)
# - 10, 20 epochs?
# + integer on xaxis (for vs. epochs plot)
# + make noise an input
#   + 1) no noise 2) exterme noise 3) comparable to signal (signal = 1 sigma)
# - precision, recall vs epoch plot?
# - add x, y labels!

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
import numpy as np

import gzip
import sys
import pickle
import pdb
import matplotlib.pyplot as plt

# :: inputs ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

num_classes = 1    
batch_size  = 1000   # >> 1000 lightcurves for each class
test_size   = 500      # >> 50 for each class
epochs      = 5
input_dim   = 100     # >> number of data points in light curve
noise       = [0.2, 0.4, 1.]    # >> signal height is 1.

all_noise   = False

# >> flare gaussian
height = 20.
center = 15.
stdev  = 10.
xmax   = 30.

n_bins = 75

# >> output filenames
fname_test = "./testdata021920.png"
fname_fig  = "./latentspace2.png"
output_dir = "./plots2:26:20/"

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def gaussian(x, a, b, c):
    '''a = height, b = position of center, c = stdev'''
    return  a * np.exp(-(x-b)**2 / 2*c**2)

# :: make model ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

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

# :: generate data :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# >> generate training data
#    >> x_train shape (1000, 100)
#    >> y_train shape (1000, 2)

# -- training data -------------------------------------------------------------

# >> make 1000 straight lines (feature = 0)
x_train_0 = np.ones((int(batch_size/2), input_dim))
y_train_0 = np.zeros((int(batch_size/2), num_classes))
if num_classes == 2:
    y_train_0[:,0] = 1.
else:
    y_train_0[:,0] = 0.

# >> make 1000 gaussians (feature = 1)
if all_noise:
    x_train_1 = np.ones((int(batch_size/2), input_dim))
    y_train_1 = np.zeros((int(batch_size/2), num_classes))
    if num_classes == 2:
        y_train_1[:,1] = 1.
    else:
        y_train_1[:,0] = 1.
else:
    x_train_1 = np.ones((int(batch_size/2), input_dim))
    x = np.linspace(0, xmax, input_dim)
    for i in range(int(batch_size/2)):
        x_train_1[i] = gaussian(x, a = height, b = center, c = stdev) + 1.
    y_train_1 = np.zeros((int(batch_size/2), num_classes))
    if num_classes == 2:
        y_train_1[:,1] = 1.
    else:
        y_train_1[:,0] = 1.

# -- test data -----------------------------------------------------------------

# >> generate test data

# >> flat
# x_test_0 = np.random.normal(size = (test_size, input_dim))
x_test_0 = np.ones((test_size, input_dim))
y_test_0 = np.zeros((test_size, num_classes))
if num_classes == 2:
    y_test_0[:,0] = 1.
else:
    y_test_0[:,0] = 0.

# >> peak
if all_noise:
    x_test_1 = np.ones((test_size, input_dim))
    y_test_1 = np.zeros((test_size, num_classes))
    if num_classes == 2:
        y_test_1[:,1] = 1.
    else:
        y_test_1[:,0] = 1.
else:    
    x_test_1 = np.ones((test_size, input_dim))
    for i in range(test_size):
        x_test_1[i] = gaussian(x, a = height, b = center, c = stdev) + 1.
    y_test_1 = np.zeros((test_size, num_classes))
    if num_classes == 2:
        y_test_1[:,1] = 1.
    else:
        y_test_1[:,0] = 1.

# >> create array to hold history
# his = []


        
# -- normalizing and plotting --------------------------------------------------
    
for j in range(len(noise)):
    # >> normalizing train data
    x_train = np.concatenate((x_train_0, x_train_1), axis=0)
    x_train = x_train/np.amax(x_train) + np.random.normal(scale = noise[j],
                                                                size = np.shape(x_train))
    y_train = np.concatenate((y_train_0, y_train_1), axis=0)

    # >> normalizing test data
    x_test = np.concatenate((x_test_0, x_test_1), axis=0)
    x_test = x_test/np.amax(x_test) + np.random.normal(scale = noise[j],
                                                             size = np.shape(x_test))
    y_test = np.concatenate((y_test_0, y_test_1), axis=0)

    # >> plot train data
    plt.ion()
    plt.figure(j)
    if all_noise: plt.title('No peak')
    else: plt.title('Noise: ' + str(noise[j]))
    plt.clf()
    x = np.linspace(0, xmax, input_dim)
    if num_classes == 2:
        inds0 = np.nonzero(y_test[:,0] == 1.)[0] # >> indices for class 0 (flat)
        inds1 = np.nonzero(y_test[:,1] == 1.)[0] # >> indices for class 1 (peak)
    else:
        inds0 = np.nonzero(y_test[:,0] == 0.)[0]
        inds1 = np.nonzero(y_test[:,0] == 1.)[0]
    for i in inds0:
        plt.plot(x, x_test[i], 'r-', alpha=0.1)
    for i in inds1:
        plt.plot(x, x_test[i], 'b-', alpha=0.1)
    plt.xlabel('time [days]')
    plt.ylabel('relative flux')
    if all_noise: plt.savefig(output_dir + 'Test_data_no_peak.png')
    else: plt.savefig(output_dir + 'Test_data_noise' + str(noise[j]) + '.png', dpi = 200)

    # :: validation ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # >> 'categorical_crossentropy': expects binary matrices(1s and 0s) of shape
    #    (samples, classes)
    if num_classes == 2:
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
    else:
        model.compile(loss='mse', optimizer='rmsprop',
                      metrics=['accuracy', keras.metrics.Precision(),
                               keras.metrics.Recall()])
    history = model.fit(x_train, y_train, epochs = epochs, batch_size = 32,
                        validation_data = (x_test, y_test))

    # pdb.set_trace()
    # -- plots vs. epochs ------------------------------------------------------
    # >> plotting accuracy
    # plt.figure(20+j)
    plt.figure(10)
    # plt.title('Noise: ' + str(noise[j]))
    plt.plot(history.history['accuracy'], label = 'Noise: ' + str(noise[j]))
    # plt.plot(history.history['val_accuracy'])
    # plt.legend(['train', 'test'], loc='upper left')

    # >> plotting loss
    # plt.figure(30+j)
    plt.figure(11)
    # plt.title('Noise: ' + str(noise[j]))
    plt.plot(history.history['loss'], label = 'Noise: ' + str(noise[j]))
    # plt.plot(history.history['val_loss'])
    # plt.legend(['train', 'test'], loc='upper left')

    plt.figure(12)
    plt.plot(history.history[list(history.history.keys())[-2]],
             label = 'Noise: ' + str(noise[j]))

    plt.figure(13)
    plt.plot(history.history[list(history.history.keys())[-1]],
             label = 'Noise: ' + str(noise[j]))

    # -- plot histogram --------------------------------------------------------
    y_predict = model.predict(x_test, verbose = 0)
    print('Prediction: ', y_predict)
    print('Actual: ', y_test)

    # >> plot y_predict
    if num_classes == 2:
        plt.figure(40+j)
        plt.title('Noise: ' + str(noise[j]))
        plt.clf()
        plt.plot(y_predict[:,0][inds0], y_predict[:,1][inds0], 'r.',
                 label='Class 0: flat')
        plt.plot(y_predict[:,0][inds1], y_predict[:,1][inds1], 'b.',
                 label='Class 1: peak')
        plt.legend()

        fig, ax = plt.subplots()
        ax.set_xlabel('p0')
        ax.set_ylabel('N(p0)')
        ax.hist([y_predict[:,0][inds0], y_predict[:,1][inds1]], n_bins,
                color = ['red', 'blue'], label = ['flat', 'peak'])
        ax.set_xlim(left=0., right=1.)
    else:
        # N, bins, patches = ax.hist(data, edgecolor='white', linewidth=1)

        # for k in range(0,3):
        #     patches[k].set_facecolor('b')
        # for k in range(3,5):    
        #     patches[k].set_facecolor('r')
        # for k in range(5, len(patches)):
        #     patches[k].set_facecolor('black')
        # >> histogram
        fig, ax = plt.subplots()
        # want y_predict to have shape (2, 50)
        ax.set_title('Noise: ' + str(noise[j]))
        ax.set_xlabel('p0')
        ax.set_ylabel('N(p0)')
        ax.hist([y_predict[:,0][inds0], y_predict[:,0][inds1]],
                n_bins,
                color=['red', 'blue'], label=['flat', 'peak'])
        if all_noise: plt.savefig(output_dir + 'Histogram_no_peak.png')
        else: plt.savefig(output_dir + 'Histogram_noise' + str(noise[j]) + '.png')

        # plt.hist(y_predict[:,0], bins = 300)
        # plt.plot(y_predict[:,0][inds0], 'r.')
        # plt.plot(y_predict[:,0][inds1], 'b.')
        #plt.show()


plt.figure(10)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.xticks(range(epochs))
plt.legend()
plt.savefig(output_dir + 'accuracy_vs_epoch.png')


plt.figure(11)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xticks(range(epochs))
plt.legend()
plt.savefig(output_dir + 'loss_vs_epoch.png')

plt.figure(12)
plt.ylabel('precision')
plt.xlabel('epoch')
plt.xticks(range(epochs))
plt.legend()
plt.savefig(output_dir + 'precision_vs_epoch.png')

plt.figure(13)
plt.ylabel('recall')
plt.xlabel('epoch')
plt.xticks(range(epochs))
plt.legend()
plt.savefig(output_dir + 'recall_vs_epoch.png')

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# # >> generate output prediction
# y_predict = model.predict(x_test, verbose = 0)
# # print(np.shape(y_predict))
# print('Prediction: ', [round(y[0]) for y in y_predict])
# print('Actual: ', np.resize(y_test, (test_size*2)))

# # >> plot prediction
# plt.figure(1)
# plt.plot(np.resize(y_test, test_size*2), [np.average(num) for num in x_test], '.')
# plt.savefig('latentspace.png')

# plt.figure(2)
# plt.plot(np.resize(y_test, test_size*2)[0:test_size],
#          [np.max(num) for num in x_test[0:test_size]], 'b.')
# plt.plot(np.resize(y_test, test_size*2)[test_size:],
#          [np.max(num) for num in x_test[test_size:]], 'r.')
# plt.savefig(fname_fig)
