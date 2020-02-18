# simple CNN (classify flares)
# feb. 2020
# adapted from https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
# and https://keras.io/getting-started/sequential-model-guide/
# * CNN works well for identifying simple patterns within data
# * effective when interesting features are shorter(fixed-length) sigments,
#   location within the segment is not very relevant

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
import numpy as np

import gzip
import sys
import pickle
import pdb
import matplotlib.pyplot as plt

# :: inputs ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

batch_size  = 2000   # >> 1000 lightcurves for each class
test_size   = 50      # >> 25 for each class
num_classes = 2
epochs      = 1
input_dim   = 100     # >> number of data points in light curve

# >> flare gaussian
height = 20.
center = 15.
stdev  = 10.
xmax   = 30.

# >> output filenames
fname_test = "./testdata2.png"
fname_fig  = "./latentspace2.png"


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def gaussian(num_data, a, b, c):
    '''a = height, b = position of center, c = stdev'''
    x = np.linspace(0, xmax, num_data)
    return  a * np.exp(-(x-b)**2 / 2*c**2) + np.random.normal(size=(num_data))

# -- generate data -------------------------------------------------------------

# >> generate training data
#    >> x_train shape (1000, 100)
#    >> y_train shape (1000, 2)

# >> make 1000 straight lines (feature = 0)
x_train_0 = np.random.normal(size = (int(batch_size/2), input_dim)) + 1.0
y_train_0 = np.zeros((int(batch_size/2), num_classes))
y_train_0[:,0] = 1.

# >> make 1000 gaussians (feature = 1)
x_train_1 = np.zeros((int(batch_size/2), input_dim))
for i in range(int(batch_size/2)):
    x_train_1[i] = gaussian(input_dim, a = height, b = center, c = stdev)
y_train_1 = np.zeros((int(batch_size/2), num_classes))
y_train_1[:,1] = 1.

x_train = np.concatenate((x_train_0, x_train_1), axis=0)
y_train = np.concatenate((y_train_0, y_train_1), axis=0)

# >> generate test data
x_test_0 = np.random.normal(size = (test_size, input_dim)) + 1.0
y_test_0 = np.zeros((test_size, num_classes))
y_test_0[:,0] = 1.
x_test_1 = np.zeros((test_size, input_dim))
for i in range(test_size):
    x_test_1[i] = gaussian(input_dim, a = height, b = center, c = stdev)
y_test_1 = np.zeros((test_size, num_classes))
y_test_1[:,1] = 1.

x_test = np.concatenate((x_test_0, x_test_1), axis=0)
y_test = np.concatenate((y_test_0, y_test_1), axis=0)

# >> plot test data
plt.ion()
plt.figure(0)
plt.title('Input light curves')
plt.plot(np.linspace(0, xmax, input_dim), x_test[0], '-',
         label = 'class 0: flat')
n = int(test_size/2)
plt.plot(np.linspace(0, xmax, input_dim), x_test[int(test_size)], '-',
         label = 'class 1: flare')
plt.xlabel('spatial frequency')
plt.ylabel('intensity')
plt.legend()
plt.savefig(fname_test)

# -- make model ----------------------------------------------------------------

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
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())

# -- validation ----------------------------------------------------------------

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 1, batch_size = 32,
          validation_data = (x_test, y_test))

y_predict = model.predict(x_test, verbose = 0)
print('Prediction: ', y_predict)
print('Actual: ', y_test)
pdb.set_trace()

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
