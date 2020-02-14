# adapted from https://keras.io/getting-started/sequential-model-guide/
# https://keras.io/examples/imdb_cnn/

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, MaxPooling2D, GlobalMaxPooling1D
import numpy as np

import gzip
import sys
import pickle
import pdb
import matplotlib.pyplot as plt

def gaussian(num_data, a, b, c):
    '''a = height, b = position of center, c = stdev'''
    x = np.linspace(0, 20, num_data)
    return  a * np.exp(-(x-b)**2 / 2*c**2) + np.random.normal(size=(num_data))

batch_size = 2000 # 1000 for each class
test_size = 50 # 25 for each class
num_classes = 2
epochs = 1

# input data dimensions
img_rows = 100

# >> generate train data

# make 1000 straight lines (feature = 0)
# >> x_train_0 shape (1000, 100)
# >> y_train_0 shape (1000, 1)
x_train_0 = np.random.normal(size = (1000, img_rows)) + 1.0
y_train_0 = np.zeros((1000, 1))

# make 1000 gaussians (feature = 1)
x_train_1 = np.zeros((1000, img_rows))
for i in range(1000):
    x_train_1[i] = gaussian(img_rows, a = 10., b = 10., c = 2.)
y_train_1 = np.ones((1000, 1))

x_train = np.concatenate((x_train_0, x_train_1), axis=0)
y_train = np.concatenate((y_train_0, y_train_1), axis=0)

# >> generate test data
x_test_0 = np.random.normal(size = (test_size, img_rows)) + 1.0
y_test_0 = np.zeros((test_size, 1))
x_test_1 = np.zeros((test_size, img_rows))
for i in range(test_size):
    x_test_1[i] = gaussian(img_rows, a = 10., b = 10., c = 2.)
y_test_1 = np.ones((test_size, 1))

x_test = np.concatenate((x_test_0, x_test_1), axis=0)
y_test = np.concatenate((y_test_0, y_test_1), axis=0)

# plot test data
plt.ion()
plt.figure(0)
plt.plot(np.linspace(0, 20, img_rows), x_test[0], '-')
n = int(test_size/2)
plt.plot(np.linspace(0, 20, img_rows), x_test[int(test_size)], '-')
plt.savefig("testdata1.png")

# filters = 250
# kernel_size = 2
# hidden_dims = 250
# >> single-input model with 2 classes (binary classification)
model = Sequential()
# model.add(Conv1D(filters, kernel_size))
# # model.add(GlobalMaxPooling1D())

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# we add a Convolution1D, which will learn filters
# # word group filters of size filter_length:
# model.add(Conv1D(filters,
#                  kernel_size,
#                  padding='valid',
#                  activation='relu',
#                  strides=1))
# # we use max pooling:
# model.add(GlobalMaxPooling1D())

# # We add a vanilla hidden layer:
# model.add(Dense(hidden_dims))
# model.add(Dropout(0.2))
# model.add(Activation('relu'))

# # We project onto a single unit output layer, and squash it with a sigmoid:
# model.add(Dense(1))
# model.add(Activation('sigmoid'))

# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
# >> optimized for a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# >> train the model, iterating on the data in batches of 32 samples
model.fit(x_train, y_train, epochs = 1, batch_size = 32, validation_data =
          (x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# >> generate output prediction
y_predict = model.predict(x_test, verbose = 0)
# print(np.shape(y_predict))
print('Prediction: ', [round(y[0]) for y in y_predict])
print('Actual: ', np.resize(y_test, (test_size*2)))

# >> plot prediction
plt.figure(1)
plt.plot(np.resize(y_test, test_size*2), [np.average(num) for num in x_test], '.')
plt.savefig('featurespace.png')

plt.figure(2)
plt.plot(np.resize(y_test, test_size*2)[0:test_size],
         [np.max(num) for num in x_test[0:test_size]], 'b.')
plt.plot(np.resize(y_test, test_size*2)[test_size:],
         [np.max(num) for num in x_test[test_size:]], 'r.')
plt.savefig('featurespace1.png')


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


# https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/
# want to: differentiate between two dfiferent light curves and figure out the
# unique feature of the light curve 2 (gaussian with amplitude a) 

# first layer: convolutional (conv) layer



# 1000 --> feature space = 0
# 1000, gaussian with amplitude a --> feature space = 1 * a

# from keras.models import Sequential
# import numpy as np
# import pdb

# # preparing synthetic data
# train_flat = np.array([[x, 1 + np.random.normal()] for x  in np.linspace(-1.0, 1.0, 100)])

# import numpy as np
# import mnist
# from keras.models import Sequential

# train_images = mnist.train_images()
# train_labels = mnist.train_labels()
# test_images = mnist.test_images()
# test_labels = mnist.test_labels()

# # Normalize the images.
# train_images = (train_images / 255) - 0.5
# test_images = (test_images / 255) - 0.5

# # Reshape the images.
# train_images = np.expand_dims(train_images, axis=3)
# test_images = np.expand_dims(test_images, axis=3)

# print(train_images.shape) # (60000, 28, 28, 1)
# print(test_images.shape)  # (10000, 28, 28, 1)

# # WIP
# model = Sequential([
#   # layers...

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=(28,28,1)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(dense(num_classes, activation='softmax'))

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])

# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))


# f = gzip.open('mnist.pkl.gz', 'rb')
# if sys.version_info < (3,):
#     data = pickle.load(f)
# else:
#     data = pickle.load(f, encoding='bytes')
# f.close()
# (x_train, y_train), (x_test, y_test) = data
# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
