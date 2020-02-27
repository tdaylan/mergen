# Pipeline to test models in modellibrary.py. Generates artificial data of
# flat lines and gaussians with some noise.
# Feb. 2020


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
import modellibrary as ml

# :: inputs ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

num_classes = 1    
batch_size  = 1000  # >> batch_size for each class
test_size   = 100   # >> test_size/2 for each class
epochs      = 5
input_dim   = 100    # >> number of data points in light curve
noise       = [0.2, 0.4, 1., 'all_noise']    # >> given as a fraction of signal
                                             #    height
                                             # >> 'all_noise' => random signal

# >> fake data: gaussian
height = 20.
center = 15.
stdev  = 10.
xmax   = 30.

n_bins = 75

# >> output filenames
# output_dir = "./"
output_dir = "./plots22620-ac/"

# >> figure numbers
#    * [0, j]     plots of light curves for noise[0] to noise[j]
#    * 10         accuracy, loss vs epochs
#    * 11         precision, recall vs epochs
#    * [12, 12+j] histograms


# :: model :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

model = ml.autoencoder(input_dim=100, kernel_size = 3)
# model = ml.simplecnn(input_dim, num_classes)

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def gaussian(x, a, b, c):
    '''a = height, b = position of center, c = stdev'''
    return  a * np.exp(-(x-b)**2 / 2*c**2)

# :: make model ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# >> make figures for vs. epoch plots
fig1, ax1 = plt.subplots(num = 10)
ax2 = ax1.twinx()

fig3, ax3 = plt.subplots(num = 11)
ax4 = ax3.twinx()

# :: generate data :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# >> generate training data
#    >> x_train shape (1000, 100)
#    >> y_train shape (1000, 2)

        
# -- normalizing and plotting --------------------------------------------------
    
for j in range(len(noise)):

    # -- training data -------------------------------------------------------------
    
    # >> make 1000 straight lines (feature = 0)
    x_train_0 = np.ones((int(batch_size/2), input_dim))
    y_train_0 = np.zeros((int(batch_size/2), num_classes))
    if num_classes == 2:
        y_train_0[:,0] = 1.
    else:
        y_train_0[:,0] = 0.

    # >> make 1000 gaussians (feature = 1)
    if noise[j] == "all_noise":
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
    if noise[j] == "all_noise":
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

    if noise[j] == "all_noise":
        noise_level = 0.4
    else:
        noise_level = noise[j]

    # >> normalizing train data
    x_train = np.concatenate((x_train_0, x_train_1), axis=0)
    x_train = x_train/np.amax(x_train) + np.random.normal(scale = noise_level,
                                                                size = np.shape(x_train))
    y_train = np.concatenate((y_train_0, y_train_1), axis=0)

    # >> normalizing test data
    x_test = np.concatenate((x_test_0, x_test_1), axis=0)
    x_test = x_test/np.amax(x_test) + np.random.normal(scale = noise_level,
                                                             size = np.shape(x_test))
    y_test = np.concatenate((y_test_0, y_test_1), axis=0)

    # >> plot train data
    plt.ion()
    plt.figure(j)
    plt.title('Noise: ' + str(noise[j]))
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
    plt.savefig(output_dir + 'Test_data_noise' + str(noise[j]) + '.png', dpi = 200)

    # :: validation ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    history = model.fit(x_train, y_train, epochs = epochs, batch_size = 32,
                        validation_data = (x_test, y_test))
    
    # -- plots vs. epochs ------------------------------------------------------

    ax1.plot(history.history['accuracy'], label = 'Accuracy (noise: ' + \
             str(noise[j]) + ')')
    ax2.plot(history.history['loss'], '--', label = 'Loss, (noise: ' + \
             str(noise[j]) + ')')
    ax3.plot(history.history[list(history.history.keys())[-2]],
             label = 'Precision (noise: ' + str(noise[j]) + ')')
    ax4.plot(history.history[list(history.history.keys())[-1]], '--',
             label = 'Recall (noise: ' + str(noise[j]) + ')')

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
        # >> histogram
        fig, ax = plt.subplots()
        ax.set_title('Noise: ' + str(noise[j]))
        ax.set_xlabel('p0')
        ax.set_ylabel('N(p0)')
        ax.hist([y_predict[:,0][inds0], y_predict[:,0][inds1]],
                n_bins,
                color=['red', 'blue'], label=['flat', 'peak'])
        plt.savefig(output_dir + 'Histogram_noise' + str(noise[j]) + '.png')

# >> label vs. epochs plots
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax2.set_ylabel('loss')
ax1.set_xticks(range(epochs))
ax1.legend(loc = 'upper right', fontsize = 'x-small')
ax2.legend(loc = 'upper left', fontsize = 'x-small')
fig1.savefig(output_dir + 'accuracy_loss_vs_epoch.png')

ax3.set_xlabel('epoch')
ax3.set_ylabel('precision')
ax4.set_ylabel('recall')
ax3.set_xticks(range(epochs))
ax3.legend(loc = 'upper right', fontsize = 'x-small')
ax4.legend(loc = 'lower right', fontsize = 'x-small')
fig3.savefig(output_dir + 'recall_precision_vs_epoch.png')

    
