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

# >> model
auto = True
cnn = False

num_classes = 1    
training_size  = 10000  # >> training_size for each class
test_size   = 10   # >> test_size/2 for each class
epochs      = 5
input_dim   = 100    # >> number of data points in light curve
noise = [0., 0.2]
# noise       = [0.2, 0.4, 1., 'all_noise']    # >> given as a fraction of signal
                                             #    height
                                             # >> 'all_noise' => random signal
batch_size = 32

latent_dims = [1]
                                             
reshape = True
                                             
# >> fake data: gaussian
height = 20.
center = 15.
stdev  = 10.
xmax   = 30.

n_bins = 75

# >> output filenames
# output_dir = "./"
output_dir = "./plots031520/"

all_gaus = False

# >> figure numbers
#    * [0, j]     plots of light curves for noise[0] to noise[j]
#    * 10         accuracy, loss vs epochs
#    * 11         precision, recall vs epochs
#    * [12, 12+j] histograms


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
    if all_gaus:
        x_train_0 = np.ones((int(training_size/2), input_dim))
        x = np.linspace(0, xmax, input_dim)
        for i in range(int(training_size/2)):
            x_train_0[i] = gaussian(x, a = height, b = center, c = stdev) + 1.
        y_train_0 = np.zeros((int(training_size/2), num_classes))
        if num_classes == 2:
            y_train_0[:,1] = 1.
        else:
            y_train_0[:,0] = 1.
    else:
        x_train_0 = np.ones((int(training_size/2), input_dim))
        y_train_0 = np.zeros((int(training_size/2), num_classes))
        if num_classes == 2:
            y_train_0[:,0] = 1.
        else:
            y_train_0[:,0] = 0.

    # >> make 1000 gaussians (feature = 1)
    if noise[j] == "all_noise":
        x_train_1 = np.ones((int(training_size/2), input_dim))
        y_train_1 = np.zeros((int(training_size/2), num_classes))
        if num_classes == 2:
            y_train_1[:,1] = 1.
        else:
            y_train_1[:,0] = 1.
    else:
        x_train_1 = np.ones((int(training_size/2), input_dim))
        x = np.linspace(0, xmax, input_dim)
        for i in range(int(training_size/2)):
            x_train_1[i] = gaussian(x, a = height, b = center, c = stdev) + 1.
        y_train_1 = np.zeros((int(training_size/2), num_classes))
        if num_classes == 2:
            y_train_1[:,1] = 1.
        else:
            y_train_1[:,0] = 1.

    # -- test data -----------------------------------------------------------------

    # >> generate test data

    # >> flat
    # x_test_0 = np.random.normal(size = (test_size, input_dim))
    if all_gaus:
        x_test_0 = np.ones((test_size, input_dim))
        for i in range(test_size):
            x_test_0[i] = gaussian(x, a = height, b = center, c = stdev) + 1.
        y_test_0 = np.zeros((test_size, num_classes))
        if num_classes == 2:
            y_test_0[:,1] = 1.
        else:
            y_test_0[:,0] = 1.
    else:
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

    # >> reshape data to (1000, 100, 1)
    if reshape:
        x_train = np.resize(x_train, (np.shape(x_train)[0], np.shape(x_train)[1], 1))
        y_train = np.resize(y_train, (np.shape(y_train)[0], np.shape(y_train)[1], 1))
        x_test = np.resize(x_test, (np.shape(x_test)[0], np.shape(x_test)[1], 1))
        y_test = np.resize(y_test, (np.shape(y_test)[0], np.shape(y_test)[1], 1))

    # :: model :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    for n in range(len(latent_dims)):
        latentDim = latent_dims[n]
        # model = ml.simpleautoencoder(input_dim = input_dim)
        if auto:
            # model, encoder = ml.autoencoder(input_dim=input_dim, kernel_size = 10)
            # model = ml.autoencoder2(input_dim=input_dim, kernel_size = 10)
            # model = ml.autoencoder3(input_dim=input_dim, latentDim = latentDim)
            model = ml.autoencoder4(input_dim=input_dim, latentDim = latentDim)
            # act_index = [1,3,5,8,11,13,15]
            # act_index = [1,4,7,10,14,17,20]
            # layer_index = [1,3,5,11,13,15]
            # layer_index = [1,4,7,14,17,20]
            layer_index = np.nonzero(['conv' in x.name for x in model.layers])[0]
            bottleneck_ind = np.nonzero(['dense' in x.name for x in \
                                         model.layers])[0][0]
            act_index = list(layer_index).append(bottleneck_ind)
        elif cnn:
            model  = ml.simplecnn(input_dim, num_classes)

        # -- fit model ---------------------------------------------------------

        if cnn:
            history = model.fit(x_train, y_train, epochs = epochs,
                                batch_size = batch_size,
                                validation_data = (x_test, y_test))

        if auto:
            history = model.fit(x_train, x_train, epochs = epochs,
                                batch_size = batch_size, shuffle = True,
                                validation_data = (x_test, x_test))

        # :: plots :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # -- plots vs. epochs --------------------------------------------------

        print('plotting accuracy vs. epoch')
        ax1.plot(history.history['accuracy'], label = 'Accuracy (noise: ' + \
                 str(noise[j]) + ')')
        print('plotting loss vs. epoch')
        ax2.plot(history.history['loss'], '--', label = 'Loss, (noise: ' + \
                 str(noise[j]) + ')')
        print('plotting precision vs. epoch')
        ax3.plot(history.history[list(history.history.keys())[-2]],
                 label = 'Precision (noise: ' + str(noise[j]) + ')')
        print('plotting recall vs. epoch')
        ax4.plot(history.history[list(history.history.keys())[-1]], '--',
                 label = 'Recall (noise: ' + str(noise[j]) + ')')

        # -- get model predict -------------------------------------------------
        y_predict = model.predict(x_test)

        # >> plot y_predict
        if cnn:
            # -- plot histogram ------------------------------------------------
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
                plt.savefig(output_dir + 'Histogram_noise' + \
                            str(noise[j]) + '.png')

        if auto:
            # -- plot decoded data ---------------------------------------------
            print('plotting decoded lc')
            plt.figure(40 + j)
            for k in range(np.shape(y_predict)[0]):
                plt.plot(x, y_predict[k][:,0], '-', alpha=0.1)
            plt.savefig(output_dir + 'latentdim' + str(latentDim) + \
                        'decoded_noise' + str(noise[j]) + '.png')

            # -- plot 8 input vs. output ---------------------------------------
            print('plotting input vs. output')
            fig, axes = plt.subplots(nrows = 2, ncols = 8, figsize = (14, 10.))
            for k in range(4):
                axes[0,k].plot(x, x_test[k][:,0])
                axes[1,k].plot(x, y_predict[k][:,0])
            for k in range(4,8):
                axes[0,k].plot(x, x_test[test_size + k][:,0])
                axes[1,k].plot(x, y_predict[test_size + k][:,0])
            axes[0,0].set_ylabel('input')
            axes[1,0].set_ylabel('output')
            plt.savefig(output_dir + 'latentdim' + str(latentDim) + 'noise' + \
                        str(noise[j]) + 'input_output' + '.png')

            # -- visualizing intermediate activations --------------------------

            from keras.models import Model
            layer_outputs = [layer.output for layer in model.layers]
            activation_model = Model(inputs=model.input, outputs=layer_outputs)
            for l in range(2):
                if l == 0: # >> get intermediate activations for x_test
                    activations = activation_model.predict(x_test)
                    # act_index = [1,3,5,8,11,13,15]
                    # act_index = [1,4,7,10,14,17,20]
                    pdb.set_trace()

                if l == 1:
                    activations = activation_model.predict(x_train)
                    act_index = [bottleneck_ind]
                for a in act_index:
                    activation = activations[a]
                    if a == bottleneck_ind: # a == 8:
                        nrows = 1
                        ncols = 1
                    elif np.shape(activation)[2] == bottleneck_ind:
                        nrows = 1
                        ncols = 8
                    else:
                        nrows = 2
                        ncols = 8
                    fig, axes = plt.subplots(nrows = nrows, ncols = ncols,
                                             squeeze=False, figsize=(14, 10))
                    fig.suptitle(layer_outputs[a].name)

                    if a == bottleneck_ind: #8:
                        if l == 0:
                            axes[0,0].hist(np.reshape(activation, test_size*2),
                                           20)
                        if l == 1:
                            axes[0,0].hist(np.reshape(activation,
                                                      training_size), 75)
                    else:
                        # >> loop through filters
                        for b in range(max(np.shape(activation)[2], 1)):
                            for k in range(len(activation)): # >> loop lc
                                if b/8 < 1:
                                    axes[0,b].plot(np.linspace(0,30,
                                                               np.shape(activation)[1]),
                                                   activation[k][:,b],
                                                   'b', alpha = 0.1)
                                else:
                                    axes[1,b-8].plot(np.linspace(0, 30,
                                                                 np.shape(activation)[1]),
                                                     activation[k][:,b],
                                                     'b', alpha = 0.1)
                    if l == 0:
                        fig.savefig(output_dir + 'latentdim' + str(latentDim) +\
                                    'noise' + str(noise[j]) + \
                                    layer_outputs[a].name.split('/')[0] +\
                                    'x_test'+ '.png')

                    if l == 1:
                        fig.savefig(output_dir + 'latentdim' + str(latentDim) +\
                                    'noise' + str(noise[j]) + \
                                    layer_outputs[a].name.split('/')[0] +\
                                    'x_train'+'.png')

            # -- visualizing kernel x filter for each layer --------------------
            # layer_index = [1,3,5,11,13,15]
            # layer_index = [1,4,7,14,17,20]
            pdb.set_trace()
            for a in layer_index:
                filters, biases = model.layers[a].get_weights()
                plt.figure()
                plt.imshow(np.reshape(filters, (np.shape(filters)[0],
                                                np.shape(filters)[2])))
                plt.savefig(output_dir + 'latentdim' + str(latentDim) + \
                            'noise' + str(noise[j]) + \
                            model.layers[a].name + 'fspace.png')

    # -- vs. epochs plots ------------------------------------------------------

    # >> label vs. epochs plots
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax2.set_ylabel('loss')
    ax1.set_xticks(range(epochs))
    ax1.legend(loc = 'upper left', fontsize = 'x-small')
    ax2.legend(loc = 'upper right', fontsize = 'x-small')
    fig1.savefig(output_dir + 'accuracy_loss_vs_epoch.png')

    ax3.set_xlabel('epoch')
    ax3.set_ylabel('precision')
    ax4.set_ylabel('recall')
    ax3.set_xticks(range(epochs))
    ax3.legend(loc = 'upper right', fontsize = 'x-small')
    ax4.legend(loc = 'lower right', fontsize = 'x-small')
    fig3.savefig(output_dir + 'recall_precision_vs_epoch.png')


    


# def display_activation(activations, col_size, row_size, act_index): 
#     activation = activations[act_index]
#     activation_index=0
#     fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
#     for row in range(0,row_size):
#         for col in range(0,col_size):
#             ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
#             activation_index += 1
