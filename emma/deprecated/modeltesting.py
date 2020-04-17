# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# 
# Pipeline to test models in modellibrary.py. Generates artificial data of
# flat lines and gaussians with some noise.
# emma feb 2020
#
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
import csv
import matplotlib.pyplot as plt
import modellibrary as ml
from sklearn.metrics import confusion_matrix

# :: inputs ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# >> model
auto = True # >> autoencoder
cnn = False # >> simple convolutional neural network

epochs = 3
n_bins = 40 # >> histogram
output_dir = "./plots/plots032520-2/"
prefix = "sector20-"

# !! parameters
# kernel_size = [11]
# strides = [1]
# filter_num = [[16,8,8,8,8,16]]
# batch_size = [8]
# latent_dims = [2]
kernel_size = [155]
strides = [1]
filter_num=[[16,8,8,8,8,16]]
batch_size = [100]
latent_dims = [1]

fake_data = False

# >> real data?
fname_time = "./tessdatasector20-time.txt"
# fname_time = "./12lightcurves-time.txt"
fname_intensity = "./tessdatasector20-intensity.csv"
# fname_intensity = './12lightcurves-intensity.csv'
cutoff = 16336 # >> truncate light curves to 16336 data points
tt_ratio = 0.9 # >> train to test ratio
train_size = 8 # >> set classified case
classified = False
fname_class = './12lightcurves-classification.txt'

# >> fake data?
artificial_data_signal = True
artificial_data_no_signal = False
noise = [0.] # >> given as a fraction of signal height

training_size  = 50000  # >> training_size for each class
test_size   = 100       # >> test_size/2 for each class
input_dim   = 1000       # >> number of data points in light curve
                                             
height = 2. # >> gaussian
center = 15.
stdev  = 0.5
xmax   = 30.

# :: parameters ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# p = {'kernel_size': [21],
#      'latent_dim': [1, 3],
#      'strides': [1],
#      'epochs': [4],
#      'dropout': [0.1],
#      'num_conv_layers': [3],
#      'num_filters': [[16, 32, 32]],
#      'batch_size': [256],
#      'activation': ['relu'],
#      'optimizer': ['adadelta'],
#      'last_activation': ['sigmoid'],
#      'losses': ['mean_squared_error', 'binary_crossentropy']}



# :: generate data :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


# >> update prefix (if varying parameter)
if type(ind) == int:
    prefix = prefix_0 + p_names[ind] + str(p[j])

if fake_data: # -- artificial data -----------------------------------------
    noise_level = noise[j]
    x_train, y_train, x_test, y_test = \
        ml.signal_data(training_size = training_size,
                       test_size = test_size,
                       input_dim = input_dim,
                       time_max = xmax,
                       noise_level = noise_level,
                       height = height,
                       center = center,
                       stdev = stdev,
                       reshape=True)
    # >> plot artificial data
    plt.figure()
    plt.title('Noise: ' + str(noise[j]))
    plt.clf()
    x = np.linspace(0, xmax, input_dim)
    inds0 = np.nonzero(y_test == 0.)[0]
    inds1 = np.nonzero(y_test == 1.)[0]
    for i in inds0:
        plt.plot(x, x_test[i], 'r-', alpha=0.1)
    for i in inds1:
        plt.plot(x, x_test[i], 'b-', alpha=0.1)
    plt.xlabel('time [days]')
    plt.ylabel('relative flux')
    plt.savefig(output_dir + prefix + 'test_data.png')
    plt.close()

else: # -- tess data -------------------------------------------------------
    print('loading tess data')
    x = np.loadtxt(fname_time)
    if classified:
        y = np.loadtxt(fname_class)
    intensity = np.loadtxt(open(fname_intensity, "rb"), delimiter=',')

    # >> truncate
    x = np.delete(x, np.arange(cutoff, np.shape(x)[0]), 0)
    intensity = np.delete(intensity, np.arange(cutoff,
                                               np.shape(intensity)[1]), 1)
    input_dim = cutoff

    # >> normalize data (divide by median)
    medians = np.median(intensity, axis = 1)
    medians = np.resize(medians, (np.shape(medians)[0], 1))
    medians = np.repeat(medians, np.shape(intensity)[1], axis = 1)
    intensity = np.divide(intensity, medians)

    # >> reshape data
    intensity = np.resize(intensity, (np.shape(intensity)[0],
                                      np.shape(intensity)[1], 1))

    # >> split test and train data
    if classified:
        split_ind = train_size
        y_train = np.copy(y[:split_ind])
        y_test = np.copy(y[split_ind:])
    else:
        split_ind = int(tt_ratio*np.shape(intensity)[0])
    x_train = np.copy(intensity[:split_ind])
    x_test = np.copy(intensity[split_ind:])

    # >> plot
    if classified:
        print('plotting training data')
        colors = ['r', 'b', 'g', 'm']
        fig1, axes1 = plt.subplots()
        fig2, axes2 = plt.subplots()
        for c in np.unique(y_test): # >> loop through classes
            inds1 = np.nonzero(y_train == c)[0]
            for i in inds1: # >> loop through light curves
                axes1.plot(x, x_train[i], color[int(c)]+'-', alpha=0.5,
                         label=str(c))
            inds2 = np.nonzero(y_test == c)[0]
            for i in inds2:
                axes2.plot(x, x_test[i], color[int(c)]+'-', alpha=0.5,
                         label=str(c))
        axes1.set_xlabel('time [days]')
        axes2.set_xlabel('time [days]')
        axes1.set_ylabel('relative flux')
        axes2.set_ylabel('relative flux')
        axes1.legend()
        axes2.legend()
        fig1.savefig(output_dir + prefix + 'train_data.png')
        fig2.savefig(output_dir + prefix + 'test_data.png')
        plt.close(fig1)
        plt.close(fig2)

    pdb.set_trace()
# :: model :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

latentDim = latent_dims[j]
if auto:
    print('loading model')
    model = ml.autoencoder5(input_dim=input_dim,
                            kernel_size = kernel_size[j],
                            latentDim = latentDim,
                            strides=strides[j],
                            filter_num=filter_num[j], lr = 0.2)
    act_index = np.nonzero(['conv' in x.name or 'lambda' in \
                              x.name for x in model.layers])[0]
    layer_index = np.nonzero(['conv' in x.name for x in model.layers])[0]
    bottleneck_ind = np.nonzero(['dense' in x.name for x in \
                                 model.layers])[0][0]
    act_index = list(act_index) + [bottleneck_ind]
    act_index = np.array(act_index) - 1
    layer_index = np.array(layer_index) - 1
    bottleneck_ind = bottleneck_ind - 1
elif cnn:
    model  = ml.simplecnn(input_dim, num_classes)

# -- fit model -------------------------------------------------------------

if cnn:
    history = model.fit(x_train, y_train, epochs = epochs,
                        batch_size = batch_size[j],
                        validation_data = (x_test, y_test))

if auto:
    print('training model')
    history = model.fit(x_train, x_train, epochs = epochs,
                        batch_size = batch_size[j], shuffle = True,
                        validation_data = (x_test, x_test))

# -- parameter summary -----------------------------------------------------

f = open(output_dir + prefix + "parameter_summary.txt", "w+")
f.write('kernel_size ' + str(kernel_size[j]) + '\n')
f.write('strides ' + str(strides[j]) + '\n')
f.write('filter_num ' + str(filter_num[j]) + '\n')
f.write('batch_size ' + str(batch_size[j]) + '\n')
f.write('latent_dim ' + str(latent_dims[j]) + '\n')
f.close()

# :: plots :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# -- plots vs. epochs ------------------------------------------------------

# >> make figures for vs. epoch plots
fig1, ax1 = plt.subplots()
ax2 = ax1.twinx()

fig3, ax3 = plt.subplots()
ax4 = ax3.twinx()
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
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax2.set_ylabel('loss')
ax1.set_xticks(range(epochs))
ax1.legend(loc = 'upper left', fontsize = 'x-small')
ax2.legend(loc = 'upper right', fontsize = 'x-small')
fig1.savefig(output_dir + prefix + 'accuracy_loss_vs_epoch.png')
plt.close(fig1)

ax3.set_xlabel('epoch')
ax3.set_ylabel('precision')
ax4.set_ylabel('recall')
ax3.set_xticks(range(epochs))
ax3.legend(loc = 'upper right', fontsize = 'x-small')
ax4.legend(loc = 'lower right', fontsize = 'x-small')
fig3.savefig(output_dir + prefix + 'recall_precision_vs_epoch.png')
plt.close(fig3)

# -- get model predict -----------------------------------------------------
y_predict = model.predict(x_test)

# -- confusion matrix ------------------------------------------------------
if fake_data:
    y_pred = np.argmax(y_predict, axis = 1)
    y_pred = np.round(np.reshape(y_pred, (np.shape(y_pred)[0])))

    print('confusion matrix')
    print(confusion_matrix(y_test, y_pred))

# if fake_data == False and classified:
#     y_pred = 0.

# >> plot y_predict
if cnn:
    # -- plot histogram ----------------------------------------------------
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
        plt.savefig(output_dir + prefix + 'histogram.png')
        plt.close(fig)

if auto:
    # -- plot decoded data ---------------------------------------------
    print('plotting decoded lc')
    if classified:
        plt.figure()
        plt.plot(x, y_predict[0][:,0], 'r-', alpha = 0.5, label = '1')
        plt.plot(x, y_predict[1][:,0], 'b-', alpha = 0.5, label = '2')
        plt.plot(x, y_predict[2][:,0], 'g-', alpha = 0.5, label = '3')
        plt.plot(x, y_predict[3][:,0], 'm-', alpha = 0.5, label = '4')
        plt.xlabel('time [days]')
        plt.ylabel('relative flux')
        plt.legend()
        plt.savefig(output_dir + prefix + 'decoded.png')
        plt.close()
    else:
        plt.figure(40 + j)
        for k in range(np.shape(y_predict)[0]):
            plt.plot(x, y_predict[k][:,0], '-', alpha=0.1)
        plt.xlabel('time [days]')
        plt.ylabel('relative flux')
        plt.savefig(output_dir + prefix + 'decoded.png')
        plt.close()

    # -- plot 8 input vs. output ---------------------------------------
    print('plotting input vs. output')
    if classified:
        fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (14, 10.))
        for k in range(4):
            axes[0, k].plot(x, x_test[k][:,0])
            axes[1, k].plot(x, y_predict[k][:,0])
        axes[0,0].set_ylabel('relative flux')
        axes[1,0].set_ylabel('relative flux')
        axes[1,2].set_xlabel('time [days]')
        axes[0,2].set_title('input')
        axes[1,2].set_title('output')
        plt.savefig(output_dir + prefix + 'input_output.png')
        plt.close(fig)
    else:
        fig, axes = plt.subplots(nrows = 2, ncols = 8, figsize = (14, 10.))
        for k in range(4):
            axes[0,k].plot(x, x_test[k][:,0])
            axes[1,k].plot(x, y_predict[k][:,0])
        for k in range(4,8):
            axes[0,k].plot(x, x_test[int(test_size/2) + k][:,0])
            axes[1,k].plot(x, y_predict[int(test_size/2) + k][:,0])
        axes[0,0].set_ylabel('relative flux')
        axes[1,0].set_ylabel('relative flux')
        axes[1,3].set_xlabel('time [days]')
        axes[0,3].set_title('input')
        axes[1,3].set_title('output')
        plt.savefig(output_dir + prefix + 'input_output.png')
        plt.close(fig)

    # -- animation ---------------------------------------------------------
    if classified:
        # >> plot first image in animation
        for m in range(np.shape(x_test)[0]):
            plt.figure()
            plt.plot(np.linspace(x[0], x[-1], np.shape(x_test)[1]),
                     np.resize(x_test[m], (np.shape(x_test[m])[0])))
            plt.xlabel('time [days]')
            plt.ylabel('relative flux')
            plt.savefig(output_dir + 'animation' + str(m) + '-act0.png')
            plt.close()

    # -- visualizing intermediate activations --------------------------

    print('plotting intermediate activations')
    from keras.models import Model
    layer_outputs = [layer.output for layer in model.layers][1:]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    for l in range(2):
        if l == 0: # >> get intermediate activations for x_test
            activations = activation_model.predict(x_test)
            # act_index = [1,3,5,8,11,13,15]
            # act_index = [1,4,7,10,14,17,20]
        if l == 1:
            activations = activation_model.predict(x_train)
            act_index = [bottleneck_ind]
        for a in act_index:
            activation = activations[a]
            if 'lambda' in model.layers[a+1].name and classified:
                for m in range(np.shape(activation)[0]):
                    plt.figure()
                    plt.plot(np.linspace(x[0], x[-1],
                                         np.shape(activation)[1]),
                             np.resize(activation[m],
                                       (np.shape(activation[m])[0])))
                    plt.xlabel('time [days]')
                    plt.ylabel('relative flux')
                    plt.savefig(output_dir + 'animation' + str(m) + '-act' + \
                                str(a) + '.png')
                    plt.close()
            if np.shape(activation)[1] == latentDim:
                nrows = latentDim
                ncols = latentDim
            elif np.shape(activation)[2] == 1: # sigmoid
                nrows = 1
                ncols = 1
            elif np.shape(activation)[2] == min(filter_num[j]):
                nrows = 1
                ncols = min(filter_num[j])
            else:
                nrows = 2
                ncols = int(max(filter_num[0])/2)
            fig, axes = plt.subplots(nrows = nrows, ncols = ncols,
                                     squeeze=False, figsize=(14, 10))
            fig.suptitle(layer_outputs[a].name)

            if a == bottleneck_ind: #8:
                if l == 0:
                    axes = ml.corner_plot(activation)[1]
                if l == 1:
                    if latentDim == 1:
                        try:
                            axes[0,0].set_ylabel('frequency')
                            axes[0,0].hist(np.reshape(activation,
                                                      np.shape(activation)[0]),
                                           75)
                        except:
                            print('activation num' + str(a + 1))
                    if latentDim == 2:
                        axes[0,0].hist(activation[:,0], 20)
                        axes[1,1].hist(activation[:,1], 20)
                    if latentDim == 3:
                        axes[0,0].hist(activation[:,0], 20)
                        axes[1,1].hist(activation[:,1], 20)
                        axes[2,2].hist(activation[:,2], 20)
            else:
                # >> loop through filters
                for b in range(max(np.shape(activation)[2], 1)):
                    for k in range(len(activation)): # >> loop lc
                        if b/8 < 1:
                            axes[0,b].plot(np.linspace(x[0],x[-1],
                                                       np.shape(activation)[1]),
                                           activation[k][:,b],
                                           'b', alpha = 0.1)
                        else:
                            axes[1,b-8].plot(np.linspace(x[0], x[-1],
                                                         np.shape(activation)[1]),
                                             activation[k][:,b],
                                             'b', alpha = 0.1)
                        # axes[-1,3].set_xlabel('time [days]')
                for r in range(np.shape(axes)[0]):
                    axes[r, 0].set_ylabel('relative flux')
                for c in range(np.shape(axes)[1]):
                    axes[-1, c].set_xlabel('time [days]')
                for ax_num in range(len(axes.flatten())):
                    axes.flatten()[ax_num].set_title('filter ' + \
                                                     str(ax_num))

            if l == 0:
                fig.savefig(output_dir + prefix + \
                            layer_outputs[a].name.split('/')[0] + \
                            '-x_test'+ '.png')
                plt.close(fig)

            if l == 1:
                fig.savefig(output_dir + prefix + \
                            layer_outputs[a].name.split('/')[0] +\
                            '-x_train'+'.png')
                plt.close(fig)

    # -- visualizing kernel x filter for each layer --------------------
    for a in layer_index:
        try:
            filters, biases = model.layers[a + 1].get_weights()
            plt.figure()
            if a  == layer_index[-1]:
                plt.imshow(np.reshape(filters, (np.shape(filters)[0],
                                                np.shape(filters)[1])))
            else:
                plt.imshow(np.reshape(filters, (np.shape(filters)[0],
                                                np.shape(filters)[2])))
            plt.xlabel('filter')
            plt.ylabel('kernel')
            plt.savefig(output_dir + prefix + \
                        model.layers[a].name + 'fspace.png')
            plt.close()
        except:
            print('layer num:' + str(a + 1))

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# def display_activation(activations, col_size, row_size, act_index): 
#     activation = activations[act_index]
#     activation_index=0
#     fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
#     for row in range(0,row_size):
#         for col in range(0,col_size):
#             ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
#             activation_index += 1

# if latentDim == 1:
#     try:
#         axes[0,0].hist(np.reshape(activation,
#                                   np.shape(activation)[0]),
#                        n_bins)
#         axes[0,0].set_ylabel('frequency')
#     except:
#         print('Could not plot latent space (layer ' + \
#               str(a + 1))
# if latentDim == 2:
#     axes[0,0].hist(activation[:,0], 20)
#     axes[1,1].hist(activation[:,1], 20)
#     axes[1,0].histogram2d(activation[:,0], activation[:,1]
# if latentDim == 3:
#     axes[0,0].hist(activation[:,0], 20)
#     axes[1,1].hist(activation[:,1], 20)
#     axes[2,2].hist(activation[:,2], 20)

            # inds1 = np.nonzero(y_train == 1.)[0]
            # inds2 = np.nonzero(y_train == 2.)[0]
            # inds3 = np.nonzero(y_train == 3.)[0]
            # inds4 = np.nonzero(y_train == 4.)[0]
            # for i in inds1:
            #     plt.plot(x, x_train[i], 'r-', alpha = 0.5, label = '1')
            # for i in inds2:
            #     plt.plot(x, x_train[i], 'b-', alpha = 0.5, label = '2')
            # for i in inds3:
            #     plt.plot(x, x_train[i], 'g-', alpha = 0.5, label = '3')
            # for i in inds4:
            #     plt.plot(x, x_train[i], 'm-', alpha = 0.5, label = '4')
