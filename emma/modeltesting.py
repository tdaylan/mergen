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

# :: inputs ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# >> model
auto = True # >> autoencoder
cnn = False # >> simple convolutional neural network

epochs = 4
n_bins = 75 # >> histogram
kernel_vis = True
output_dir = "./plots/plots032220/"
prefix = "sector20-"

# !! parameters
kernel_size = [10]
strides = [1]
filter_num = [[16,8,8,8,8,16]]
batch_size = [100]
latent_dims = [10]

fake_data = False

# >> real data?
fname_time = "./tessdatasector20-time.txt"
fname_intensity = "./tessdatasector20-intensity.csv"
cutoff = 16336 # >> truncate light curves to 17840 data points
tt_ratio = 0.9 # >> train to test ratio

# >> fake data?
artificial_data_signal = True
artificial_data_no_signal = False
noise = [0.] # >> given as a fraction of signal height

training_size  = 10000  # >> training_size for each class
test_size   = 100       # >> test_size/2 for each class
input_dim   = 17840       # >> number of data points in light curve
                                             
height = 20. # >> gaussian
center = 15.
stdev  = 10.
xmax   = 30.

# :: setting up parameters to vary :::::::::::::::::::::::::::::::::::::::::::::

p = [kernel_size, strides, filter_num, noise, batch_size, latent_dims]
p_names = ['kernel', 'strides', 'filter', 'noise', 'batch_size', 'latentdim']

ind = np.nonzero(np.array([len(param) for param in p]) != 1)[0]
if len(ind) != 0:
    ind = int(ind)
    # ind = np.nonzero(params)[0]
    for i in range(len(p)):
        if i != ind:
            p[i] = np.repeat(p[i], len(p[ind]), axis=0)

    prefix_0 = prefix
            
# len(params[ind])

# >> save summary txt file


# :: make model ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# >> make figures for vs. epoch plots
fig1, ax1 = plt.subplots(num = 10)
ax2 = ax1.twinx()

fig3, ax3 = plt.subplots(num = 11)
ax4 = ax3.twinx()

# :: generate data :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

for j in range(len(noise)):
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
        plt.figure(j)
        plt.title('Noise: ' + str(noise[j]))
        plt.clf()
        x = np.linspace(0, xmax, input_dim)
        inds0 = np.nonzero(y_test[:,0] == 0.)[0]
        inds1 = np.nonzero(y_test[:,0] == 1.)[0]
        for i in inds0:
            plt.plot(x, x_test[i], 'r-', alpha=0.)
        for i in inds1:
            plt.plot(x, x_test[i], 'b-', alpha=0.1)
        plt.xlabel('time [days]')
        plt.ylabel('relative flux')
        plt.savefig(output_dir + prefix + 'Test_data_noise' + str(noise[j]) + \
                    '.png', dpi = 200)
        plt.close()

    else: # -- tess data -------------------------------------------------------
        print('loading tess data')
        x = np.loadtxt(fname_time)
        # with open(fname_intensity) as f_input:
        #     intensity = np.genfromtxt((','.join(row) for row in \
        #                                csv.reader(f_input)), delimiter=',')

        intensity = np.loadtxt(open(fname_intensity, "rb"), delimiter=',')
        # intensity = np.loadtxt(fname_intensity)
        # intensity = intensity.astype(np.float)
        
        # >> truncate
        x = np.delete(x, np.arange(cutoff, np.shape(x)[0]), 0)
        intensity = np.delete(intensity, np.arange(cutoff,
                                                   np.shape(intensity)[1]), 1)
        input_dim = cutoff

        # >> reshape data
        intensity = np.resize(intensity, (np.shape(intensity)[0],
                                          np.shape(intensity)[1], 1))
        # >> split test and train data
        split_ind = int(tt_ratio*np.shape(intensity)[0])
        x_train = np.copy(intensity[:split_ind])
        x_test = np.copy(intensity[split_ind:])

        # !! normalize data

    # # >> reshape data
    # if auto:
    #     x_train = np.resize(x_train, (np.shape(x_train)[0],
    #                                   np.shape(x_train)[1], 1))
    #     y_train = np.resize(y_train, (np.shape(y_train)[0],
    #                                   np.shape(y_train)[1], 1))
    #     x_test = np.resize(x_test, (np.shape(x_test)[0],
    #                                 np.shape(x_test)[1], 1))
    #     y_test = np.resize(y_test, (np.shape(y_test)[0],
    #                                 np.shape(y_test)[1], 1))

    # :: model :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    latentDim = latent_dims[j]
    if auto:
        print('loading model')
        model = ml.autoencoder5(input_dim=input_dim,
                                kernel_size = kernel_size[j],
                                latentDim = latentDim,
                                strides=strides[j],
                                filter_num=filter_num[j]) 
        layer_index = np.nonzero(['conv' in x.name or 'lambda' in \
                                  x.name for x in model.layers])[0]
        bottleneck_ind = np.nonzero(['dense' in x.name for x in \
                                     model.layers])[0][0]
        act_index = list(layer_index) + [bottleneck_ind]
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
            plt.savefig(output_dir + prefix + 'Histogram_noise' + \
                        str(noise[j]) + '.png')
            plt.close(fig)

    if auto:
        # -- plot decoded data ---------------------------------------------
        print('plotting decoded lc')
        plt.figure(40 + j)
        for k in range(np.shape(y_predict)[0]):
            plt.plot(x, y_predict[k][:,0], '-', alpha=0.1)
        plt.savefig(output_dir + prefix + 'latentdim' + str(latentDim) + \
                    'decoded_noise' + str(noise[j]) + '.png')
        plt.close()

        # -- plot 8 input vs. output ---------------------------------------
        print('plotting input vs. output')
        fig, axes = plt.subplots(nrows = 2, ncols = 8, figsize = (14, 10.))
        for k in range(4):
            axes[0,k].plot(x, x_test[k][:,0])
            axes[1,k].plot(x, y_predict[k][:,0])
        for k in range(4,8):
            axes[0,k].plot(x, x_test[int(test_size/2) + k][:,0])
            axes[1,k].plot(x, y_predict[int(test_size/2) + k][:,0])
        axes[0,0].set_ylabel('input')
        axes[1,0].set_ylabel('output')
        plt.savefig(output_dir + prefix + 'latentdim' + str(latentDim) + 'noise' + \
                    str(noise[j]) + 'input_output' + '.png')
        plt.close(fig)

        # -- visualizing intermediate activations --------------------------

        from keras.models import Model
        layer_outputs = [layer.output for layer in model.layers]
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
                if np.shape(activation)[1] == 1:
                    nrows = 1
                    ncols = 1
                elif np.shape(activation)[2] == 1: # a == 8:
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
                        axes[0,0].hist(np.reshape(activation, test_size),
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
                    fig.savefig(output_dir + prefix + 'latentdim' + str(latentDim) +\
                                'noise' + str(noise[j]) + \
                                layer_outputs[a].name.split('/')[0] +\
                                'x_test'+ '.png')
                    plt.close(fig)

                if l == 1:
                    fig.savefig(output_dir + prefix + 'latentdim' + str(latentDim) +\
                                'noise' + str(noise[j]) + \
                                layer_outputs[a].name.split('/')[0] +\
                                'x_train'+'.png')
                    plt.close(fig)

        # -- visualizing kernel x filter for each layer --------------------
        # layer_index = [1,3,5,11,13,15]
        # layer_index = [1,4,7,14,17,20]
        if kernel_vis:
            for a in layer_index:
                filters, biases = model.layers[a].get_weights()
                plt.figure()
                if a == layer_index[-1]:
                    plt.imshow(np.reshape(filters, (np.shape(filters)[0],
                                                    np.shape(filters)[1])))
                else:
                    plt.imshow(np.reshape(filters, (np.shape(filters)[0],
                                                    np.shape(filters)[2])))
                plt.savefig(output_dir + prefix + 'latentdim' + str(latentDim) + \
                            'noise' + str(noise[j]) + \
                            model.layers[a].name + 'fspace.png')
                plt.close()

    # -- vs. epochs plots ------------------------------------------------------

    # >> label vs. epochs plots
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





# def display_activation(activations, col_size, row_size, act_index): 
#     activation = activations[act_index]
#     activation_index=0
#     fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
#     for row in range(0,row_size):
#         for col in range(0,col_size):
#             ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
#             activation_index += 1
