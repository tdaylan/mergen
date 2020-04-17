# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# 
# keras autoencoder model and plotting functions
# emma feb 2020
# 
# * convolutional autoencoder
# * autoencoder
#
#
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 

import os
import pdb
import matplotlib.pyplot as plt
import numpy as np

# :: autoencoder ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def autoencoder(x_train, x_test, params, dual_input = False, rms_train=False,
                rms_test=False, supervised = False, y_train=False,
                y_test=False, num_classes=False):
    '''If supervised = True, must provide y_train, y_test, num_classes'''
    from keras import optimizers
    import keras.metrics
    from keras.models import Model
    from keras.layers import Dense, concatenate

    # encoded = encoder(x_train, params) # >> max of channels
    encoded = encoder1(x_train, params) # >> mean of channels
    
    if dual_input:
        mlp = create_mlp(np.shape(rms_train)[1])
        shared_input = concatenate([mlp.output,encoded.output])
        shared_output = Dense(params['latent_dim'],
                              activation='relu')(shared_input)

    if supervised:
        if dual_input:
            x = Dense(num_classes, activation='softmax')(shared_output)
            model = Model(inputs=[encoded.input,mlp.input], outputs=x)
        else:
            x = Dense(num_classes,
                  activation='softmax')(encoded.output)
            model = Model(encoded.input, x)
        model.summary()
    else:
        decoded = decoder(x_train, encoded.output, params)
        model = Model(encoded.input, decoded)
        print(model.summary())
    
    if params['optimizer'] == 'adam':
        opt = optimizers.adam(lr = params['lr'], 
                              decay=params['lr']/params['epochs'])
    elif params['optimizer'] == 'adadelta':
        opt = optimizers.adadelta(lr = params['lr'])
        
    model.compile(optimizer=opt, loss=params['losses'],
                  metrics=['accuracy', keras.metrics.Precision(),
                  keras.metrics.Recall()])

    if supervised and dual_input:
        history = model.fit([x_train, rms_train], y_train,
                            epochs=params['epochs'],
                            batch_size=params['batch_size'], shuffle=True)
    elif supervised and not dual_input:
        history = model.fit(x_train, y_train, epochs=params['epochs'],
                            batch_size=params['batch_size'], shuffle=True)
    elif dual_input and not supervised:
        history = model.fit([x_train, rms_train], x_train,
                            epochs=params['epochs'],
                            batch_size=params['batch_size'], shuffle=True)
    else:
        history = model.fit(x_train, x_train, epochs=params['epochs'],
                            batch_size=params['batch_size'], shuffle=True,
                            validation_data=(x_test, x_test))
        
    return history, model

def encoder(x_train, params):
    from keras.layers import Input,Conv1D,MaxPooling1D,Dropout,Flatten,Dense
    from keras.models import Model
    
    input_dim = np.shape(x_train)[1]
    num_iter = int((params['num_conv_layers'] - 1)/2)
    
    input_img = Input(shape = (input_dim, 1))
    x = Conv1D(params['num_filters'][0], params['kernel_size'],
               activation=params['activation'], padding='same')(input_img)
    for i in range(num_iter):
        x = MaxPooling1D(2, padding='same')(x)
        x = Dropout(params['dropout'])(x)
        x = MaxPooling1D([params['num_filters'][i]],
                         data_format='channels_first')(x)
        x = Conv1D(params['num_filters'][1+i], params['kernel_size'],
                   activation=params['activation'], padding='same')(x)
    x = MaxPooling1D([params['num_filters'][i]], 
                     data_format='channels_first')(x)
    x = Flatten()(x)
    encoded = Dense(params['latent_dim'], activation=params['activation'])(x)
    # return encoded
    encoder = Model(input_img, encoded)
    return encoder

def encoder1(x_train, params):
    '''https://machinelearningmastery.com/introduction-to-1x1-convolutions-to
    -reduce-the-complexity-of-convolutional-neural-networks/
    Using convolutions over channels to downsample feature maps'''
    from keras.layers import Input,Conv1D,MaxPooling1D,Dropout,Flatten,Dense
    from keras.models import Model
    
    input_dim = np.shape(x_train)[1]
    num_iter = int((params['num_conv_layers'] - 1)/2)
    
    input_img = Input(shape = (input_dim, 1))
    x = Conv1D(params['num_filters'][0], params['kernel_size'],
               activation=params['activation'], padding='same')(input_img)
    for i in range(num_iter):
        x = MaxPooling1D(2, padding='same')(x)
        x = Dropout(params['dropout'])(x)
        x = Conv1D(1, 1, activation='relu')(x)
        # x = MaxPooling1D([params['num_filters'][i]],
        #                  data_format='channels_first')(x)
        x = Conv1D(params['num_filters'][1+i], params['kernel_size'],
                   activation=params['activation'], padding='same')(x)
    x = MaxPooling1D([params['num_filters'][i]], 
                     data_format='channels_first')(x)
    x = Flatten()(x)
    encoded = Dense(params['latent_dim'], activation=params['activation'])(x)
    # return encoded
    encoder = Model(input_img, encoded)
    return encoder


def decoder(x_train, bottleneck, params):
    from keras.layers import Dense,Reshape,Conv1D,UpSampling1D,Dropout
    from keras.layers import MaxPooling1D
    # from keras.models import Model
    input_dim = np.shape(x_train)[1]
    num_iter = int((params['num_conv_layers'] - 1)/2)
    
    # encoded = Input(shape = (params['latent_dim'],))
    x = Dense(int(input_dim/(2**(num_iter))))(bottleneck)
    x = Reshape((int(input_dim/(2**(num_iter))), 1))(x)
    for i in range(num_iter):
        x = Conv1D(params['num_filters'][num_iter+1], params['kernel_size'],
                   activation=params['activation'], padding='same')(x)
        x = UpSampling1D(2)(x)
        x = Dropout(params['dropout'])(x)
        x = MaxPooling1D([params['num_filters'][num_iter+1]],
                         data_format='channels_first')(x)
    decoded = Conv1D(1, params['kernel_size'],
                     activation=params['last_activation'], padding='same')(x)
    return decoded
    # decoder = Model(bottleneck, decoded)
    # return decoder
    

def create_mlp(input_dim):
    '''Build multi-layer perceptron neural network model for numerical data
    (rms)'''
    from keras.models import Model
    from keras.layers import Dense, Input
    input_img = Input(shape = (input_dim,))
    x = Dense(8, activation='relu')(input_img)
    x = Dense(4, activation='relu')(x)
    x = Dense(1, activation='linear')(x)
    
    model = Model(input_img, x)
    # model = Sequential()
    # model.add(Dense(8, input_dim=dim, activation='relu'))
    # model.add(Dense(4, activation='relu'))
    # model.add(Dense(1, activation='linear'))
    return model
    
# :: partitioning data ::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def split_data(fname, train_test_ratio = 0.9, cutoff=16336,supervised=False,
               classes=False):
    intensity = np.loadtxt(open(fname, 'rb'), delimiter=',')

    # >> truncate
    intensity = np.delete(intensity,np.arange(cutoff,np.shape(intensity)[1]),1)

    # >> reshape data
    intensity = np.resize(intensity, (np.shape(intensity)[0],
                                      np.shape(intensity)[1], 1))

    # >> split test and train data
    if supervised:
        train_inds = []
        test_inds = []
        class_types = np.unique(classes)
        num_classes = len(class_types)
        y_train = []
        y_test = []
        # y_train=np.zeros((np.shape(x_train)[0], num_classes))
        # y_test=np.zeros((np.shape(x_test)[0], num_classes))
        for c in class_types:
            inds= np.nonzero(classes==c)[0]
            extent = int(len(inds)*train_test_ratio)
            train_inds.extend(inds[:extent])
            test_inds.extend(inds[extent:])
            labels = np.zeros((len(inds), num_classes))
            labels[:,int(c)] = 1.
            y_train.extend(labels[:extent])
            y_test.extend(labels[extent:])

        y_train = np.array(y_train)
        y_test - np.array(y_test)
        x_train = np.copy(intensity[train_inds])
        x_test = np.copy(intensity[test_inds])
    else:
        split_ind = int(train_test_ratio*np.shape(intensity)[0])
        x_train = np.copy(intensity[:split_ind])
        x_test = np.copy(intensity[split_ind:])
        y_test, y_train = [False, False]

    return x_train, x_test, y_train, y_test
    

def rms(x):
    rms = np.sqrt(np.mean(x**2, axis = 1))
    return rms

def standardize(x):
    cutoff = np.shape(x)[1]
    # >> subtract by mean
    means = np.mean(x, axis = 1, keepdims=True)
    # means = np.reshape(means, (np.shape(means)[0], 1, 1))
    # means = np.repeat(means, cutoff, axis = 1)
    x = x - means
    
    # >> divide by standard deviations
    stdevs = np.std(x, axis = 1, keepdims=True)
    x = x / stdevs
        
    return x
        
    
def normalize(x):
    medians = np.median(x, axis = 1, keepdims=True)
    x = x / medians
    return x

def normalize1(x):
    xmin = np.min(x, axis=1, keepdims=True)
    x = x - xmin
    xmax = np.max(x, axis=1, keepdims=True)
    x = x * 2 / xmax
    x = x - 1.
    # scale = 2/(xmax-xmin)
    # offset = (xmin - xmax)/(xmax-xmin)
    # x = x*scale + offset
    return x

# :: fake data ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def gaussian(x, a, b, c):
    '''a = height, b = position of center, c = stdev'''
    import numpy as np
    return a * np.exp(-(x-b)**2 / (2*c**2))

def signal_data(training_size = 10000, test_size = 100, input_dim = 100,
                 time_max = 30., noise_level = 0.0, height = 1., center = 15.,
                 stdev = 0.8, h_factor = 0.2, center_factor = 5.,
                 reshape=False):
    '''Generate training data set with flat light curves and gaussian light
    curves, with variable height, center, noise level as a fraction of gaussian
    height)
    '''

    x = np.empty((training_size + test_size, input_dim))
    y = np.empty((training_size + test_size))
    l = int(np.shape(x)[0]/2)
    
    # >> no peak data
    x[:l] = np.zeros((l, input_dim))
    y[:l] = 0.

    # >> with peak data
    time = np.linspace(0, time_max, input_dim)
    for i in range(l):
        a = height + h_factor*np.random.normal()
        b = center + center_factor*np.random.normal()
        x[l+i] = gaussian(time, a = a, b = b, c = stdev)
    y[l:] = 1.

    # >> add noise
    x += np.random.normal(scale = noise_level, size = np.shape(x))

    # >> partition training and test datasets
    x_train = np.concatenate((x[:int(training_size/2)], 
                              x[l:-int(test_size/2)]))
    y_train = np.concatenate((y[:int(training_size/2)], 
                              y[l:-int(test_size/2)]))
    x_test = np.concatenate((x[int(training_size/2):l], 
                             x[-int(test_size/2):]))
    y_test = np.concatenate((y[int(training_size/2):l], 
                             y[-int(test_size/2):]))

    if reshape:
        x_train = np.reshape(x_train, (np.shape(x_train)[0],
                                       np.shape(x_train)[1], 1))
        x_test = np.reshape(x_test, (np.shape(x_test)[0],
                                     np.shape(x_test)[1], 1))

    return x_train, y_train, x_test, y_test

def no_signal_data(training_size = 10000, test_size = 100, input_dim = 100,
                   noise_level = 0., min0max1=True, reshape=False):
    import numpy as np

    x = np.empty((training_size + test_size, input_dim))
    y = np.empty((training_size + test_size))
    l = int(np.shape(x)[0]/2)
    
    # >> no peak data
    if min0max1:
        x = np.zeros(np.shape(x))
    else:
        x = np.ones(np.shape(x))
    y = 0.

    # >> add noise
    x += np.random.normal(scale = noise_level, size = np.shape(x))

    # >> partition training and test datasets
    x_train = np.concatenate((x[:int(training_size/2)], 
                              x[l:-int(test_size/2)]))
    y_train = np.concatenate((y[:int(training_size/2)], 
                              y[l:-int(test_size/2)]))
    x_test = np.concatenate((x[int(training_size/2):l], 
                             x[-int(test_size/2):]))
    y_test = np.concatenate((y[int(training_size/2):l], 
                             y[-int(test_size/2):]))

    if reshape:
        x_train = np.reshape(x_train, (np.shape(x_train)[0],
                                       np.shape(x_train)[1], 1))
        x_test = np.reshape(x_test, (np.shape(x_test)[0],
                                     np.shape(x_test)[1], 1))
    
    return x_train, y_train, x_test, y_test

# :: plotting :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


def corner_plot(activation, p, n_bins = 50, log = True):
    '''Creates corner plot for latent space.
    '''
    from matplotlib.colors import LogNorm
    # latentDim = np.shape(activation)[1]
    latentDim = p['latent_dim']

    fig, axes = plt.subplots(nrows = latentDim, ncols = latentDim,
                             figsize = (10, 10))

    # >> deal with 1 latent dimension case
    if latentDim == 1:
        axes.hist(np.reshape(activation, np.shape(activation)[0]), n_bins,
                  log=log)
        axes.set_ylabel('\u03C61')
        axes.set_ylabel('frequency')
    else:
        # >> row 1 column 1 is first latent dimension (phi1)
        for i in range(latentDim):
            axes[i,i].hist(activation[:,i], n_bins, log=log)
            axes[i,i].set_aspect(aspect=1)
            for j in range(i):
                if log:
                    norm = LogNorm()
                axes[i,j].hist2d(activation[:,j], activation[:,i],
                                 bins=n_bins, norm=norm)
                # >> remove axis frame of empty plots
                axes[latentDim-1-i, latentDim-1-j].axis('off')

            # >> x and y labels
            axes[i,0].set_ylabel('\u03C6' + str(i))
            axes[latentDim-1,i].set_xlabel('\u03C6' + str(i))

        # >> removing axis
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        plt.subplots_adjust(hspace=0, wspace=0)

    return fig, axes

def input_output_plot(x, x_test, x_predict, out = '', reshape = True,
                      inds = [0, -14, -10, 1, 2], addend = 0., sharey=False):
    '''Plots input light curve, output light curve and the residual.
    !!Can only handle len(inds) divisible by 3 or 5'''
    if len(inds) % 5 == 0:
        ncols = 5
    elif len(inds) % 3 == 0:
        ncols = 3
    ngroups = int(len(inds)/ncols)
    nrows = int(3*ngroups)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8*1.6, 3*1.3*3),
                             sharey=sharey)
    for i in range(ncols):
        for ngroup in range(ngroups):
            if reshape:
                ind = int(ngroup*ncols + i)
                addend = 1. - np.median(x_test[inds[ind]])
                axes[ngroup*3, i].plot(x,
                                       x_test[inds[ind]][:,0]+addend, '.')
                axes[ngroup*3+1,i].plot(x,
                                        x_predict[inds[ind]][:,0]+addend, '.')
                residual = (x_test[inds[ind]][:,0] - \
                            x_predict[inds[ind]][:,0])
                # residual = (x_test[inds[ind]][:,0] - \
                #             x_predict[inds[ind]][:,0])/ \
                #             x_test[inds[ind]][:,0]
                axes[ngroup*3+2, i].plot(x, residual, '.')
            else:
                axes[ngroup*3, i].plot(x, x_test[inds[ind]]+addend, '.')
                axes[ngroup*3+1, i].plot(x, x_predict[inds[ind]]+addend, '.')
                residual = (x_test[inds[ind]] - x_predict[inds[ind]])
                # residual = (x_test[inds[ind]] - x_predict[inds[ind]])/ \
                #     x_test[inds[ind]]
                axes[ngroup*3+2, i].plt(x, residual, '.')
        axes[-1, i].set_xlabel('time [days]')
    for i in range(ngroups):
        axes[3*i, 0].set_ylabel('input\nrelative flux')
        axes[3*i+1, 0].set_ylabel('output\nrelative flux')
        axes[3*i+2, 0].set_ylabel('residual')
    # for ax in axes.flatten():
    #     ax.set_aspect(aspect=3./8.)
    fig.tight_layout()
    if out != '':
        plt.savefig(out)
        plt.close(fig)
    return fig, axes
    
def get_activations(model, x_test, dual_input = False, rms_test = False):
    from keras.models import Model
    layer_outputs = [layer.output for layer in model.layers][1:]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    if dual_input:
        activations = activation_model.predict([x_test, rms_test])
    else:
        activations = activation_model.predict(x_test)
    return activations

def latent_space_plot(model, activations, params, out):
    # >> get ind for plotting latent space
    dense_inds = np.nonzero(['dense' in x.name for x in \
                                 model.layers])[0]
    for ind in dense_inds:
        if np.shape(activations[ind-1])[1] == params['latent_dim']:
            bottleneck_ind = ind - 1
    fig, axes = corner_plot(activations[bottleneck_ind-1], params)
    plt.savefig(out)
    plt.close(fig)
    return fig, axes

def kernel_filter_plot(model, out_dir):
    # >> get inds for plotting kernel and filters
    layer_inds = np.nonzero(['conv' in x.name for x in model.layers])[0]
    for a in layer_inds: # >> loop through conv layers
        filters, biases = model.layers[a].get_weights()
        fig, ax = plt.subplots()
        ax.imshow(np.reshape(filters, (np.shape(filters)[0],
                                       np.shape(filters)[2])))
        ax.set_xlabel('filter')
        ax.set_ylabel('kernel')
        plt.savefig(out_dir + 'layer' + str(a) + '.png')
        plt.close(fig)

def intermed_act_plot(x, model, activations, x_test, out_dir, addend=0.5,
                      inds = [0, -1], movie = True):
    '''Visualizing intermediate activations
    activation.shape = (test_size, input_dim, filter_num) = (116, 16272, 32)'''
    # >> get inds for plotting intermediate activations
    act_inds = np.nonzero(['conv' in x.name or \
                           'max_pool' in x.name or \
                           'dropout' in x.name or \
                           'reshape' in x.name for x in \
                           model.layers])[0]
    act_inds = np.array(act_inds) -1

    for c in range(len(inds)): # >> loop through light curves
        fig, axes = plt.subplots(figsize=(4,3))
        addend = 1. - np.median(x_test[inds[c]])
        axes.plot(np.linspace(np.min(x), np.max(x), np.shape(x_test)[1]),
                x_test[inds[c]] + addend, '.')
        axes.set_xlabel('time [days]')
        axes.set_ylabel('relative flux')
        plt.tight_layout()
        fig.savefig(out_dir+str(c)+'ind-0input.png')
        plt.close(fig)
        for a in act_inds: # >> loop through layers
            activation = activations[a]
            if np.shape(activation)[2] == 1:
                nrows = 1
                ncols = 1
            else:
                ncols = 4
                nrows = int(np.shape(activation)[2]/ncols)
            fig, axes = plt.subplots(nrows,ncols,figsize=(8*ncols*0.5,3*nrows))
            for b in range(np.shape(activation)[2]): # >> loop through filters
                if ncols == 1:
                    ax = axes
                else:
                    ax = axes.flatten()[b]
                x1 = np.linspace(np.min(x), np.max(x), np.shape(activation)[1])
                ax.plot(x1, activation[inds[c]][:,b] + addend, '.')
            if nrows == 1:
                axes.set_xlabel('time [days]')
                axes.set_ylabel('relative flux')
                # axes.set_aspect(aspect=3./8.)
            else:
                for i in range(nrows):
                    axes[i,0].set_ylabel('relative\nflux')
                for j in range(ncols):
                    axes[-1,j].set_xlabel('time [days]')
            fig.tight_layout()
            fig.savefig(out_dir+str(c)+'ind-'+str(a+1)+model.layers[a+1].name\
                        +'.png')
            plt.close(fig)

def epoch_plots(history, p, out_dir):
    label_list = [['loss', 'accuracy'], ['precision', 'recall']]
    key_list = [['loss', 'accuracy'], [list(history.history.keys())[-2],
                                       list(history.history.keys())[-1]]]
    for i in range(2):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(history.history[key_list[i][0]], label=label_list[i][0])
        ax1.set_ylabel(label_list[i][0])
        ax2.plot(history.history[key_list[i][1]], '--', label=label_list[i][1])
        ax2.set_ylabel(label_list[i][1])
        ax1.set_xlabel('epoch')
        ax1.set_xticks(range(p['epochs']))
        ax1.legend(loc = 'upper left', fontsize = 'x-small')
        ax2.legend(loc = 'upper right', fontsize = 'x-small')
        fig.tight_layout()
        if i == 0:
            plt.savefig(out_dir + 'acc_loss.png')
        else:
            plt.savefig(out_dir + 'prec_recall.png')
        plt.close(fig)
    
def input_bottleneck_output_plot(x, x_test, x_predict, activations, model,
                                 out = '',
                                 reshape = True,
                                 inds = [0, 1, -1, -2, -3],
                                 addend = 0.5, sharey=False):
    '''Can only handle len(inds) divisible by 3 or 5'''
    bottleneck_ind = np.nonzero(['dense' in x.name for x in \
                                 model.layers])[0][0]
    bottleneck = activations[bottleneck_ind - 1]
    if len(inds) % 5 == 0:
        ncols = 5
    elif len(inds) % 3 == 0:
        ncols = 3
    ngroups = int(len(inds)/ncols)
    nrows = int(3*ngroups)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8*1.6, 3*1.3*3),
                             sharey=sharey)
    for i in range(ncols):
        for ngroup in range(ngroups):
            if reshape:
                ind = int(ngroup*ncols + i)
                addend = 1. - np.median(x_test[inds[ind]])
                axes[ngroup*3, i].plot(x, x_test[inds[ind]][:,0]+addend, '.')
                img = np.reshape(bottleneck[inds[ind]],
                                 (1, np.shape(bottleneck[inds[ind]])[0]))
                axes[ngroup*3+1, i].imshow(img)
                axes[ngroup*3+2, i].plot(x, x_predict[inds[ind]][:,0]+addend,
                                         '.')
            else:
                addend = 1. - np.median(x_test[inds[ind]])
                axes[ngroup*3, i].plot(x, x_test[inds[ind]]+addend, '.')
                axes[ngroup*3+2, i].plot(x, x_predict[inds[ind]]+addend, '.')
        axes[-1, i].set_xlabel('time [days]')
    for i in range(ngroups):
        axes[3*i, 0].set_ylabel('input\nrelative flux')
        axes[3*i+1, 0].set_ylabel('bottleneck')
        axes[3*i+2, 0].set_ylabel('output\nrelative flux')
    fig.tight_layout()
    if out != '':
        plt.savefig(out)
        plt.close(fig)
    return fig, axes
    

def movie(x, model, activations, x_test, p, out_dir, inds = [0, -1],
          addend=0.5):
    for c in range(len(inds)):
        fig, axes = plt.subplots(figsize=(8,3))
        ymin = []
        ymax = []
        for activation in activations:
            if np.shape(activation)[1] == p['latent_dim']:
                ymin.append(min(activation[inds[c]]))
                ymax.append(max(activation[inds[c]]))
            elif len(np.shape(activation)) > 2:
                if np.shape(activation)[2] == 1:
                    ymin.append(min(activation[inds[c]]))
                    ymax.append(max(activation[inds[c]]))
        ymin = np.min(ymin) + addend
        ymax = np.max(ymax) + addend
        addend = 1. - np.median(x_test[inds[c]])

        # >> plot input
        axes.plot(np.linspace(np.min(x), np.max(x), np.shape(x_test)[1]),
                  x_test[inds[c]] + addend, '.')
        axes.set_xlabel('time [days]')
        axes.set_ylabel('relative flux')
        axes.set_ylim(ymin=ymin, ymax=ymax)
        fig.tight_layout()
        fig.savefig('./image-000.png')

        # >> plot intermediate activations
        n=1
        for a in range(len(activations)):
            activation = activations[a]
            if np.shape(activation)[1] == p['latent_dim']:
                length = p['latent_dim']
                axes.cla()
                axes.plot(np.linspace(np.min(x), np.max(x), length),
                          activation[inds[c]] + addend, '.')
                axes.set_xlabel('time [days]')
                axes.set_ylabel('relative flux')
                axes.set_ylim(ymin=ymin, ymax =ymax)
                fig.tight_layout()
                fig.savefig('./image-' + f'{n:03}.png')
                n += 1
            elif len(np.shape(activation)) > 2:
                if np.shape(activation)[2] == 1:
                    length = np.shape(activation)[1]
                    y = np.reshape(activation[inds[c]], (length))
                    axes.cla()
                    axes.plot(np.linspace(np.min(x), np.max(x), length),
                              y + addend, '.')
                    axes.set_xlabel('time [days]')
                    axes.set_ylabel('relative flux')
                    axes.set_ylim(ymin = ymin, ymax = ymax)
                    fig.tight_layout()
                    fig.savefig('./image-' + f'{n:03}.png')
                    n += 1
        os.system('ffmpeg -framerate 2 -i ./image-%03d.png -pix_fmt yuv420p '+\
                  out_dir+str(c)+'ind-movie.mp4')

def latent_space_clustering(activation, x_test, x, ticid, out = './', 
                            n_bins = 50, addend=1., scatter = True):
    '''Clustering latent space
    '''
    from matplotlib.colors import LogNorm
    # from sklearn.cluster import DBSCAN
    from sklearn.neighbors import LocalOutlierFactor
    latentDim = np.shape(activation)[1]

    # >> deal with 1 latent dimension case
    if latentDim == 1: # TODO !!
        fig, axes = plt.subplots(figsize = (15,15))
        axes.hist(np.reshape(activation, np.shape(activation)[0]), n_bins,
                  log=True)
        axes.set_ylabel('\u03C61')
        axes.set_ylabel('frequency')
    else:
        # >> row 1 column 1 is first latent dimension (phi1)
        for i in range(latentDim):
            # axes[i,i].hist(activation[:,i], n_bins, log=log)
            # axes[i,i].set_aspect(aspect=1)
            for j in range(i):
                # -- calculate lof --------------------------------------------
                z1, z2 = activation[:,j], activation[:,i]
                X = np.array((z1, z2)).T                
                clf = LocalOutlierFactor()
                clf.fit_predict(X)
                lof = -1 * clf.negative_outlier_factor_
                inds = np.argsort(lof)[-20:] # >> outliers
                inds2 = np.argsort(lof)[:20] # >> inliers
                
                # -- plot latent space w/ inset plots -------------------------
                fig, ax = plt.subplots(figsize = (15,15))
                
                if scatter:
                    ax.plot(z1, z2, '.')
                else:
                    ax.hist2d(z1, z2, bins=n_bins, norm=LogNorm())
                
                plt.xticks(fontsize='xx-large')
                plt.yticks(fontsize='xx-large')
                
                h = 0.047
                x0 = 0.85
                y0 = 0.9
                xstep = h*8/3 + 0.025
                ystep = h + 0.025
                
                # >> sort to clean up plot
                inds0 = inds[:10]
                inds0 = sorted(inds, key=lambda z: ((z1[z]-np.max(z1))+\
                                                    (z2[z]-np.min(z2)))**2)
                
                for k in range(10):
                    # >> make inset axes
                    if k < 5:
                        axins = ax.inset_axes([x0 - k*xstep, y0, h*8/3, h])
                    else:
                        axins = ax.inset_axes([x0, y0 - (k-4)*ystep, h*8/3, h])
                    xp, yp = z1[inds0[k]], z2[inds0[k]]
            
                    xextent = ax.get_xlim()[1] - ax.get_xlim()[0]
                    yextent = ax.get_ylim()[1] - ax.get_ylim()[0]
                    x1, x2 = xp-0.01*xextent, xp+0.01*xextent
                    y1, y2 = yp-0.01*yextent, yp+0.01*yextent
                    axins.set_xlim(x1, x2)
                    axins.set_ylim(y1, y2)
                    ax.indicate_inset_zoom(axins)
                    
                    # >> plot light curves
                    axins.set_xlim(min(x), max(x))
                    axins.set_ylim(min(x_test[inds0[k]]),
                                   max(x_test[inds0[k]]))
                    axins.plot(x, x_test[inds0[k]] + addend, '.k',
                               markersize=3)
                    axins.set_xticklabels('')
                    axins.set_yticklabels('')
                    axins.patch.set_alpha(0.5)

                # >> x and y labels
                ax.set_ylabel('\u03C61' + str(i), fontsize='xx-large')
                ax.set_xlabel('\u03C61' + str(j), fontsize='xx-large')
                fig.savefig(out + 'phi' + str(j) + 'phi' + str(i) + '.png')
                
                # -- plot 20 light curves -------------------------------------
                # >> plot light curves with lof label
                fig1, ax1 = plt.subplots(20, figsize = (7,28))
                fig1.subplots_adjust(hspace=0)
                fig2, ax2 = plt.subplots(20, figsize = (7,28))
                fig2.subplots_adjust(hspace=0)
                fig3, ax3 = plt.subplots(20, figsize = (7,28))
                fig3.subplots_adjust(hspace=0)
                for k in range(20):
                    # >> outlier plot
                    ax1[k].plot(x, x_test[inds[19-k]]+addend, '.k', markersize=3)
                    # ax1[k].set_aspect(1/4)
                    ax1[k].set_xticks([])
                    ax1[k].set_ylabel('relative\nflux')
                    ax1[k].text(0.8, 0.65,
                                'LOF {}\nTIC {}'.format(str(lof[inds[19-k]])[:9],
                                                        str(int(ticid[inds[19-k]]))),
                                transform = ax1[k].transAxes)
                    
                    # >> inlier plot
                    ax2[k].plot(x, x_test[inds2[k]]+addend, '.k', markersize=3)
                    # ax2[k].set_aspect(1/4)
                    ax2[k].set_xticks([])
                    ax2[k].set_ylabel('rellative\nflux')
                    ax2[k].text(0.8, 0.65,
                                'LOF {}\nTIC {}'.format(str(lof[inds2[k]])[:9],
                                                        str(int(ticid[inds2[k]]))),
                                transform = ax2[k].transAxes)
                    
                    # >> random lof plot
                    ind = np.random.choice(range(len(lof)-1))
                    ax3[k].plot(x, x_test[ind] + addend, '.k', markersize=3)
                    # ax3[k].set_aspect(1/4)
                    ax3[k].set_xticks([])
                    ax3[k].set_ylabel('relative\nflux')
                    ax3[k].text(0.8, 0.65,
                                'LOF {}\nTIC {}'.format(str(lof[ind])[:9],
                                                        str(int(ticid[ind]))),
                                transform = ax3[k].transAxes)
                
                ax1[-1].set_xlabel('time [BJD - 2457000]')
                ax2[-1].set_xlabel('time [BJD - 2457000]')
                ax3[-1].set_xlabel('time [BJD - 2457000]')
                fig1.savefig(out + 'phi' + str(j) + 'phi' + str(i) + \
                            '-outliers.png')
                fig2.savefig(out + 'phi' + str(j) + 'phi' + str(i) + \
                             '-inliers.png')
                fig3.savefig(out + 'phi' + str(j) + 'phi'  + str(i) + \
                             '-randomlof.png')
                

        # >> removing axis
        # for ax in axes.flatten():
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        # plt.subplots_adjust(hspace=0, wspace=0)

    return fig, ax

def training_test_plot(x, x_train, x_test, y_train_classes, y_test_classes,
                       y_predict, num_classes, out):
    colors = ['r', 'g', 'b', 'm']
    # >> training data set
    fig, ax = plt.subplots(nrows = 7, ncols = num_classes, figsize=(15,10),
                           sharex=True)
    plt.subplots_adjust(hspace=0)
    # >> test data set
    fig1, ax1 = plt.subplots(nrows = 7, ncols = num_classes, figsize=(15,10),
                             sharex=True)
    plt.subplots_adjust(hspace=0)
    for i in range(num_classes): # >> loop through classes
        inds = np.nonzero(y_train_classes == i)[0]
        inds1 = np.nonzero(y_test_classes == i)[0]
        for j in range(min(7, len(inds))): # >> loop through rows
            ax[j,i].plot(x, x_train[inds[j]], '.'+colors[i], markersize=3)
        for j in range(min(7, len(inds1))):
            ax1[j,i].plot(x, x_test[inds1[j]], '.'+colors[y_predict[inds1[j]]],
                          markersize=3)
            pdb.set_trace()
            ax1[j,i].text(0.8, 0.65, 'True: '+str(i)+'\nPredicted: '+\
                          str(y_predict[inds1[j]]),
                          transform=ax1[j,i].transAxes)
    for i in range(num_classes):
        ax[0,i].set_title('True class '+str(i))
        ax1[0,i].set_title('True class '+str(i))
        
        ax[-1,i].set_xlabel('time [BJD - 2457000]')
        ax1[-1,i].set_xlabel('time [BJD - 2457000]')
    for j in range(7):
        ax[j,0].set_ylabel('relative\nflux')
    for j in range(7):
        ax1[j,0].set_ylabel('relative\nflux')
        
    # fig.tight_layout()
    # fig1.tight_layout()
    
    fig.savefig(out+'train.png')
    fig1.savefig(out+'test.png')



    
# def autoencoder_dual_input1(x_train, x_test, rms_train, rms_test, params,
#                             supervised=False, y_train = False, y_test=False,
#                             num_classes = False):
#     '''Adapted from: https://www.pyimagesearch.com/2019/02/04/keras-multiple-
#     inputs-and-mixed-data/'''
#     from keras.layers import concatenate
#     from keras.layers import Conv1D, Dense, Reshape
#     from keras.models import Model
#     from keras import optimizers
#     import keras.metrics
#     input_dim = np.shape(x_train)[1]
#     # >> create the MLP and autoencoder models
#     mlp = create_mlp(np.shape(rms_train)[1])
#     encoded = encoder1(x_train, params)
#     # autoencoder = create_conv_layers(x_train, params)

#     # x = Reshape((1,1))(mlp.output)
#     # x = concatenate([autoencoder.output, Reshape((1,1))(mlp.output)], axis = 1)
#     # x = Reshape((input_dim,))(autoencoder.output)
#     x = concatenate([mlp.output,
#                      Reshape((input_dim,))(encoded.output)], axis = 1)
#     # pdb.set_trace()
#     # x = Reshape((input_dim+1,))(x)
    
    
#     # x = Dense(4, activation='relu')(combinedInput)
#     # x = Dense(1, activation='linear')(x)
#     x = Dense(input_dim, activation='relu')(x)
#     x = Reshape((input_dim,1))(x)
#     # x = Conv1D(1, params['kernel_size'], activation=params['last_activation'],
#     #            padding='same')(combinedInput)
#     if supervised:
#         x = Dense(num_classes,activation='softmax')(encoded.output)
#         model = Model(inputs=[encoded.input, mlp.input], output=x)
#         print(model.summary())
#     else:
#         model = Model(inputs=[encoded.input, mlp.input], outputs=x)
#         print(model.summary())
    
#     # !! find a better way to do this
#     if params['optimizer'] == 'adam':
#         opt = optimizers.adam(lr = params['lr'], 
#                               decay=params['lr']/params['epochs'])
#     elif params['optimizer'] == 'adadelta':
#         opt = optimizers.adadelta(lr = params['lr'])
        
#     model.compile(optimizer=opt, loss=params['losses'],
#                   metrics=['accuracy', keras.metrics.Precision(),
#                            keras.metrics.Recall()])
#     history = model.fit([x_train, rms_train], x_train, epochs=params['epochs'],
#                         batch_size=params['batch_size'], shuffle=True,
#                         validation_data=([x_test, rms_test], x_test))
#     return history, model

# def autoencoder_dual_input2(x_train, x_test, rms_train, rms_test, params,
#                             supervised=False, y_train=False, y_test=False,
#                             num_classes=False):
#     '''Adapted from: https://stackoverflow.com/questions/52435274/how-to-use-
#     keras-merge-layer-for-autoencoder-with-two-ouput'''
#     from keras.layers import concatenate
#     from keras.layers import Dense
#     from keras import optimizers
#     import keras.metrics
#     from keras.models import Model
    
#     # >> create the MLP and encoder models
#     mlp = create_mlp(np.shape(rms_train)[1])
#     encoded = encoder(x_train, params)

#     # >> shared representation layer
#     shared_input = concatenate([mlp.output,encoded.output])
#     shared_output = Dense(params['latent_dim'], activation='relu')(shared_input)
#     if supervised:
#         x = Dense(num_classes,activation='softmax')(shared_output)
#         model = Model(inputs=[encoded.input, mlp.input], output=x)
#         print(model.summary())
#     else:
#         decoded = decoder(x_train, shared_output, params)
#         model = Model(inputs=[encoded.input, mlp.input], outputs=decoded)
    
#     # >> get model
#     # model = Model(inputs=[encoded.input, mlp.input], outputs=decoded.output)
    
    
#     # !! find a better way to do this
#     if params['optimizer'] == 'adam':
#         opt = optimizers.adam(lr = params['lr'], 
#                               decay=params['lr']/params['epochs'])
#     elif params['optimizer'] == 'adadelta':
#         opt = optimizers.adadelta(lr = params['lr'])
        
#     model.compile(optimizer=opt, loss=params['losses'],
#                   metrics=['accuracy', keras.metrics.Precision(),
#                            keras.metrics.Recall()])
#     history = model.fit([x_train, rms_train], x_train, epochs=params['epochs'],
#                         batch_size=params['batch_size'], shuffle=True,
#                         validation_data=([x_test, rms_test], x_test))
#     return history, model
    
    
    
    # def encoder2(x_train, params):
#     '''https://towardsdatascience.com/applied-deep-learning-part-4
#     -convolutional-neural-networks-584bc134c1e2
#     stacked'''
#     from keras.layers import Input,Conv1D,MaxPooling1D,Dropout,Flatten,Dense
#     from keras.models import Model
    
#     input_dim = np.shape(x_train)[1]
#     num_iter = int((params['num_conv_layers'] - 1)/2)
    
#     input_img = Input(shape = (input_dim, 1))
#     x = Conv1D(params['num_filters'][0], params['kernel_size'],
#                activation=params['activation'], padding='same')(input_img)
#     for i in range(num_iter):
#         x = MaxPooling1D(2, padding='same')(x)
#         x = Dropout(params['dropout'])(x)
#         x = Conv1D(1, 1, activation='relu')(x)
#         # x = MaxPooling1D([params['num_filters'][i]],
#         #                  data_format='channels_first')(x)
#         x = Conv1D(params['num_filters'][1+i], params['kernel_size'],
#                    activation=params['activation'], padding='same')(x)
#         x = Conv1D(params['num_filters'][1+i], params['kernel_size'],
#                    activation=params['activation'], padding='same')(x)
#     x = MaxPooling1D([params['num_filters'][i]], 
#                      data_format='channels_first')(x)
#     x = Flatten()(x)
#     encoded = Dense(params['latent_dim'], activation=params['activation'])(x)
#     # return encoded
#     encoder = Model(input_img, encoded)
#     return encoder
    
    
    # def create_conv_layers(x_train, params, supervised = False):
#     from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
#     from keras.layers import Reshape, Dense, Flatten, Dropout
#     from keras.models import Model

#     input_dim = np.shape(x_train)[1]
#     num_iter = int((params['num_conv_layers'] - 1)/2)
    
#     input_img = Input(shape = (input_dim, 1))
#     x = Conv1D(params['num_filters'][0], params['kernel_size'],
#                 activation=params['activation'], padding='same')(input_img)
#     for i in range(num_iter):
#         x = MaxPooling1D(2, padding='same')(x)
#         x = Dropout(params['dropout'])(x)
#         x = MaxPooling1D([params['num_filters'][i]],
#                           data_format='channels_first')(x)
#         x = Conv1D(params['num_filters'][1+i], params['kernel_size'],
#                     activation=params['activation'], padding='same')(x)
#     x = MaxPooling1D([params['num_filters'][i]], 
#                       data_format='channels_first')(x)
#     x = Flatten()(x)
#     encoded = Dense(params['latent_dim'], activation=params['activation'])(x)

#     x = Dense(int(input_dim/(2**(i+1))))(encoded)
#     x = Reshape((int(input_dim/(2**(i+1))), 1))(x)
#     for i in range(num_iter):
#         x = Conv1D(params['num_filters'][num_iter+1], params['kernel_size'],
#                     activation=params['activation'], padding='same')(x)
#         x = UpSampling1D(2)(x)
#         x = Dropout(params['dropout'])(x)
#         x = MaxPooling1D([params['num_filters'][num_iter+1]],
#                           data_format='channels_first')(x)
#     decoded = Conv1D(1, params['kernel_size'],
#                       activation=params['last_activation'], padding='same')(x)
    
#     if supervised:
#         model = Model(input_img, encoded)
#         print(model.summary())
          
#     else:
#         model = Model(input_img, decoded)
#         print(model.summary())
    
#     return model
