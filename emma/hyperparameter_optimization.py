# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
# pipeline for retrieving outliers in tess data
# etc 04-20
#
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

import modellibrary as ml
import numpy as np
import random
import pdb
from itertools import product
from sklearn.metrics import confusion_matrix


output_dir = './plots/plots041120/'

fname_time = './section1-time.txt'
fname_intensity = './section1-intensity.csv'
fname_ticid = './section1-ticid.txt'
# fname_time = './supervised100-time.txt'
# fname_intensity = './supervised100-intensity.csv'
# fname_class = './supervised100-classification.txt'
cutoff = 8896 # !! get rid of this

# >> lc index in x_test
intermed_inds = [6,0] # >> plot_intermed_act
input_bottle_inds = [0,1,2,3,4] # >> in_bottle_out
inds = [0,1,2,3,4,5,6,7,-1,-2,-3,-4,-5,-6,-7] # >> input_output_plot

# >> parameters
p = {'kernel_size': [3],
     'latent_dim': [15],
     'strides': [1],
     'epochs': [50],
     'dropout': [0.0],
     'num_conv_layers': [5],
     'num_filters': [[8,8,8,8,8]],
     'batch_size': [32],
     'activation': ['relu'],
     'optimizer': ['adam'],
     'last_activation': ['relu'],
     'losses': ['mean_squared_error'],
     'lr': [0.001]}

grid_search = True
randomized_search = False
n_iter = 200 # >> for randomized_search

dual_input = True

plot_epoch         = True
plot_in_out        = True
plot_in_bottle_out = False
plot_kernel        = False
plot_intermed_act  = False
plot_latent_test   = True
plot_latent_train  = False
plot_clustering    = True
make_movie         = False

supervised = False
addend = 0.

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# -- x_train, x_test ----------------------------------------------------------
x = np.loadtxt(fname_time)
x = np.delete(x, np.arange(cutoff, np.shape(x)[0]), 0)
x_train, x_test = ml.split_data(fname_intensity, cutoff=cutoff,
                                train_test_ratio=0.90)

ticid = np.loadtxt(fname_ticid)
ticid_train = ticid[:np.shape(x_train)[0]]
ticid_test = ticid[-1 * np.shape(x_test)[0]:]

if supervised:
    classes = np.loadtxt(fname_class)
    y_train = classes[:np.shape(x_train)[0]]
    y_test = classes[np.shape(x_train)[0]:]
    
if dual_input:
    rms_train = ml.rms(x_train)
    rms_test = ml.rms(x_test)

# x_train = np.delete(x_train, 5606, axis = 0) # buggy
# x_train = np.delete(x_train, 2688, axis = 0)
# x_train = np.delete(x_train, 1154, axis = 0)

# x_train = x_train[:500]
    
# -- initialize parameters ----------------------------------------------------

p_list = []
if grid_search:
    p_combinations = list(product(*p.values()))
    for i in range(len(p_combinations)):
        p1 = {}
        for j in range(len(p.keys())):
            p1[list(p.keys())[j]] = p_combinations[i][j]
        p_list.append(p1)
elif randomized_search:
    for n in range(n_iter):
        p_dict = {}
        # >> randomized parameter search
        for i in range(len(list(p.keys()))):
            key = list(p.keys())[i]
            if key != 'num_filters':
                p_dict[key] = random.choice(p[key])
                if key == 'kernel_size':
                    p_dict[key] = [p_dict[key]]
                if key == 'num_conv_layers':
                    num_filts = np.repeat(random.choice(p['num_filters']),
                                                        p_dict[key])
                    # p_dict['num_filters'] = list(map(lambda el: [el],
                    #                                  num_filts))
                    p_dict['num_filters'] = list(num_filts)
        p_list.append(p_dict)
    
# plt.ion()

# -- run model ----------------------------------------------------------------

for i in range(len(p_list)):

    p = p_list[i]
    print(p)

    if supervised:
        history, model = ml.autoencoder(ml.standardize(x_train),
                                        ml.standardize(x_test), p,
                                        supervised=True,
                                        y_train=y_train, y_test=y_test)
    elif dual_input:
        history, model = ml.autoencoder_dual_input(ml.standardize(x_train),
                                                   ml.standardize(x_test),
                                                   rms_train, rms_test, p)
    # history, model = ml.autoencoder21(x_train, x_test, p)
    else:
        history, model = ml.autoencoder(ml.standardize(x_train),
                                        ml.standardize(x_test), p,
                                        supervised=False)
        
    x_predict = model.predict(x_test)
    pdb.set_trace()

    # -- param summary txt ----------------------------------------------------
        
    with open(output_dir + 'param_summary.txt', 'a') as f:
        f.write('parameter set ' + str(i) + '\n')
        f.write(str(p.items()) + '\n')
        label_list = ['loss', 'accuracy', 'precision', 'recall']
        key_list =['loss', 'accuracy', list(history.history.keys())[-2],
                   list(history.history.keys())[-1]]
        for j in range(4):
            f.write(label_list[j]+' '+str(history.history[key_list[j]][-1])+\
                    '\n')
        if supervised:
            chi_2 = np.average(np.sum((x_predict - y_test)**2 / (y_test),
                                      axis = 1))
        else:
            # >> assuming uncertainty of 0.02
            chi_2 = np.average((x_predict-ml.standardize(x_test))**2 / 0.02)
            # chi_2 = np.average(np.sum((x_predict - x_test)**2 / (x_test),
            #                           axis = 1))
        f.write('chi_squared ' + str(chi_2) + '\n')
        if supervised:
            # >> confusion matrix
            # y_true = np.argmax(x_test, axis = 1)
            # y_pred = np.max(x_predict, axis = 1)
            # y_pred = np.round(np.reshape(y_pred, (np.shape(y_pred)[0])))
            # y_pred = np.argmax(x_predict, axis = 1)
            # cm = confusion_matrix(y_test, y_pred, labels=[0.,1.])
            # cm = confusion_matrix(y_test, y_pred, labels=[0.,1.])
            pdb.set_trace()
            cm = confusion_matrix(x_predict, np.roudn(y_test))
            f.write('confusion matrix\n')
            f.write(str(cm))
            # f.write('    0   1\n')
            # f.write('0 ' + str(cm[0]) + '\n')
            # f.write('1 ' + str(cm[1]) + '\n')
        f.write('\n')


    # -- data visualization ---------------------------------------------------
    
    # >> plot loss, accuracy, precision, recall vs. epochs
    if plot_epoch:
        ml.epoch_plots(history, p, output_dir + 'epoch-' + str(i) + '-')
    
    # >> plot some decoded light curves
    if plot_in_out:
        fig, axes = ml.input_output_plot(x, np.standardize(x_test), x_predict,
                                         inds=inds, out=output_dir+\
                                         'input_output-x_test-'+\
                                         str(i)+'.png', addend=addend,
                                         sharey=False)
    # >> plot latent space
    activations = ml.get_activations(model, np.standardize(x_test))
    if plot_latent_test:
        fig, axes = ml.latent_space_plot(model, activations,
                                         output_dir+'latent_space-'+str(i)+\
                                             '.png')
    if plot_latent_train:
        fig, axes = ml.latent_space_plot(model,
                                         ml.get_activations(model,
                                                            ml.standardize(x_train)),
                                         output_dir+'latent_space-x_train-'+\
                                             str(i)+'.png')

    # >> plot kernel vs. filter
    if plot_kernel:
        ml.kernel_filter_plot(model, output_dir+'kernel-'+str(i)+'-')

    # >> plot intermediate activations
    if plot_intermed_act:
        ml.intermed_act_plot(x, model, activations, ml.normalize(x_test),
                             output_dir+'intermed_act-'+str(i)+'-',
                             addend=addend, inds=intermed_inds)
    
    if make_movie:
        ml.movie(x, model, activations, x_test, p,
                 output_dir+'movie-'+str(i)+'-', addend=addend,
                 inds=intermed_inds)

    # >> plot input, bottleneck, output
    if plot_in_bottle_out:
        ml.input_bottleneck_output_plot(x, ml.standardize(x_test), x_predict,
                                        activations,
                                        model, out=output_dir+\
                                        'input_bottleneck_output-'+\
                                        str(i)+'.png', addend=addend,
                                        inds = input_bottle_inds, sharey=False)
            
    if plot_clustering:
        bottleneck_ind = np.nonzero(['dense' in x.name for x in \
                                     model.layers])[0][0]
        bottleneck = activations[bottleneck_ind - 1]        
        ml.latent_space_clustering(bottleneck, ml.normalize(x_test), x,
                                   ticid_test, out=output_dir+\
                                       'clustering-x_test-'+str(i)+'-',
                                   addend=addend)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# # >> parameters
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

# p = {'kernel_size': [21],
#      'latent_dim': [1,2,3,4],
#      'strides': [1],
#      'epochs': [20],
#      'dropout': [0.1],
#      'num_conv_layers': [3],
#      'num_filters': [[16, 32, 32]],
#      'batch_size': [256],
#      'activation': ['relu'],
#      'optimizer': ['adadelta'],
#      'last_activation': ['sigmoid'],
#      'losses': ['mean_squared_error']}

# >> parameters
# p = {'kernel_size': 21,
#      'latent_dim': 3,
#      'strides': 1,
#      'epochs': 4,
#      'dropout': 0.5,
#      'num_conv_layers': 3,
#      'num_filters': [32, 32, 32],
#      'batch_size': 128,
#      'activation': 'relu',
#      'optimizer': 'adadelta',
#      'last_activation': 'sigmoid',
#      'losses': 'mean_squared_loss'}

# p = {'kernel_size': np.arange(5, 501, 2),
#      'latent_dim': [25],
#      'strides' : [1],
#      'epochs': [2],
#      'dropout': np.arange(0, 1., 0.05),
#      'num_conv_layers': [3,5,7,9,11,13],
#      'num_filters': [8,16,32,64, 128],
#      'batch_size': [32,64,128,256],
#      'activation': ['relu'],
#      'optimizer': ['adadelta', 'sgd', 'adam'],
#      'last_activation': ['relu', 'sigmoid'],
#      'losses': ['mean_squared_error', 'binary_crossentropy'],
#      'noise': [0.]}
