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

topo_test = True # >> tests on mock data, supervised data and unsupervised data
# * runs unsupervised mode
# * runs supervised mode
# * runs mock datasets
#   * runs no noise
#   * runs .1, .5, ... noise
#   * !! runs all noise (no light curves with peak)
noise = [0., .1, .5]
output_dir = './plots/plots042920-6/'

# >> unsupervised mode
fname_time  = 'section1-time.txt'
fname_flux  = 'section1-flux.csv'
fname_ticid = 'section1-ticid.txt'
# fname_time = 'tessdatasector20-time.txt'
# fname_flux = 'tessdatasector20-intensity.csv'
# fname_ticid = 'tessdatasector20-ticid.txt'

# >> supervised mode
fname_class = 's0020-348-class.csv'
fname_time_supervised  = 's0020-348-time.csv'
fname_flux_supervised  = 's0020-348-flux.csv'
fname_err_supervised   = 's0020-348-flux_err.csv'
fname_ticid_supervised = 's0020-348-ticid.csv'

# >> topography settings
split_lc=False
input_rms = False
supervised = False

# >> parameters
p = {'kernel_size': [[3,3,3,3,3,3,3,3]],
      'latent_dim': [25],
      'strides': [1],
      'epochs': [7],
      'dropout': [0.2],
      'num_conv_layers': [8],
      'num_filters': [[8,16,32,64,64,32,16,8]],
      'batch_size': [32],
      'activation': ['relu'],
      'optimizer': ['adam'],
      'last_activation': ['relu'],
      'losses': ['mean_squared_error'],
      'lr': [0.001]}

# >> supervised
# p = {'kernel_size': [3],
#       'latent_dim': [25,45],
#       'strides': [1],
#       'epochs': [15],
#       'dropout': [0.5,0.1],
#       'num_conv_layers': [9,7],
#       'num_filters': [[32,32,32,32,32,32,32,32,32],[64,64,64,64,64,64,64]],
#       'batch_size': [32],
#       'activation': ['relu'],
#       'optimizer': ['adam'],
#       'last_activation': ['relu'],
#       'losses': ['categorical_crossentropy'],
#       'lr': [0.001]}

# # >> random search
# p = {'kernel_size': [3,5,7],
#       'latent_dim': np.arange(3,100,5),
#       'strides' : [1],
#       'epochs': [20],
#       'dropout': np.arange(0, .5, 0.05),
#       'num_conv_layers': [4,6,8,10,12],
#       'num_filters': [8,16,32,64,128],
#       'batch_size': [32,64,128,256],
#       'activation': ['relu'],
#       'optimizer': ['adam'],
#       'last_activation': ['relu', 'tanh', 'sigmoid'],
#       'losses': ['mean_squared_error'],
#       'lr': [0.001]}

grid_search = True
randomized_search = False
n_iter = 200 # >> for randomized_search

# >> data visualization settings
plot_epoch         = True
plot_in_out        = True
plot_in_bottle_out = True
plot_kernel        = False
plot_intermed_act  = False
plot_latent_test   = True
plot_latent_train  = True
plot_clustering    = False
make_movie         = False

addend = 1. # !!
# >> lc index in x_test
intermed_inds = [6,0] # >> plot_intermed_act
input_bottle_inds = [0,1,2,3,4] # >> in_bottle_out
inds = [0,1,2,3,4,5,6,7,-1,-2,-3,-4,-5,-6,-7] # >> input_output_plot
# inds = [0,1,2,3,4,5,6,7,8,9,0,1,2,3,4]

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

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

if topo_test:
    iterations = 5
else:
    iterations = 1

for k in range(iterations):
    
    if k == 0:
        supervised = False
        mock_data = False
    elif k == 1:
        supervised = True
        mock_data = False
    else:
        supervised = False
        mock_data = True
        noise_level = noise[k-2]
    
    # -- x_train, x_test ------------------------------------------------------
    
    if not mock_data:
        # >> load classes
        if supervised:
            classes     = np.loadtxt(fname_class)
            num_classes = len(np.unique(classes))
            
            x     = np.loadtxt(fname_time_supervised)
            flux  = np.loadtxt(fname_flux_supervised, delimiter=',')
            ticid = np.loadtxt(fname_ticid_supervised)
            
            # !!
            orbit_gap_start = np.argmax(np.diff(x))
            orbit_gap_end = orbit_gap_start + 1
            
        else:
            classes     = False
            num_classes = False
            
            x     = np.loadtxt(fname_time)
            flux  = np.loadtxt(fname_flux, delimiter=',')
            ticid = np.loadtxt(fname_ticid)
        
            # !! find orbit gap
            orbit_gap_start = len(x)-1
            orbit_gap_end = orbit_gap_start + 1
        
        # >> truncate (must be a multiple of 2**num_conv_layers)
        new_length = int(np.shape(flux)[1] / \
                     (2**(np.max(p['num_conv_layers'])/2)))*\
                     int((2**(np.max(p['num_conv_layers'])/2)))
        flux = np.delete(flux, np.arange(new_length, np.shape(flux)[1]), 1)
        x = x[:new_length]
    
        x_train, x_test, y_train, y_test, x = ml.split_data(flux, x,
                                                            train_test_ratio=0.90,
                                                            supervised=supervised,
                                                            classes=classes,
                                                            interpolate=False)

        if not supervised and not mock_data:
            training_size, test_size = np.shape(x_train)[0],np.shape(x_test)[0]
        
        ticid_train = ticid[:np.shape(x_train)[0]]
        ticid_test = ticid[-1 * np.shape(x_test)[0]:]
        
        # if not split_lc:
        #     x_train = x_train[:,:orbit_gap_start]
        #     x_test = x_test[:,:orbit_gap_start]
        #     x = x[:orbit_gap_start]
        
        # >> normalize
        x_train = ml.normalize(x_train)
        x_test = ml.normalize(x_test)
        
    else:
        # !! try sinusoids (vary period, amplitude) and gaussians
        x, x_train, y_train, x_test, y_test = \
            ml.signal_data(training_size=training_size, test_size=test_size,
                           input_dim=new_length,
                           noise_level=noise_level)
            
    if input_rms:
        rms_train, rms_test = ml.rms(x_train), ml.rms(x_test)
    else:
        rms_train, rms_test = False, False
    
    # -- run model ----------------------------------------------------------------
    
    for i in range(len(p_list)):
    
        p = p_list[i]
        print(p)
        
        history, model = ml.autoencoder(x_train, x_test, p,
                                        input_rms=input_rms,
                                        rms_train=rms_train, rms_test=rms_test,
                                        supervised=supervised,
                                        num_classes=num_classes,
                                        y_train=y_train, y_test=y_test,
                                        split_lc=split_lc,
                                        orbit_gap=[orbit_gap_start,
                                                   orbit_gap_end])
    
        if input_rms: x_predict = model.predict([x_test, ml.rms(x_test)])
        x_predict = model.predict(x_test)
    
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
                y_predict = np.argmax(x_predict, axis=-1)
                y_true = np.argmax(y_test, axis=-1)
                cm = confusion_matrix(np.round(y_predict), y_true)
                f.write('confusion matrix\n')
                f.write(str(cm))
                f.write('\ny_true\n')
                f.write(str(y_true)+'\n')
                f.write('y_predict\n')
                f.write(str(y_predict)+'\n')
            else:
                # >> assuming uncertainty of 0.02
                chi_2 = np.average((x_predict-ml.normalize(x_test))**2 / 0.02)
                f.write('chi_squared ' + str(chi_2) + '\n')
                mse = np.average((x_predict - ml.normalize(x_test))**2)
                f.write('mse '+ str(mse) + '\n')
            f.write('\n')
    
        # -- data visualization ---------------------------------------------------
        
        # >> plot loss, accuracy, precision, recall vs. epochs
        if plot_epoch:
            ml.epoch_plots(history,p,output_dir+'epoch-'+str(i)+'-'+str(k)+'-')
        
        # >> plot some decoded light curves
        if plot_in_out and not supervised:
            fig, axes = ml.input_output_plot(x, x_test, x_predict, inds=inds,
                                             out=output_dir+\
                                                    'input_output-x_test-'+\
                                              str(i)+'-'+str(k)+'.png',
                                              addend=addend, sharey=False)
        # >> plot latent space
        activations = ml.get_activations(model, x_test, rms_test = rms_test,
                                         input_rms=input_rms)
    
        if plot_latent_test:
            fig, axes = ml.latent_space_plot(model, activations, p,
                                             output_dir+'latent_space-'+\
                                                 str(i)+'-'+str(k)+'.png')
        if plot_latent_train:
            activations_train = ml.get_activations(model, x_train, 
                                                   rms_test=rms_train,
                                                   input_rms=input_rms)
            fig, axes = ml.latent_space_plot(model, activations_train, p,
                                             output_dir+\
                                                 'latent_space-x_train-'+\
                                                 str(i)+'-'+str(k)+'.png')
    
        # >> plot kernel vs. filter
        if plot_kernel:
            ml.kernel_filter_plot(model, output_dir+'kernel-'+str(i)+'-'+\
                                  str(k)+'-')
    
        # >> plot intermediate activations
        if plot_intermed_act:
            ml.intermed_act_plot(x, model, activations, ml.normalize(x_test),
                                 output_dir+'intermed_act-'+str(i)+'-'+str(k)+'-',
                                 addend=addend, inds=intermed_inds)
        
        if make_movie:
            ml.movie(x, model, activations, x_test, p,
                     output_dir+'movie-'+str(i)+'-'+str(k)+'-', addend=addend,
                     inds=intermed_inds)
    
        # >> plot input, bottleneck, output
        if plot_in_bottle_out and not supervised:
            ml.input_bottleneck_output_plot(x, x_test, x_predict,
                                            activations,
                                            model, out=output_dir+\
                                            'input_bottleneck_output-'+\
                                            str(i)+'-'+str(k)+'.png', addend=addend,
                                            inds = input_bottle_inds, sharey=False)
                
        if plot_clustering:
            bottleneck_ind = np.nonzero(['dense' in x.name for x in \
                                         model.layers])[0][0]
            bottleneck = activations[bottleneck_ind - 1]        
            ml.latent_space_clustering(bottleneck, ml.normalize(x_test), x,
                                       ticid_test, out=output_dir+\
                                           'clustering-x_test-'+str(i)+'-'+\
                                               str(k)+'-',
                                       addend=addend)
                
        if supervised:
            y_train_classes = classes[:np.shape(x_train)[0]]
            ml.training_test_plot(x,x_train,x_test,
                                  y_train_classes,y_true,y_predict,num_classes,
                                  output_dir+'lc-'+str(i)+'-'+str(k)+'-',
                                  ticid_train, ticid_test)
    
    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
