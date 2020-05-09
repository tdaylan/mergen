# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
# pipeline for retrieving outliers in tess data
# etc 0420
#
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

import modellibrary as ml
import numpy as np
import random
import pdb
from itertools import product
from sklearn.metrics import confusion_matrix

test_mock_data = True
test_supervised = True
test_unsupervised = True

topo_test = True # >> tests on mock data, supervised data and unsupervised data
# * runs unsupervised mode
# * runs supervised mode
# * runs mock datasets
#   * runs no noise
#   * runs .1, .5, ... noise
#   * !! runs all noise (no light curves with peak)
noise = [.1, .5]
training_size, test_size, input_dim = 1039, 116, 8896
training_size_supervised, test_size_supervised = 313, 35
input_dim_supervised = 8896
output_dir = './plots/plots050820/'

# >> unsupervised mode
fname_time  = 's0020-before_orbit-1155-time.txt' # s0020-1155-
fname_flux  = 's0020-before_orbit-1155-flux.csv'
fname_ticid = 's0020-before_orbit-1155-ticid.txt'

# >> supervised mode
fname_class = 's0020-348-class.csv'
fname_time_supervised  = 's0020-before_orbit-348-time.txt'
fname_flux_supervised  = 's0020-before_orbit-348-flux.csv'
fname_ticid_supervised = 's0020-before_orbit-348-ticid.txt'

# >> topography settings
split_lc=False
input_rms = False
supervised = False

# >> parameters
p = {'kernel_size': [[3,3]],
      'latent_dim': [5, 25, 50, 75],
      'strides': [1],
      'epochs': [50],
      'dropout': [0.],
      'num_filters': [[1,1]],
      'batch_size': [128],
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
input_bottle_inds = [0,1,2,-6,-7] # >> in_bottle_out
inds = [0,1,2,3,4,5,6,7,-1,-2,-3,-4,-5,-6,-7] # >> input_output_plot
# inds = [0,1,2,3,4,5,6,7,8,9,0,1,2,3,4]

center_factor = 0. # 5.
h_factor = 0. # 0.2

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# -- initialize parameters ----------------------------------------------------

p_list = []
if grid_search:
    p_combinations = list(product(*p.values()))
    for i in range(len(p_combinations)):
        p1 = {}
        for j in range(len(p.keys())):
            p1[list(p.keys())[j]] = p_combinations[i][j]
        p1['num_conv_layers'] = len(p1['num_filters'])
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

iterations = 0
if test_mock_data: iterations += 2*len(noise)
if test_supervised: iterations += 1
if test_unsupervised: iterations += 1

# == loop through parameter sets ==============================================

for i in range(len(p_list)):
    p = p_list[i]
    print(p)
    
    # -- loop through supervised, unsupervised modes --------------------------
    for k in range(iterations):
        
        if test_mock_data:
            if k < len(noise):            # >> run mock data unsupervised
                supervised, mock_data, noise_level = False, True, noise[k]
                title = "mock data noise "+str(noise_level)
            elif k < 2*len(noise):        # >> run mock data supervised
                supervised, mock_data = True, True
                noise_level = noise[k-len(noise)]
                title = 'mock data supervised noise '+str(noise_level)
            if test_unsupervised:
                if k == iterations - 2:
                    supervised, mock_data = False, False
                    title='unsupervised mode'
                if test_supervised and k == iterations-1:
                    supervised, mock_data = True, False
                    title='supervised mode'
            elif test_supervised:
                supervised, mock_data = True, False
                title='supervised mode'
                
        elif test_unsupervised:
            if k == 0:
                supervised, mock_data = False, False
                title='unsupervised mode'
            if test_supervised and k ==1:
                supervised, mock_data = True, False
                title='supervised mode'
                
        else:
            supervised, mock_data = True, False
            title='supervised mode'
        
        print(title)
        
        # == x_train, x_test ==================================================
        if not mock_data:
            # -- load tess data -----------------------------------------------
            if supervised:
                classes     = np.loadtxt(fname_class)
                num_classes = len(np.unique(classes))
                x     = np.loadtxt(fname_time_supervised)
                flux  = np.loadtxt(fname_flux_supervised, delimiter=',')
                ticid = np.loadtxt(fname_ticid_supervised)
                orbit_gap_start = np.argmax(np.diff(x))
                orbit_gap_end = orbit_gap_start + 1     
            else:
                classes     = False
                num_classes = False
                x     = np.loadtxt(fname_time)
                flux  = np.loadtxt(fname_flux, delimiter=',')
                ticid = np.loadtxt(fname_ticid)
                orbit_gap_start = len(x)-1 # !!
                orbit_gap_end = orbit_gap_start + 1
        
            # >> normalize
            flux, x = ml.normalize(flux, x)

            # >> take before orbit gap            
            if not split_lc:
                flux = flux[:,:orbit_gap_start]
                x = x[:orbit_gap_start]
            
            x_train, x_test, y_train, y_test, x = \
                ml.split_data(flux, x, p, train_test_ratio=0.90, 
                              supervised=supervised, classes=classes)
            
            ticid_train = ticid[:np.shape(x_train)[0]]
            ticid_test = ticid[-1 * np.shape(x_test)[0]:]
            
        else: # -- mock data --------------------------------------------------
            # !! try sinusoids (vary period, amplitude) and 
            if supervised:
                x, x_train, y_train, x_test, y_test = \
                    ml.signal_data(training_size=training_size_supervised,
                                   test_size=test_size_supervised,
                                   input_dim=input_dim_supervised,
                                   noise_level=noise_level,
                                   center_factor=center_factor,
                                   h_factor=h_factor)
                num_classes = 2
                orbit_gap_start, orbit_gap_end = False, False
                ticid_train, ticid_test = False, False
            else:
                x, x_train, y_train, x_test, y_test = \
                    ml.signal_data(training_size=training_size,
                                   test_size=test_size,
                                   input_dim=input_dim,
                                   noise_level=noise_level,
                                   center_factor=center_factor,
                                   h_factor=h_factor)
                num_classes = False
                orbit_gap_start, orbit_gap_end = False, False
                ticid_train, ticid_test = False, False
                
        if input_rms:
            rms_train, rms_test = ml.rms(x_train), ml.rms(x_test)
        else:
            rms_train, rms_test = False, False
    
        # == run model ========================================================
            
        history, model = ml.autoencoder(x_train, x_test, p,
                                        input_rms=input_rms,
                                        rms_train=rms_train, rms_test=rms_test,
                                        supervised=supervised,
                                        num_classes=num_classes,
                                        y_train=y_train, y_test=y_test,
                                        split_lc=split_lc,
                                        orbit_gap=[orbit_gap_start,
                                                   orbit_gap_end],
                                        simple = True)
    
        if input_rms: x_predict = model.predict([x_test, ml.rms(x_test)])
        x_predict = model.predict(x_test)
    
        # -- param summary txt ------------------------------------------------
            
        with open(output_dir + 'param_summary.txt', 'a') as f:
            f.write('parameter set ' + str(i) + ' - ' + title +'\n')
            f.write(str(p.items()) + '\n')
            if supervised:
                label_list = ['loss', 'accuracy', 'precision', 'recall']
                key_list =['loss', 'accuracy', list(history.history.keys())[-2],
                       list(history.history.keys())[-1]]
            else:
                label_list = ['loss', 'accuracy']
                key_list = ['loss', 'accuracy']

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
                chi_2 = np.average((x_predict-x_test)**2 / 0.02)
                f.write('chi_squared ' + str(chi_2) + '\n')
                mse = np.average((x_predict - x_test)**2)
                f.write('mse '+ str(mse) + '\n')
            f.write('\n')
            
        if k == 0:
            with open(output_dir + 'model_summary.txt', 'a') as f:
                model.summary(print_fn=lambda line: f.write(line + "\n"))
    
        # == data visualization ===============================================
        
        # >> plot loss, accuracy, precision, recall vs. epochs
        if plot_epoch:
            ml.epoch_plots(history,p,output_dir+'epoch-'+str(i)+'-'+str(k)+'-',
                           supervised=supervised)
        
        # >> plot some decoded light curves
        if plot_in_out and not supervised:
            fig, axes = ml.input_output_plot(x, x_test, x_predict,
                                             output_dir+'input_output-x_test-'+\
                                              str(i)+'-'+str(k)+'.png',
                                              ticid_test=ticid_test,
                                              inds=inds,
                                              addend=addend, sharey=False,
                                              mock_data=mock_data)
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
            ml.intermed_act_plot(x, model, activations, x_test,
                                 output_dir+'intermed_act-'+str(i)+'-'+str(k)+'-',
                                 addend=addend, inds=intermed_inds)
        
        if make_movie:
            ml.movie(x, model, activations, x_test, p,
                     output_dir+'movie-'+str(i)+'-'+str(k)+'-', addend=addend,
                     inds=intermed_inds)
    
        # >> plot input, bottleneck, output
        if plot_in_bottle_out and not supervised:
            ml.input_bottleneck_output_plot(x, x_test, x_predict,
                                            activations, model, ticid_test,
                                            output_dir+\
                                            'input_bottleneck_output-'+\
                                            str(i)+'-'+str(k)+'.png',
                                            addend=addend,
                                            inds = input_bottle_inds,
                                            sharey=False, mock_data=mock_data)
                
        if plot_clustering:
            bottleneck_ind = np.nonzero(['dense' in x.name for x in \
                                         model.layers])[0][0]
            bottleneck = activations[bottleneck_ind - 1]        
            ml.latent_space_clustering(bottleneck, x_test, x,
                                       ticid_test, out=output_dir+\
                                           'clustering-x_test-'+str(i)+'-'+\
                                               str(k)+'-',
                                       addend=addend)
                
        if supervised:
            if mock_data:
                y_train_classes = y_train[:,1]
            else:
                y_train_classes = classes[:np.shape(x_train)[0]]
            ml.training_test_plot(x,x_train,x_test,
                                  y_train_classes,y_true,y_predict,num_classes,
                                  output_dir+'lc-'+str(i)+'-'+str(k)+'-',
                                  ticid_train, ticid_test, mock_data=mock_data)
    
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
