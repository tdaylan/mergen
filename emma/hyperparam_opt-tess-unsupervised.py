# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
# 2020-07-14 - hyperparam_opt-tess-unsupervised.py
# Runs a convolutional autoencoder on TESS data. Run with:
# 1. First download data folders from Dropbox (named Sector*Cam*CCD*/) for all
#    groups you want to run on. Move data folders to data_dir
# 2. Download Table_of_momentum_dumps.csv, and change path in mom_dump
# 3. Run this script in the command line with 
#    $ python hyperparam_opt-tess-unsupervised.py
# 
# Emma Chickles
# 
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

sirius=True
run_cpu=False

# data_dir = '../../' # >> directory with input data (ending with /)

if sirius:
    data_dir = '/nfs/ger/home/echickle/data/'
    output_dir = '/nfs/ger/home/echickle/Ensemble-Sector_1/'
    mom_dump = '/nfs/ger/home/echickle/data/Table_of_momentum_dumps.csv'
    database_dir = '/nfs/ger/home/echickle/data/databases/'
else:
    data_dir = '/Users/studentadmin/Dropbox/TESS_UROP/data/'
    output_dir = '../../plots120820/'
    mom_dump = '../../Table_of_momentum_dumps.csv'
    database_dir = data_dir + 'databases/'

lib_dir = '../main/' # >> directory containing model.py, data_functions.py
                     # >> and plotting_functions.py
# database_dir = '../../databases/' # >> directory containing text files for
                                  # >> cross-checking classifications
single_file = False
# database_dir= '/Users/studentadmin/Dropbox/TESS_UROP/data/databases/'


# database_dir = output_dir + 'all_simbad_classifications.txt'
simbad_database_dir = ''
# >> input data
sectors = [1]
cams = [1,2,3,4]
ccds = [[1,2,3,4]]*4
fast=False
n_tot=100
n_components=200

# model_init = output_dir + 'model'
model_init = None
load_saved_model = False
load_weights = False
weights_path = output_dir+'model.hdf5'

train_test_ratio = 1.0 # >> fraction of training set size to testing set size

# >> what this script will run:
preprocessing = True
hyperparameter_optimization = False # >> run hyperparameter search
run_model = False # >> train autoencoder with parameter set p
save_model_epoch = False
diag_plots = False # >> creates diagnostic plots

plot_feat_space = False
novelty_detection=False
classification_param_search=False
classification=False # >> runs DBSCAN on learned features

run_dbscan = False
run_hdbscan= False
run_gmm = False

iterative=True
plot_only=False
train_split=False

# >> normalization options:
#    * standardization : sets mean to 0. and standard deviation to 1.
#    * median_normalization : divides by median
#    * minmax_normalization : sets range of values from 0. to 1.
#    * none : no normalization
if train_split or iterative:
    norm_type = 'none'
else:
    norm_type = 'standardization'

input_rms=False# >> concatenate RMS to learned features
input_psd=False # >> also train on PSD
# n_pgram = 1500
n_pgram = 128

load_psd=False # >> if psd_train.fits, psd_test.fits already exists
use_tess_features = True
use_tls_features = False
input_features=False # >> this option cannot be used yet
split_at_orbit_gap=False
DAE = False
concat_ext_feats=False

model_name = 'best_model.hdf5'
# >> move targets out of training set and into testing set (integer)
# !! TODO: print failure if target not in sector
# targets = [219107776] # >> EX DRA # !!
validation_targets = []

if 1 in sectors:
    custom_mask = list(range(800)) + list(range(15800, 17400)) + list(range(19576, 20075))
elif 4 in sectors:
    custom_mask = list(range(9100, 9800))
else:
    custom_mask = []

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

import numpy as np
import pdb
import os
from astropy.io import fits

if run_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
if sirius:
    print('GPU devices: ')
    print(tf.config.list_physical_devices('GPU'))
    print('CPU devices: ')
    print(tf.config.list_physical_devices('CPU'))

import gc
# tf.enable_eager_execution()

import sys
sys.path.insert(0, lib_dir)     # >> needed if scripts not in current dir
import model as ml              # >> for autoencoder
import data_functions as df     # >> for classification, pre-processing
import plotting_functions as pf # >> for vsualizations
from sklearn.mixture import GaussianMixture

# from keras.models import load_model
from tensorflow.keras.models import load_model

# >> hyperparameters
# if hyperparameter_optimization:
p_opt = {'kernel_size': [3,5],
      'latent_dim': [25, 35],
      'strides': [1,2],
      'epochs': [10],
      'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
      'num_filters': [32,64,128],
      'num_conv_layers': [4,6,8,10],
      'batch_size': [128],
      'activation': [tf.keras.activations.softplus,
                     tf.keras.activations.selu,
                     tf.keras.activations.relu,
                     tf.keras.activations.exponential,
                     tf.keras.activations.elu],
      'optimizer': ['adam', 'adadelta'],
      'last_activation': ['elu'],
      'losses': ['mean_squared_error'],
      'lr': [0.001, 0.0001, 0.00001],
      'initializer': ['random_normal'],
      'num_consecutive': [2, 3],
       'kernel_regularizer': [None],

       'bias_regularizer': [None],

       'activity_regularizer': [None],
     'fully_conv': [False], 'encoder_decoder_skip': [False],
     'encoder_skip':[False], 'decoder_skip': [False],
     'full_feed_forward_highway': [False], 'cvae': [False],
     'share_pool_inds': [False],
      'pool_size': [1, 2], 'batch_norm': [True]}    


# else:    
p = {'kernel_size': 5,
      'latent_dim': 35,
      'strides': 1,
      'epochs': 30,
      'dropout': 0.2,
      'num_filters': 32,
      'num_conv_layers': 6,
      'batch_size': 32,
      'activation': 'elu',
      'optimizer': 'adam',
      'last_activation': 'linear',
      'losses': 'mean_squared_error',
      'lr': 0.0001,
      'initializer': 'random_normal',
      'num_consecutive': 2,
      'pool_size': 2, 
      'pool_strides': 2,
      'units': [1024, 512, 64, 16],
      'kernel_regularizer': None,
      'bias_regularizer': None,
      'activity_regularizer': None,
      'fully_conv': False,
      'encoder_decoder_skip': False,
      'encoder_skip': False,
      'decoder_skip': False,
      'full_feed_forward_highway': False,
      'cvae': False,
      'share_pool_inds': False,
      'batch_norm': True}      
    
# -- create output directory --------------------------------------------------
    
if os.path.isdir(output_dir) == False: # >> check if dir already exists
    os.mkdir(output_dir)
    
# -- load data ----------------------------------------------------------------

if preprocessing:
    if len(sectors) > 1:
        # flux, x, ticid,target_info = \
        #     df.combine_sectors_by_time_axis(sectors, data_dir, 0.2,
        #                                     custom_mask=custom_mask, order=5,
        #                                     tol=0.5, norm_type=norm_type,
        #                                     output_dir=output_dir)
        
        flux, time, ticid, target_info =\
            df.combine_sectors_by_lc(sectors, data_dir, custom_mask=custom_mask,
                                     output_dir=output_dir)
        
    else:
        # >> currently only handles one sector
        flux, time, ticid, target_info = \
            df.load_data_from_metafiles(data_dir, sectors[0], cams=cams, ccds=ccds,
                                        DEBUG=True, fast=fast,
                                        output_dir=output_dir, nan_mask_check=True,
                                        custom_mask=custom_mask)
        

    flux_train, flux_test, x_train, x_test, ticid_train, ticid_test, \
        target_info_train, target_info_test, rms_train, rms_test, time, time_plot = \
        ml.autoencoder_preprocessing(flux, time, p, ticid, target_info,
                                     mock_data=False,
                                     sector=sectors[0],
                                     validation_targets=validation_targets,
                                     norm_type=norm_type,
                                     input_rms=input_rms, input_psd=input_psd,
                                     load_psd=load_psd, n_pgram=n_pgram,
                                     train_test_ratio=train_test_ratio,
                                     split=split_at_orbit_gap,
                                     output_dir=output_dir, 
                                     data_dir=data_dir,
                                     use_tess_features=use_tess_features,
                                     use_tls_features=use_tls_features,
                                     concat_ext_feats=concat_ext_feats)
        

if input_psd:
    p['concat_ext_feats'] = True

# == talos experiment =========================================================
# if hyperparameter_optimization:
#     print('Starting hyperparameter optimization...')
#     import talos
#     experiment_name='TESS-unsupervised'
#     t = talos.Scan(x=x_test, y=x_test, params=p, model=ml.conv_autoencoder,
#                    experiment_name=experiment_name, reduction_metric='val_loss',
#                    minimize_loss=True, reduction_method='correlation',
#                    fraction_limit=0.001)      
#     analyze_object = talos.Analyze(t)
#     data_frame, best_param_ind,p = pf.hyperparam_opt_diagnosis(analyze_object,
#                                                        output_dir,
#                                                        supervised=False)

# == run model ================================================================
if run_model:
    gc.collect()
    print('Training autoencoder...') 
    history, model, bottleneck_train, bottleneck, x_predict = \
        ml.conv_autoencoder(x_train, x_train, x_test, x_test, p, val=False,
                            split=split_at_orbit_gap,
                            ticid_train=ticid_train, ticid_test=ticid_test,
                            save_model=True, predict=True,
                            save_bottleneck=True,
                            output_dir=output_dir,
                            model_init=model_init, train=True) 
    
    if split_at_orbit_gap:
        x_train = np.concatenate(x_train, axis=1)
        x_test = np.concatenate(x_test, axis=1)
        x_predict = np.concatenate(x_predict, axis=1)
    plot_epoch = True
elif load_saved_model:
    if load_weights:
        print('Loading weights...')
        history, model, bottleneck_train, bottleneck, x_predict, x_predict_train = \
                ml.conv_autoencoder(x_train, x_train, x_test, x_test, p, val=False,
                                    split=split_at_orbit_gap,
                                    ticid_train=ticid_train, ticid_test=ticid_test,
                                    save_model=True, predict=True,
                                    save_bottleneck=True,
                                    output_dir=output_dir,
                                    model_init=model_init, train=False,
                                    weights_path=weights_path,
                                    concat_ext_feats=concat_ext_feats)        
        
    else:
        model = load_model(output_dir+'model')
    plot_epoch = False
    
    
# == Plots ====================================================================
if diag_plots:
    print('Creating plots...')
    pf.diagnostic_plots(history, model, p, output_dir, time, x_train,
                        x_test, x_predict, x_predict_train, mock_data=False,
                        target_info_test=target_info_test,
                        target_info_train=target_info_train,
                        ticid_train=ticid_train,
                        ticid_test=ticid_test, percentage=False,
                        input_features=input_features,
                        input_rms=input_rms, rms_test=rms_test,
                        input_psd=input_psd,
                        rms_train=rms_train, n_tot=40,
                        plot_epoch = plot_epoch,
                        load_bottleneck=True)     

# :: novelty detection and classification ::::::::::::::::::::::::::::::::::::::

if novelty_detection or classification:         
    for i in range(3):
        if i == 0:
            use_learned_features=True
            use_tess_features=False
            use_tls_features=False
            use_engineered_features=False
            use_rms=False
            prefix='_0_learned'
            DAE=False
        elif i == 1:
            use_learned_features=False
            use_tess_features=True
            use_tls_features=False
            use_engineered_features=False        
            use_rms=False
            prefix='_1_ext'
            DAE_hyperparam_opt=True
            DAE=True
            p_DAE = {'max_dim': [9, 11, 13, 15, 17, 19], 'step': [1,2,3,4,5,6],
                      'latent_dim': [3,4,5],
                      'activation': ['relu', 'elu'],
                    'last_activation': ['relu', 'elu'],
                      'optimizer': ['adam'],
                      'lr':[0.001, 0.005, 0.01], 'epochs': [20],
                      'losses': ['mean_squared_error'],
                      'batch_size':[128],
                      'initializer': ['glorot_normal', 'glorot_uniform'],
                      'fully_conv': [False]}             
        elif i == 2:
            use_learned_features=True
            use_tess_features=True
            use_tls_features=False
            use_engineered_features=False        
            use_rms=True
            prefix='2_learned_RMS_ext_'  
            DAE_hyperparam_opt=True
            DAE=True
            p_DAE = {'max_dim': list(np.arange(40, 70, 5)), 'step': [1,2,3,4,5,6],
                      'latent_dim': list(np.arange(12, 50, 5)),
                      'activation': ['relu', 'elu'],
                    'last_activation': ['relu', 'elu'],
                      'optimizer': ['adam'],
                      'lr':[0.001, 0.005, 0.01], 'epochs': [20],
                      'losses': ['mean_squared_error'],
                      'batch_size':[128],
                      'initializer': ['glorot_normal', 'glorot_uniform'],
                      'fully_conv': [False]}                
            
        print('Creating feature space')
        
        if input_rms:
            rms = np.concatenate([rms_train, rms_test])
        else:
            rms = None

        ml.post_process(time_plot, flux_train, flux_test, ticid_train, ticid_test,
                        target_info_train, target_info_test, p, output_dir, sectors,
                        prefix=prefix, data_dir=data_dir, database_dir=database_dir,
                        cams=cams, ccds=ccds,
                        use_learned_features=use_learned_features,
                        use_tess_features=use_tess_features, 
                        use_engineered_features=use_engineered_features,
                        use_tls_features=use_tls_features, log=False,
                        momentum_dump_csv=mom_dump)
                
# == iterative training =======================================================
        
if iterative:
    # pdb.set_trace()
    # ticid_err = np.loadtxt(output_dir+'iteration1-ticid_highest_error_train.txt')
    # inter, comm1, comm2 = np.intersect1d(ticid_err, ticid_train, return_indices=True)
    # x_train = x_train[comm2]
    # ticid_train = ticid_train[comm2]
    # target_info_train = target_info_train[comm2]
    if plot_only:
        run=False
    else:
        run=True
    ml.iterative_cae(x_train, x_test, time, p, ticid_train, 
                      ticid_test, target_info_train, target_info_test,
                     iterations=2, n_split=[4,8], latent_dim=[16,8],
                      output_dir=output_dir, split=split_at_orbit_gap,
                      input_psd=input_psd, database_dir=database_dir,
                      data_dir=data_dir, train_psd_only=False,
                     momentum_dump_csv=mom_dump, sectors=sectors,
                     concat_ext_feats=concat_ext_feats, plot=plot_only,
                     run=run, hyperparam_opt=hyperparameter_optimization,
                     p_opt=p_opt) 

if train_split:
    ml.split_cae(time, x_train, x_test, p, target_info_train, target_info_test,
                 ticid_train, ticid_test, sectors, data_dir=data_dir, 
                 database_dir=database_dir, output_dir=output_dir, 
                 momentum_dump_csv=mom_dump, save_model_epoch=save_model_epoch)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
