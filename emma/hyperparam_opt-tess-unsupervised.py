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

run_cpu=True

# data_dir = '../../' # >> directory with input data (ending with /)
# data_dir = '/Users/studentadmin/Dropbox/TESS_UROP/data/'
data_dir = '/nfs/ger/home/echickle/data/'

# >> directory to save plots (will make dir if doesn't exist)
output_dir = '/nfs/ger/home/echickle/Ensemble-Sector_2_split/'

# mom_dump = '../../Table_of_momentum_dumps.csv'
mom_dump = '/nfs/ger/home/echickle/data/Table_of_momentum_dumps.csv'

lib_dir = '../main/' # >> directory containing model.py, data_functions.py
                     # >> and plotting_functions.py
# database_dir = '../../databases/' # >> directory containing text files for
                                  # >> cross-checking classifications
single_file = False
# database_dir= '/Users/studentadmin/Dropbox/TESS_UROP/data/databases/'
database_dir = '/nfs/ger/home/echickle/data/databases/'

# database_dir = output_dir + 'all_simbad_classifications.txt'
simbad_database_dir = ''
# >> input data
sectors = [2]
cams = [1,2,3,4]
# cams = [1]
# ccds =  [[2,3,4], [2,3,4], [1,2,4], [1,2,4]]
ccds = [[1,2,3,4]]*4
fast=False
n_tot=100
n_components=200

# weights init
# model_init = output_dir + 'model'
model_init = None
load_saved_model = False
load_weights = False
weights_path = output_dir+'model.hdf5'

# train_test_ratio = 0.1 # >> fraction of training set size to testing set size
train_test_ratio = 0.9
# train_test_ratio = 1.

# >> what this script will run:
preprocessing = True
hyperparameter_optimization = False # >> run hyperparameter search
run_model = True # >> train autoencoder on a parameter set p
diag_plots = False # >> creates diagnostic plots. If run_model==False, then will
                  # >> load bottleneck*.fits for plotting

plot_feat_space = False
novelty_detection=False
classification_param_search=False
classification=False # >> runs DBSCAN on learned features

run_dbscan = False
run_hdbscan= False
run_gmm = False

iterative=True


# >> normalization options:
#    * standardization : sets mean to 0. and standard deviation to 1.
#    * median_normalization : divides by median
#    * minmax_normalization : sets range of values from 0. to 1.
#    * none : no normalization
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

import talos                    # >> a hyperparameter optimization library
import numpy as np
import pdb
import os
from astropy.io import fits

if run_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
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
if hyperparameter_optimization:
    p = {'kernel_size': [3,5],
          'latent_dim': [25],
          'strides': [2],# 3
          'epochs': [5],
          'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
          'num_filters': [32,64,128],
          'num_conv_layers': [4,6,8,10],
          'batch_size': [128],
          'activation': [tf.keras.activations.softplus,
                         tf.keras.activations.selu,
                         tf.keras.activations.relu,
                         'swish',
                         tf.keras.activations.exponential,
                         tf.keras.activations.elu, 'linear'],
          'optimizer': ['adam', 'adadelta'],
          'last_activation': ['linear'],
          'losses': ['mean_squared_error'],
          'lr': [0.001],
          'initializer': ['random_normal'],
          'num_consecutive': [2],
           'kernel_regularizer': [None],
          
           'bias_regularizer': [None],
          
           'activity_regularizer': [None],
         
          'pool_size': [1]}     

    p = {'kernel_size': [5,15,25],
          'latent_dim': [35, 55, 75],
          'strides': [1,2,3],
          'epochs': [10],
          'dropout': [0.1, 0.3, 0.5],
          'num_filters': [8, 32, 64],
          'num_conv_layers': [4, 6, 8, 10],
          'batch_size': [32],
          'activation': ['elu'],
          'optimizer': ['adam'],
          'last_activation': ['elu', 'relu', 'linear'],
          'losses': ['mean_squared_error'],
          'lr': [0.0001],
          'initializer': ['random_normal', 'glorot_normal', 'glorot_uniform'],
          'num_consecutive': [1,2,3],
          'pool_size': [2,4], 
          'pool_strides': [1,2],
          'units': [[1024, 512, 64, 16]],
          'kernel_regularizer': [None],
          'bias_regularizer': [None],
          'activity_regularizer': [None],
          'fully_conv': [False],
          'encoder_decoder_skip': [False],
          'encoder_skip': [False],
          'decoder_skip': [False],
          'full_feed_forward_highway': [False],
          'cvae': [False],
          'share_pool_inds': [False],
          'batch_norm': [True]}      


else:    
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
        # norm_type='none'
        
        flux, x, ticid, target_info = df.combine_sectors_by_lc(sectors, data_dir,
                                                                custom_mask=custom_mask,
                                                                output_dir=output_dir)
        
    else:
        # >> currently only handles one sector
        flux, x, ticid, target_info = \
            df.load_data_from_metafiles(data_dir, sectors[0], cams=cams, ccds=ccds,
                                        DEBUG=True, fast=fast,
                                        output_dir=output_dir, nan_mask_check=True,
                                        custom_mask=custom_mask)
        # flux_plot = None
        
    x_train, x_test, y_train, y_test, ticid_train, ticid_test, target_info_train, \
        target_info_test, rms_train, rms_test, x = \
        ml.autoencoder_preprocessing(flux, x, p, ticid, target_info,
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
        
    # hdr = fits.Header()
    # hdu = fits.PrimaryHDU(x_train, header=hdr)
    # hdu.writeto(output_dir + 'x_train.fits')
    # hdu = fits.PrimaryHDU(x_test, header=hdr)
    # hdu.writeto(output_dir + 'x_test.fits')
    # hdu = fits.PrimaryHDU(ticid_train, header=hdr)
    # hdu.writeto(output_dir + 'ticid_train.fits')
    # hdu = fits.PrimaryHDU(ticid_test, header=hdr)
    # hdu.writeto(output_dir + 'ticid_test.fits')
    # # hdu = fits.PrimaryHDU(target_info_train, header=hdr)
    # # hdu.writeto(output_dir + 'target_info_train.fits')
    # # hdu = fits.PrimaryHDU(target_info_test, header=hdr)
    # # hdu.writeto(output_dir + 'target_info_test.fits')    
    # if input_rms:
    #     hdu = fits.PrimaryHDU(rms_train, header=hdr) 
    #     hdu.writeto(output_dir + 'rms_train.fits')
    #     hdu = fits.PrimaryHDU(rms_test, header=hdr)
    #     hdu.writeto(output_dir + 'rms_test.fits')     


else:
    f = fits.open(output_dir + 'x_train.fits')
    x_train = f[0].data
    f = fits.open(output_dir + 'x_test.fits')
    x_test = f[0].data
    f = fits.open(output_dir + 'ticid_train.fits')
    ticid_train = f[0].data
    f = fits.open(output_dir + 'ticid_test.fits')
    ticid_test = f[0].data

if input_psd:
    p['concat_ext_feats'] = True

title='TESS-unsupervised'

# == talos experiment =========================================================
if hyperparameter_optimization:
    print('Starting hyperparameter optimization...')
    t = talos.Scan(x=x_test,
                    y=x_test,
                    params=p,
                    model=ml.conv_autoencoder,
                    experiment_name=title, 
                    reduction_metric = 'val_loss',
                    minimize_loss=True,
                    reduction_method='correlation',
                    fraction_limit = 0.001)      
    # fraction_limit = 0.001
    analyze_object = talos.Analyze(t)
    data_frame, best_param_ind,p = pf.hyperparam_opt_diagnosis(analyze_object,
                                                       output_dir,
                                                       supervised=False)

# == run model ================================================================
if run_model:
    gc.collect()
    print('Training autoencoder...') 
    history, model, x_predict = \
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
        history, model, x_predict = \
                ml.conv_autoencoder(x_train, x_train, x_test, x_test, p, val=False,
                                    split=split_at_orbit_gap,
                                    ticid_train=ticid_train, ticid_test=ticid_test,
                                    save_model=True, predict=True,
                                    save_bottleneck=True,
                                    output_dir=output_dir,
                                    model_init=model_init, train=False,
                                    weights_path=weights_path,
                                    concat_ext_feats=concat_ext_feats)        
        
        # # >> create model
        # import model as ml
        # from keras.models import Model
        # encoded = ml.encoder(x_train, p)
        # decoded = ml.decoder(x_train, encoded.output, p)
        # model = Model(encoded.input, decoded)
        # model.summary()
        # model.load_weights(output_dir+weights_path)
    else:
        model = load_model(output_dir+'model')
    plot_epoch = False
    
    
# == Plots ====================================================================
if diag_plots:
    print('Creating plots...')
    pf.diagnostic_plots(history, model, p, output_dir, x, x_train,
                        x_test, x_predict, mock_data=False,
                        addend=0.,
                        target_info_test=target_info_test,
                        target_info_train=target_info_train,
                        ticid_train=ticid_train,
                        ticid_test=ticid_test, percentage=False,
                        input_features=input_features,
                        input_rms=input_rms, rms_test=rms_test,
                        input_psd=input_psd,
                        rms_train=rms_train, n_tot=40,
                        
                        plot_epoch = plot_epoch,
                        plot_in_out = True,
                        plot_in_bottle_out=False,
                        plot_latent_test = True,
                        plot_latent_train = True,
                        plot_kernel=False,
                        plot_intermed_act=False,
                        make_movie = False,
                        plot_lof_test=False,
                        plot_lof_train=False,
                        plot_lof_all=False,
                        plot_reconstruction_error_test=True,
                        plot_reconstruction_error_all=False,
                        load_bottleneck=True)     
# flux_train=flux_train, flux_test=flux_test, time=x,     
    
 
# if input_psd:
#     x = x[0]  
if novelty_detection or classification:         
    for i in [0]: # !!
        if i == 0:
            use_learned_features=True
            use_tess_features=False
            use_tls_features=False
            use_engineered_features=False
            use_rms=False
            description='_0_learned'
            DAE=False
        elif i == 1:
            use_learned_features=False
            use_tess_features=True
            use_tls_features=False
            use_engineered_features=False        
            use_rms=False
            description='_1_ext'
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
            description='_2_learned_RMS_ext'        
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
            
        # !! 
        flux_train = x_train
        flux_test = x_test
        
        if p['concat_ext_feats'] or input_psd:
            features, flux_feat, ticid_feat, info_feat = \
                ml.bottleneck_preprocessing(sectors[0],
                                            np.concatenate([x_train[0], x_test[0]], axis=0),
                                            np.concatenate([ticid_train, ticid_test]),
                                            np.concatenate([target_info_train,
                                                            target_info_test]),
                                            rms=np.concatenate([rms_train, rms_test]),
                                            data_dir=data_dir, bottleneck_dir=output_dir,
                                            output_dir=output_dir,
                                            use_learned_features=use_learned_features,
                                            use_tess_features=use_tess_features,
                                            use_engineered_features=use_engineered_features,
                                            use_tls_features=use_tls_features,
                                            use_rms=use_rms, norm=True,
                                            cams=cams, ccds=ccds, log=True)    
                
        
        else:
            features, flux_feat, ticid_feat, info_feat = \
                ml.bottleneck_preprocessing(sectors[0],
                                            np.concatenate([flux_train, flux_test], axis=0),
                                            np.concatenate([ticid_train, ticid_test]),
                                            np.concatenate([target_info_train,
                                                            target_info_test]),
                                            rms=rms,
                                            data_dir=data_dir,
                                            bottleneck_dir=output_dir,
                                            output_dir=output_dir,
                                            use_learned_features=True,
                                            use_tess_features=use_tess_features,
                                            use_engineered_features=False,
                                            use_tls_features=use_tls_features,
                                            use_rms=use_rms, norm=True,
                                            cams=cams, ccds=ccds, log=True)  
                
        if plot_feat_space:
            print('Plotting feature space')
            pf.latent_space_plot(features, output_dir + 'feature_space_'+str(i)+'.png')    
        
        if DAE:
            if DAE_hyperparam_opt:
    
          
                t = talos.Scan(x=features,
                                y=features,
                                params=p_DAE,
                                model=ml.deep_autoencoder,
                                experiment_name='DAE', 
                                reduction_metric = 'val_loss',
                                minimize_loss=True,
                                reduction_method='correlation',
                                fraction_limit = 0.1)            
                analyze_object = talos.Analyze(t)
                data_frame, best_param_ind,p_best = pf.hyperparam_opt_diagnosis(analyze_object,
                                                                    output_dir,
                                                                    supervised=False) 
                p_DAE=p_best
                p_DAE['epochs'] = 100
                
            else:
                    
                p_DAE = {'max_dim': 50, 'step': 4, 'latent_dim': 30,
                         'activation': 'elu', 'last_activation': 'elu',
                         'optimizer': 'adam',
                         'lr':0.001, 'epochs': 100, 'losses': 'mean_squared_error',
                         'batch_size': 128, 'initializer': 'glorot_uniform',
                         'fully_conv': False}    
            
                
                # p_DAE = {'max_dim': 9, 'step': 5, 'latent_dim': 4,
                #          'activation': 'elu', 'last_activation': 'elu',
                #          'optimizer': 'adam',
                #          'lr':0.01, 'epochs': 100, 'losses': 'mean_squared_error',
                #          'batch_size': 128, 'initializer': 'glorot_normal',
                #          'fully_conv': False}               
                
            history_DAE, model_DAE = ml.deep_autoencoder(features, features,
                                                           features, features,
                                                           p_DAE, resize=False,
                                                           batch_norm=True)
            new_features = ml.get_bottleneck(model_DAE, features, p_DAE, DAE=True)
            features=new_features
            
            pf.epoch_plots(history_DAE, p_DAE, output_dir)
            
            print('Plotting feature space')
            pf.latent_space_plot(features, output_dir + 'feature_space' + \
                                 ''+'_DAE_'+str(i)+'.png')        
    
            
        
    
        if novelty_detection:
            print('Novelty detection')
            pf.plot_lof(x, flux_feat, ticid_feat, features, 20, output_dir,
                        n_tot=n_tot, target_info=info_feat, prefix=str(i),
                        database_dir=database_dir, debug=True, addend=0.,
                        single_file=single_file, log=True, n_pgram=n_pgram,
                        plot_psd=True, momentum_dump_csv=mom_dump)
            
            pf.plot_lof_summary(x, flux_feat, ticid_feat, features, 20,
                                output_dir, target_info=info_feat,
                                database_dir=database_dir,
                                momentum_dump_csv=mom_dump)
    
        if classification:
            if classification_param_search and run_dbscan:
                df.KNN_plotting(output_dir +'str(i)-', features, [10, 20, 100])
        
                print('DBSCAN parameter search')
                parameter_sets, num_classes, silhouette_scores, db_scores, ch_scores, acc = \
                df.dbscan_param_search(features, x, flux_feat, ticid_feat,
                                        info_feat, DEBUG=False, 
                                        output_dir=output_dir+str(i), 
                                        simbad_database_txt=simbad_database_dir,
                                        leaf_size=[30], algorithm=['auto'],
                                        min_samples=[5],
                                        metric=['minkowski'], p=[3,4],
                                        database_dir=database_dir,
                                        eps=list(np.arange(1.5, 4., 0.1)),
                                        confusion_matrix=False, pca=False, tsne=False,
                                        tsne_clustering=False)      
                
                print('Classification with best parameter set')
                best_ind = np.argmax(silhouette_scores)
                best_param_set = parameter_sets[best_ind]
                
            else:
                best_param_set=[2.0, 3, 'minkowski', 'auto', 30, 4]    
          
        
            
            if classification_param_search and run_hdbscan:
                print('HDBSCAN parameter search')
                acc = df.hdbscan_param_search(features, x, flux_feat, ticid_feat,
                                              info_feat, output_dir=output_dir,
                                              p0=[3,4], single_file=single_file,
                                              database_dir=database_dir, metric=['all'],
                                              min_samples=[3], min_cluster_size=[3],
                                              data_dir=data_dir)
            elif not classification_param_search and run_hdbscan:
                # best_param_set = [3, 3, 'manhattan', None]
                best_param_set = [3, 3, 'canberra', None]
                print('Run HDBSCAN')
                _, _, acc = df.hdbscan_param_search(features, x, flux_feat, ticid_feat,
                                              info_feat, output_dir=output_dir,
                                              p0=[best_param_set[3]], single_file=single_file,
                                              database_dir=database_dir,
                                              metric=[best_param_set[2]],
                                              min_cluster_size=[best_param_set[0]],
                                              min_samples=[best_param_set[1]],
                                              DEBUG=True, pca=True, tsne=True,
                                              data_dir=data_dir, save=False)  
              
            if run_hdbscan:
                import hdbscan
                clusterer = hdbscan.HDBSCAN(min_cluster_size=best_param_set[0],
                                            min_samples=best_param_set[1],
                                            metric=best_param_set[2]).fit(features)
                assigned_labels, assigned_classes, recalls = \
                    pf.assign_classes(ticid_feat, clusterer.labels_, database_dir=database_dir,
                                      output_dir=output_dir, prefix='hdbscan-')
                    
    
                
                with open(output_dir + 'param_summary.txt', 'a') as f:
                    f.write('accuracy: ' + str(np.max(acc)))   
                
            # df.gmm_param_search(features, x, flux_feat, ticid_feat, info_feat,
            #                  output_dir=output_dir+'gmm_'+str(i), database_dir=database_dir, 
            #                  data_dir=data_dir) 
    
            if run_gmm:
                if os.path.exists(output_dir+'gmm_fit.txt'):
                    _, labels = np.loadtxt(output_dir+'gmm_fit.txt')
                else:
                    gmm = GaussianMixture(n_components=n_components)
                    labels = gmm.fit_predict(features)
                    np.savetxt(output_dir+'gmm_fit.txt', np.array([ticid_feat, labels]))
                acc = pf.plot_confusion_matrix(ticid_feat, labels,
                                               database_dir=database_dir,
                                               single_file=single_file,
                                               output_dir=output_dir,
                                               prefix='gmm-'+str(i)+'_')          
                pf.quick_plot_classification(x, flux_feat,ticid_feat,info_feat, 
                                             features, labels,path=output_dir,
                                             prefix='gmm-'+str(i)+'_',
                                             database_dir=database_dir)
                pf.plot_cross_identifications(x, flux_feat, ticid_feat,
                                              info_feat, features,
                                              labels, path=output_dir,
                                              database_dir=database_dir,
                                              data_dir=data_dir, prefix='gmm-'+str(i)+'_')
        
                class_info = df.get_true_classifications(ticid_feat, database_dir)
                pf.ensemble_summary(ticid_feat, labels, database_dir,
                                    output_dir, 'gmm-', data_dir=data_dir,
                                    class_info=class_info)
                
                
                cm, assignments, ticid_true, y_true, class_info_new, recalls, false_discovery_rates,\
                        counts_true, counts_pred, precisions, accuracy = \
                            pf.assign_real_labels(ticid_feat, labels, database_dir, data_dir, class_info)
                pf.ensemble_summary_tables(assignments, recalls, false_discovery_rates, precisions, accuracy, counts_true, counts_pred, output_dir)
                pf.ensemble_summary_tables(assignments, recalls, false_discovery_rates, precisions, accuracy, counts_true, counts_pred, output_dir, target_labels=[])
                inter, comm1, comm2 = np.intersect1d(ticid_feat, ticid_true, return_indices=True)
                y_pred = labels[comm1]
                
                flux_in = flux_feat[comm1]
                flux_pred = model.predict(flux_in)
                
                pf.plot_fail_cases(x, flux_in, ticid_true, y_true, y_pred, assignments, class_info, info_feat[comm1], output_dir)
                
                # >> find top 20 most popular classes
                classes, counts = np.unique(y_true, return_counts=True)
                classes = classes[np.argsort(counts)]
                for class_label in classes[-20:]:
                    pf.plot_class_dists(assignments, ticid_true, y_pred, y_true,
                                        data_dir, sectors, true_label=class_label,
                                        output_dir=output_dir)
                
                
                # pf.plot_class_dists(assignments, ticid_true, y_pred, y_true, data_dir, sectors, output_dir=output_dir)
                pf.sector_dists(data_dir, sectors, output_dir=output_dir)
                
                true_label = 'E'
                pf.plot_fail_reconstructions(x, flux_in, flux_pred, ticid_true,
                                             y_true, y_pred, assignments,
                                             class_info, info_feat[comm1],
                                             output_dir=output_dir,
                                             true_label='E')
                
                
# == iterative training =======================================================
        
if iterative:
    ml.iterative_cae(x_train, y_train, x_test, y_test, x, p, ticid_train, 
                      ticid_test, target_info_train, target_info_test, num_split=2,
                      output_dir=output_dir, split=split_at_orbit_gap,
                      input_psd=input_psd, database_dir=database_dir,
                      data_dir=data_dir, train_psd_only=False,
                     momentum_dump_csv=mom_dump, sectors=sectors,
                     concat_ext_feats=concat_ext_feats) 
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
