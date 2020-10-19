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

# data_dir = '../../' # >> directory with input data (ending with /)
data_dir = '/Users/studentadmin/Dropbox/TESS_UROP/data/'
output_dir = '../../plots/Ensemble-Sectors_4_5/' # >> directory to save diagnostic plots
                                     # >> will make dir if doesn't exist
mom_dump = '../../Table_of_momentum_dumps.csv'
lib_dir = '../main/' # >> directory containing model.py, data_functions.py
                     # >> and plotting_functions.py
# database_dir = '../../databases/' # >> directory containing text files for
                                  # >> cross-checking classifications
single_file = False
database_dir= '/Users/studentadmin/Dropbox/TESS_UROP/data/databases/'
# database_dir = output_dir + 'all_simbad_classifications.txt'
simbad_database_dir = ''
# >> input data
sectors = [4,5]
cams = [1,2,3,4]
# cams = [1]
# ccds =  [[2,3,4], [2,3,4], [1,2,4], [1,2,4]]
ccds = [[1,2,3,4]]*4
fast=False

# weights init
# model_init = output_dir + 'model'
model_init = None


# train_test_ratio = 0.1 # >> fraction of training set size to testing set size
train_test_ratio = 0.9
# train_test_ratio = 1.

# >> what this script will run:
hyperparameter_optimization = False # >> run hyperparameter search
run_model = True # >> train autoencoder on a parameter set p
iterative=True
diag_plots = True # >> creates diagnostic plots. If run_model==False, then will
                  # >> load bottleneck*.fits for plotting

novelty_detection=True
classification_param_search=False
classification=True # >> runs DBSCAN on learned features

# >> normalization options:
#    * standardization : sets mean to 0. and standard deviation to 1.
#    * median_normalization : divides by median
#    * minmax_normalization : sets range of values from 0. to 1.
#    * none : no normalization
norm_type = 'standardization'

input_rms=True# >> concatenate RMS to learned features
input_psd=False # >> also train on PSD
# n_pgram = 1500
n_pgram = 128

load_psd=False # >> if psd_train.fits, psd_test.fits already exists
use_tess_features = True
use_tls_features = False
input_features=False # >> this option cannot be used yet
split_at_orbit_gap=False
DAE = False

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
import tensorflow as tf
import gc
# tf.enable_eager_execution()

import sys
sys.path.insert(0, lib_dir)     # >> needed if scripts not in current dir
import model as ml              # >> for autoencoder
import data_functions as df     # >> for classification, pre-processing
import plotting_functions as pf # >> for vsualizations

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

else:
    # >> strides: list, len = num_consecutive
    p = {'kernel_size': 5,
          'latent_dim': 35,
          'strides': 1,
          'epochs': 3,
          'dropout': 0.2,
          'num_filters': 16,
          'num_conv_layers': 4,
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
          'batchnorm_before_act': True,
          'concat_ext_feats': False}      
    
# -- create output directory --------------------------------------------------
    
if os.path.isdir(output_dir) == False: # >> check if dir already exists
    os.mkdir(output_dir)
    
# -- load data ----------------------------------------------------------------


if len(sectors) > 1:
    flux, x, ticid,target_info = \
        df.combine_sectors_by_time_axis(sectors, data_dir, 0.2,
                                        custom_mask=custom_mask, order=5,
                                        tol=0.5, norm_type=norm_type,
                                        output_dir=output_dir)
    norm_type='none'
    
    # flux, x, ticid, target_info = df.combine_sectors_by_lc(sectors, data_dir,
    #                                                        custom_mask=custom_mask,
    #                                                        output_dir=output_dir)
    
else:
    # >> currently only handles one sector
    flux, x, ticid, target_info = \
        df.load_data_from_metafiles(data_dir, sectors[0], cams=cams, ccds=ccds,
                                    DEBUG=True, fast=fast,
                                    output_dir=output_dir, nan_mask_check=True,
                                    custom_mask=custom_mask)
    
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
                                 use_tls_features=use_tls_features)
    
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
                            model_init=model_init) 
    
    if split_at_orbit_gap:
        x_train = np.concatenate(x_train, axis=1)
        x_test = np.concatenate(x_test, axis=1)
        x_predict = np.concatenate(x_predict, axis=1)
    
    
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
                        plot_epoch = True,
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
    
 
# if input_psd:
#     x = x[0]  
if novelty_detection or classification:         
    for i in [0,1,2]:
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
                                            np.concatenate([x_train, x_test], axis=0),
                                            np.concatenate([ticid_train, ticid_test]),
                                            np.concatenate([target_info_train,
                                                            target_info_test]),
                                            rms=np.concatenate([rms_train, rms_test]),
                                            data_dir=data_dir,
                                            bottleneck_dir=output_dir,
                                            output_dir=output_dir,
                                            use_learned_features=True,
                                            use_tess_features=use_tess_features,
                                            use_engineered_features=False,
                                            use_tls_features=use_tls_features,
                                            use_rms=use_rms, norm=True,
                                            cams=cams, ccds=ccds, log=True)  
                
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
                        n_tot=200, target_info=info_feat, prefix=str(i),
                        cross_check_txt=database_dir, debug=True, addend=0.,
                        single_file=single_file, log=True, n_pgram=n_pgram,
                        plot_psd=True)
    
        if classification:
            if classification_param_search:
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
          
        
            
            if classification_param_search:
                print('HDBSCAN parameter search')
                acc = df.hdbscan_param_search(features, x, flux_feat, ticid_feat,
                                              info_feat, output_dir=output_dir,
                                              p0=[3,4], single_file=single_file,
                                              database_dir=database_dir, metric=['all'],
                                              min_samples=[3], min_cluster_size=[3],
                                              data_dir=data_dir)
            else:
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
            
            with open(output_dir + 'param_summary.txt', 'a') as f:
                f.write('accuracy: ' + str(np.max(acc)))   
                
            # df.gmm_param_search(features, x, flux_feat, ticid_feat, info_feat,
            #                  output_dir=output_dir+'gmm_'+str(i), database_dir=database_dir, 
            #                  data_dir=data_dir) 
    
            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(n_components=100)
            labels = gmm.fit_predict(features)
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
    
        
# == iterative training =======================================================
        
if iterative:
    ml.iterative_cae(x_train, y_train, x_test, y_test, x, p, ticid_train, 
                      ticid_test, target_info_train, target_info_test, num_split=2,
                      output_dir=output_dir, split=split_at_orbit_gap,
                      input_psd=input_psd, database_dir=database_dir,
                      data_dir=data_dir) 
        
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
    # if DBSCAN_parameter_search:
    #     from sklearn.cluster import DBSCAN
    #     with fits.open(output_dir + 'bottleneck_test.fits') as hdul:
    #         bottleneck = hdul[0].data
    #     # !! already standardized
    #     # bottleneck = ml.standardize(bottleneck, ax=0)
    #     eps = list(np.arange(0.1,5.0,0.1))
    #     min_samples = [10]# [2, 5,10,15]
    #     metric = ['euclidean'] # ['euclidean', 'minkowski']
    #     # algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    #     algorithm=['auto']
    #     # leaf_size = [30, 40, 50]
    #     leaf_size=[30]
    #     # p = [1,2,3,4]
    #     p=[None]
    #     classes = []
    #     num_classes = []
    #     counts = []
    #     num_noisy= []
    #     parameter_sets=[]
    #     for i in range(len(eps)):
    #         for j in range(len(min_samples)):
    #             for k in range(len(metric)):
    #                 for l in range(len(algorithm)):
    #                     for m in range(len(leaf_size)):
    #                         for n in range(len(p)):
    #                             db = DBSCAN(eps=eps[i],
    #                                         min_samples=min_samples[j],
    #                                         metric=metric[k],
    #                                         algorithm=algorithm[l],
    #                                         leaf_size=leaf_size[m],
    #                                         p=p[n]).fit(bottleneck)
    #                             print(db.labels_)
    #                             print(np.unique(db.labels_, return_counts=True))
    #                             classes_1, counts_1 = \
    #                                 np.unique(db.labels_, return_counts=True)
    #                             classes.append(classes_1)
    #                             num_classes.append(len(classes_1))
    #                             counts.append(counts_1)
    #                             num_noisy.append(counts[0])
    #                             parameter_sets.append([eps[i], min_samples[j],
    #                                                    metric[k],
    #                                                    algorithm[l],
    #                                                    leaf_size[m],
    #                                                    p[n]])
    #                             with open(output_dir + 'dbscan_param_search.txt', 'a') as f:
    #                                 f.write('{} {} {} {} {} {}\n'.format(eps[i],
    #                                                                    min_samples[j],
    #                                                                    metric[k],
    #                                                                    algorithm[l],
    #                                                                    leaf_size[m],
    #                                                                    p[n]))
    #                                 f.write(str(np.unique(db.labels_, return_counts=True)))
    #                                 f.write('\n\n')
    #     # >> get best parameter set (want to maximize)
    #     # for i in range(2, max(num_classes)+1):
    #     for i in np.unique(num_classes):
    #         # print('num classes: ' + str(max(num_classes)))
    #         print('num classes: ' + str(i))
    #         inds = np.nonzero(np.array(num_classes)==i)
    #         best = np.argmin(np.array(num_noisy)[inds])
    #         best = inds[0][best]
    #         print('best_parameter_set: ' + str(parameter_sets[best]))
    #         print(str(counts[best]))
    #         p=parameter_sets[best]
    #         classes = pf.features_plotting_2D(bottleneck, bottleneck,
    #                                           output_dir, 'dbscan',
    #                                           x, x_test, ticid_test,
    #                                           target_info=target_info_test,
    #                                           feature_engineering=False,
    #                                           eps=p[0], min_samples=p[1],
    #                                           metric=p[2], algorithm=p[3],
    #                                           leaf_size=p[4], p=p[5],
    #                                           folder_suffix='_'+str(i)+\
    #                                               'classes',
    #                                           momentum_dump_csv=mom_dump)
    
    # else:
    #     classes = pf.features_plotting_2D(bottleneck, bottleneck, output_dir,
    #                                       'dbscan', x, x_test, ticid_test,
    #                                       target_info=target_info_test,
    #                                       feature_engineering=False, eps=2.9,
    #                                       min_samples=2, metric='minkowski',
    #                                       algorithm='auto', leaf_size=30, p=4,
    #                                       momentum_dump_csv=mom_dump)        
    # pf.features_plotting_2D(bottleneck, bottleneck, output_dir, 'kmeans',
    #                         x, x_test, ticid_test,
    #                         feature_engineering=False,
    #                         momentum_dump_csv=mom_dump)
    
    # pf.plot_pca(bottleneck, classes, output_dir=output_dir)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # targets = []
    # for i in ticid_test:
    #     targets.append('TIC ' + str(int(i)))
    # ff.features_insets2D(x, x_test, bottleneck, targets, output_dir)    

# activations = ml.get_activations(model, x_test)
# bottleneck = ml.get_bottleneck(model, activations, p)

# orbit_gap_start = len(x)-1 # !!
# orbit_gap_end = orbit_gap_start + 1

# # >> take before orbit gap            
# flux = flux[:,:orbit_gap_start]
# x = x[:orbit_gap_start]
    
    
# if input_rms:
#     rms = np.loadtxt(data_dir + fname_rms[i])
#     rms_list.append(rms)
    
    
# fname_time = []
# fname_flux = []
# fname_ticid = []
# fname_rms = []
# for i in [4]:
#     for j in [1,2,3,4]:
#         fname_time.append('Sector20Cam'+str(i)+'CCD'+str(j)+'-time.txt')
#         fname_flux.append('Sector20Cam'+str(i)+'CCD'+str(j)+'-flux.csv')
#         fname_ticid.append('Sector20Cam'+str(i)+'CCD'+str(j)+'-ticid.txt') 
#         fname_rms.append('Sector20Cam'+str(i)+'CCD'+str(j)+'-rms.txt')

# # fname_time=['Sector20Cam1CCD1-time.txt', 'Sector20Cam1CCD2-time.txt',
# #             'Sector20Cam1CCD3-time.txt', 'Sector20Cam1CCD4-time.txt',
# #             'Sector20Cam2CCD1-time.txt', 'Sector20Cam2CCD2-time.txt',
# #             'Sector20Cam2CCD3-time.txt', 'Sector20Cam3CCD4-time.txt']
# # fname_flux=['Sector20Cam1CCD1-flux.csv', 'Sector20Cam1CCD1-flux.csv',
# #             'Sector20Cam1CCD1-flux.csv', 'Sector20Cam1CCD1-flux.csv',
# #             'Sector20Cam1CCD1-flux.csv','Sector20Cam1CCD1-flux.csv',
# #             'Sector20Cam1CCD1-flux.csv']
# # fname_ticid=['Sector20Cam1CCD1-ticid.txt']
# # fname_rms=['Sector20Cam1CCD1-rms.txt']
    
# x     = np.loadtxt(data_dir+fname_time[i])
# flux  = np.loadtxt(data_dir+fname_flux[i], delimiter=',')
# ticid = np.loadtxt(data_dir+fname_ticid[i])


    # !!
# np.savetxt(output_dir+'x_predict.txt',
#            np.reshape(x_predict, (np.shape(x_predict)[0],
#                                   np.shape(x_predict)[1])),
#            delimiter=',')
# model.save(output_dir+"model.h5")
# print("Saved model!")

# with open(output_dir+'historydict.p', 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)
        
    
# model = load_model(output_dir + 'model.h5')
# history = pickle.load( open(output_dir + 'historydict.p', 'rb'))
    
# x_predict = np.loadtxt(output_dir+'x_predict.txt', delimiter=',')
# activations, bottleneck = pf.diagnostic_plots(history, model, p,
#                                               output_dir, x, x_train,
#                                               x_test,
#                     x_predict, mock_data=False,
#                     ticid_train=ticid_train,
#                     ticid_test=ticid_test,
#                     plot_intermed_act=False,
#                     plot_latent_test=False,
#                     plot_latent_train=False,
#                     make_movie=False, percentage=False,
#                     input_features=input_features,
#                     input_rms=input_rms, rms_test=rms_test,
#                     rms_train=rms_train)        
# ml.param_summary(history, x_test, x_predict, p, output_dir, 0,
#                   'tess-unsupervised')
# ml.model_summary_txt(output_dir, model)    

# p = {'kernel_size': [3,5,7],
#   'latent_dim': list(np.arange(5, 30, 2)),
#   'strides': [1],
#   'epochs': [50],
#   'dropout': list(np.arange(0.1, 0.5, 0.02)),
#   'num_filters': [8, 16, 32, 64],
#   'num_conv_layers': [2,4,6,8,10],
#   'batch_size': [128],
#   'activation': ['elu', 'selu'],
#   'optimizer': ['adam', 'adadelta'],
#   'last_activation': ['linear'],
#   'losses': ['mean_squared_error'],
#   'lr': list(np.arange(0.001, 0.1, 0.01)),
#   'initializer': ['random_normal', 'random_uniform', 'glorot_normal',
#                   'glorot_uniform', 'zeros']}
    
    
    # p = {'kernel_size': 7,
    #       'latent_dim': 21,
    #       'strides': 1,
    #       'epochs': 20,
    #       'dropout': 0.5,
    #       'num_filters': 64,
    #       'num_conv_layers': 4,
    #       'batch_size': 128,
    #       'activation': 'elu',
    #       'optimizer': 'adam',
    #       'last_activation': 'linear',
    #       'losses': 'mean_squared_error',
    #       'lr': 0.001,
    #       'initializer': 'random_uniform'}

# run_tic = False # >> input a TICID (temporary)
# tics = [219107776.0, 185336364.0] # >> for run_tic
        
        # >> load everything
        # with fits.open(output_dir + 'x.fits') as hdul:
        #     x = hdul[0].data
            
        # with fits.open(output_dir + 'x_train.fits') as hdul:
        #     x_train = hdul[0].data          
            
        # with fits.open(output_dir + 'x_test.fits') as hdul:
        #     x_test = hdul[0].data        
    
# if run_tic:
#     # new_length = np.min(lengths)
#     new_length = 18757 # !! length of tabby star
#     x = []
#     flux = []
#     ticid = []
    
#     # >> truncate all light curves
#     for i in range(len(fnames)):
#         x = time_list[i][:new_length]
#         flux.append(flux_list[i][:,:new_length])
#         ticid.extend(ticid_list[i])
        
#     # >> load and truncate tabby star, ex dra
#     for i in range(len(tics)):
#         x_tmp, flux_tmp = ml.get_lc(str(int(tics[i])), out=output_dir,
#                                     DEBUG_INTERP=True,
#                                     download_fits=False)
#         flux_tmp = ml.interpolate_lc(flux_tmp, x_tmp,
#                                      prefix=str(int(tics[i])),
#                                      DEBUG_INTERP=True)
#         flux.append([flux_tmp[:new_length]])
#         ticid.extend([tics[i]])
        
#     # >> concatenate all light curves
#     flux = np.concatenate(flux, axis=0)
# else:    
    # # >> save time, x_train, x_test
    # hdr = fits.Header()
    # hdu = fits.PrimaryHDU(x, header=hdr)
    # hdu.writeto(output_dir + 'x.fits')
    # hdr = fits.Header()
    # hdu = fits.PrimaryHDU(x_test, header=hdr)
    # hdu.writeto(output_dir + 'x_test.fits')
    # hdr = fits.Header()
    # hdu = fits.PrimaryHDU(x_train, header=hdr)
    # hdu.writeto(output_dir + 'x_train.fits')    
        # pf.diagnostic_plots(history, model, p, output_dir, x, x_train,
        #                     x_test, x_predict, mock_data=False,
        #                     ticid_train=ticid_train,
        #                     ticid_test=ticid_test, percentage=False,
        #                     input_features=input_features,
        #                     input_rms=input_rms, rms_test=rms_test,
        #                     rms_train=rms_train,
        #                     plot_intermed_act=False,
        #                     plot_latent_test=False,
        #                     plot_latent_train=False,
        #                     plot_reconstruction_error_all=False,
        #                     make_movie=False,
        #                     plot_epoch=False,
        #                     plot_kernel=False)      
            # 2.4000000000000004 2 minkowski auto 30 4
            # (array([-1,  0,  1,  2]), array([24, 84,  6,  2]))
            
            # 2.9000000000000004 2 minkowski auto 30 4
            # (array([-1,  0,  1,  2]), array([23, 85,  6,  2]))    
        # x_predict = np.reshape(x_predict, (np.shape(x_predict)[0],
        #                                    np.shape(x_predict)[1], 1))
# ticid_train = ticid[:np.shape(x_train)[0]]
# ticid_test = ticid[-1 * np.shape(x_test)[0]:]
# target_info_train = target_info[:np.shape(x_train)[0]]
# target_info_test = target_info[-1 * np.shape(x_test)[0]:]    
    
# lengths = []
# time_list = []
# flux_list = []
# ticid = np.empty((0, 1))
# target_info = np.empty((0, 3))
# for i in range(len(fnames)):
#     print('Loading ' + fnames[i] + '...')
#     with fits.open(data_dir + fnames[i]) as hdul:
#         x = hdul[0].data
#         flux = hdul[1].data
#         ticid_list = hdul[2].data

#     lengths.append(len(x))
#     time_list.append(x)
#     flux_list.append(flux)
#     ticid = np.append(ticid, ticid_list)
#     target_info = np.append(target_info,
#                             np.repeat([fname_info[i]], len(flux), axis=0),
#                             axis=0)

# # !! truncate if using multiple sectors
# new_length = np.min([np.shape(i)[1] for i in flux_list])
# flux = []
# for i in range(len(fnames)):
#     flux.append(flux_list[i][:,:new_length])

# flux = np.concatenate(flux, axis=0)
# x = time_list[0][:new_length]

# # >> shuffle flux array
# inds = np.arange(len(flux))
# np.random.shuffle(inds)
# flux = flux[inds]
# ticid = ticid[inds]
# target_info = target_info[inds].astype('int')

# # >> moves target object to the testing set (and will be plotted in the
# # >> input-output-residual plot)
# if len(targets) > 0:
#     for t in targets:
#         target_ind = np.nonzero( ticid == t )[0][0]
#         flux = np.insert(flux, -1, flux[target_ind], axis=0)
#         flux = np.delete(flux, target_ind, axis=0)
#         ticid = np.insert(ticid, -1, ticid[target_ind])
#         ticid = np.delete(ticid, target_ind)
#         target_info = np.insert(target_info, -1, target_info[target_ind],
#                                 axis=0)
#         target_info = np.delete(target_info, target_ind, axis=0)        
    
# -- nan mask -----------------------------------------------------------------
# >> apply nan mask
# print('Applying NaN mask...')
# flux, x = df.nan_mask(flux, x, output_dir=output_dir, ticid=ticid, 
#                       DEBUG=True, debug_ind=10, target_info=target_info)

# -- partition data -----------------------------------------------------------
# >> calculate rms and standardize
# if input_rms:
#     print('Calculating RMS..')
#     rms = df.rms(flux)
    
#     if norm_type == 'standardization':
#         print('Standardizing fluxes...')
#         flux = df.standardize(flux)

#     elif norm_type == 'median_normalization':
#         print('Normalizing fluxes (dividing by median)...')
#         flux = df.normalize(flux)
        
#     elif norm_type == 'minmax_normalization':
#         print('Normalizing fluxes (changing minimum and range)...')
#         mins = np.min(flux, axis = 1, keepdims=True)
#         flux = flux - mins
#         maxs = np.max(flux, axis=1, keepdims=True)
#         flux = flux / maxs
        
#     else:
#         print('Light curves are not normalized!')

# print('Partitioning data...')
# x_train, x_test, y_train, y_test, ticid_train, ticid_test, target_info_train, \
#     target_info_test, x = \
#     ml.split_data(flux, ticid, target_info, x, p,
#                   train_test_ratio=train_test_ratio,
#                   supervised=False)

# if input_rms:
#     rms_train = rms[:np.shape(x_train)[0]]
#     rms_test = rms[-1 * np.shape(x_test)[0]:]
# else:
#     rms_train, rms_test = False, False   
        
        
        # # >> file names
# fnames = []
# fname_info = []
# for sector in sectors:
#     for cam in cams:
#         for ccd in ccds:
#             s = 'Sector{sector}/Sector{sector}Cam{cam}CCD{ccd}/' + \
#                 'Sector{sector}Cam{cam}CCD{ccd}_lightcurves.fits'
#             fnames.append(s.format(sector=sector, cam=cam, ccd=ccd))
#             fname_info.append([sector, cam, ccd])
        
          # 'kernel_regularizer': [tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
          #                        tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001),
          #                        tf.keras.regularizers.l1_l2(l1=0.1, l2=0.1),
          #                        tf.keras.regularizers.l1_l2(l1=0., l2=0.01),
          #                        tf.keras.regularizers.l1_l2(l1=0., l2=0.001),
          #                        tf.keras.regularizers.l1_l2(l1=0., l2=0.1),
          #                        tf.keras.regularizers.l1_l2(l1=0.01, l2=0.),
          #                        tf.keras.regularizers.l1_l2(l1=0.001, l2=0.),
          #                        tf.keras.regularizers.l1_l2(l1=0.1, l2=0.),
          #                        None],
          
          # 'bias_regularizer': [tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
          #                        tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001),
          #                        tf.keras.regularizers.l1_l2(l1=0.1, l2=0.1),
          #                        tf.keras.regularizers.l1_l2(l1=0., l2=0.01),
          #                        tf.keras.regularizers.l1_l2(l1=0., l2=0.001),
          #                        tf.keras.regularizers.l1_l2(l1=0., l2=0.1),
          #                        tf.keras.regularizers.l1_l2(l1=0.01, l2=0.),
          #                        tf.keras.regularizers.l1_l2(l1=0.001, l2=0.),
          #                        None],
          
          # 'activity_regularizer': [tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),
          #                        tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001),
          #                        tf.keras.regularizers.l1_l2(l1=0.1, l2=0.1),
          #                        tf.keras.regularizers.l1_l2(l1=0., l2=0.01),
          #                        tf.keras.regularizers.l1_l2(l1=0., l2=0.001),
          #                        tf.keras.regularizers.l1_l2(l1=0., l2=0.1),
          #                        tf.keras.regularizers.l1_l2(l1=0.01, l2=0.),
          #                        tf.keras.regularizers.l1_l2(l1=0.001, l2=0.),
          #                        tf.keras.regularizers.l1_l2(l1=0.1, l2=0.),
          #                        None],       
            
    # if split_at_orbit_gap or input_psd or not run_model:

    #     history = []
            
    #     with fits.open(output_dir + 'x_predict.fits') as hdul:
    #         x_predict = hdul[0].data
            
    #     import keras
    #     import tensorflow as tf
    #     model = keras.models.load_model(output_dir+'model', custom_objects={'tf': tf}) 
        
    #     # !! re-arrange x_predict
        
    #     pf.diagnostic_plots(history, model, p, output_dir, x, x_train,
    #                         x_test, x_predict, mock_data=False,
    #                         target_info_test=target_info_test,
    #                         target_info_train=target_info_train,
    #                         ticid_train=ticid_train, ticid_test=ticid_test,
    #                         rms_test=rms_test, rms_train=rms_train,
    #                         input_features=input_features, n_tot=40,
    #                         input_rms=input_rms, input_psd=input_psd,
    #                         percentage=False,
    #                         plot_epoch = False,
    #                         plot_in_out = True,
    #                         plot_in_bottle_out=False,
    #                         plot_latent_test = True,
    #                         plot_latent_train = True,
    #                         plot_kernel=False,
    #                         plot_intermed_act=False,
    #                         make_movie = False,
    #                         plot_lof_test=True,
    #                         plot_lof_train=True,
    #                         plot_lof_all=True,
    #                         plot_reconstruction_error_test=False,
    #                         plot_reconstruction_error_all=True,
    #                         load_bottleneck=True)
    # else: 
    #     if p['concat_ext_feats'] and not input_psd:
    #         pf.diagnostic_plots(history, model, p, output_dir, x, x_train[0],
    #                             x_test[0], x_predict[0], mock_data=False,
    #                             addend=0.,
    #                             target_info_test=target_info_test,
    #                             target_info_train=target_info_train,
    #                             ticid_train=ticid_train,
    #                             ticid_test=ticid_test, percentage=False,
    #                             input_features=input_features,
    #                             input_rms=input_rms, rms_test=rms_test,
    #                             input_psd=input_psd,
    #                             rms_train=rms_train, n_tot=40,
    #                             plot_epoch = False,
    #                             plot_in_out = True,
    #                             plot_in_bottle_out=False,
    #                             plot_latent_test = True,
    #                             plot_latent_train = True,
    #                             plot_kernel=False,
    #                             plot_intermed_act=False,
    #                             make_movie = False,
    #                             plot_lof_test=False,
    #                             plot_lof_train=False,
    #                             plot_lof_all=False,
    #                             plot_reconstruction_error_test=True,
    #                             plot_reconstruction_error_all=False,
    #                             load_bottleneck=True)         
    #     elif input_psd:
    #         pf.diagnostic_plots(history, model, p, output_dir, x[0], x_train[0],
    #                             x_test[0], x_predict[0], mock_data=False,
    #                             addend=0.,
    #                             target_info_test=target_info_test,
    #                             target_info_train=target_info_train,
    #                             ticid_train=ticid_train,
    #                             ticid_test=ticid_test, percentage=False,
    #                             input_features=input_features,
    #                             input_rms=input_rms, rms_test=rms_test,
    #                             input_psd=False,
    #                             rms_train=rms_train, n_tot=40,
    #                             plot_epoch = False,
    #                             plot_in_out = True,
    #                             plot_in_bottle_out=False,
    #                             plot_latent_test = True,
    #                             plot_latent_train = True,
    #                             plot_kernel=False,
    #                             plot_intermed_act=False,
    #                             make_movie = False,
    #                             plot_lof_test=False,
    #                             plot_lof_train=False,
    #                             plot_lof_all=False,
    #                             plot_reconstruction_error_test=True,
    #                             plot_reconstruction_error_all=False,
    #                             load_bottleneck=True)                
    #         fig, axes = pf.input_output_plot(x[1], x_test[1], x_predict[1],
    #                                       output_dir+'input_output_PSD.png',
    #                                       ticid_test=ticid_test,
    #                                       target_info=target_info_test)             
    #     else:
    #         pf.diagnostic_plots(history, model, p, output_dir, x, x_train,
    #                             x_test, x_predict, mock_data=False, addend=0.,
    #                             target_info_test=target_info_test,
    #                             target_info_train=target_info_train,
    #                             ticid_train=ticid_train,
    #                             ticid_test=ticid_test, percentage=False,
    #                             input_features=input_features,
    #                             input_rms=input_rms, rms_test=rms_test,
    #                             input_psd=input_psd,
    #                             rms_train=rms_train, n_tot=40,
    #                             plot_epoch = False,
    #                             plot_in_out = True,
    #                             plot_in_bottle_out=False,
    #                             plot_latent_test = True,
    #                             plot_latent_train = True,
    #                             plot_kernel=False,
    #                             plot_intermed_act=True,
    #                             make_movie = False,
    #                             plot_lof_test=False,
    #                             plot_lof_train=False,
    #                             plot_lof_all=False,
    #                             plot_reconstruction_error_test=False,
    #                             plot_reconstruction_error_all=True,
    #                             load_bottleneck=True)                            

# for i in range(4):               
            
        # parameter_sets, num_classes, silhouette_scores, db_scores, ch_scores, acc = \
        # df.dbscan_param_search(features, x, flux_feat, ticid_feat,
        #                         info_feat, DEBUG=True, 
        #                         output_dir=output_dir+str(i), single_file=single_file,
        #                         simbad_database_txt=simbad_database_dir,
        #                         leaf_size=[best_param_set[4]],
        #                         algorithm=[best_param_set[3]],
        #                         min_samples=[best_param_set[1]],
        #                         metric=[best_param_set[2]], p=[best_param_set[5]],
        #                         database_dir=database_dir,
        #                         eps=[best_param_set[0]])  


# df.representation_learning(flux, x, ticid, target_info, output_dir=output_dir,
#                            p=p)

# !! tmp
# # >> train and test on high frequency only
# import random
# ticid_all = []
# err = []
# with open(output_dir+'reconstruction_error_all.txt', 'r') as f:
#     lines = f.readlines()
#     for i in range(len(lines)):
#         line = lines[i].split()
#         ticid_all.append(float(line[0]))
#         err.append(float(line[1]))      
# ticid_all=np.array(ticid_all)
# err=np.array(err)
# inds = np.nonzero(err > np.sort(err)[19800])
# random.Random(4).shuffle(inds)
# ticid_all = ticid_all[inds]
# new_flux=[]
# new_ticid=[]
# new_target_info=[]
# for i in range(len(ticid_all)):
#     new_ind = np.nonzero(ticid == ticid_all[i])
#     new_flux.append(flux[new_ind])
#     new_ticid.append(ticid[new_ind])
#     new_target_info.append(target_info[new_ind])
# flux = np.array(new_flux).reshape(len(new_flux), np.shape(new_flux)[2])
# ticid = np.array(new_ticid).reshape(len(new_ticid))
# target_info = np.array(new_target_info).reshape(len(new_target_info), 5)
# # !! tmp              