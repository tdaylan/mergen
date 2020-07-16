# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
# 2020-07-14 - hyperparam_opt-tess-unsupervised.py
# Runs a convolutional autoencoder on TESS data. Run with:
# 1. First download data folders from Dropbox (named Sector*Cam*CCD*/) for all
#    groups you want to run on. Move data folders to dat_dir
# 2. Download Table_of_momentum_dumps.csv, and change path in mom_dump
# 3. Run this script in the command line with 
#    $ python hyperparam_opt-tess-unsupervised.py
#
# TODO:
# * make load data function (nan mask, etc.), also integrate into mlp.py
# 
# Emma Chickles
# 
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

dat_dir = '../../' # >> directory with input data (ending with /)
output_dir = '../../plots/CAE/' # >> directory to save diagnostic plots
                                     # >> will make dir if doesn't exist
mom_dump = '../../Table_of_momentum_dumps.csv'
lib_dir = '../main/' # >> directory containing model.py, data_functions.py
                     # >> and plotting_functions.py
# >> input data
sectors = [20]
cams = [1, 2, 3, 4]
ccds =  [1, 2, 3, 4]
# train_test_ratio = 0.1 # >> fraction of training set size to testing set size
train_test_ratio = 0.85

# >> what this script will run:
hyperparameter_optimization = True # >> run hyperparameter search
run_model = True # >> train autoencoder on a parameter set p
diag_plots = True # >> creates diagnostic plots. If run_model==False, then will
                  # >> load bottleneck*.fits for plotting
classification=True # >> runs DBSCAN on learned features
DBSCAN_parameter_search=True # >> runs grid search for DBSCAN

# >> normalization options:
#    * standardization : sets mean to 0. and standard deviation to 1.
#    * median_normalization : divides by median
#    * minmax_normalization : sets range of values from 0. to 1.
#    * none : no normalization
norm_type = 'median_normalization'

input_rms=True # >> concatenate RMS to learned features
use_tess_features = True
input_features=False # >> this option cannot be used yet

# >> move targets out of training set and into testing set (integer)
targets = [219107776] # >> EX DRA # !!
# targets = []

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

import talos                    # >> hyperparameter optimization library
import numpy as np
import pdb
import os
from astropy.io import fits

import sys
sys.path.insert(0, lib_dir)  # >> needed if scripts not in current dir
import model as ml              # >> for autoencoder
import data_functions as df     # >> for classification, pre-processing
import plotting_functions as pl # >> for vsualizations

# >> file names
fnames = []
fname_info = []
for sector in sectors:
    for cam in cams:
        for ccd in ccds:
            s = 'Sector{sector}Cam{cam}CCD{ccd}/' + \
                'Sector{sector}Cam{cam}CCD{ccd}_lightcurves.fits'
            fnames.append(s.format(sector=sector, cam=cam, ccd=ccd))
            fname_info.append([sector, cam, ccd])

# >> hyperparameters
if hyperparameter_optimization: # !! change epochs
    p = {'kernel_size': [3,5,7],
      'latent_dim': list(np.arange(5, 30, 5)),
      'strides': [1],
      'epochs': [8],
      'dropout': list(np.arange(0.1, 0.5, 0.1)),
      'num_filters': [8, 16, 32, 64],
      'num_conv_layers': [2,4,6,8,10],
      'batch_size': [128],
      'activation': ['elu'],
      'optimizer': ['adam', 'adadelta'],
      'last_activation': ['linear'],
      'losses': ['mean_squared_error'],
      'lr': [0.001, 0.005, 0.01, 0.05, 0.1],
      'initializer': ['random_normal', 'random_uniform', 'glorot_normal',
                      'glorot_uniform']}
else:
    # p = {'kernel_size': 3,
    #       'latent_dim': 25,
    #       'strides': 1,
    #       'epochs': 15,
    #       'dropout': 0.3,
    #       'num_filters': 32,
    #       'num_conv_layers': 6,
    #       'batch_size': 128,
    #       'activation': 'elu',
    #       'optimizer': 'adadelta',
    #       'last_activation': 'linear',
    #       'losses': 'mean_squared_error',
    #       'lr': 0.001,
    #       'initializer': 'glorot_normal'}    
    # p = {'kernel_size': 3,
    #       'latent_dim': 10,
    #       'strides': 1,
    #       'epochs': 25,
    #       'dropout': 0.5,
    #       'num_filters': 32,
    #       'num_conv_layers': 2,
    #       'batch_size': 128,
    #       'activation': 'elu',
    #       'optimizer': 'adam',
    #       'last_activation': 'linear',
    #       'losses': 'mean_squared_error',
    #       'lr': 0.001,
    #       'initializer': 'random_normal'}    
    p = {'kernel_size': 3,
          'latent_dim': 17,
          'strides': 1,
          'epochs': 15,
          'dropout': 0.3,
          'num_filters': 32,
          'num_conv_layers': 2,
          'batch_size': 128,
          'activation': 'elu',
          'optimizer': 'adadelta',
          'last_activation': 'linear',
          'losses': 'mean_squared_error',
          'lr': 0.001,
          'initializer': 'glorot_normal'}        

# -- create output directory --------------------------------------------------
    
if os.path.isdir(output_dir) == False: # >> check if dir already exists
    os.mkdir(output_dir)
    
# -- load data ----------------------------------------------------------------
    
flux, x, ticid, target_info = \
    df.load_data_from_metafiles(dat_dir, sector, DEBUG=True,
                                output_dir=output_dir, nan_mask_check=True)
    
x_train, x_test, y_train, y_test, ticid_train, ticid_test, target_info_train, \
    target_info_test, rms_train, rms_test, x = \
    ml.autoencoder_preprocessing(flux, ticid, x, target_info, p,
                                 targets=targets, norm_type=norm_type,
                                 input_rms=input_rms,
                                 train_test_ratio=train_test_ratio)

title='TESS-unsupervised'

# == talos experiment =========================================================
if hyperparameter_optimization:
    print('Starting hyperparameter optimization...')
    # t = talos.Scan(x=np.concatenate([x_train, x_test]),
    #                 y=np.concatenate([x_train, x_test]),
    #                 params=p,
    #                 model=ml.conv_autoencoder,
    #                 experiment_name=title, 
    #                 reduction_metric = 'val_loss',
    #                 minimize_loss=True,
    #                 reduction_method='correlation',
    #                 fraction_limit=0.0001) 
    t = talos.Scan(x=x_test,
                   y=x_test,
                    params=p,
                    model=ml.conv_autoencoder,
                    experiment_name=title, 
                    reduction_metric = 'val_loss',
                    minimize_loss=True,
                    reduction_method='correlation',
                    fraction_limit=0.0001)     
    # fraction_limit = 0.001
    analyze_object = talos.Analyze(t)
    df, best_param_ind,p = pl.hyperparam_opt_diagnosis(analyze_object,
                                                       output_dir,
                                                       supervised=False)

# == run model ================================================================
if run_model:
    print('Training autoencoder...') 
    history, model = ml.conv_autoencoder(x_train, x_train, x_test, x_test, p)
    x_predict = model.predict(x_test)
    
    ml.param_summary(history, x_test, x_predict, p, output_dir, 0, title)
    ml.model_summary_txt(output_dir, model)
    
    # >> only plot epoch
    pl.epoch_plots(history, p, output_dir+'epoch-', supervised=False)
    
    # >> save x_predict as .fits file
    # tmp = np.reshape(x_predict, (np.shape(x_predict)[0],
    #                              np.shape(x_predict)[1]))
    hdr = fits.Header()
    hdu = fits.PrimaryHDU(x_predict, header=hdr)
    hdu.writeto(output_dir + 'x_predict.fits')
    
    # >> save bottleneck_test, bottleneck_train
    bottleneck = ml.get_bottleneck(model, x_test, input_rms=input_rms,
                                   rms=rms_test)    
    hdr = fits.Header()
    hdu = fits.PrimaryHDU(bottleneck, header=hdr)
    hdu.writeto(output_dir + 'bottleneck_test.fits')       
    
    bottleneck_train = ml.get_bottleneck(model, x_train, input_rms=input_rms,
                                         rms=rms_train)
    hdr = fits.Header()
    hdu = fits.PrimaryHDU(bottleneck_train, header=hdr)
    hdu.writeto(output_dir + 'bottleneck_train.fits')

# == Plots ====================================================================
if diag_plots:
    print('Creating plots...')
    if run_model == False:

        model, history = [], []    
            
        with fits.open(output_dir + 'x_predict.fits') as hdul:
            x_predict = hdul[0].data
        
        pl.diagnostic_plots(history, model, p, output_dir, x, x_train,
                            x_test, x_predict, mock_data=False,
                            target_info_test=target_info_test,
                            target_info_train=target_info_train,
                            ticid_train=ticid_train, ticid_test=ticid_test,
                            rms_test=rms_test, rms_train=rms_train,
                            input_features=input_features, n_tot=40,
                            input_rms=input_rms, percentage=False,
                            plot_epoch = False,
                            plot_in_out = True,
                            plot_in_bottle_out=False,
                            plot_latent_test = True,
                            plot_latent_train = True,
                            plot_kernel=False,
                            plot_intermed_act=False,
                            plot_clustering=False,
                            make_movie = False,
                            plot_lof_test=True,
                            plot_lof_train=True,
                            plot_lof_all=True,
                            plot_reconstruction_error_test=False,
                            plot_reconstruction_error_all=False,
                            load_bottleneck=True)
    else: 
        pl.diagnostic_plots(history, model, p, output_dir, x, x_train,
                            x_test, x_predict, mock_data=False,
                            target_info_test=target_info_test,
                            target_info_train=target_info_train,
                            ticid_train=ticid_train,
                            ticid_test=ticid_test, percentage=False,
                            input_features=input_features,
                            input_rms=input_rms, rms_test=rms_test,
                            rms_train=rms_train, n_tot=40,
                            plot_epoch = False,
                            plot_in_out = True,
                            plot_in_bottle_out=False,
                            plot_latent_test = True,
                            plot_latent_train = True,
                            plot_kernel=True,
                            plot_intermed_act=False,
                            plot_clustering=False,
                            make_movie = False,
                            plot_lof_test=True,
                            plot_lof_train=True,
                            plot_lof_all=True,
                            plot_reconstruction_error_test=False,
                            plot_reconstruction_error_all=False)                            

# >> Feature plots

if classification:
    features, flux_feat, ticid_feat, info_feat = \
        ml.bottleneck_preprocessing(sectors[0],
                                    np.concatenate([x_train, x_test], axis=0),
                                    np.concatenate([ticid_train, ticid_test]),
                                    np.concatenate([target_info_train,
                                                    target_info_test]),
                                    data_dir=dat_dir,
                                    output_dir=output_dir,
                                    use_learned_features=True,
                                    use_tess_features=use_tess_features,
                                    use_engineered_features=False)
        
    pl.latent_space_plot(features, {'latent_dim': np.shape(features)[1]},
                         output_dir + 'latent_space-tessfeats.png')

    parameter_sets, num_classes, silhouette_scores, db_scores, ch_scores = \
    df.dbscan_param_search(features, x, flux_feat, ticid_feat,
                           info_feat, DEBUG=True, 
                           output_dir=output_dir, 
                           simbad_database_txt='../../simbad_database.txt')      
    
    # parameter_sets, num_classes, silhouette_scores, db_scores, ch_scores = \
    # df.dbscan_param_search(bottleneck, x, x_test, ticid_test,
    #                        target_info_test, DEBUG=True, 
    #                        output_dir=output_dir, 
    #                        simbad_database_txt='../../simbad_database.txt',
    #                        leaf_size=[30], algorithm=['auto']) 
    
        
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
    #         classes = pl.features_plotting_2D(bottleneck, bottleneck,
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
    #     classes = pl.features_plotting_2D(bottleneck, bottleneck, output_dir,
    #                                       'dbscan', x, x_test, ticid_test,
    #                                       target_info=target_info_test,
    #                                       feature_engineering=False, eps=2.9,
    #                                       min_samples=2, metric='minkowski',
    #                                       algorithm='auto', leaf_size=30, p=4,
    #                                       momentum_dump_csv=mom_dump)        
    # pl.features_plotting_2D(bottleneck, bottleneck, output_dir, 'kmeans',
    #                         x, x_test, ticid_test,
    #                         feature_engineering=False,
    #                         momentum_dump_csv=mom_dump)
    
    # pl.plot_pca(bottleneck, classes, output_dir=output_dir)

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
#     rms = np.loadtxt(dat_dir + fname_rms[i])
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
    
# x     = np.loadtxt(dat_dir+fname_time[i])
# flux  = np.loadtxt(dat_dir+fname_flux[i], delimiter=',')
# ticid = np.loadtxt(dat_dir+fname_ticid[i])


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
# activations, bottleneck = pl.diagnostic_plots(history, model, p,
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
        # pl.diagnostic_plots(history, model, p, output_dir, x, x_train,
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
#     with fits.open(dat_dir + fnames[i]) as hdul:
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