# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
# 2020-05-26 - hyperparam_opt-tess-unsupervised.py
# Runs a convolutional autoencoder on TESS data.
# / Emma Chickles
# 
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

import modellibrary as ml       # >> autoencoder and pre-processing
import feature_functions as ff  # >> DBSCAN classification
import talos                    # >> hyperparameter optimization library

import numpy as np
import pdb
import os
import sys
# import pickle
# from keras.models import load_model
from astropy.io import fits
sys.path.insert(0, '../main/')
import model as ml
import data_functions as df
import plotting_functions as pl

output_dir = '../../plots/cae-cam4ccd1/'
# output_dir = './plots/test/'

hyperparameter_optimization = False
run_model = True
diag_plots = True
classification=True # >> runs dbscan, classifies light curves
run_tic = False # >> input a TICID (temporary)
DBSCAN_parameter_search=True
input_features=False
input_rms=True

tics = [219107776.0, 185336364.0] # >> for run_tic
targets = [219107776]

# >> file names
# dat_dir = '/Users/studentadmin/Dropbox/TESS_UROP/Sector_20_LC/'
dat_dir = './'
fnames = ['../../Sector20Cam4CCD1_raw_lightcurves.fits']

# >> hyperparameters
if hyperparameter_optimization:
    p = {'kernel_size': [3,5,7],
      'latent_dim': list(np.arange(5, 30, 2)),
      'strides': [1],
      'epochs': [50],
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
    p = {'kernel_size': 7,
          'latent_dim': 21,
          'strides': 1,
          'epochs': 20,
          'dropout': 0.5,
          'num_filters': 64,
          'num_conv_layers': 4,
          'batch_size': 128,
          'activation': 'elu',
          'optimizer': 'adam',
          'last_activation': 'linear',
          'losses': 'mean_squared_error',
          'lr': 0.001,
          'initializer': 'random_uniform'}

# -- create output directory --------------------------------------------------
    
if os.path.isdir(output_dir) == False:
    os.mkdir(output_dir)
    
# -- load data ----------------------------------------------------------------
    
lengths = []
time_list = []
flux_list = []
ticid_list = []
rms_list = []
for i in range(len(fnames)):
    print('Loading ' + fnames[i] + '...')
    with fits.open(dat_dir + fnames[i]) as hdul:
        x = hdul[0].data
        flux = hdul[1].data
        ticid = hdul[2].data

    lengths.append(len(x))
    time_list.append(x)
    flux_list.append(flux)
    ticid_list.append(ticid)
    

# !! tmp
# new_length = np.shape(flux)[1]
if run_tic:
    # new_length = np.min(lengths)
    new_length = 18757 # !! length of tabby star
    x = []
    flux = []
    ticid = []
    
    # >> truncate all light curves
    for i in range(len(fnames)):
        x = time_list[i][:new_length]
        flux.append(flux_list[i][:,:new_length])
        ticid.extend(ticid_list[i])
        
    # >> load and truncate tabby star, ex dra
    for i in range(len(tics)):
        x_tmp, flux_tmp = ml.get_lc(str(int(tics[i])), out=output_dir,
                                    DEBUG_INTERP=True,
                                    download_fits=False)
        flux_tmp = ml.interpolate_lc(flux_tmp, x_tmp,
                                     prefix=str(int(tics[i])),
                                     DEBUG_INTERP=True)
        flux.append([flux_tmp[:new_length]])
        ticid.extend([tics[i]])
        
    # >> concatenate all light curves
    flux = np.concatenate(flux, axis=0)
else:
    flux = np.concatenate(flux_list, axis=0)
    x = time_list[0]
    
    ticid = []
    for i in range(len(fnames)):
        ticid.extend(ticid_list[i])
    ticid = np.array(ticid)
        
    # !! tmp
    # >> moves target object to the testing set (and will be plotted first
    # >> in the input-output-residual plot)
    if len(targets) > 0:
        for t in targets:
            target_ind = np.nonzero( ticid == t )[0][0]
            flux = np.insert(flux, -1, flux[target_ind], axis=0)
            flux = np.delete(flux, target_ind, axis=0)
            ticid = np.insert(ticid, -1, ticid[target_ind])
            ticid = np.delete(ticid, target_ind)
    
# -- nan mask -----------------------------------------------------------------
# >> apply nan mask
print('Applying NaN mask...')
flux, x = df.nan_mask(flux, x, output_dir=output_dir, ticid=ticid, 
                      DEBUG=True, debug_ind=10)

# -- partition data -----------------------------------------------------------
# >> calculate rms and standardize
if input_rms:
    print('Calculating RMS..')
    rms = df.rms(flux)
    
    # print('Standardizing fluxes...')
    # flux = df.standardize(flux)
    
    print('Normalizing fluxes (dividing by median)...')
    flux = df.normalize(flux)
    
    # print('Normalizing fluxes (changing minimum and range)...')
    # mins = np.min(flux, axis = 1, keepdims=True)
    # flux = flux - mins
    # maxs = np.max(flux, axis=1, keepdims=True)
    # flux = flux / maxs

print('Partitioning data...')
x_train, x_test, y_train, y_test, x = \
    ml.split_data(flux, x, p, train_test_ratio=0.90, supervised=False)

ticid_train = ticid[:np.shape(x_train)[0]]
ticid_test = ticid[-1 * np.shape(x_test)[0]:]

if input_rms:
    rms_train = rms[:np.shape(x_train)[0]]
    rms_test = rms[-1 * np.shape(x_test)[0]:]
else:
    rms_train, rms_test = False, False

title='TESS-unsupervised'

# == talos experiment =========================================================
if hyperparameter_optimization:
    print('Starting hyperparameter optimization...')
    t = talos.Scan(x=np.concatenate([x_train, x_test]),
                    y=np.concatenate([x_train, x_test]),
                    params=p,
                    model=ml.conv_autoencoder,
                    experiment_name=title, 
                    reduction_metric = 'val_loss',
                    minimize_loss=True,
                    reduction_method='correlation') # fraction_limit=0.0001
    analyze_object = talos.Analyze(t)
    df, best_param_ind,p = ml.hyperparam_opt_diagnosis(analyze_object,
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
        
        # >> load everything
        # with fits.open(output_dir + 'x.fits') as hdul:
        #     x = hdul[0].data
            
        # with fits.open(output_dir + 'x_train.fits') as hdul:
        #     x_train = hdul[0].data          
            
        # with fits.open(output_dir + 'x_test.fits') as hdul:
        #     x_test = hdul[0].data        
            
        with fits.open(output_dir + 'x_predict.fits') as hdul:
            x_predict = hdul[0].data
    
        # x_predict = np.reshape(x_predict, (np.shape(x_predict)[0],
        #                                    np.shape(x_predict)[1], 1))
        
        pl.diagnostic_plots(history, model, p, output_dir, x, x_train,
                            x_test, x_predict, mock_data=False,
                            ticid_train=ticid_train, ticid_test=ticid_test,
                            rms_test=rms_test, rms_train=rms_train,
                            input_features=input_features,
                            input_rms=input_rms, percentage=False,
                            plot_intermed_act=False,
                            plot_latent_test=False,
                            plot_latent_train=False,
                            make_movie=False,
                            plot_epoch=False,
                            load_bottleneck=True)
    else:
        pl.diagnostic_plots(history, model, p, output_dir, x, x_train,
                            x_test, x_predict, mock_data=False,
                            ticid_train=ticid_train,
                            ticid_test=ticid_test, percentage=False,
                            input_features=input_features,
                            input_rms=input_rms, rms_test=rms_test,
                            rms_train=rms_train,
                            plot_intermed_act=False,
                            plot_latent_test=False,
                            plot_latent_train=False,
                            plot_reconstruction_error_all=False,
                            make_movie=False,
                            plot_epoch=False,
                            plot_kernel=False)    


# >> Feature plots
if classification:
    
    if DBSCAN_parameter_search:
        from sklearn.cluster import DBSCAN
        with fits.open(output_dir + 'bottleneck_test.fits') as hdul:
            bottleneck = hdul[0].data
        # !!
        # bottleneck = ml.standardize(bottleneck, ax=0)
        eps = list(np.arange(0.1,5.0,0.1))
        min_samples = [2, 5,10,15]
        metric = ['euclidean', 'minkowski']
        algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
        leaf_size = [30, 40, 50]
        p = [1,2,3,4]
        classes = []
        num_classes = []
        counts = []
        num_noisy= []
        parameter_sets=[]
        for i in range(len(eps)):
            for j in range(len(min_samples)):
                for k in range(len(metric)):
                    for l in range(len(algorithm)):
                        for m in range(len(leaf_size)):
                            for n in range(len(p)):
                                db = DBSCAN(eps=eps[i],
                                            min_samples=min_samples[j],
                                            metric=metric[k],
                                            algorithm=algorithm[l],
                                            leaf_size=leaf_size[m],
                                            p=p[n]).fit(bottleneck)
                                print(db.labels_)
                                print(np.unique(db.labels_, return_counts=True))
                                classes_1, counts_1 = \
                                    np.unique(db.labels_, return_counts=True)
                                classes.append(classes_1)
                                num_classes.append(len(classes_1))
                                counts.append(counts_1)
                                num_noisy.append(counts[0])
                                parameter_sets.append([eps[i], min_samples[j],
                                                       metric[k],
                                                       algorithm[l],
                                                       leaf_size[m],
                                                       p[n]])
                                with open(output_dir + 'dbscan_param_search.txt', 'a') as f:
                                    f.write('{} {} {} {} {} {}\n'.format(eps[i],
                                                                       min_samples[j],
                                                                       metric[k],
                                                                       algorithm[l],
                                                                       leaf_size[m],
                                                                       p[n]))
                                    f.write(str(np.unique(db.labels_, return_counts=True)))
                                    f.write('\n\n')
        # >> get best parameter set (want to maximize)
        for i in range(2, max(num_classes)+1):
            # print('num classes: ' + str(max(num_classes)))
            print('num classes: ' + str(i))
            inds = np.nonzero(np.array(num_classes)==i)
            best = np.argmin(np.array(num_noisy)[inds])
            best = inds[0][best]
            print('best_parameter_set: ' + str(parameter_sets[best]))
            print(str(counts[best]))
            p=parameter_sets[best]
            # 2.4000000000000004 2 minkowski auto 30 4
            # (array([-1,  0,  1,  2]), array([24, 84,  6,  2]))
            
            # 2.9000000000000004 2 minkowski auto 30 4
            # (array([-1,  0,  1,  2]), array([23, 85,  6,  2]))
            classes = ff.features_plotting_2D(bottleneck, bottleneck,
                                              output_dir, 'dbscan',
                                              x, x_test, ticid_test,
                                              feature_engineering=False,
                                              eps=p[0], min_samples=p[1],
                                              metric=p[2], algorithm=p[3],
                                              leaf_size=p[4], p=p[5],
                                              folder_suffix='_'+str(i)+\
                                                  'classes')
    
    else:
        classes = ff.features_plotting_2D(bottleneck, bottleneck, output_dir,
                                          'dbscan', x, x_test, ticid_test,
                                          feature_engineering=False, eps=2.9,
                                          min_samples=2, metric='minkowski',
                                          algorithm='auto', leaf_size=30, p=4)
    ff.features_plotting_2D(bottleneck, bottleneck, output_dir, 'kmeans',
                            x, x_test, ticid_test,
                            feature_engineering=False)
    
    pl.plot_pca(bottleneck, classes, output_dir=output_dir)
    
    targets = []
    for i in ticid_test:
        targets.append('TIC ' + str(int(i)))
    ff.features_insets2D(x, x_test, bottleneck, targets, output_dir)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

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
