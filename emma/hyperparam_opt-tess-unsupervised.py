# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
# 2020-05-26 - hyperparam_opt-tess-unsupervised.py
# Runs a convolutional autoencoder on TESS data.
# / Emma Chickles
# 
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

import modellibrary as ml
import numpy as np
import talos
import pdb
import os
import pickle
import feature_functions as ff
import plots_lib as pl
from keras.models import load_model

output_dir = './plots/plots062420-0/'
# dat_dir = './plots/plots062420/'
dat_dir = '/Users/studentadmin/Dropbox/TESS_UROP/Sector_20_LC/'

hyperparameter_optimization = False
run_model = True
diag_plots = True
classification=False # >> runs dbscan, classifies light curves
run_tic = False # >> input a TICID (temporary)
DBSCAN_parameter_search=True
input_features=False
input_rms=False


tics = [219107776.0, 185336364.0] # >> for run_tic

# >> file names
fname_time = []
fname_flux = []
fname_ticid = []
fname_rms = []

for i in [4]:
    for j in [1,2,3,4]:
        fname_time.append('Sector20Cam'+str(i)+'CCD'+str(j)+'-time.txt')
        fname_flux.append('Sector20Cam'+str(i)+'CCD'+str(j)+'-flux.csv')
        fname_ticid.append('Sector20Cam'+str(i)+'CCD'+str(j)+'-ticid.txt') 
        fname_rms.append('Sector20Cam'+str(i)+'CCD'+str(j)+'-rms.txt')

# fname_time=['Sector20Cam1CCD1-time.txt', 'Sector20Cam1CCD2-time.txt',
#             'Sector20Cam1CCD3-time.txt', 'Sector20Cam1CCD4-time.txt',
#             'Sector20Cam2CCD1-time.txt', 'Sector20Cam2CCD2-time.txt',
#             'Sector20Cam2CCD3-time.txt', 'Sector20Cam3CCD4-time.txt']
# fname_flux=['Sector20Cam1CCD1-flux.csv', 'Sector20Cam1CCD1-flux.csv',
#             'Sector20Cam1CCD1-flux.csv', 'Sector20Cam1CCD1-flux.csv',
#             'Sector20Cam1CCD1-flux.csv','Sector20Cam1CCD1-flux.csv',
#             'Sector20Cam1CCD1-flux.csv']
# fname_ticid=['Sector20Cam1CCD1-ticid.txt']
# fname_rms=['Sector20Cam1CCD1-rms.txt']

# >> hyperparameters
if hyperparameter_optimization:
    p = {'kernel_size': [3,5,7],
      'latent_dim': [50],
      'strides': [1],
      'epochs': [10],
      'dropout': list(np.arange(0.1, 0.5, 0.02)),
      'num_filters': [8, 16, 32, 64],
      'num_conv_layers': [2,4,6,8,10],
      'batch_size': [128],
      'activation': ['elu', 'selu'],
      'optimizer': ['adam', 'adadelta'],
      'last_activation': ['elu', 'selu', 'linear'],
      'losses': ['mean_squared_error'],
      'lr': list(np.arange(0.001, 0.1, 0.01)),
      'initializer': ['random_normal', 'random_uniform', 'glorot_normal',
                      'glorot_uniform', 'zeros']}
else:
    p = {'kernel_size': 7,
          'latent_dim': 21,
          'strides': 1,
          'epochs': 10,
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
for i in range(len(fname_time)):
    x     = np.loadtxt(dat_dir+fname_time[i])
    flux  = np.loadtxt(dat_dir+fname_flux[i], delimiter=',')
    ticid = np.loadtxt(dat_dir+fname_ticid[i])
    lengths.append(len(x))
    time_list.append(x)
    flux_list.append(flux)
    ticid_list.append(ticid)
    
    if input_rms:
        rms = np.loadtxt(dat_dir + fname_rms[i])
        rms_list.append(rms)

# !! tmp
# new_length = np.shape(flux)[1]
if run_tic:
    # new_length = np.min(lengths)
    new_length = 18757 # !! length of tabby star
    x = []
    flux = []
    ticid = []
    
    # >> truncate all light curves
    for i in range(len(fname_time)):
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
    if input_rms:
        rms = np.concatenate(rms_list, axis=0)
    
    ticid = []
    for i in range(len(fname_ticid)):
        ticid.extend(ticid_list[i])
    
# >> apply nan mask
flux, x = ml.nan_mask(flux, x, output_dir=output_dir, ticid=ticid, 
                      DEBUG=True, debug_ind=10)
title='TESS-unsupervised'

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

x_train, x_test, y_train, y_test, x = \
    ml.split_data(flux, x, p, train_test_ratio=0.90, supervised=False)

ticid_train = ticid[:np.shape(x_train)[0]]
ticid_test = ticid[-1 * np.shape(x_test)[0]:]

if input_rms:
    rms_train = rms[:np.shape(x_train)[0]]
    rms_test = rms[-1 * np.shape(x_test)[0]:]
else:
    rms_train, rms_test = False, False

# == talos experiment =========================================================
if hyperparameter_optimization:
    t = talos.Scan(x=np.concatenate([x_train, x_test]),
                    y=np.concatenate([x_train, x_test]),
                    params=p,
                    model=ml.conv_autoencoder,
                    experiment_name=title, 
                    reduction_metric = 'val_loss',
                    minimize_loss=True,
                    reduction_method='correlation',
                    fraction_limit=0.0001)
    analyze_object = talos.Analyze(t)
    df, best_param_ind,p = ml.hyperparam_opt_diagnosis(analyze_object,
                                                       output_dir,
                                                       supervised=False)


# == run model ================================================================
if run_model:
    history, model = ml.conv_autoencoder(x_train, x_train, x_test, x_test, p)
    x_predict = model.predict(x_test)
    # !!
    # np.savetxt(output_dir+'x_predict.txt',
    #            np.reshape(x_predict, (np.shape(x_predict)[0],
    #                                   np.shape(x_predict)[1])),
    #            delimiter=',')
    # model.save(output_dir+"model.h5")
    # print("Saved model!")
    
    # with open(output_dir+'historydict.p', 'wb') as file_pi:
    #     pickle.dump(history.history, file_pi)
        
    ml.param_summary(history, x_test, x_predict, p, output_dir, 0,
                      'tess-unsupervised')
    ml.model_summary_txt(output_dir, model)
    

if diag_plots:
    if run_model == False:
        # model = load_model(output_dir + 'model.h5')
        # history = pickle.load( open(output_dir + 'historydict.p', 'rb'))
        history = []
        x_predict = np.loadtxt(output_dir+'x_predict.txt', delimiter=',')
        x_predict = np.reshape(x_predict, (np.shape(x_predict)[0],
                                           np.shape(x_predict)[1], 1))
        
        activations, bottleneck = pl.diagnostic_plots(history, model, p,
                                                      output_dir, x, x_train, x_test,
                            x_predict, mock_data=False, ticid_train=ticid_train,
                            ticid_test=ticid_test,plot_intermed_act=False,
                            plot_latent_test=False, plot_latent_train=False,
                            make_movie=False, percentage=False, plot_epoch=False,
                            input_features=input_features, input_rms=input_rms,
                            rms_test=rms_test, rms_train=rms_train)
    else:
        activations, bottleneck = pl.diagnostic_plots(history, model, p,
                                                      output_dir, x, x_train,
                                                      x_test,
                            x_predict, mock_data=False,
                            ticid_train=ticid_train,
                            ticid_test=ticid_test,
                            plot_intermed_act=False,
                            plot_latent_test=False,
                            plot_latent_train=False,
                            plot_reconstruction_error_all=False,
                            make_movie=False, percentage=False,
                            input_features=input_features,
                            input_rms=input_rms, rms_test=rms_test,
                            rms_train=rms_train)    
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


# >> Feature plots
if classification:
    # activations = ml.get_activations(model, x_test)
    # bottleneck = ml.get_bottleneck(model, activations, p)
    
    if DBSCAN_parameter_search:
        from sklearn.cluster import DBSCAN
        bottleneck = ml.standardize(bottleneck, ax=0)
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
            classes = ff.features_plotting_2D(bottleneck, bottleneck, output_dir, 'dbscan',
                                    x, x_test, ticid_test,
                                    feature_engineering=False, eps=p[0],
                                    min_samples=p[1],
                                    metric=p[2], algorithm=p[3], leaf_size=p[4],
                                    p=p[5], folder_suffix='_'+str(i)+'classes')
    
    else:
        classes = ff.features_plotting_2D(bottleneck, bottleneck, output_dir, 'dbscan',
                                x, x_test, ticid_test,
                                feature_engineering=False, eps=2.9, min_samples=2,
                                metric='minkowski', algorithm='auto', leaf_size=30,
                                p=4)
    ff.features_plotting_2D(bottleneck, bottleneck, output_dir, 'kmeans',
                            x, x_test, ticid_test,
                            feature_engineering=False)
    
    ml.plot_pca(bottleneck, classes, output_dir=output_dir)
    
    targets = []
    for i in ticid_test:
        targets.append('TIC ' + str(int(i)))
    ff.features_insets2D(x, x_test, bottleneck, targets, output_dir)


# orbit_gap_start = len(x)-1 # !!
# orbit_gap_end = orbit_gap_start + 1

# # >> take before orbit gap            
# flux = flux[:,:orbit_gap_start]
# x = x[:orbit_gap_start]
