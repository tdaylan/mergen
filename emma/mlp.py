# get x_train, x_test, y_train, y_test, time for MLP on engineered features

lib_dir = '../main/'

import numpy as np
import talos
import pdb
from astropy.io import fits
import sys
sys.path.insert(0, lib_dir)
import model as ml
import plotting_functions as pl
import data_functions as df
import pandas as pd


sectors = [20]
cams = [1]
ccds = [1,2]

preprocessing = False
supervised = False
use_tess_features = True # >> 5 features 
use_engineered_features = True # >> 16 features
use_tls_features = True # >> 4 features

hyperparameter_optimization = False
run_model = False
classification = True
# toi_train = True # >> train only on TOIs
# toi_validation = True # >> validate classifications with TOIs

output_dir = '../../plots/DAE/' # >> make dir if doesn't exist
data_dir = '/Users/studentadmin/Dropbox/TESS_UROP/data/'
# data_dir = '../../'
# prefix = 'Sector20Cam1CCD1_2'
prefix = 'Sector20Cam1CCD1'

num_classes = 4
# validation_targets = [219107776]
validation_targets = []

# tois = pd.read_csv('../../tois.csv', skiprows=[0,1,2,3])

if hyperparameter_optimization:
    # p = {'max_dim': list(range(21, 60)),
    #      'step': list(range(1, 8)),
    #      'latent_dim': list(range(4, 20)),
    #       'activation': ['elu'], 'last_activation': ['linear'],
    #       'optimizer': ['adam', 'adadelta'],
    #       'lr':list(np.arange(0.001, 0.1, 0.01)),
    #       'epochs': [50], 'losses': ['mean_squared_error'],
    #       'batch_size': [128], 'initializer': ['random_normal',
    #                                            'random_uniform',
    #                                            'glorot_normal',
    #                                            'glorot_uniform', 'zeros']}    
    p = {'max_dim': list(range(25, 60)),
         'step': list(range(1, 8)),
         'latent_dim': list(range(2, 25)),
          'activation': ['elu'], 'last_activation': ['linear'],
          'optimizer': ['adam', 'adadelta'],
          'lr':list(np.arange(0.001, 0.1, 0.01)),
          'epochs': [50], 'losses': ['mean_squared_error'],
          'batch_size': [128], 'initializer': ['random_normal',
                                               'random_uniform',
                                               'glorot_normal',
                                               'glorot_uniform', 'zeros']}        
else:     
    # p = {'max_dim': 50, 'step': 5, 'latent_dim': 17,
    #       'activation': 'elu', 'last_activation': 'linear',
    #       'optimizer': 'adadelta',
    #       'lr':0.081, 'epochs': 300, 'losses': 'mean_squared_error',
    #       'batch_size': 128, 'initializer': 'random_uniform'}      
    p = {'max_dim': 41, 'step': 6, 'latent_dim': 23,
          'activation': 'elu', 'last_activation': 'linear',
          'optimizer': 'adam',
          'lr':0.011, 'epochs': 300, 'losses': 'mean_squared_error',
          'batch_size': 128, 'initializer': 'glorot_uniform'}         

if preprocessing:
    if supervised:
        # !!
        features = np.loadtxt(data_dir+'sector_20_cam1_ccd1_features.txt')
        intensity = np.loadtxt(data_dir+'sector20_cam1_ccd1_processed_intensities.txt')
        time = np.loadtxt(data_dir+'sector20_cam1_ccd1_interp_times.txt')[:,0]
        ticid = np.loadtxt(data_dir+'sector20_cam1_ccd1_targets.txt').astype('int')
            
        # >> get rid of buggy lc (couldn't download fits files I think)
        features = np.delete(features, [553, 904, 916], axis=0)
        intensity = np.delete(intensity, [553, 904, 916], axis=0)
        ticid = np.delete(ticid, [553, 904, 916])
    
        
        # >> retrieve classes [545, 333,  13,  28]
        classes = []
        with open('classified_Sector20Cam1CCD1.txt', 'r') as f:
            lines = f.readlines()
            ticid_classified = [line.split()[0] for line in lines]
            classified = [int(line.split()[1]) for line in lines]
        for i in ticid:
            classes.append(int(classified[ticid_classified.index(str(i))]))
        classes = np.array(classes)
            
        x_train, x_test, y_train, y_test, flux_train, flux_test, ticid_train, ticid_test, time = \
            ml.split_data_features(intensity, features, time, ticid, classes, p,
                                   supervised=True,
                          resize_arr=False, truncate=False)
            
    else:
        
        features = np.loadtxt(data_dir+'sector_20_cam1_ccd1_features.txt')
        intensity = np.loadtxt(data_dir+'sector20_cam1_ccd1_processed_intensities.txt')
        time = np.loadtxt(data_dir+'sector20_cam1_ccd1_interp_times.txt')[:,0]
        ticid = np.loadtxt(data_dir+'sector20_cam1_ccd1_targets.txt').astype('int')

        x_train, x_test, y_train, y_test, flux_train, flux_test, ticid_train, ticid_test, time = \
            ml.split_data_features(intensity, features, time, ticid, False, p,
                                   supervised=False,
                                   resize_arr=False, truncate=False)
        
        
    np.savetxt(prefix+'x_train.csv', x_train)
    np.savetxt(prefix+'x_test.csv', x_test)
    np.savetxt(prefix+'ticid_train.txt', ticid_train)
    np.savetxt(prefix+'ticid_test.txt', ticid_test)
    np.savetxt(prefix+'flux_train.csv', flux_train)
    np.savetxt(prefix+'flux_test.csv', flux_test)
    
    if supervised:
        np.savetxt(prefix+'y_train.csv', y_train)
        np.savetxt(prefix+'y_test.csv', y_test)

else:
  
    flux, time, ticid, target_info = \
    df.load_data_from_metafiles(data_dir, sectors[0], nan_mask_check=False,
                                cams=cams, ccds=ccds)
    
    features, flux, ticid, target_info = \
        ml.bottleneck_preprocessing(sectors[0], flux, ticid, target_info,
                                    output_dir=output_dir,
                                    data_dir=data_dir,
                                    use_engineered_features=use_engineered_features,
                                    use_tls_features=use_tls_features,
                                    use_tess_features=use_tess_features,
                                    use_learned_features=False,
                                    cams=cams, ccds=ccds)


    x_train, x_test, y_train, y_test, flux_train, flux_test, \
        ticid_train, ticid_test, target_info_train, target_info_test, time =\
            ml.autoencoder_preprocessing(flux, ticid, time, target_info, p,
                                         validation_targets=validation_targets,
                                         DAE=True,
                                         features=features)

    if supervised:
        history, model = ml.mlp(x_train, y_train, x_test, y_test, p, 
                                resize=False)
    else:
        if hyperparameter_optimization:
            t = talos.Scan(x=np.concatenate([x_train, x_test]),
                           y=np.concatenate([x_train, x_test]),
                           params=p, model=ml.simple_autoencoder,
                           experiment_name='feature autoencoder',
                           reduction_metric = 'val_loss',
                           minimize_loss = True, fraction_limit=0.01)
            
            analyze_object = talos.Analyze(t)
            
            df, best_param_ind, p = pl.hyperparam_opt_diagnosis(analyze_object,
                                                                output_dir,
                                                                supervised=supervised)
            
        
    if run_model:
        history, model = ml.simple_autoencoder(x_train, x_train, x_test,
                                               x_test, p, resize=False,
                                               batch_norm=True)
        x_predict = model.predict(x_test)
        # >> save to fits file
        hdr = fits.Header()
        hdu = fits.PrimaryHDU(x_predict, header=hdr)
        hdu.writeto(output_dir + 'x_predict.fits')
        
        # >> save bottleneck_test, bottleneck_train
        bottleneck = ml.get_bottleneck(model, x_test, DAE=True)    
        hdr = fits.Header()
        hdu = fits.PrimaryHDU(bottleneck, header=hdr)
        hdu.writeto(output_dir + 'bottleneck_test.fits')       
        
        bottleneck_train = ml.get_bottleneck(model, x_train, DAE=True)
        hdr = fits.Header()
        hdu = fits.PrimaryHDU(bottleneck_train, header=hdr)
        hdu.writeto(output_dir + 'bottleneck_train.fits')    
        
        ml.model_summary_txt(output_dir, model)
        ml.param_summary(history, x_test, x_predict, p, output_dir, 0, '',
                         supervised=supervised, y_test=y_test)
    
        if supervised:
            ml.epoch_plots(history, p, output_dir+'epoch-', supervised=True)
            y_train_classes = np.argmax(y_train, axis = 1)
            y_predict = np.argmax(x_predict, axis=-1)
            y_true = np.argmax(y_test, axis=-1)
            ml.training_test_plot(time, flux_train, flux_test, y_train_classes,
                                  y_true, y_predict, num_classes, output_dir+'lc-',
                                  ticid_train, ticid_test, mock_data=False)
            
        else:
            x = np.arange(np.shape(x_test)[1])
            
            pl.diagnostic_plots(history, model, p, output_dir, x, x_train,
                                x_test, x_predict, feature_vector=True,
                                flux_train=flux_train, flux_test=flux_test,
                                time=time,
                                mock_data=False, n_tot=40,
                                ticid_train=ticid_train, ticid_test=ticid_test,
                                target_info_test=target_info_test,
                                target_info_train=target_info_train,
                                DAE=True,
                                plot_epoch = True,
                                plot_in_out = True,
                                plot_in_bottle_out=False,
                                plot_latent_test = True,
                                plot_latent_train = True,
                                plot_kernel=False,
                                plot_intermed_act=False,
                                make_movie = False,
                                plot_lof_test=True,
                                plot_lof_train=True,
                                plot_lof_all=True,
                                plot_reconstruction_error_test=False,
                                plot_reconstruction_error_all=False,
                                load_bottleneck=False)   
            pl.latent_space_plot(x_train, 
                                 output_dir+'latent_space-original_features.png',
                                 units='psi')
    
    if classification:
        with fits.open(output_dir + 'bottleneck_train.fits') as hdul:
            bottleneck = hdul[0].data
        parameter_sets, num_classes, silhouette_scores, db_scores, ch_scores =\
            df.dbscan_param_search(bottleneck, time, flux_train, ticid_train,
                                    target_info_train, DEBUG=True,
                                    leaf_size=[30], algorithm=['auto'],
                                    min_samples=[3],
                                    output_dir=output_dir,
                                    simbad_database_txt='../../simbad_database.txt',
                                    database_dir='../../databases/',
                                    eps=list(np.arange(1.,3.,0.1)))
        # parameter_sets, num_classes, silhouette_scores, db_scores, ch_scores =\
        #     df.dbscan_param_search(bottleneck, time, flux_train, ticid_train,
        #                            target_info_train, DEBUG=True, 
        #                            output_dir=output_dir,
        #                            simbad_database_txt='../../simbad_database.txt',
        #                            database_dir='../../databases/')            
                                
    # mom_dump = '../../Table_of_momentum_dumps.csv'
    # # >> get best parameter set (want to maximize)
    # # print('num classes: ' + str(max(num_classes)))
    # print('num classes: ' + str(i))
    # inds = np.nonzero(np.array(num_classes)>10)
    # # !!
    # # best = np.argmin(np.array(num_noisy)[inds])
    # # best = np.argmax(np.array(silhouette_scores)[inds])
    # # best = np.argmin(np.array(db_scores)[inds])
    # best = np.argmax(np.array(ch_scores)[inds])
    # best = inds[0][best]
    # print('best_parameter_set: ' + str(parameter_sets[best]))
    # # print(str(counts[best]))
    # p=parameter_sets[best]
    # classes = pl.features_plotting_2D(bottleneck, bottleneck,
    #                                   output_dir, 'dbscan',
    #                                   time, flux_train, ticid_train,
    #                                   target_info=target_info_train,
    #                                   feature_engineering=False,
    #                                   eps=p[0], min_samples=p[1],
    #                                   metric=p[2], algorithm=p[3],
    #                                   leaf_size=p[4], p=p[5],
    #                                   folder_suffix='_'+str(i)+\
    #                                       'classes',
    #                                   momentum_dump_csv=mom_dump)          


# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
            # ml.diagnostic_plots(history, model, p, output_dir, '', x, x_train,
            #                     x_test, x_predict, ticid_train=ticid_train,
            #                     ticid_test=ticid_test, addend=0.,
            #                     flux_test=flux_test, flux_train=flux_train,
            #                     time=time, feature_vector=True,
            #                     plot_latent_test=True, plot_latent_train=True,
            #                     plot_intermed_act=True, plot_lof_test=True,
            #                     plot_lof_train=True, plot_kernel=True)
        
    # for i in np.unique(num_classes)[::-1]:
    #     # print('num classes: ' + str(max(num_classes)))
    #     print('num classes: ' + str(i))
    #     inds = np.nonzero(np.array(num_classes)==i)
    #     # !!
    #     # best = np.argmin(np.array(num_noisy)[inds])
    #     # best = np.argmax(np.array(silhouette_scores)[inds])
    #     # best = np.argmin(np.array(db_scores))
    #     best = np.argmax(np.array(ch_scores))
    #     # best = inds[0][best]
    #     print('best_parameter_set: ' + str(parameter_sets[best]))
    #     print(str(counts[best]))
    #     p=parameter_sets[best]
    #     classes = pl.features_plotting_2D(bottleneck, bottleneck,
    #                                       output_dir, 'dbscan',
    #                                       time, flux_train, ticid_train,
    #                                       target_info=target_info_train,
    #                                       feature_engineering=False,
    #                                       eps=p[0], min_samples=p[1],
    #                                       metric=p[2], algorithm=p[3],
    #                                       leaf_size=p[4], p=p[5],
    #                                       folder_suffix='_'+str(i)+\
    #                                           'classes',
    #                                       momentum_dump_csv=mom_dump)        
        # classes = pl.features_plotting_2D(bottleneck, bottleneck,
        #                                   output_dir, 'dbscan',
        #                                   time, flux_test, ticid_test,
        #                                   target_info=target_info_test,
        #                                   feature_engineering=False,
        #                                   eps=p[0], min_samples=p[1],
        #                                   metric=p[2], algorithm=p[3],
        #                                   leaf_size=p[4], p=p[5],
        #                                   folder_suffix='_'+str(i)+\
        #                                       'classes',
        #                                   momentum_dump_csv=mom_dump)
        # activations = ml.get_activations(model, x_test)
        # bottleneck = ml.get_bottleneck(model, activations, p)
        # ml.plot_lof(time, flux_test, ticid_test, bottleneck, 20, output_dir)        
        # features = []
        # intensity = []
        # time = []
        # ticid = []
        
        # features.append(np.loadtxt(data_dir+'sector_20_cam1_ccd1_features.txt'))
        # features.append(np.loadtxt(data_dir+'Sector20Cam1CCD2_features.txt'))
        # intensity.append(np.loadtxt(data_dir+'sector20_cam1_ccd1_processed_intensities.txt'))
        # intensity.append(np.loadtxt(data_dir+'Sector20Cam1CCD2_ints_processed.txt'))
        
        # time = np.loadtxt(data_dir+'sector20_cam1_ccd1_interp_times.txt')[:,0]
        # # time.append(np.loadtxt(data_dir+'sector20_cam1_ccd1_interp_times.txt')[:,0])
        # # time.append(np.loadtxt(data_dir+'Sector20Cam1CCD2_interp_times.txt')[:,0])
        
        # ticid.append(np.loadtxt(data_dir+'sector20_cam1_ccd1_targets.txt').astype('int'))
        # ticid.append(np.loadtxt(data_dir+'Sector20Cam1CCD2_targets.txt').astype('int'))
        
        # features = np.concatenate(features)
        # intensity = np.concatenate(intensity)
        # ticid = np.concatenate(ticid)
        
    # x_train = x_train - np.min(x_train, axis=0)
    # x_train = x_train / np.max(x_train, axis=0)
    # x_test = x_test - np.min(x_test, axis=0)
    # x_test = x_test / np.max(x_test, axis=0)
    # x_train = ml.standardize(x_train, ax=0)
    # x_test = ml.standardize(x_test, ax=0)        

    # x_train = np.loadtxt(prefix+'x_train.csv')
    # x_test = np.loadtxt(prefix+'x_test.csv')
    # y_train = np.loadtxt(prefix+'y_train.csv')
    # y_test = np.loadtxt(prefix+'y_test.csv')
    # ticid_train = np.loadtxt(prefix+'ticid_train.txt')
    # ticid_test = np.loadtxt(prefix+'ticid_test.txt')
    # flux_train = np.loadtxt(prefix+'flux_train.csv')
    # flux_test = np.loadtxt(prefix+'flux_test.csv')
    # time = np.loadtxt(data_dir+'sector20_cam1_ccd1_interp_times.txt')[0]  
    # # >> get TESS features (Tmag, rad, mass, GAIAmag, d, objType)
    # TESS_features = []
    # for i in range(len(ticid)):
    #     print(i)
    #     TESS_features.append(pl.get_features(ticid[i]))
    # # >> get features
    # with fits.open(data_dir + 'Sector20_v0_features/' +\
    #                'Sector20_features_v0_all.fits') as hdul:
    #     features = hdul[0].data
    #     ticid_features = hdul[1].data
    
    # # >> get light curves
    # fnames = []
    # fname_info = []
    # for sector in sectors:
    #     for cam in cams:
    #         for ccd in ccds:
    #             s = 'Sector{sector}Cam{cam}CCD{ccd}/' + \
    #                 'Sector{sector}Cam{cam}CCD{ccd}_lightcurves.fits'
    #             fnames.append(s.format(sector=sector, cam=cam, ccd=ccd))
    #             fname_info.append([sector, cam, ccd])
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
    # flux = np.concatenate(flux_list, axis=0)
    # # x = time_list[0][:new_length]
    # time = time_list[0]
        
    # # >> shuffle flux array
    # inds = np.arange(len(flux))
    # np.random.shuffle(inds)
    # flux = flux[inds]
    # ticid = ticid[inds]
    # target_info = target_info[inds].astype('int')         
        
    # tess_features = np.loadtxt('../../tess_features_sector20.txt',
    #                            delimiter=' ',
    #                            usecols=[1,2,3,4,5,6])
    # # >> take out any light curves with nans
    # inds = np.nonzero(np.prod(~np.isnan(tess_features), axis=1))
    # tess_features = tess_features[inds]
    
    # # >> only use these light curves
    # intersection, comm1, comm2 = np.intersect1d(tess_features[:,0], ticid,
    #                                             return_indices=True)
    # ticid = ticid[comm1]
    # flux = flux[comm1]
    # features = features[comm1]
    # ticid = intersection

    # # >> re-arrange tess features
    # tmp = []
    # for i in range(len(ticid)):
    #     ind = np.nonzero(tess_features[:,0] == ticid[i])[0][0]
    #     tmp.append(tess_features[ind][1:])
    # tess_features = np.array(tmp)
    
    # # >> concatenate features
    # features = np.concatenate([features, tess_features], axis=1)
    
    
    # 
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
            
            
    # x_train, x_test, y_train, y_test, flux_train, flux_test, ticid_train,\
    #     ticid_test, target_info_train, target_info_test, time = \
    #     ml.split_data_features(flux, features, time, ticid, target_info,
    #                            False, p,
    #                            supervised=False,
    #                            resize_arr=False, truncate=False,
    #                            train_test_ratio=0.9) 
    
    # # !! stanadrdize
    # x_train = df.standardize(x_train, ax=-1)
    # x_test = df.standardize(x_test, ax=-1)
        
    # from sklearn.cluster import DBSCAN
    # from sklearn.metrics import silhouette_score, calinski_harabasz_score
    # from sklearn.metrics import davies_bouldin_score
    # # with fits.open(output_dir + 'bottleneck_test.fits') as hdul:
    # #     bottleneck = hdul[0].data
    # with fits.open(output_dir + 'bottleneck_train.fits') as hdul:
    #     bottleneck = hdul[0].data        
    # # !! already standardized
    # # bottleneck = ml.standardize(bottleneck, ax=0)
    # eps = list(np.arange(0.1,5.0,0.1))
    # min_samples = [10]# [2, 5,10,15]
    # metric = ['euclidean'] # ['euclidean', 'minkowski'] # >> default
    # # algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    # algorithm = ['auto'] # >> default
    # # leaf_size = [30, 40, 50]
    # leaf_size = [30] # >> default
    # # p = [1,2,3,4]
    # p = [None] # >> default
    # classes = []
    # num_classes = []
    # counts = []
    # num_noisy= []
    # parameter_sets=[]
    # silhouette_scores=[]
    # ch_scores = []
    # db_scores = []
    # for i in range(len(eps)):
    #     for j in range(len(min_samples)):
    #         for k in range(len(metric)):
    #             for l in range(len(algorithm)):
    #                 for m in range(len(leaf_size)):
    #                     for n in range(len(p)):
    #                         db = DBSCAN(eps=eps[i],
    #                                     min_samples=min_samples[j],
    #                                     metric=metric[k],
    #                                     algorithm=algorithm[l],
    #                                     leaf_size=leaf_size[m],
    #                                     p=p[n]).fit(bottleneck)
    #                         print(db.labels_)
    #                         print(np.unique(db.labels_, return_counts=True))
    #                         classes_1, counts_1 = \
    #                             np.unique(db.labels_, return_counts=True)
    #                         if len(classes_1) > 1:
    #                             classes.append(classes_1)
    #                             num_classes.append(len(classes_1))
    #                             counts.append(counts_1)
    #                             num_noisy.append(counts[0])
    #                             parameter_sets.append([eps[i], min_samples[j],
    #                                                    metric[k],
    #                                                    algorithm[l],
    #                                                    leaf_size[m],
    #                                                    p[n]])
                                
    #                             # >> compute silhouette
    #                             silhouette = silhouette_score(bottleneck,
    #                                                           db.labels_)
    #                             silhouette_scores.append(silhouette)
                                
    #                             # >> compute calinski harabasz score
    #                             score = calinski_harabasz_score(bottleneck,
    #                                                             db.labels_)
    #                             ch_scores.append(score)
                                
    #                             # >> compute davies-bouldin score
    #                             dav_boul_score = davies_bouldin_score(bottleneck,
    #                                                          db.labels_)
    #                             db_scores.append(dav_boul_score)
                                
    #                             with open(output_dir + 'dbscan_param_search.txt', 'a') as f:
    #                                 # f.write('{} {} {} {} {} {}\n'.format(eps[i],
    #                                 #                                    min_samples[j],
    #                                 #                                    metric[k],
    #                                 #                                    algorithm[l],
    #                                 #                                    leaf_size[m],
    #                                 #                                    p[n]))
    #                                 f.write('{} {} {} {} {} {} {} {} {} {}\n'.format(eps[i],
    #                                                                    min_samples[j],
    #                                                                    metric[k],
    #                                                                    algorithm[l],
    #                                                                    leaf_size[m],
    #                                                                    p[n],
    #                                                                    len(classes_1),
    #                                                                    silhouette,
    #                                                                    score,
    #                                                                    dav_boul_score))
                                    # f.write(str(np.unique(db.labels_, return_counts=True)))
                                    # f.write('\n\n')
                                
    # dbscan_param_search=False
    # if not dbscan_param_search:
    #     with open(output_dir + 'dbscan_param_search.txt', 'r') as f:
    #         lines = f.readlines()        
            
        # if use_engineered_features and use_learned_features:
        #     features = np.concatenate([engineered_feature_vector,
        #                                learned_feature_vector,
        #                                tess_features], axis=1)
        # elif use_engineered_features:
        #     features = np.concatenate([engineered_feature_vector,
        #                                tess_features], axis=1)  
        # elif use_learned_features:
        #     features = np.concatenate([learned_feature_vector,
        #                                tess_features], axis=1)
        # else:
        #     features = tess_features        
    # hidden = []
    # for i in range(5): # >> num hidden layers
    #     for j in [1,2,3,4]: # >> step size
    #         hidden.append(list(np.arange(1, 16-j*i, -j)))
    #     for j in range(8, 17):
    #         hidden.append(list(np.repeat(j, i)))
    # filter_num = []
    # for i in range(5): # >> num hidden layers
    #     for j in range(16, 32, 8): # >> highest dimension
    #         for k in range(4, 16, 2): # >> lowest dimesion
    #             for l in range(1,4): # >> number of hidden layers
    #                 step = max(int(float(j) / k / l), 1)
    #                 hidden.append(list(np.arange(j, k-1, -step)))
    # hidden = []
    
    # max_dims = range(12, 30, 2)
    # latent_dim = [4,6,8,10,12]
    # steps = range(1, 8)
    # for max_dim in max_dims:
    #     for l_dim in latent_dim:
    #         for step in steps:
    #             hidden.append(list(range(max_dim, l_dim, -step)))
                    
            
    # p = {'hidden_units': hidden, 'latent_dim': [4,6,8,10,12],
    #       'activation': ['relu','elu'], 'last_activation': ['linear'],
    #       'optimizer': ['adam', 'adadelta'],
    #       'lr':list(np.arange(0.001, 0.1, 0.01)),
    #       'epochs': [100], 'losses': ['mean_squared_error'],
    #       'batch_size': [128], 'initializer': ['random_normal',
    #                                            'random_uniform',
    #                                            'glorot_normal',
    #                                            'glorot_uniform', 'zeros']}        