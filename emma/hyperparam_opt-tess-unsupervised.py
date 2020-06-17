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
import pickle
import feature_functions as ff
from keras.models import load_model

output_dir = './plots/plots061720/'

hyperparameter_optimization = False
run_model = True
diag_plots = True
run_tic = True # >> input a TICID
DBSCAN_parameter_search=True

tics = [219107776.0] # >> for run_tic

# >> unsupervised mode
fname_time='./sector_20_lc/Sector20Cam1CCD1_times_processed.txt'
fname_flux='./sector_20_lc/Sector20Cam1CCD1_intensities_processed.txt'
fname_ticid='./sector_20_lc/Sector20Cam1CCD1_targets.txt'


# x     = np.loadtxt(fname_time)
# flux  = np.loadtxt(fname_flux, delimiter=',')
# ticid = np.loadtxt(fname_ticid)

fname_time=['./sector_20_lc/Sector20Cam1CCD1_times_processed.txt',
            './sector_20_lc/Sector20Cam1CCD2_times_processed.txt',
            './sector_20_lc/Sector20Cam1CCD3_times_processed.txt']
fname_flux=['./sector_20_lc/Sector20Cam1CCD1_intensities_processed.txt',
            './sector_20_lc/Sector20Cam1CCD2_intensities_processed.txt',
            './sector_20_lc/Sector20Cam1CCD3_intensities_processed.txt']
fname_ticid=['./sector_20_lc/Sector20Cam1CCD1_targets.txt',
             './sector_20_lc/Sector20Cam1CCD2_targets.txt',
             './sector_20_lc/Sector20Cam1CCD3_targets.txt']

lengths = []
time_list = []
flux_list = []
ticid_list = []
for i in range(len(fname_time)):
    x     = np.loadtxt(fname_time[i])
    flux  = np.loadtxt(fname_flux[i])
    ticid = np.loadtxt(fname_ticid[i])
    lengths.append(len(x))
    time_list.append(x)
    flux_list.append(flux)
    ticid_list.append(ticid)
    
# !! won't need to concatenate once interpolation scheme changes 
new_length = np.min(lengths)
x = []
flux = []
ticid = []
for i in range(len(fname_time)):
    x = time_list[i][:new_length]
    flux.append(flux_list[i][:,:new_length])
    ticid.extend(ticid_list[i])

# !! tmp
if run_tic:
    for i in range(len(tics)):
        flux_tmp = ml.get_lc(str(int(tics[i])), out=output_dir, DEBUG_INTERP=True,
                             download_fits=False)
        flux_tmp = np.delete(flux_tmp, np.nonzero(np.isnan(flux_tmp)))
        flux.append([flux_tmp[:new_length]])
        ticid.extend([tics[i]])
    
flux = np.concatenate(flux, axis=0)

# fname_time  = 's0020-before_orbit_only-time.txt'
# fname_flux  = 's0020-before_orbit_only-flux.csv'
# fname_ticid = 's0020-before_orbit_only-ticid.txt'
# fname_time  = 's0020-full_orbit-time.txt'
# fname_flux  = 's0020-full_orbit-flux.csv'
# fname_ticid = 's0020-full_orbit-ticid.txt'
# fname_time  = 's0020-before_orbit-time.txt'
# fname_flux  = 's0020-before_orbit-flux.csv'
# fname_ticid = 's0020-before_orbit-ticid.txt'
# fname_time  = 's0020-before_orbit-1155-time.txt' # s0020-1155-
# fname_flux  = 's0020-before_orbit-1155-flux.csv'
# fname_ticid = 's0020-before_orbit-1155-ticid.txt'

if hyperparameter_optimization:
    p = {'kernel_size': [3,5,7],
      'latent_dim': [10],
      'strides': [1],
      'epochs': [10],
      'dropout': [0.1,0.3,0.5],
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


title='TESS-unsupervised'
mock_data=False

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

orbit_gap_start = len(x)-1 # !!
orbit_gap_end = orbit_gap_start + 1

# >> normalize
# flux, x = ml.normalize(flux, x)
flux = ml.standardize(flux)

# >> take before orbit gap            
flux = flux[:,:orbit_gap_start]
x = x[:orbit_gap_start]

x_train, x_test, y_train, y_test, x = \
    ml.split_data(flux, x, p, train_test_ratio=0.90, supervised=False)

ticid_train = ticid[:np.shape(x_train)[0]]
ticid_test = ticid[-1 * np.shape(x_test)[0]:]
        
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
    model.save(output_dir+"model.h5")
    print("Saved model!")
    
    with open(output_dir+'historydict.p', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
        
    ml.param_summary(history, x_test, x_predict, p, output_dir, 0,
                      'tess-unsupervised')
    ml.model_summary_txt(output_dir, model)
    

if diag_plots:
    if run_model == False:
        model = load_model(output_dir + 'model.h5')
        # history = pickle.load( open(output_dir + 'historydict.p', 'rb'))
        history = []
        x_predict = model.predict(x_test)
        ml.diagnostic_plots(history, model, p, output_dir, x, x_train, x_test,
                            x_predict, mock_data=mock_data, ticid_train=ticid_train,
                            ticid_test=ticid_test,plot_intermed_act=True,
                            plot_latent_test=True, plot_latent_train=True,
                            make_movie=False, percentage=True, plot_epoch=False)
    else:
        # ml.diagnostic_plots(history, model, p, output_dir, x, x_train, x_test,
        #                     x_predict, mock_data=mock_data, ticid_train=ticid_train,
        #                     ticid_test=ticid_test,plot_intermed_act=True,
        #                     plot_latent_test=True, plot_latent_train=True,
        #                     make_movie=False, percentage=True)
        ml.diagnostic_plots(history, model, p, output_dir, x, x_train, x_test,
                            x_predict, mock_data=mock_data, ticid_train=ticid_train,
                            ticid_test=ticid_test,plot_intermed_act=True,
                            plot_latent_test=True, plot_latent_train=False,
                            make_movie=False, percentage=False)        
        # ml.param_summary(history, x_test, x_predict, p, output_dir, 0,
        #                   'tess-unsupervised')
        # ml.model_summary_txt(output_dir, model)
    ml.plot_reconstruction_error(x, x_test, x_test, x_predict, ticid_test,
                                  output_dir=output_dir)
    
    # >> Feature plots
    
    activations = ml.get_activations(model, x_test)
    bottleneck = ml.get_bottleneck(model, activations, p)
    
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
