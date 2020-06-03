# get x_train, x_test, y_train, y_test, time for MLP on engineered features

import numpy as np
import modellibrary as ml
import talos
import pdb
# from sklearn.preprocessing import StandardScaler

preprocessing = False
supervised = False
hyperparameter_optimization = True

output_dir = 'plots/plots053120/'
dat_dir = './Sector20Cam1CCD1/'
# prefix = 'Sector20Cam1CCD1_2'
prefix = 'Sector20Cam1CCD1'

num_classes = 4

      
if hyperparameter_optimization:
    hidden = []
    # for i in range(5): # >> num hidden layers
    #     for j in [1,2,3,4]: # >> step size
    #         hidden.append(list(np.arange(1, 16-j*i, -j)))
    #     for j in range(8, 17):
    #         hidden.append(list(np.repeat(j, i)))
    filter_num = []
    for i in range(5): # >> num hidden layers
        for j in range(16, 32, 8): # >> highest dimension
            for k in range(4, 16, 2): # >> lowest dimesion
                for l in range(1,4): # >> number of hidden layers
                    step = max(int(float(j) / k / l), 1)
                    hidden.append(list(np.arange(j, k-1, -step)))
                    
            
    p = {'hidden_units': hidden, 'latent_dim': [4,6,8,10],
          'activation': ['elu'], 'last_activation': ['linear', 'elu'],
          'optimizer': ['adam', 'adadelta'],
          'lr':list(np.arange(0.001, 0.1, 0.01)),
          'epochs': [200], 'losses': ['mean_squared_error'],
          'batch_size': [128], 'initializer': ['random_normal',
                                               'random_uniform',
                                               'glorot_normal',
                                               'glorot_uniform', 'zeros']}
else:
    p = {'hidden_units': [14,12], 'latent_dim': 8,
          'activation': 'elu', 'last_activation': 'elu', 'optimizer': 'adam',
          'lr':0.001, 'epochs': 500, 'losses': 'mean_squared_error',
          'batch_size': 128, 'initializer': 'random_normal'}

if preprocessing:
    if supervised:
        features = np.loadtxt(dat_dir+'sector_20_cam1_ccd1_features.txt')
        intensity = np.loadtxt(dat_dir+'sector20_cam1_ccd1_processed_intensities.txt')
        time = np.loadtxt(dat_dir+'sector20_cam1_ccd1_interp_times.txt')[:,0]
        ticid = np.loadtxt(dat_dir+'sector20_cam1_ccd1_targets.txt').astype('int')
            
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
        features = np.loadtxt(dat_dir+'sector_20_cam1_ccd1_features.txt')
        intensity = np.loadtxt(dat_dir+'sector20_cam1_ccd1_processed_intensities.txt')
        time = np.loadtxt(dat_dir+'sector20_cam1_ccd1_interp_times.txt')[:,0]
        ticid = np.loadtxt(dat_dir+'sector20_cam1_ccd1_targets.txt').astype('int')
            
        # features = []
        # intensity = []
        # time = []
        # ticid = []
        
        # features.append(np.loadtxt(dat_dir+'sector_20_cam1_ccd1_features.txt'))
        # features.append(np.loadtxt(dat_dir+'Sector20Cam1CCD2_features.txt'))
        # intensity.append(np.loadtxt(dat_dir+'sector20_cam1_ccd1_processed_intensities.txt'))
        # intensity.append(np.loadtxt(dat_dir+'Sector20Cam1CCD2_ints_processed.txt'))
        
        # time = np.loadtxt(dat_dir+'sector20_cam1_ccd1_interp_times.txt')[:,0]
        # # time.append(np.loadtxt(dat_dir+'sector20_cam1_ccd1_interp_times.txt')[:,0])
        # # time.append(np.loadtxt(dat_dir+'Sector20Cam1CCD2_interp_times.txt')[:,0])
        
        # ticid.append(np.loadtxt(dat_dir+'sector20_cam1_ccd1_targets.txt').astype('int'))
        # ticid.append(np.loadtxt(dat_dir+'Sector20Cam1CCD2_targets.txt').astype('int'))
        
        # features = np.concatenate(features)
        # intensity = np.concatenate(intensity)
        # ticid = np.concatenate(ticid)

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
    x_train = np.loadtxt(prefix+'x_train.csv')
    x_test = np.loadtxt(prefix+'x_test.csv')
    y_train = np.loadtxt(prefix+'y_train.csv')
    y_test = np.loadtxt(prefix+'y_test.csv')
    ticid_train = np.loadtxt(prefix+'ticid_train.txt')
    ticid_test = np.loadtxt(prefix+'ticid_test.txt')
    flux_train = np.loadtxt(prefix+'flux_train.csv')
    flux_test = np.loadtxt(prefix+'flux_test.csv')
    time = np.loadtxt(dat_dir+'sector20_cam1_ccd1_interp_times.txt')[0]
    
    # x_train = x_train - np.min(x_train, axis=0)
    # x_train = x_train / np.max(x_train, axis=0)
    # x_test = x_test - np.min(x_test, axis=0)
    # x_test = x_test / np.max(x_test, axis=0)
    # x_train = ml.standardize(x_train, ax=0)
    # x_test = ml.standardize(x_test, ax=0)
    
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
            
            df, best_param_ind, p = ml.hyperparam_opt_diagnosis(analyze_object,
                                                                output_dir,
                                                                supervised=supervised)
            
        
        
        history, model = ml.simple_autoencoder(x_train, x_train, x_test,
                                               x_test, p, resize=False,
                                               batch_norm=False)
    x_predict = model.predict(x_test)
    
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
        ml.diagnostic_plots(history, model, p, output_dir, '', x, x_train,
                            x_test, x_predict, ticid_train=ticid_train,
                            ticid_test=ticid_test, addend=0.,
                            flux_test=flux_test, flux_train=flux_train,
                            time=time, feature_vector=True,
                            plot_latent_test=True, plot_latent_train=True,
                            plot_intermed_act=True, plot_lof_test=True,
                            plot_lof_train=True, plot_kernel=True)
        # activations = ml.get_activations(model, x_test)
        # bottleneck = ml.get_bottleneck(model, activations, p)
        # ml.plot_lof(time, flux_test, ticid_test, bottleneck, 20, output_dir)



