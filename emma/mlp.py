# get x_train, x_test, y_train, y_test, time for MLP on engineered features

import numpy as np
import modellibrary as ml

preprocessing = False
supervised = False

output_dir = 'plots/plots052220/'
dat_dir = './Sector20Cam1CCD1/'
prefix = 'Sector20Cam1CCD1'

num_classes = 4
p = {'units': [32, 16], 
     'activation': 'relu', 'optimizer': 'adam', 'lr':0.01,
     'epochs':10, 'losses': 'mean_squared_error', 'batch_size':64}

if preprocessing:
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
        
    np.savetxt(prefix+'x_train.csv', x_train)
    np.savetxt(prefix+'x_test.csv', x_test)
    np.savetxt(prefix+'y_train.csv', y_train)
    np.savetxt(prefix+'y_test.csv', y_test)
    np.savetxt(prefix+'ticid_train.txt', ticid_train)
    np.savetxt(prefix+'ticid_test.txt', ticid_test)
    np.savetxt(prefix+'flux_train.csv', flux_train)
    np.savetxt(prefix+'flux_test.csv', flux_test)

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
    
    if supervised:
        history, model = ml.mlp(x_train, y_train, x_test, y_test, p, 
                                resize=False)
    else:
        histoyr, model = ml.simple_autoencoder(x_train, x_train, x_test,
                                               x_test, p, resize=False)
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
        ml.diagnostic_plots(history, model, p, output_dir, '', time, x_train,
                            x_test, x_predict, ticid_train=ticid_train,
                            ticid_test=ticid_test)


