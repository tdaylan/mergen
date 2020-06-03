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

output_dir = './plots/plots060220/'

hyperparameter_optimization = True

# >> unsupervised mode
fname_time  = 's0020-before_orbit-1155-time.txt' # s0020-1155-
fname_flux  = 's0020-before_orbit-1155-flux.csv'
fname_ticid = 's0020-before_orbit-1155-ticid.txt'

if hyperparameter_optimization:
    p = {'kernel_size': [3,5,7],
      'latent_dim': list(np.arange(5, 35, 2)),
      'strides': [1],
      'epochs': [20],
      'dropout': list(np.arange(0., 0.6, 0.1)),
      'num_filters': [8, 16, 32, 64],
      'num_conv_layers': [1,2,3,4,5,6],
      'batch_size': [64,128],
      'activation': ['elu', 'selu'],
      'optimizer': ['adam', 'adadelta'],
      'last_activation': ['elu', 'selu', 'linear'],
      'losses': ['mean_squared_error'],
      'lr': list(np.arange(0.001, 0.1, 0.01)),
      'initializer': ['random_normal', 'random_uniform', 'glorot_normal',
                      'glorot_uniform', 'zeros']}
else:
    p = {'kernel_size': 5,
          'latent_dim': 21,
          'strides': 1,
          'epochs': 50,
          'dropout': 0.5,
          'num_filters': 32,
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

x     = np.loadtxt(fname_time)
flux  = np.loadtxt(fname_flux, delimiter=',')
ticid = np.loadtxt(fname_ticid)
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
                    fraction_limit=0.000005)
    analyze_object = talos.Analyze(t)
    df, best_param_ind,p = ml.hyperparam_opt_diagnosis(analyze_object,
                                                       output_dir,
                                                       supervised=False)


# == run model ================================================================
history, model = ml.conv_autoencoder(x_train, x_train, x_test, x_test, p)
x_predict = model.predict(x_test)

ml.diagnostic_plots(history, model, p, output_dir, '', x, x_train, x_test,
                    x_predict, mock_data=mock_data, ticid_train=ticid_train,
                    ticid_test=ticid_test,plot_intermed_act=True,
                    plot_latent_test=True, plot_latent_train=True,
                    make_movie=False)
ml.param_summary(history, x_test, x_predict, p, output_dir, 0,
                 'tess-unsupervised')
ml.model_summary_txt(output_dir, model)

