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
import feature_functions as ff

output_dir = './plots/plots060620-2/'

hyperparameter_optimization = False

# >> unsupervised mode
fname_time  = 's0020-before_orbit-time.txt'
fname_flux  = 's0020-before_orbit-flux.csv'
fname_ticid = 's0020-before_orbit-ticid.txt'
# fname_time  = 's0020-before_orbit-1155-time.txt' # s0020-1155-
# fname_flux  = 's0020-before_orbit-1155-flux.csv'
# fname_ticid = 's0020-before_orbit-1155-ticid.txt'

if hyperparameter_optimization:
    p = {'kernel_size': [3,5,7],
      'latent_dim': list(np.arange(5, 35, 2)),
      'strides': [1],
      'epochs': [20],
      'dropout': list(np.arange(0., 0.6, 0.1)),
      'num_filters': [8, 16, 32, 64],
      'num_conv_layers': [2,4,6,8,10],
      'batch_size': [64,128],
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
          'epochs': 50,
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
                    fraction_limit=0.00005)
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
                    make_movie=False, percentage=True)
ml.param_summary(history, x_test, x_predict, p, output_dir, 0,
                  'tess-unsupervised')
ml.model_summary_txt(output_dir, model)
ml.plot_reconstruction_error(x, x_test, x_test, x_predict, ticid_test,
                              output_dir=output_dir)

# >> Feature plots


activations = ml.get_activations(model, x_test)
bottleneck = ml.get_bottleneck(model, activations, p)

DEBUG_DBSCAN=False
if DEBUG_DBSCAN:
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
    print('num classes: ' + str(max(num_classes)))
    inds = np.nonzero(np.array(num_classes)==max(num_classes))
    for i in inds[0]:
        print(counts[i])
    # 2.4000000000000004 2 minkowski auto 30 4
    # (array([-1,  0,  1,  2]), array([24, 84,  6,  2]))
    
    # 2.9000000000000004 2 minkowski auto 30 4
    # (array([-1,  0,  1,  2]), array([23, 85,  6,  2]))

ff.features_plotting_2D(bottleneck, bottleneck, output_dir, 'dbscan',
                        x, x_test, ticid_test,
                        feature_engineering=False, eps=2.9, min_samples=2,
                        metric='minkowski', algorithm='auto', leaf_size=30,
                        p=4)
ff.features_plotting_2D(bottleneck, bottleneck, output_dir, 'kmeans',
                        x, x_test, ticid_test,
                        feature_engineering=False)

targets = []
for i in ticid_test:
    targets.append('TIC ' + str(int(i)))
ff.features_insets2D(x, x_test, bottleneck, targets, output_dir)
