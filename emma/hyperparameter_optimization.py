
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
# hyperparameter optimization
# etc 032620
#
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

import modellibrary as ml
import numpy as np
import matplotlib.pyplot as plt
import pdb
import copy
import random
from keras.models import Model
from sklearn.metrics import confusion_matrix


output_dir = './plots/plots040320-5/'
fake_data = True

fname_time = './section1-time.txt'
fname_intensity = './section1-intensity.csv'
cutoff = 8896 # >> divisible by 4
training_size = 1036 # >> even number
test_size = 116
test_noise = True
input_dim = cutoff
# fname_time = './tesssector20-all-time.txt'
# fname_time = './tessdatasector20-time.txt'
# fname_intensity = './tesssector20-all-intensity.csv'
# fname_intensity = './tessdatasector20-intensity.csv'

if fake_data:
    intermed_inds = [0, -1]
    input_bottle_inds = [0, 1, 2, -1, -2]
else:
    intermed_inds = [6,0]
    input_bottle_inds = [0,1,2,3,4]

# >> parameters
p = {'kernel_size': [71, 81, 93, 101],
     'latent_dim': [25],
     'strides': [1],
     'epochs': [20],
     'dropout': [0.2],
     'num_conv_layers': [9],
     'num_filters': [[32,32,32,32,32,32,32,32,32]],
     'batch_size': [32],
     'activation': ['relu'],
     'optimizer': ['adam'],
     'last_activation': ['relu'],
     'losses': ['mean_squared_error'],
     'noise': [0.2,0.5,1.],
     'lr': ['default']} # !! TODO changing optimizer


# p = {'kernel_size': np.arange(5, 100, 2),
#      'latent_dim': [25],
#      'strides' : [1],
#      'epochs': [10],
#      'dropout': np.arange(0, 0.5, 0.05),
#      'num_conv_layers': [3,5,7,9,11],
#      'num_filters': [16,32,64],
#      'batch_size': [32,64,128,256],
#      'activation': ['relu'],
#      'optimizer': ['adadelta', 'sgd', 'adam'],
#      'last_activation': ['relu', 'sigmoid'],
#      'losses': ['mean_squared_error', 'binary_crossentropy'],
#      'noise': [0.]}

grid_search = True # >> modified from sklearn.model_selection
randomized_search = False
n_iter = 10

plot_epoch = True
plot_in_bottle_out = True
plot_kernel = True
plot_feature = True
plot_latent_test = True
plot_latent_train = True

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# -- x_train, x_test -----------------------------------------------------------
if fake_data:
    if test_noise:
        # x_train, y_train, x_test, y_test = ml.signal_data(training_size = training_size,
        #                                                   test_size = test_size,
        #                                                   input_dim = input_dim,
        #                                                   reshape=True)
        x_train, y_train, x_test, y_test = ml.signal_data1(training_size=training_size,
                                                           test_size=test_size,
                                                           input_dim=input_dim,
                                                           reshape=True)
    x = np.linspace(0, 30, input_dim)
    # >> inds for plotting decoded vs input
    inds = [0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7]
else:
    # cutoff = 16272
    # cutoff = 16336
    # cutoff = 8899
    # cutoff = 8896 # TODO 
    x = np.loadtxt(fname_time)
    x = np.delete(x, np.arange(cutoff, np.shape(x)[0]), 0)
    x_train, x_test = ml.split_data(fname_intensity, cutoff=cutoff)
    # inds = [0, -14, -10] # >> for plotting decoded vs input
    inds = [0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7]
    
# -- initialize parameters -----------------------------------------------------

p_list = []

if grid_search:
    for a in range(len(p['kernel_size'])):
        for b in range(len(p['latent_dim'])):
            for c in range(len(p['strides'])):
                for d in range(len(p['epochs'])):
                    for e in range(len(p['dropout'])):
                        for f in range(len(p['num_conv_layers'])):
                            for h in range(len(p['batch_size'])):
                                for i in range(len(p['activation'])):
                                    for j in range(len(p['optimizer'])):
                                        for k in range(len(p['last_activation'])):
                                            for l in range(len(p['losses'])):
                                                for m in range(len(p['noise'])):
                                                    p1 = {'kernel_size': p['kernel_size'][a],
                                                          'latent_dim': p['latent_dim'][b],
                                                          'strides': p['strides'][c],
                                                          'epochs': p['epochs'][d],
                                                          'dropout': p['dropout'][e],
                                                          'num_conv_layers': p['num_conv_layers'][f],
                                                          'num_filters': p['num_filters'][f],
                                                          'batch_size': p['batch_size'][h],
                                                          'activation': p['activation'][i],
                                                          'optimizer': p['optimizer'][j],
                                                          'last_activation': p['last_activation'][k],
                                                          'losses': p['losses'][l],
                                                          'noise': p['noise'][m]}
                                                    if p1 not in p_list:
                                                        p_list.append(copy.deepcopy(p1))
else:
    for n in range(n_iter):
        p_dict = {}
        # >> randomized parameter search
        for i in range(len(list(p.keys()))):
            key = list(p.keys())[i]
            if key != 'num_filters':
                p_dict[key] = random.choice(p[key])
                if key == 'kernel_size':
                    p_dict[key] = [p_dict[key]]
                if key == 'num_conv_layers':
                    num_filts = np.repeat(random.choice(p['num_filters']),
                                                        p_dict[key])
                    # p_dict['num_filters'] = list(map(lambda el: [el],
                    #                                  num_filts))
                    p_dict['num_filters'] = list(num_filts)
        p_list.append(p_dict)
    
# plt.ion()

# -- run model -----------------------------------------------------------------

for i in range(len(p_list)):

    if fake_data and test_noise:
        if p['noise'] == 'all':
            x_train, y_train, x_test, y_test = \
                ml.no_signal_data(training_size=training_size,
                                  test_size = test_size,
                                  input_dim = input_dim,
                                  reshape=True,
                                  noise_level=0.2)
        else:
            # x_train, y_train, x_test, y_test = \
            #     ml.signal_data(training_size=training_size,
            #                    test_size=test_size,
            #                    input_dim = input_dim,
            #                    reshape=True,
            #                    noise_level=p['noise'])
            x_train, y_train, x_test, y_test =\
                ml.signal_data1(training_size=training_size,
                                test_size=test_size,
                                input_dim=input_dim, reshape=True,
                                noise_level=p_list[i]['noise'])
    p = p_list[i]
    print(p)

    
    history, model = ml.autoencoder21(x_train, x_test, p)
    x_predict = model.predict(x_test)
        
    with open(output_dir + 'param_summary.txt', 'a') as f:
        f.write('parameter set ' + str(i) + '\n')
        f.write(str(p.items()) + '\n')
        label_list = ['loss', 'accuracy', 'precision', 'recall']
        key_list =['loss', 'accuracy', list(history.history.keys())[-2],
                   list(history.history.keys())[-1]]
        for j in range(4):
            f.write(label_list[j]+' '+str(history.history[key_list[j]][-1])+\
                    '\n')
        if fake_data:
            # >> confusion matrix
            y_true = np.argmax(x_test, axis = 1)
            # y_pred = np.max(x_predict, axis = 1)
            # y_pred = np.round(np.reshape(y_pred, (np.shape(y_pred)[0])))
            y_pred = np.argmax(x_predict, axis = 1)
            # cm = confusion_matrix(y_test, y_pred, labels=[0.,1.])
            cm = confusion_matrix(y_test, y_pred, labels=[0.,1.])
            f.write('confusion matrix\n')
            f.write('    0   1\n')
            f.write('0 ' + str(cm[0]) + '\n')
            f.write('1 ' + str(cm[1]) + '\n')
        f.write('\n')


    # >> plot loss, accuracy, precision, recall vs. epochs
    if plot_epoch:
        ml.epoch_plots(history, p, output_dir + 'epoch-' + str(i) + '-')
    
    # >> plot some decoded light curves
    if fake_data: addend = 1.
    else: addend = 0.5
    fig, axes = ml.input_output_plot(x, x_test, x_predict, inds=inds,
                                     out=output_dir+'input_output-x_test-'+\
                                     str(i)+'.png', addend=addend,
                                     sharey=False)
    # >> plot latent space
    activations = ml.get_activations(model, x_test)
    if plot_latent_test:
        fig, axes = ml.latent_space_plot(model, activations,
                                         output_dir+'latent_space-'+str(i)+'.png')
    if plot_latent_train:
        fig, axes = ml.latent_space_plot(model, ml.get_activations(model, x_train),
                                         output_dir+'latent_space-x_train-'+str(i)+\
                                         '.png')

    # >> plot kernel vs. filter
    if plot_kernel:
        ml.kernel_filter_plot(model, output_dir+'kernel-'+str(i)+'-')

    # >> plot intermediate activations
    if plot_feature:
        ml.intermed_act_plot(x, model, activations, x_test,
                             output_dir+'intermed_act-'+str(i)+'-',
                             addend=addend, inds=intermed_inds) # >> TODO make inds variable
        ml.movie(x, model, activations, x_test, p,
                 output_dir+'movie-'+str(i)+'-', addend=addend,
                 inds=intermed_inds)

    if plot_in_bottle_out:
        ml.input_bottleneck_output_plot(x, x_test, x_predict, activations, model,
                                        out=output_dir+'input_bottleneck_output-'+\
                                        str(i)+'.png', addend=addend,
                                        inds = input_bottle_inds, sharey=False)

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# # >> parameters
# p = {'kernel_size': [21],
#      'latent_dim': [1, 3],
#      'strides': [1],
#      'epochs': [4],
#      'dropout': [0.1],
#      'num_conv_layers': [3],
#      'num_filters': [[16, 32, 32]],
#      'batch_size': [256],
#      'activation': ['relu'],
#      'optimizer': ['adadelta'],
#      'last_activation': ['sigmoid'],
#      'losses': ['mean_squared_error', 'binary_crossentropy']}

# p = {'kernel_size': [21],
#      'latent_dim': [1,2,3,4],
#      'strides': [1],
#      'epochs': [20],
#      'dropout': [0.1],
#      'num_conv_layers': [3],
#      'num_filters': [[16, 32, 32]],
#      'batch_size': [256],
#      'activation': ['relu'],
#      'optimizer': ['adadelta'],
#      'last_activation': ['sigmoid'],
#      'losses': ['mean_squared_error']}

# >> parameters
# p = {'kernel_size': 21,
#      'latent_dim': 3,
#      'strides': 1,
#      'epochs': 4,
#      'dropout': 0.5,
#      'num_conv_layers': 3,
#      'num_filters': [32, 32, 32],
#      'batch_size': 128,
#      'activation': 'relu',
#      'optimizer': 'adadelta',
#      'last_activation': 'sigmoid',
#      'losses': 'mean_squared_loss'}

