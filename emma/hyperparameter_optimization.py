# without using other people's code

import modellibrary as ml
import numpy as np
import matplotlib.pyplot as plt
import pdb
import copy
from keras.models import Model
from sklearn.metrics import confusion_matrix

output_dir = './plots/plots032620-2/'
fake_data = False

# >> x_train, x_test
if fake_data:
    x_train, y_train, x_test, y_test = ml.signal_data(training_size = 45000,
                                                      input_dim = 128,
                                                      reshape=True)
    x = np.linspace(0, 30, 128)
    inds = [0, -1, -2] # >> for plotting decoded vs input
else:
    cutoff = 16336
    fname_time = './tessdatasector20-time.txt'
    fname_intensity = './tessdatasector20-intensity.csv'
    x = np.loadtxt(fname_time)
    x = np.delete(x, np.arange(cutoff, np.shape(x)[0]), 0)
    x_train, x_test = ml.split_data(fname_intensity)
    inds = [0, -14, -10] # >> for plotting decoded vs input

# >> parameters
p = {'kernel_size': [3, 21],
     'latent_dim': [3],
     'strides': [1],
     'epochs': [4],
     'dropout': [0.1],
     'num_conv_layers': [3],
     'num_filters': [[16, 32, 32]],
     'batch_size': [256],
     'activation': ['relu'],
     'optimizer': ['adadelta'],
     'last_activation': ['sigmoid'],
     'losses': ['mean_squared_error']}

p_list = []
for a in range(len(p['kernel_size'])):
    for b in range(len(p['latent_dim'])):
        for c in range(len(p['strides'])):
            for d in range(len(p['epochs'])):
                for e in range(len(p['dropout'])):
                    for f in range(len(p['num_conv_layers'])):
                        for g in range(len(p['num_filters'])):
                            for h in range(len(p['batch_size'])):
                                for i in range(len(p['activation'])):
                                    for j in range(len(p['optimizer'])):
                                        for k in range(len(p['last_activation'])):
                                            for l in range(len(p['losses'])):
                                                p1 = {'kernel_size': p['kernel_size'][a],
                                                      'latent_dim': p['latent_dim'][b],
                                                      'strides': p['strides'][c],
                                                      'epochs': p['epochs'][d],
                                                      'dropout': p['dropout'][e],
                                                      'num_conv_layers': p['num_conv_layers'][f],
                                                      'num_filters': p['num_filters'][g],
                                                      'batch_size': p['batch_size'][h],
                                                      'activation': p['activation'][i],
                                                      'optimizer': p['optimizer'][j],
                                                      'last_activation': p['last_activation'][k],
                                                      'losses': p['losses'][l]}
                                                if p1 not in p_list:
                                                    p_list.append(copy.deepcopy(p1))

plt.ion()

for i in range(len(p_list)):
    p = p_list[i]
    print(p)
    history, model = ml.autoencoder19(x_train, x_test, p)

    x_predict = model.predict(x_test)
        
    with open(output_dir + 'param_summary.txt', 'a') as f:
        f.write('parameter set ' + str(i) + '\n')
        f.write(str(p.items()) + '\n')
        f.write('loss ' + str(history.history['loss'][-1]) + '\n')
        f.write('accuracy ' + str(history.history['accuracy'][-1]) + '\n')
        if fake_data:
            y_pred = np.max(x_predict, axis = 1)
            y_pred = np.round(np.reshape(y_pred, (np.shape(y_pred)[0])))
            cm = confusion_matrix(y_test, y_pred, labels=[0.,1.])
            f.write('confusion matrix\n')
            f.write('    0   1\n')
            f.write('0 ' + str(cm[0]) + '\n')
            f.write('1 ' + str(cm[1]) + '\n')
        f.write('\n')

    # >> plot some decoded light curves
    fig, axes = ml.input_output_plot(x, x_test, x_predict, inds=inds)
    plt.savefig(output_dir + 'input_output-' + str(i) + '.png')
    plt.close(fig)

    # >> plot latent dim
    bottleneck_layer = np.nonzero(['dense' in x.name for x in model.layers])[0][0]
    layer_outputs = [layer.output for layer in model.layers][1:]
    activation_model = Model(input=model.input, output=layer_outputs)
    activations = activation_model.predict(x_test)
    activation = activations[bottleneck_layer-1]
    fig, axes = ml.corner_plot(activation)
    plt.savefig(output_dir + 'latent_space-' + str(i) + '.png')
    plt.close(fig)
    
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
