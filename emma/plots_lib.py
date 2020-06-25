# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
# 2020-06-20 - plots_lib.py
# Visualization library for TESS novelty detection pipeline.
# Emma Chickles
# 
# * diagnostic_plots : make all plots
# * input_output_plot : plots input, output and residual of CAE pipeline
# 
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
def diagnostic_plots(history, model, p, output_dir, 
                     x, x_train, x_test, x_predict, sharey=False, prefix='',
                     mock_data=False, ticid_train=False, ticid_test=False,
                     supervised=False, y_true=False, y_predict=False,
                     y_train=False, y_test=False,
                     flux_test=False, flux_train=False, time=False,
                     rms_train=False, rms_test = False, input_rms = False,
                     inds = [0,1,2,3,4,5,6,7,-1,-2,-3,-4,-5,-6,-7],
                     intermed_inds = [6,0],
                     input_bottle_inds = [0,1,2,-6,-7],
                     addend = 1., feature_vector=False, percentage=False,
                     input_features = False,
                     plot_epoch = True,
                     plot_in_out = True,
                     plot_in_bottle_out=False,
                     plot_latent_test = False,
                     plot_latent_train = False,
                     plot_kernel=False,
                     plot_intermed_act=False,
                     plot_clustering=False,
                     make_movie = False,
                     plot_lof_test=False,
                     plot_lof_train=False,
                     plot_lof_all=True,
                     plot_reconstruction_error_test=False,
                     plot_reconstruction_error_all=True):
    '''Produces all plots.
    Parameters:
        * history : Keras model.history
        * model : Keras Model()
        * p : parameter set given as a dictionary, e.g. {'latent_dim': 21, ...}
        * outout_dir : directory to save plots in
        * x : time array
        * x_train : 
    TODO: get rid fo rms_* options (integrate into input_features)
        '''

    # !! TODO: change supervised inputs to just y_train, y_test
    plt.rcParams.update(plt.rcParamsDefault)
    # !!
    # activations = get_activations(model, x_test, rms_test = rms_test,
    #                               input_rms=input_rms)
    activations = get_activations(model, x_test)    
    
    # >> plot loss, accuracy, precision, recall vs. epochs
    if plot_epoch:
        epoch_plots(history, p, output_dir+prefix+'epoch-',
                    supervised=supervised)   

    # -- unsupervised ---------------------------------------------------------
    # >> plot some decoded light curves
    if plot_in_out and not supervised:
        fig, axes = input_output_plot(x, x_test, x_predict,
                                      output_dir+prefix+'input_output.png',
                                      ticid_test=ticid_test,
                                      inds=inds,
                                      addend=addend, sharey=sharey,
                                      mock_data=mock_data,
                                      feature_vector=feature_vector,
                                      percentage=percentage)
        
    # >> plot input, bottleneck, output
    if plot_in_bottle_out and not supervised:
        input_bottleneck_output_plot(x, x_test, x_predict,
                                     activations, model, ticid_test,
                                     output_dir+prefix+\
                                     'input_bottleneck_output.png',
                                     addend=addend, inds = input_bottle_inds,
                                     sharey=False, mock_data=mock_data,
                                     feature_vector=feature_vector)
            
    # >> plot light curves with highest, smallest and random reconstruction
    #    error
    if plot_reconstruction_error_test:
        plot_reconstruction_error(x, x_test, x_test, x_predict, ticid_test,
                                  output_dir=output_dir)
    
    if plot_reconstruction_error_all:
        # >> concatenate test and train sets
        tmp = np.concatenate([x_test, x_train], axis=0)
        tmp_predict = model.predict(tmp)
        plot_reconstruction_error(x, tmp, tmp, tmp_predict, 
                                  np.concatenate([ticid_test, ticid_train],
                                                 axis=0),
                                  output_dir=output_dir)
        
    # -- supervised -----------------------------------------------------------
    if supervised:
        y_train_classes = np.argmax(y_train, axis = 1)
        num_classes = len(np.unique(y_train_classes))
        training_test_plot(x,x_train,x_test,
                              y_train_classes,y_true,y_predict,num_classes,
                              output_dir+prefix+'lc-', ticid_train, ticid_test,
                              mock_data=mock_data)
        
    # -- latent space visualization -------------------------------------------
    if input_features:
        features = []
        for ticid in ticid_test:
            res = get_features(ticid)
            features.append([res[1:6]])
        features = np.array(features)
    else: features=False
        
    if plot_latent_test:
        fig, axes = latent_space_plot(model, activations, p,
                                      output_dir+prefix+'latent_space.png')

        
    if plot_latent_train:
        # !!
        # activations_train = get_activations(model, x_train, rms_test=rms_train,
        #                                     input_rms=input_rms)
        activations_train = get_activations(model, x_train)        
        fig, axes = latent_space_plot(model, activations_train, p,
                                      output_dir+prefix+\
                                          'latent_space-x_train.png')
            
    if plot_lof_test:
        bottleneck = get_bottleneck(model, activations, p)
        for n in [20]: # [20, 50, 100]:
            if type(flux_test) != bool:
                plot_lof(time, flux_test, ticid_test, bottleneck, 20,
                         output_dir, prefix='test-', n_neighbors=n,
                         mock_data=mock_data, feature_vector=feature_vector)
            else:
                plot_lof(x, x_test, ticid_test, bottleneck, 20, output_dir,
                         prefix = 'test-', n_neighbors=n, mock_data=mock_data,
                         feature_vector=feature_vector)
            
    if plot_lof_train:
        # !! repeated code + rms
        # activations_train = get_activations(model, x_train, rms_test=rms_train,
        #                             input_rms=input_rms)
        activations_train = get_activations(model, x_train)
        bottleneck_train = get_bottleneck(model, activations_train, p)
        for n in [20]: # [20, 50, 100]:
            if type(flux_train) != bool:
                plot_lof(time, flux_train, ticid_train, bottleneck_train, 20,
                         output_dir, prefix='train-', n_neighbors=n,
                         mock_data=mock_data, feature_vector=feature_vector)
            else:
                plot_lof(x, x_train, ticid_train, bottleneck_train, 20,
                         output_dir, prefix = 'train-', n_neighbors=n,
                         mock_data=mock_data, feature_vector=feature_vector)   
                
    if plot_lof_all:
        # !! repeated code + rms
        # activations_train = get_activations(model, x_train, rms_test=rms_train,
        #                             input_rms=input_rms)
        activations_train = get_activations(model, x_train)        
        bottleneck_train = get_bottleneck(model, activations_train, p,
                                          input_features=input_features,
                                          features=features,
                                          input_rms=input_rms,
                                          rms=rms_train)
        bottleneck = get_bottleneck(model, activations, p,
                                    input_features=input_features,
                                    features=features,
                                    input_rms=input_rms, rms=rms_test)
        bottleneck_all = np.concatenate([bottleneck, bottleneck_train], axis=0)
        # !!
        # np.savetxt(output_dir+'latent_space.txt', bottleneck_all,
        #            delimiter=',')
        plot_lof(x, np.concatenate([x_test, x_train], axis=0),
                 np.concatenate([ticid_test, ticid_train], axis=0),
                 bottleneck_all, 20, output_dir, prefix='all-',
                 n_neighbors=20,
                 mock_data=mock_data, feature_vector=feature_vector)

    # >> plot kernel vs. filter
    if plot_kernel:
        kernel_filter_plot(model, output_dir+prefix+'kernel-')
        

    # if plot_clustering:
    #     bottleneck_ind = np.nonzero(['dense' in x.name for x in \
    #                                  model.layers])[0][0]
    #     bottleneck = activations[bottleneck_ind - 1]        
    #     latent_space_clustering(bottleneck, x_test, x, ticid_test,
    #                             out=output_dir+prefix+\
    #                                 'clustering-x_test-', addend=addend)
        

    # -- intermediate activations visualization -------------------------------
    if plot_intermed_act:
        intermed_act_plot(x, model, activations, x_test,
                          output_dir+prefix+'intermed_act-', addend=addend,
                          inds=intermed_inds, feature_vector=feature_vector)
    
    if make_movie:
        movie(x, model, activations, x_test, p, output_dir+prefix+'movie-',
              ticid_test, addend=addend, inds=intermed_inds)
        
    return activations, bottleneck

def epoch_plots(history, p, out_dir, supervised):
    '''Plot metrics vs. epochs.
    Parameters:
        * history : dictionary, output from model.history
        * model = Keras Model()
        * activations
        * '''
    if supervised:
        label_list = [['loss', 'accuracy'], ['precision', 'recall']]
        key_list = [['loss', 'accuracy'], [list(history.history.keys())[-2],
                                           list(history.history.keys())[-1]]]

        for i in range(len(key_list)):
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(history.history[key_list[i][0]], label=label_list[i][0])
            ax1.set_ylabel(label_list[i][0])
            ax2.plot(history.history[key_list[i][1]], '--', label=label_list[i][1])
            ax2.set_ylabel(label_list[i][1])
            ax1.set_xlabel('epoch')
            ax1.set_xticks(np.arange(0, int(p['epochs']),
                                     max(int(p['epochs']/10),1)))
            ax1.tick_params('both', labelsize='x-small')
            ax2.tick_params('both', labelsize='x-small')
            fig.tight_layout()
            if i == 0:
                plt.savefig(out_dir + 'acc_loss.png')
            else:
                plt.savefig(out_dir + 'prec_recall.png')
            plt.close(fig)
            
    else:
        fig, ax1 = plt.subplots()
        ax1.plot(history.history['loss'], label='loss')
        ax1.set_ylabel('loss')
        ax1.set_xlabel('epoch')
        ax1.set_xticks(np.arange(0, int(p['epochs']),
                                 max(int(p['epochs']/10),1)))
        ax1.tick_params('both', labelsize='x-small')
        fig.tight_layout()
        plt.savefig(out_dir + 'loss.png')
        plt.close(fig)

# == visualizations for unsupervised pipeline =================================

def input_output_plot(x, x_test, x_predict, out, ticid_test=False,
                      inds = [0, -14, -10, 1, 2], addend = 0., sharey=False,
                      mock_data=False, feature_vector=False,
                      percentage=False):
    '''Plots input light curve, output light curve and the residual.
    Can only handle len(inds) divisible by 3 or 5.
    Parameters:
        * x : time array
        * x_test
        * x_predict : output of model.predict(x_test)
        * out : output directory
        * ticid_test : list/array of TICIDs, required if mock_data=False
        * inds : indices of light curves in x_test to plot (len(inds)=15)
        * addend : constant to add to light curves when plotting
        * sharey : share y axis
        * mock_data : if mock_data, includes TICID, mass, rad, ... in titles
        * feature_vector : if feature_vector, assumes x-axis is latent space
                           not time
        * percentage : if percentage, plots residual as a fraction of x_test
    '''

    if len(inds) % 5 == 0:
        ncols = 5
    elif len(inds) % 3 == 0:
        ncols = 3
    ngroups = int(len(inds)/ncols)
    nrows = int(3*ngroups)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15,12), sharey=sharey,
                             sharex=True)
    plt.subplots_adjust(hspace=0)
    for i in range(ncols):
        for ngroup in range(ngroups):
            ind = int(ngroup*ncols + i)
            if not mock_data:
                ticid_label(axes[ngroup*3,i], ticid_test[inds[ind]],title=True)
                
            # >> plot input
            axes[ngroup*3,i].plot(x,x_test[inds[ind]]+addend,'.k',markersize=2)
            
            # >> plot output
            axes[ngroup*3+1,i].plot(x,x_predict[inds[ind]]+addend,'.k',
                                    markersize=2)
            # >> calculate residual
            residual = (x_test[inds[ind]] - x_predict[inds[ind]])
            if percentage:
                residual = residual / x_test[inds[ind]]
                
            # >> plot residual
            axes[ngroup*3+2, i].plot(x, residual, '.k', markersize=2)
            for j in range(3):
                format_axes(axes[ngroup*3+j,i])
            
        if feature_vector: # >> x-axis is latent dims
            axes[-1, i].set_xlabel('\u03C8', fontsize='small')
        else: # >> x-axis is time
            axes[-1, i].set_xlabel('time [BJD - 2457000]', fontsize='small')
            
    # >> make y-axis labels
    for i in range(ngroups):
        axes[3*i,   0].set_ylabel('input\nrelative flux',  fontsize='small')
        axes[3*i+1, 0].set_ylabel('output\nrelative flux', fontsize='small')
        axes[3*i+2, 0].set_ylabel('residual', fontsize='small') 
        
    fig.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    return fig, axes

def latent_space_plot(model, activations, params, out):
    ''' Plots corner plot of latent space.
    Parameters:
        * model : Keras Model()
        * activations : list of intermediate activations (from
                        get_activations())
        * params : dictionary of hyperparameters
        * out : output directory (ending with '/')
    '''
    # >> get index of bottleneck layer
    dense_inds = np.nonzero(['dense' in x.name for x in \
                                 model.layers])[0]
    # dense_inds = np.nonzero(['flatten' in x.name for x in model.layers])[0]
    for ind in dense_inds:
        if np.shape(activations[ind-1])[1] == params['latent_dim']:
            bottleneck_ind = ind - 1
    
    # >> make corner plot of latent spcae
    fig, axes = corner_plot(activations[bottleneck_ind-1], params)
    plt.savefig(out)
    plt.close(fig)
    return fig, axes

def kernel_filter_plot(model, out_dir):
    '''Plots kernel against filters, i.e. an image with dimension
    (kernel_size, num_filters).
    Parameters:
        * model : Keras Model()
        * out_dir : output directory (ending with '/')'''
    # >> get inds for plotting kernel and filters
    layer_inds = np.nonzero(['conv' in x.name for x in model.layers])[0]
    for a in layer_inds: # >> loop through conv layers
        filters, biases = model.layers[a].get_weights()
        fig, ax = plt.subplots()
        ax.imshow(np.reshape(filters, (np.shape(filters)[0],
                                       np.shape(filters)[2])))
        ax.set_xlabel('filter')
        ax.set_ylabel('kernel')
        plt.savefig(out_dir + 'layer' + str(a) + '.png')
        plt.close(fig)

def intermed_act_plot(x, model, activations, x_test, out_dir, addend=0.,
                      inds = [0, -1], feature_vector=False):
    '''Visualizing intermediate activations.
    Parameters:
        * x: time array
        * model: Keras Model()
        * activations: list of intermediate activations (from
                       get_activations())
        * x_test : array of fluxes, shape=(num light curves, num data points)
        * out_dir : output directory
        * append : constant to add to light curve when plotting
        * inds : indices of light curves in x_test to plot
        * feature_vector : if feature_vector, assumes x is latent dimensions,
                           not time
    Note that activation.shape = (num light curves, num data points,
                                  num_filters)'''
    # >> get inds for plotting intermediate activations
    act_inds = np.nonzero(['conv' in x.name or \
                           'max_pool' in x.name or \
                           'dropout' in x.name or \
                               'dense' in x.name or \
                           'reshape' in x.name for x in \
                           model.layers])[0]
    act_inds = np.array(act_inds) -1

    for c in range(len(inds)): # >> loop through light curves
        
        # -- plot input -------------------------------------------------------
        fig, axes = plt.subplots(figsize=(8,3))
        addend = 1. - np.median(x_test[inds[c]])
        axes.plot(np.linspace(np.min(x), np.max(x), np.shape(x_test)[1]),
                x_test[inds[c]] + addend, '.k', markersize=2)
        if feature_vector:
            axes.set_xlabel('\u03C8')
        else:
            axes.set_xlabel('time [BJD - 2457000]')
        axes.set_ylabel('relative flux')
        plt.tight_layout()
        fig.savefig(out_dir+str(c)+'ind-0input.png')
        plt.close(fig)
        
        # -- plot intermediate activations ------------------------------------
        for a in act_inds: # >> loop through layers
            activation = activations[a]
            
            if len(np.shape(activation)) == 2:
                ncols, nrows = 1, 1
                num_filters=1
                
            else:   
                if np.shape(activation)[2] == 1:
                    nrows = 1
                    ncols = 1
                    num_filters=1
                else:
                    num_filters = np.shape(activation)[2]
                    ncols = 4
                    nrows = int(num_filters/ncols)
                    
            fig, axes = plt.subplots(nrows,ncols,figsize=(8*ncols,3*nrows))                    
            for b in range(num_filters): # >> loop through filters
                if ncols == 1:
                    ax = axes
                else:
                    ax = axes.flatten()[b]
                    
                # >> make new time array and plot
                x1 = np.linspace(np.min(x), np.max(x), np.shape(activation)[1])
                if num_filters > 1:
                    ax.plot(x1, activation[inds[c]][:,b]+addend,'.k',
                            markersize=2)
                else:
                    ax.plot(x1, activation[inds[c]]+addend, '.k', markersize=2)
                
            # >> make x-axis and y-axis labels
            if nrows == 1:
                if feature_vector:
                    axes.set_xlabel('\u03C8')
                else:
                    axes.set_xlabel('time [BJD - 2457000]')        
                axes.set_ylabel('relative flux')
            else:
                for i in range(nrows):
                    axes[i,0].set_ylabel('relative\nflux')
                for j in range(ncols):
                    if feature_vector:
                        axes[-1,j].set_xlabel('\u03C8')
                    else:
                        axes[-1,j].set_xlabel('time [BJD - 2457000]')
            fig.tight_layout()
            fig.savefig(out_dir+str(c)+'ind-'+str(a+1)+model.layers[a+1].name\
                        +'.png')
            plt.close(fig)



            
    
def input_bottleneck_output_plot(x, x_test, x_predict, activations, model,
                                 ticid_test, out, inds=[0,1,-1,-2,-3],
                                 addend = 1., sharey=False, mock_data=False,
                                 feature_vector=False):
    '''Can only handle len(inds) divisible by 3 or 5'''
    bottleneck_ind = np.nonzero(['dense' in x.name for x in \
                                 model.layers])[0][0]
    bottleneck = activations[bottleneck_ind - 1]
    if len(inds) % 5 == 0:
        ncols = 5
    elif len(inds) % 3 == 0:
        ncols = 3
    ngroups = int(len(inds)/ncols)
    nrows = int(3*ngroups)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15,5), sharey=sharey,
                             sharex=False)
    plt.subplots_adjust(hspace=0)
    for i in range(ncols):
        for ngroup in range(ngroups):
            ind = int(ngroup*ncols + i)
            axes[ngroup*3,i].plot(x,x_test[inds[ind]]+addend,'.k',markersize=2)
            axes[ngroup*3+1,i].plot(np.linspace(np.min(x),np.max(x),
                                              len(bottleneck[inds[ind]])),
                                              bottleneck[inds[ind]], '.k',
                                              markersize=2)
            axes[ngroup*3+2,i].plot(x,x_predict[inds[ind]]+addend,'.k',
                                    markersize=2)
            if not mock_data:
                ticid_label(axes[ngroup*3,i],ticid_test[inds[ind]], title=True)
            for j in range(3):
                format_axes(axes[ngroup*3+j,i])
        axes[1, i].set_xlabel('\u03C6', fontsize='small')
        axes[1,i].set_xticklabels([])
        if feature_vector:
            axes[0, i].set_xlabel('\u03C8', fontsize='small')            
            axes[-1, i].set_xlabel('\u03C8', fontsize='small') 
        else:
            axes[0, i].set_xlabel('time [BJD - 2457000]', fontsize='small')        
            axes[-1, i].set_xlabel('time [BJD - 2457000]', fontsize='small')
    for i in range(ngroups):
        axes[3*i,   0].set_ylabel('input\nrelative flux',  fontsize='small')
        axes[3*i+1, 0].set_ylabel('bottleneck', fontsize='small')
        axes[3*i+2, 0].set_ylabel('output\nrelative flux', fontsize='small')
    fig.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    return fig, axes
    

def movie(x, model, activations, x_test, p, out_dir, ticid_test, inds = [0, -1],
          addend=0.5):
    '''Make a .mp4 file of intermediate activations.
    Parameters:
        * x : time array
        * model : Keras Model()
        * activations : output from get_activations()
        * x_test
        * p : parameter set
        * out_dir : output directory
        * inds : light curve indices in x_test'''
    for c in range(len(inds)):
        fig, axes = plt.subplots(figsize=(8,3*1.5))
        ymin = []
        ymax = []
        for activation in activations:
            # if np.shape(activation)[1] == p['latent_dim']:
            ymin.append(min(np.min(activation[inds[c]]),
                            np.min(x_test[inds[c]])))
                # ymax.append(max(activation[inds[c]]))
            ymax.append(max(np.max(activation[inds[c]]),
                            np.max(x_test[inds[c]])))
            # elif len(np.shape(activation)) > 2:
                # if np.shape(activation)[2] == 1:
                    # ymin.append(min(activation[inds[c]]))
                    # ymax.append(max(activation[inds[c]]))
        ymin = np.min(ymin) + addend + 0.3*np.median(x_test[inds[c]])
        ymax = np.max(ymax) + addend - 0.3*np.median(x_test[inds[c]])
        addend = 1. - np.median(x_test[inds[c]])

        # >> plot input
        axes.plot(np.linspace(np.min(x), np.max(x), np.shape(x_test)[1]),
                  x_test[inds[c]] + addend, '.k', markersize=2)
        axes.set_xlabel('time [BJD - 2457000]')
        axes.set_ylabel('relative flux')
        axes.set_ylim(ymin=ymin, ymax=ymax)
        # fig.tight_layout()
        fig.savefig('./image-000.png')
        plt.close(fig)

        # >> plot intermediate activations
        n=1
        for a in range(len(activations)):
            activation = activations[a]
            if np.shape(activation)[1] == p['latent_dim']:
                length = p['latent_dim']
                axes.cla()
                axes.plot(np.linspace(np.min(x), np.max(x), length),
                          activation[inds[c]] + addend, '.k', markersize=2)
                axes.set_xlabel('time [BJD - 2457000]')
                axes.set_ylabel('relative flux')
                # format_axes(axes, xlabel=True, ylabel=True)
                ticid_label(axes, ticid_test[inds[c]])
                axes.set_ylim(ymin=ymin, ymax =ymax)
                # fig.tight_layout()
                fig.savefig('./image-' + f'{n:03}.png')
                plt.close(fig)
                n += 1
            elif len(np.shape(activation)) > 2:
                # >> don't plot activations with multiple filters
                if np.shape(activation)[2] == 1:
                    length = np.shape(activation)[1]
                    y = np.reshape(activation[inds[c]], (length))
                    axes.cla()
                    axes.plot(np.linspace(np.min(x), np.max(x), length),
                              y + addend, '.k', markersize=2)
                    axes.set_xlabel('time [BJD - 2457000]')
                    axes.set_ylabel('relative flux')
                    # format_axes(axes, xlabel=True, ylabel=True)
                    ticid_label(axes, ticid_test[inds[c]])
                    axes.set_ylim(ymin = ymin, ymax = ymax)
                    # fig.tight_layout()
                    fig.savefig('./image-' + f'{n:03}.png')
                    plt.close(fig)
                    n += 1
        os.system('ffmpeg -framerate 2 -i ./image-%03d.png -pix_fmt yuv420p '+\
                  out_dir+str(c)+'ind-movie.mp4')

def training_test_plot(x, x_train, x_test, y_train_classes, y_test_classes,
                       y_predict, num_classes, out, ticid_train, ticid_test,
                       mock_data=False):
    # !! add more rows
    colors = ['r', 'g', 'b', 'm'] # !! add more colors
    # >> training data set
    fig, ax = plt.subplots(nrows = 7, ncols = num_classes, figsize=(15,10),
                           sharex=True)
    plt.subplots_adjust(hspace=0)
    # >> test data set
    fig1, ax1 = plt.subplots(nrows = 7, ncols = num_classes, figsize=(15,10),
                             sharex=True)
    plt.subplots_adjust(hspace=0)
    for i in range(num_classes): # >> loop through classes
        inds = np.nonzero(y_train_classes == i)[0]
        inds1 = np.nonzero(y_test_classes == i)[0]
        for j in range(min(7, len(inds))): # >> loop through rows
            ax[j,i].plot(x, x_train[inds[j]], '.'+colors[i], markersize=2)
            if not mock_data:
                ticid_label(ax[j,i], ticid_train[inds[j]])
        for j in range(min(7, len(inds1))):
            ax1[j,i].plot(x, x_test[inds1[j]], '.'+colors[y_predict[inds1[j]]],
                          markersize=2)
            if not mock_data:
                ticid_label(ax1[j,i], ticid_test[inds1[j]])    
            ax1[j,i].text(0.98, 0.02, 'True: '+str(i)+'\nPredicted: '+\
                          str(y_predict[inds1[j]]),
                          transform=ax1[j,i].transAxes, fontsize='xx-small',
                          horizontalalignment='right',
                          verticalalignment='bottom')
    for i in range(num_classes):
        ax[0,i].set_title('True class '+str(i), color=colors[i])
        ax1[0,i].set_title('True class '+str(i), color=colors[i])
        
        for axis in [ax[-1,i], ax1[-1,i]]:
            axis.set_xlabel('time [BJD - 2457000]', fontsize='small')
    for j in range(7):
        for axis in [ax[j,0],ax1[j,0]]:
            axis.set_ylabel('relative\nflux', fontsize='small')
            
    for axis in  ax.flatten():
        format_axes(axis)
    for axis in ax1.flatten():
        format_axes(axis)
    # fig.tight_layout()
    # fig1.tight_layout()
    fig.savefig(out+'train.png')
    fig1.savefig(out+'test.png')
    plt.close(fig)
    plt.close(fig1)

def plot_lof(time, intensity, targets, features, n, path,
             momentum_dump_csv = './Table_of_momentum_dumps.csv',
             n_neighbors=20,
             prefix='', mock_data=False, addend=1., feature_vector=False,
             n_tot=200):
    """ Adapted from Lindsey Gordon's feature_functions.py
    Plots the 20 most and least interesting light curves based on LOF.
    Parameters:
        * time : array with shape 
        * intensity
        * targets : list of TICIDs
        * feature vector
        * n : number of curves to plot in each figure
        * n_tot : total number of light curves to plots (number of figures =
                  n_tot / n)
        * path : output directory
    """
    from sklearn.neighbors import LocalOutlierFactor

    # -- calculate LOF -------------------------------------------------------
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)
    fit_predictor = clf.fit_predict(features)
    negative_factor = clf.negative_outlier_factor_
    
    lof = -1 * negative_factor
    ranked = np.argsort(lof)
    largest_indices = ranked[::-1][:n_tot] # >> outliers
    smallest_indices = ranked[:n_tot] # >> inliers
    
    # >> save LOF values in txt file 
    with open(path+'lof-'+prefix+'.txt', 'w') as f:
        for i in range(len(targets)):
            f.write('{} {}\n'.format(targets[i], lof[i]))
            
    # >> make histogram of LOF values
    plt.figure()
    plt.hist(lof, bins=50)
    plt.ylabel('Number of light curves')
    plt.xlabel('Local Outlier Factor (LOF)')
    plt.savefig(path+'lof-histogram.png')
    plt.close()
    
    # -- momentum dumps ------------------------------------------------------
    # >> get momentum dump times
    with open(momentum_dump_csv, 'r') as f:
        lines = f.readlines()
        mom_dumps = [ float(line.split()[3][:-1]) for line in lines[6:] ]
        inds = np.nonzero((mom_dumps >= np.min(time)) * \
                          (mom_dumps <= np.max(time)))
        mom_dumps = np.array(mom_dumps)[inds]

    # -- plot smallest and largest LOF light curves --------------------------
    num_figs = int(n_tot/n) # >> number of figures to generate
    
    for j in range(num_figs):
        
        for i in range(2): # >> loop through smallest and largest LOF plots
            fig, ax = plt.subplots(n, 1, sharex=True, figsize = (8, 3*n))
            
            for k in range(n): # >> loop through each row
                if i == 0: ind = largest_indices[j*n + k]
                elif i == 1: ind = smallest_indices[j*n + k]\
                
                # >> plot momentum dumps
                for t in mom_dumps:
                    ymin = 0.85*np.min(intensity[ind])
                    ymax = 1.15*np.max(intensity[ind])
                    # ax[k].plot([t,t], [0, 1], '--g', alpha=0.5)
                    ax[k].plot([t,t], [ymin, ymax], '--g', alpha=0.5)
                    
                # >> plot light curve
                ax[k].plot(time, intensity[ind] + addend, '.k', markersize=2)
                ax[k].text(0.98, 0.02, '%.3g'%lof[ind],
                           transform=ax[k].transAxes,
                           horizontalalignment='right',
                           verticalalignment='bottom',
                           fontsize='xx-small')
                format_axes(ax[k], ylabel=True)
                if not mock_data:
                    ticid_label(ax[k], targets[ind], title=True)
    
            # >> label axes
            if feature_vector:
                ax[n-1].set_xlabel('\u03C8')
            else:
                ax[n-1].set_xlabel('time [BJD - 2457000]')
                
            # >> save figures
            if i == 0:
                fig.suptitle(str(n) + ' largest LOF targets', fontsize=16,
                             y=0.9)
                fig.savefig(path + 'lof-' + prefix + 'kneigh' + \
                            str(n_neighbors) + '-largest_' + str(j*n) + 'to' +\
                            str(j*n + n) + '.png',
                            bbox_inches='tight')
                plt.close(fig)
            elif i == 1:
                fig.suptitle(str(n) + ' smallest LOF targets', fontsize=16,
                             y=0.9)
                fig.savefig(path + 'lof-' + prefix + 'kneigh' + \
                            str(n_neighbors) + '-smallest' + str(j*n) + 'to' +\
                            str(j*n + n) + '.png',
                            bbox_inches='tight')
                plt.close(fig)
                    
    # -- plot n random LOF light curves --------------------------------------
    fig, ax = plt.subplots(n, 1, sharex=True, figsize = (8, 3*n))   
                 
    for k in range(n):
        ind = np.random.choice(range(len(lof)-1))
            
        # >> plot momentum dumps
        for t in mom_dumps:
            ymin = 0.85*np.min(intensity[ind])
            ymax = 1.15*np.max(intensity[ind])
            # ax[k].plot([t,t], [0, 1], '--g', alpha=0.5)
            ax[k].plot([t,t], [ymin, ymax], '--g', alpha=0.5)
            
        # >> plot light curve
        ax[k].plot(time, intensity[ind] + addend, '.k', markersize=2)
        ax[k].text(0.98, 0.02, '%.3g'%lof[ind], transform=ax[k].transAxes,
                   horizontalalignment='right', verticalalignment='bottom',
                   fontsize='xx-small')
        
        # >> formatting
        format_axes(ax[k], ylabel=True)
        if not mock_data:
            ticid_label(ax[k], targets[ind], title=True)
    if feature_vector:
        ax[n-1].set_xlabel('\u03C8')
    else:
        ax[n-1].set_xlabel('time [BJD - 2457000]')     
    fig.suptitle(str(n) + ' random LOF targets', fontsize=16, y=0.9)
    
    # >> save figure
    fig.savefig(path + 'lof-' + prefix + 'kneigh' + str(n_neighbors) \
                + "-random.png", bbox_inches='tight')
    plt.close(fig)
    
def hyperparam_opt_diagnosis(analyze_object, output_dir, supervised=False):
    import pandas as pd
    import matplotlib.pyplot as plt
    # analyze_object = talos.Analyze('talos_experiment.csv')
    
    print(analyze_object.data)
    print(analyze_object.low('val_loss'))
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    df = analyze_object.data
    print(df.iloc[[np.argmin(df['val_loss'])]])
    
    with open(output_dir + 'best_params.txt', 'a') as f: 
        best_param_ind = np.argmin(df['val_loss'])
        f.write(str(df.iloc[best_param_ind]) + '\n')
    
    if supervised:
        label_list = ['val_loss', 'val_acc', 'val_precision',
                      'val_recall']
        key_list = ['val_loss', 'val_accuracy', 'val_precision_1',
                    'val_recall_1']
    else:
        label_list = ['val_loss']
        key_list = ['val_loss']
        
    for i in range(len(label_list)):
        analyze_object.plot_line(key_list[i])
        plt.xlabel('round')
        plt.ylabel(label_list[i])
        plt.savefig(output_dir + label_list[i] + '_plot.png')
    
    # >> kernel density estimation
    analyze_object.plot_kde('val_loss')
    plt.xlabel('val_loss')
    plt.ylabel('kernel density\nestimation')
    plt.savefig(output_dir + 'kde.png')
    
    analyze_object.plot_hist('val_loss', bins=50)
    plt.xlabel('val_loss')
    plt.ylabel('num observations')
    plt.tight_layout()
    plt.savefig(output_dir + 'hist_val_loss.png')
    
    # >> heat map correlation
    analyze_object.plot_corr('val_loss', ['acc', 'loss', 'val_acc'])
    plt.tight_layout()
    plt.savefig(output_dir + 'correlation_heatmap.png')
    
    # >> get best parameter set
    hyperparameters = list(analyze_object.data.columns)
    for col in ['round_epochs', 'val_loss', 'val_accuracy', 'val_precision_1',
            'val_recall_1', 'loss', 'accuracy', 'precision_1', 'recall_1']:
        hyperparameters.remove(col)
        
    p = {}
    for key in hyperparameters:
        p[key] = df.iloc[best_param_ind][key]
    
    return df, best_param_ind, p

def plot_reconstruction_error(time, intensity, x_test, x_predict, ticid_test,
                              output_dir='./', addend=1., mock_data=False,
                              feature_vector=False, n=20):
    '''For autoencoder, intensity = x_test'''
    # >> calculate reconstruction error (mean squared error)
    err = (x_test - x_predict)**2
    err = np.mean(err, axis=1)
    err = err.reshape(np.shape(err)[0])
    
    # >> get top n light curves
    ranked = np.argsort(err)
    largest_inds = ranked[::-1][:n]
    smallest_inds = ranked[:n]
    
    # >> save in txt file
    with open(output_dir+'reconstruction_error.txt', 'w') as f:
        for i in range(len(ticid_test)):
            f.write('{} {}\n'.format(ticid_test[i], err[i]))
    
    for i in range(2):
        fig, ax = plt.subplots(n, 1, sharex=True, figsize = (8, 3*n))
        for k in range(n): # >> loop through each row
            if i == 0: ind = largest_inds[k]
            else: ind = smallest_inds[k]
            
            # >> plot light curve
            ax[k].plot(time, intensity[ind]+addend, '.k', markersize=2)
            if not feature_vector:
                ax[k].plot(time, x_predict[ind]+addend, '-')
            ax[k].text(0.98, 0.02, 'mse: ' +str(err[ind]),
                       transform=ax[k].transAxes, horizontalalignment='right',
                       verticalalignment='bottom', fontsize='xx-small')
            format_axes(ax[k], ylabel=True)
            if not mock_data:
                ticid_label(ax[k], ticid_test[ind], title=True)
                
        if feature_vector:
            ax[n-1].set_xlabel('\u03C8')
        else:
            ax[n-1].set_xlabel('time [BJD - 2457000]')
        if i == 0:
            fig.suptitle('largest reconstruction error', fontsize=16, y=0.9)
            fig.savefig(output_dir + 'reconstruction_error-largest.png',
                        bbox_inches='tight')
        else:
            fig.suptitle('smallest reconstruction error', fontsize=16, y=0.9)
            fig.savefig(output_dir + 'reconstruction_error-smallest.png',
                        bbox_inches='tight')            
    
def plot_classification(time, intensity, targets, labels, path,
             momentum_dump_csv = './Table_of_momentum_dumps.csv',
             n=20,
             prefix='', mock_data=False, addend=1., feature_vector=False):
    """ 
    """

    classes, counts = np.unique(labels, return_counts=True)
    
    # >> get momentum dump times
    with open(momentum_dump_csv, 'r') as f:
        lines = f.readlines()
        mom_dumps = [ float(line.split()[3][:-1]) for line in lines[6:] ]
        inds = np.nonzero((mom_dumps >= np.min(time)) * \
                          (mom_dumps <= np.max(time)))
        mom_dumps = np.array(mom_dumps)[inds]
        
    for i in range(len(classes)): # >> loop through each class
        fig, ax = plt.subplots(n, 1, sharex=True, figsize = (8, 3*n))
        class_inds = np.nonzero(labels == classes[i])[0]
        if classes[i] == 0:
            color = 'red'
        elif classes[i] == -1:
            color = 'black'
        elif classes[i] == 1:
            color = 'blue'
        elif classes[i] == 2:
            color = 'green'
        else:
            color = 'purple'
        
        for k in range(min(n, counts[i])): # >> loop through each row
            ind = class_inds[k]
            
            # >> plot momentum dumps
            for t in mom_dumps:
                ax[k].plot([t,t], [0, 1], '--g', alpha=0.5,
                           transform=ax[k].transAxes)            
            
            # >> plot light curve
            ax[k].plot(time, intensity[ind] + addend, '.k', markersize=2)
            ax[k].text(0.98, 0.02, str(labels[ind]), transform=ax[k].transAxes,
                       horizontalalignment='right', verticalalignment='bottom',
                       fontsize='xx-small')
            format_axes(ax[k], ylabel=True)
            if not mock_data:
                ticid_label(ax[k], targets[ind], title=True)

        if feature_vector:
            ax[n-1].set_xlabel('\u03C8')
        else:
            ax[n-1].set_xlabel('time [BJD - 2457000]')
    
        if classes[i] == -1:
            fig.suptitle('Class -1 (outliers)', fontsize=16, y=0.9,
                         color=color)
        else:
            fig.suptitle('Class ' + str(classes[i]), fontsize=16, y=0.9,
                         color=color)
        fig.savefig(path + prefix + '-class' + str(classes[i]) + '.png',
                    bbox_inches='tight')
        plt.close(fig)
        
def plot_pca(bottleneck, classes, n_components=2, output_dir='./'):
    from sklearn.decomposition import PCA
    import pandas as pd
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(bottleneck)
    # principalDf = pd.DataFrame(data = principalComponents,
    #                            columns=['principal component 1',
    #                                     'principal component 2'])
    fig, ax = plt.subplots()
    ax.set_ylabel('Principal Component 1')
    ax.set_xlabel('Principal Component 2')
    ax.set_title('2 component PCA')
    
    # >> loop through classes
    class_labels = np.unique(classes)
    for i in range(len(class_labels)):
        inds = np.nonzero(classes == class_labels[i])
        if class_labels[i] == 0:
            color='r'
        elif class_labels[i] == 1:
            color = 'b'
        elif class_labels[i] == 2:
            color='g'
        elif class_labels[i] == 3:
            color='m'
        else:
            color='k'
        
        ax.plot(principalComponents[inds][:,0], principalComponents[inds][:,1],
                '.'+color, markersize=2)
    fig.savefig(output_dir + 'PCA_plot.png')

# == helper functions =========================================================

def ticid_label(ax, ticid, title=False):
    '''Query catalog data and add text to axis.'''

    # >> query catalog data
    target, Teff, rad, mass, GAIAmag, d, objType = get_features(ticid)
    
    # >> change sigfigs for effective temperature
    if np.isnan(Teff):
        Teff = 'nan'
    else: Teff = '%.4d'%Teff
    
    info = target+'\nTeff {}\nrad {}\nmass {}\nG {}\nd {}\nO {}'
    info1 = target+', Teff {}, rad {}, mass {},\nG {}, d {}, O {}'
    
    # >> make text
    if title:
        ax.set_title(info1.format(Teff, '%.2g'%rad, '%.2g'%mass, 
                                  '%.3g'%GAIAmag, '%.3g'%d, objType),
                     fontsize='xx-small')
    else:
        ax.text(0.98, 0.98, info.format(Teff, '%.2g'%rad, '%.2g'%mass, 
                                        '%.3g'%GAIAmag, '%.3g'%d, objType),
                  transform=ax.transAxes, horizontalalignment='right',
                  verticalalignment='top', fontsize='xx-small')
    
def get_features(ticid):
    '''Query catalog data https://arxiv.org/pdf/1905.10694.pdf'''
    from astroquery.mast import Catalogs

    target = 'TIC '+str(int(ticid))
    catalog_data = Catalogs.query_object(target, radius=0.02, catalog='TIC')
    Teff = catalog_data[0]["Teff"]

    rad = catalog_data[0]["rad"]
    mass = catalog_data[0]["mass"]
    GAIAmag = catalog_data[0]["GAIAmag"]
    d = catalog_data[0]["d"]
    # Bmag = catalog_data[0]["Bmag"]
    # Vmag = catalog_data[0]["Vmag"]
    objType = catalog_data[0]["objType"]
    # Tmag = catalog_data[0]["Tmag"]
    # lum = catalog_data[0]["lum"]

    return target, Teff, rad, mass, GAIAmag, d, objType
    
def format_axes(ax, xlabel=False, ylabel=False):
    '''Helper function to plot TESS light curves. Aspect ratio is 3/8.
    Parameters:
        * ax : matplotlib axis'''
    # >> force aspect = 3/8
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.set_aspect(abs((xlim[1]-xlim[0])/(ylim[1]-ylim[0])*(3./8.)))
    # ax.set_aspect(3./8., adjustable='box')
    
    if list(ax.get_xticklabels()) == []:
        ax.tick_params('x', bottom=False) # >> remove ticks if no label
    else:
        ax.tick_params('x', labelsize='small')
    ax.tick_params('y', labelsize='small')
    ax.ticklabel_format(useOffset=False)
    if xlabel:
        ax.set_xlabel('Time [BJD - 2457000]')
    if ylabel:
        ax.set_ylabel('Relative flux')
    
def get_activations(model, x_test, input_rms = False, rms_test = False):
    '''Returns intermediate activations.'''
    from keras.models import Model
    layer_outputs = [layer.output for layer in model.layers][1:]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    if input_rms:
        activations = activation_model.predict([x_test, rms_test])
    else:
        activations = activation_model.predict(x_test)
    return activations

def get_bottleneck(model, activations, p, input_features=False, 
                   features=False, input_rms=False, rms=False):
    '''Get bottleneck layer, with shape (num light curves, latent dimension)
    Parameters:
        * model : Keras Model()
        * activations : from get_activations()
        * p : parameter set, with p['latent_dim'] = dimension of latent space
        * input_features : bool
        * features : array of features to concatenate with bottleneck, must be
                     given if input_features=True
        * rms : list of RMS must be given if input_rms=True
    '''
    # >> first find all Dense layers
    inds = np.nonzero(['dense' in x.name for x in model.layers])[0]
    
    # >> now check which Dense layers has number of units = latent_dim
    for ind in inds:
        ind = ind - 1 # >> len(activations) = len(model.layers) - 1, since
                      #    activations doesn't include the Input layer
        num_units = np.shape(activations[ind])[1]
        if num_units == p['latent_dim']:
            bottleneck_ind = ind
    
    bottleneck = activations[bottleneck_ind]
    
    if input_features: # >> concatenate features to bottleneck
        bottleneck = np.concatenate([bottleneck, input_features], axis=1)
    if input_rms:
        bottleneck = np.concatenate([bottleneck,
                                     np.reshape(rms, (np.shape(rms)[0],1))],
                                    axis=1)
        
    return bottleneck

def corner_plot(activation, p, n_bins = 50, log = True):
    '''Creates corner plot.'''
    from matplotlib.colors import LogNorm
    
    latentDim = p['latent_dim']

    fig, axes = plt.subplots(nrows = latentDim, ncols = latentDim,
                             figsize = (10, 10))

    # >> deal with 1 latent dimension case
    if latentDim == 1:
        axes.hist(np.reshape(activation, np.shape(activation)[0]), n_bins,
                  log=log)
        axes.set_ylabel('\u03C61')
        axes.set_ylabel('frequency')
    else:
        # >> row 1 column 1 is first latent dimension (phi1)
        for i in range(latentDim):
            axes[i,i].hist(activation[:,i], n_bins, log=log)
            axes[i,i].set_aspect(aspect=1)
            for j in range(i):
                if log:
                    norm = LogNorm()
                axes[i,j].hist2d(activation[:,j], activation[:,i],
                                 bins=n_bins, norm=norm)
                # >> remove axis frame of empty plots
                axes[latentDim-1-i, latentDim-1-j].axis('off')

            # >> x and y labels
            axes[i,0].set_ylabel('\u03C6' + str(i))
            axes[latentDim-1,i].set_xlabel('\u03C6' + str(i))

        # >> removing axis
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        plt.subplots_adjust(hspace=0, wspace=0)

    return fig, axes
     
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
