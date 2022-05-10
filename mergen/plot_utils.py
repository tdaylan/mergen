"""
Created on Thu Jun  4 21:58:45 2020

Plotting functions only. 

@author: Lindsey Gordon @lcgordon and Emma Chickles @emmachickles

Last updated: Feb 26 2021

Wrapper functions
    * produce_feature_visualizations
    * produce_clustering_visualizations
    * produce_novelty_visualizations
    * produce_ae_visualizations

Feature space visualizations
    * features_plotting_2D : n-choose-2 plots of pairs of features against
                             each other.
                             * can be colored based on classification results.
    * features2D_with_insets : same as features_plotting_2D but with insets of
                               the extrema light curves
    * histo_features       : produces histograms of all features
    * features_insets
    * inset_plotting
    * inset_plotting_colored
    * plot_pca
    * latent_space_plot
    * plot_tsne
    * presentation_feature

Novelty visualizations
    * generate_novelty_scores !!
    * load_novelty_scores !!
    * plot_lof             : plots the n top, bottom, and random light curves
                             ranked on their LOF scores
    * plot_nvlty_lc
    * isolate_plot_feature_outliers
    * lof_and_insets_on_sector
    * plot_lof_summary
    * paper_plot_lof
    * presentation_LOF_by_feature
    * presentation_plot_classifications
    * plot_lof_2col

Clustering visualizations
    * two_years_ensemble_summary
    * ensemble_summary_plots
    * ensemble_budget
    * plot_confusion_matrix
    * classification_plots
    * quick_plot_classification : to plot the first n in each class
    * plot_paramscan_metrics
    * plot_paramscan_classes
    * classification_diagnosis
    * plot_classification
    * plot_cross_identifications
    * plot_clusters
    * evaluate_classifications
    * plot_fail_cases
    * sector_dists

Autoencoder visualizations
    * hyperparam_opt_diagnosis
    * plot_filter_vis
    * plot_bottleneck_vis
    * plot_saliency_map
    * diagnostic_plots
    * epoch_plots
    * input_output_plot
    * kernel_filter_plot
    * intermed_act_plot
    * input_bottleneck_output_plot
    * movie
    * training_test_plot
    * reconstruction_error_power
    * plot_fail_reconstructions
    * presentation_act
    * presentation_kernel_plots
    * presentation_validation

Preprocessing visualizations
    * sector_nan_mask_diag

Helper functions
    * ENF_labels
    * CAE_labels
    * get_colors
    * astroquery_pull_data !!
    * get_extrema
    * plot_histogram
    * plot_lygos
    * ticid_label
    * classification_label
    * format_axes
    * plot_light_curves
    * plot_lc

To Do List: 
    - Reorganize by importance
    - Easy forward-facing function calls
    - Better documentation on all functions
    - Move imports to init
    - Merge repeated functions
    - fix insets on histograms    

"""

import numpy as np
import numpy.ma as ma 
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)                
import pdb # >> debugging tool
# import model as ml
from . import learn_utils as lt
from astropy.timeseries import LombScargle


import scipy.signal as signal
from scipy.stats import moment
from scipy import stats
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
rcParams["lines.markersize"] = 2
# rcParams['lines.color'] = 'k'
from scipy.signal import argrelextrema

from matplotlib import rc
rc("text", usetex=False)

import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import fnmatch

from datetime import datetime
import os
import shutil
from scipy.stats import moment, sigmaclip

# import seaborn as sn

# import data_functions as df
from . import data_utils as dt
import random

def produce_feature_visualizations():
    return

def produce_clustering_visualizations(feats, numtot, numpot, tsne, output_dir,
                                      otdict, objid, sector, datapath, metapath,
                                      prefix='', anim=False, elev=45,
                                      crot_analysis=False):
    # prefix = 'perplexity'+str(perplexity)+'_elev'+str(elev)+'_'
    output_dir = output_dir + 'imgs/'
    dt.create_dir(output_dir)

    # ensemble_summary_plots(objid, numtot, numpot, otdict, sector,
    #                        datapath, output_dir, prefix, metapath)
    # pdb.set_trace()

    # >> color with clustering results
    prefix = 'pred_'
    plot_tsne(feats, numpot, X=tsne, output_dir=output_dir,
              prefix=prefix, animate=anim, elev=elev, otypedict=otdict)

    # >> color with classifications from GCVS, SIMBAD, ASAS-SN
    prefix = 'true_'
    plot_tsne(feats, numtot, X=tsne, output_dir=output_dir,
              prefix=prefix, animate=anim, elev=elev, otypedict=otdict)

    # >> specific science case: complex rotators
    # !! TODO : input list of objects [CROT, ...]
    if crot_analysis:
        prefix='crot_'
        # >> known complex rotators (sectors 1, 2)
        # targets      = [38820496, 177309964, 206544316, 234295610, 289840928,
        #                 425933644, 425937691] # >> sector 1 only
        targets = [38820496, 177309964, 201789285, 206544316, 224283342,\
                   234295610, 289840928, 332517282, 425933644, 425937691]
        numcot = [] # >> numerized complex rotator ensemble object type
        cotd = {0: 'NONE', 1:'CROT'}
        for ticid in objid:
            if int(ticid) in targets:
                numcot.append(1)
            else:
                numcot.append(0)
        plot_tsne(feats, numcot, X=tsne, output_dir=output_dir,
                  prefix=prefix, animate=anim, elev=elev, otypedict=cotd,
                  class_marker='x', class_ms=20, debug=True)

    return

def produce_novelty_visualizations(lof, output_dir, objid, sector, feats,
                                   datapath, tsne=None, mdumpcsv=None, bins=20,
                                   datatype='SPOC'):
    print("Producing novelty visualizations...")
    # plot_histogram(lof, bins=bins, x_label="Local Outlier Factor (LOF)",
    #                filename=output_dir+'lof-histogram.png', insetx=time,
    #                insety=flux, targets=objid)
    # plot_histogram(lof, bins=bins, x_label="Local Outlier Factor (LOF)",
    #                filename=output_dir+'nvlty/lof-histogram.png', insets=False)

    # # >> plot tsne
    # fig, ax = plt.subplots()
    # ax.plot(tsne[:,0], tsne[:,1], '.k', ms=2)
    # for i in range(20):
    #     ind = np.argsort(lof)[-i-1]
    #     ax.plot(tsne[ind,0], tsne[ind,1], 'xr', ms=20)
    # fig.tight_layout()
    # fig.savefig(output_dir+'nvlty/lof_tsne.png')
    # plt.close(fig)
    # print('Wrote '+output_dir+'nvlty/lof_tsne.png')

    # # >> plot light curves
    # plot_nvlty_lc(output_dir, datapath, lof, objid, sector, feats,
    #               mdumpcsv=mdumpcsv, datatype=datatype)

    # >> plot saliency maps
    objid_targ = [140512085]
    plot_saliency_map(output_dir, objid, sector, objid_targ,
                      bottleneck_name='bottleneck', feat=0, smooth_samples=20,
                      smooth_noise=0.20, 
                      log=False)
    return

def produce_ae_visualizations(x, x_train, x_pred, output_dir, ticid, target_info,
                              psd=False):

    fig, ax = input_output_plot(x, x_train, x_pred,
                                output_dir+'input_output.png', ticid_test=ticid,
                                target_info=target_info, psd=psd)

    plot_reconstruction_error(x, x_train, x_pred, ticid, output_dir=output_dir,
                              target_info=target_info, psd=psd)

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Feature Space Visualizations ::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def features_plotting_2D(savepath, features, labels = None, engineered_features = True, version = 0,
                         clustering = 'unclustered', plot_LC_classes = None):
    """ Plot (n 2) features against each other to visualize distribution in feature space.
    Parameters:
        * savepath: save path for subfolder containing all plots to be saved into
        * features: array of all feature vectors for all targets
        * labels: classifier labels to use in color coded plotting
        * engineered_features: True if using ENF features, False if CAE features. Affects labels.
        * version: if using ENF features, which version (0/1/2). Value irrelevant for CAE features.
        * clustering: clustering algorithm used (optional). doubles as prefix on filelabels
        * plot_LC_classes: provides information to plot light curve classes, only runs if clustering != None
            structure like: [time_axis, rel_fluxes, target_labels, target_info, momentum_dump_csv]
            
    Returns: nothing
    
    """ 
    rcParams['figure.figsize'] = 10,10
        
    #makes folder and saves to it    
    folder_path = savepath + clustering + '/'
    try:
        os.makedirs(folder_path)
    except OSError:
        print ("Save directory already exists")
        
    #optional clustering plotting
    if plot_LC_classes is not None:
        time = plot_LC_classes[0]
        intensity = plot_LC_classes[1]
        targets = plot_LC_classes[2]
        target_info = plot_LC_classes[3]
        momentum_dump = plot_LC_classes[4]
        plot_classification(time, intensity, targets, labels,
                            folder_path+'/', prefix=clustering,
                            momentum_dump_csv=momentum_dump,
                            target_info=target_info)
        plot_pca(features, labels, output_dir=folder_path+'/')
    
    colors = get_colors()
    #Create labels
    if engineered_features: #ENF features
        print("Using ENF features")
        graph_labels, fname_labels = ENF_labels(version=version)
        num_features = len(features[0])
    else: #CAE features
        print("Using CAE features")
        num_features = np.shape(features)[1]
        graph_labels, fname_labels = CAE_labels(num_features)
     
    if labels is None:
        labels = np.ones((1, num_features)) * -1
    
    for n in range(num_features):
        feat1 = features[:,n]
        for m in range(num_features):
            if m == n:
                continue                
            feat2 = features[:,m]
 
            plt.figure()
            plt.clf()
            for n in range(len(features)):
                plt.scatter(feat1[n], feat2[n], c=colors[labels[n]], s=2)
            plt.xlabel(graph_labels[n])
            plt.ylabel(graph_labels[m])
            plt.savefig((folder_path+ fname_labels[n] + "-" + fname_labels[m]  + "-" + clustering + ".png"))
            plt.close()


def features2D_with_insets(savepath, time, intensity, targets, features, engineered_features = True,
                                 labels = None, classifier = "", version = 0):
    """ Similar to features_plotting_2D but plots the 2D distributions color-coded by class
    with the light curves of the extrema plotted along the top and bottom. Useful for visualizing outliers.
    
    Parameters: 
        * savepath: where you want the subfolder of all plots saved
        * time: time axis for all llight curve subplots
        * intensity: intensity arrays for all light curves
        * targets: labels for all light curves (in same order as intensities)
        * features: all features
        * engienered_features: CAE = False, ENF = True
        * labels: classification label array from the classifier used. 
            If None, will plot all as black/unclassified
        * classifier: string indicating classifier used, will be used in filenames
        * version
    
    Returns: nothing
    [Updated LCG 02272021]
    """   
    folderpath = savepath + "2DFeatures-insets-colored" + classifier + "/"
    try:
        os.makedirs(folderpath)
    except OSError:
        print ("Directory already exists.")
        
    if engineered_features: #ENF features
        print("Using ENF features")
        graph_labels, fname_labels = ENF_labels(version=version)
        num_features = len(features[0])
    else: #CAE features
        print("Using CAE features")
        num_features = np.shape(features)[1]
        graph_labels, fname_labels = CAE_labels(num_features)
    
    if labels is None:
        labels = np.ones((1,len(features))) * -1 #array of all -1 classes (same as no classifications)
    
    for n in range(num_features):
        for m in range(num_features):
            if m == n:
                continue 
                      
            filename = folderpath + classifier + "-" + fname_labels[n] + "-vs-" + fname_labels[m] + ".png"     
            inset_indexes = get_extrema(features, n, m)
            inset_plotting_colored(features[:,n], features[:,m], graph_labels[n], 
                                   graph_labels[m], time, intensity, inset_indexes, targets, filename, 
                                   labels, labels)
  
def histo_features(savepath, features, bins, engineered_features = True, version=0):
    """ Plots and saves a histogram of each feature. 
    Parameters:
        * savepath
        * features
        * bins for histogram
        * engineered_features: True= ENF, False = CAE
        * version: for what version of ENF (defaults to 0 and is ignored for CAE)
        
    Returns: nothing
    **** put the insets back in/fix insets for scaling **** """
    folderpath = savepath + "feature_histograms_binning_" + str(bins) + "/"
    try:
        os.makedirs(folderpath)
    except OSError:
        print ("Directory %s already exists" % folderpath)
    
    if engineered_features: #ENF features
        print("Using ENF features")
        graph_labels, fname_labels = ENF_labels(version=version)
        num_features = len(features[0])
    else: #CAE features
        print("Using CAE features")
        num_features = np.shape(features)[1]
        graph_labels, fname_labels = CAE_labels(num_features)

    for n in range(num_features):
        filename = folderpath + fname_labels[n] + "histogram.png"
        plot_histogram(features[:,n], bins, fname_labels[n])
    return
 
def features_insets(time, intensity, feature_vectors, targets, path, version = 0):
    """ Plots 2 features against each other with the extrema points' associated
    light curves plotted as insets along the top and bottom of the plot. 
    
    time is the time axis for the group
    intensity is the full list of intensities
    feature_vectors is the complete list of feature vectors
    targets is the complete list of targets
    folder is the folder into which you wish to save the folder of plots. it 
    should be formatted as a string, ending with a /
    modified [lcg 07202020]
    """   
    path = path + "2DFeatures-insets"
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
        print("New folder created will have -new at the end. Please rename.")
        path = path + "-new"
        os.makedirs(path)
    else:
        print ("Successfully created the directory %s" % path) 
        
    if version==0:
        graph_labels = ["Average", "Variance", "Skewness", "Kurtosis", "Log Variance",
                            "Log Skewness", "Log Kurtosis", "Maximum Power", "Log Maximum Power", 
                            "Period of Maximum Power (0.1 to 10 days)","Slope" , "Log Slope",
                            "P0", "P1", "P2", "Period of Maximum Power (0.001 to 0.1 days)"]
        fname_labels = ["Avg", "Var", "Skew", "Kurt", "LogVar", "LogSkew", "LogKurt",
                            "MaxPower", "LogMaxPower", "Period0_1to10", "Slope", "LogSlope",
                            "P0", "P1", "P2", "Period0to0_1"]
    elif version == 1: 
            
        graph_labels = ["TLS Best fit Period (days)", "TLS Best fit duration (days)", "TLS best fit depth (ppt from transit bottom",
                            "TLS Best fit Power"]
        fname_labels = ["TLSPeriod", "TLSDuration", "TLSDepth", "TLSPower"]

    for n in range(len(feature_vectors[0])):
        graph_label1 = graph_labels[n]
        fname_label1 = fname_labels[n]
        for m in range(len(feature_vectors[0])):
            if m == n:
                continue
            graph_label2 = graph_labels[m]
            fname_label2 = fname_labels[m]  

            filename = path + "/" + fname_label1 + "-vs-" + fname_label2 + ".png"     
            
            inset_indexes = get_extrema(feature_vectors, n,m)
            
            inset_plotting(feature_vectors[:,n], feature_vectors[:,m], graph_label1, graph_label2, time, intensity, inset_indexes, targets, filename)

def inset_plotting(datax, datay, label1, label2, insetx, insety, inset_indexes, targets, filename):
    """ Plots the extrema of a 2D feature plot as insets on the top and bottom border
    datax and datay are the features being plotted as a scatter plot beneath it
    label1 and label2 are the x and y labels
    insetx is the time axis for the insets
    insety is the complete list of intensities 
    inset_indexes are the identified extrema to be plotted
    targets is the complete list of target TICs
    filename is the exact path that the plot is to be saved to.
    modified [lcg 06302020]"""
    
    x_range = datax.max() - datax.min()
    y_range = datay.max() - datay.min()
    y_offset = 0.2 * y_range
    x_offset = 0.01 * x_range
    
    fig, ax1 = plt.subplots()

    ax1.scatter(datax,datay, s=2)
    ax1.set_xlim(datax.min() - x_offset, datax.max() + x_offset)
    ax1.set_ylim(datay.min() - y_offset,  datay.max() + y_offset)
    ax1.set_xlabel(label1)
    ax1.set_ylabel(label2)
    
    i_height = y_offset / 2
    i_width = x_range/4.5
    
    x_init = datax.min() 
    y_init = datay.max() + (0.4*y_offset)
    n = 0
    inset_indexes = inset_indexes[0:8]
    while n < (len(inset_indexes)):
        axis_name = "axins" + str(n)
        
    
        axis_name = ax1.inset_axes([x_init, y_init, i_width, i_height], transform = ax1.transData) #x pos, y pos, width, height
        axis_name.scatter(insetx, insety[inset_indexes[n]], c='black', s = 0.1, rasterized=True)
        
        #this sets where the pointer goes to
        x1, x2 = datax[inset_indexes[n]], datax[inset_indexes[n]] + 0.001*x_range
        y1, y2 =  datay[inset_indexes[n]], datay[inset_indexes[n]] + 0.001*y_range
        axis_name.set_xlim(x1, x2)
        axis_name.set_ylim(y1, y2)
        ax1.indicate_inset_zoom(axis_name)
              
        #this sets the actual axes limits    
        axis_name.set_xlim(insetx[0], insetx[-1])
        axis_name.set_ylim(insety[inset_indexes[n]].min(), insety[inset_indexes[n]].max())
        axis_name.set_title("TIC " + str(int(targets[inset_indexes[n]])) + " \n" + astroquery_pull_data(targets[inset_indexes[n]]), fontsize=6)
        axis_name.set_xticklabels([])
        axis_name.set_yticklabels([])
        
        x_init += 1.1* i_width
        n = n + 1
        
        if n == 4: 
            y_init = datay.min() - (0.8*y_offset)
            x_init = datax.min()
            
    plt.savefig(filename)   
    plt.close()


            
def inset_plotting_colored(datax, datay, label1, label2, insetx, insety, inset_indexes, targets, filename, realclasses, guessclasses):
    """ Plots the extrema of a 2D feature plot as insets on the top and bottom border
    Variant on inset_plotting. Colors insets by guessed classes, and the 
    connecting lines by the real classes.
    datax and datay are the features being plotted as a scatter plot beneath it
    label1 and label2 are the x and y labels
    insetx is the time axis for the insets
    insety is the complete list of intensities 
    inset_indexes are the identified extrema to be plotted
    targets is the complete list of target TICs
    filename is the exact path that the plot is to be saved to.
    realclasses is the array of hand labeled classes
    guessclasses are the predicted classes
    modified [lcg 06302020]"""
    
    x_range = datax.max() - datax.min()
    y_range = datay.max() - datay.min()
    y_offset = 0.2 * y_range
    x_offset = 0.01 * x_range
    colors = get_colors()
    
    fig, ax1 = plt.subplots()
    
    for n in range(len(datax)):
        c = colors[int(guessclasses[n])]
        ax1.scatter(datax[n], datay[n], s=2)

    ax1.set_xlim(datax.min() - x_offset, datax.max() + x_offset)
    ax1.set_ylim(datay.min() - y_offset,  datay.max() + y_offset)
    ax1.set_xlabel(label1)
    ax1.set_ylabel(label2)
    
    i_height = y_offset / 2
    i_width = x_range/4.5
    
    x_init = datax.min() 
    y_init = datay.max() + (0.4*y_offset)
    n = 0
    inset_indexes = inset_indexes[0:8]
    
    while n < (len(inset_indexes)):
        axis_name = "axins" + str(n)
        real_class = int(realclasses[inset_indexes[n]])
        guessed_class = int(guessclasses[inset_indexes[n]])
        
    
        axis_name = ax1.inset_axes([x_init, y_init, i_width, i_height], transform = ax1.transData) #x pos, y pos, width, height
        axis_name.scatter(insetx, insety[inset_indexes[n]], c=colors[guessed_class], s = 0.1, rasterized=True)
        
        #this sets where the pointer goes to
        x1, x2 = datax[inset_indexes[n]], datax[inset_indexes[n]] + 0.001*x_range
        y1, y2 =  datay[inset_indexes[n]], datay[inset_indexes[n]] + 0.001*y_range
        axis_name.set_xlim(x1, x2)
        axis_name.set_ylim(y1, y2)
        ax1.indicate_inset_zoom(axis_name, edgecolor=colors[real_class])
              
        #this sets the actual axes limits    
        axis_name.set_xlim(insetx[0], insetx[-1])
        axis_name.set_ylim(insety[inset_indexes[n]].min(), insety[inset_indexes[n]].max())
        axis_name.set_title("TIC " + str(int(targets[inset_indexes[n]])) + " " + astroquery_pull_data(targets[inset_indexes[n]]), fontsize=8)
        axis_name.set_xticklabels([])
        axis_name.set_yticklabels([])
        
        x_init += 1.1* i_width
        n = n + 1
        
        if n == 4: 
            y_init = datay.min() - (0.8*y_offset)
            x_init = datax.min()
            
    plt.savefig(filename)
    plt.close()

def plot_pca(bottleneck, classes, n_components=2, output_dir='./', prefix=''):
    from sklearn.decomposition import PCA
    import pandas as pd
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(bottleneck)
    fig, ax = plt.subplots()
    ax.set_ylabel('Principal Component 1')
    ax.set_xlabel('Principal Component 2')
    ax.set_title('2 component PCA')
    colors = get_colors() 
    # >> loop through classes
    class_labels = np.unique(classes)
    for i in range(len(class_labels)):
        inds = np.nonzero(classes == class_labels[i])
        if class_labels[i] == -1:
            color = 'black'
        elif class_labels[i] < len(colors)-1:
            color = colors[class_labels[i]]
        else:
            color='black'
        
        ax.plot(principalComponents[inds][:,0], principalComponents[inds][:,1],
                '.', color=color)
    fig.savefig(output_dir + prefix + 'PCA_plot.png')
    plt.close(fig)

def latent_space_plot(activation, out='./latent_space.png', n_bins = 50,
                      log = True, save=True,
                      units='phi', figsize=(10,10), fontsize='x-small'):
    '''Creates corner plot of latent space.
        Parameters:
        * bottleneck : bottleneck layer, shape=(num light curves, num features)
        * out : output filename
        * n_bins : number of bins in histogram (int)
        * log : if True, plots log histogram
        * units : either 'phi' (learned features), or 'psi'
          (engineered features)
    '''
        
    from matplotlib.colors import LogNorm
    
    latentDim = np.shape(activation)[1]

    fig, axes = plt.subplots(nrows = latentDim, ncols = latentDim,
                             figsize = figsize)

    if units == 'phi':
        ax_label = '\u03C6'
    elif units == 'psi':
        ax_label = '\u03C8'

    # >> deal with 1 latent dimension case
    if latentDim == 1:
        axes.hist(np.reshape(activation, np.shape(activation)[0]), n_bins,
                  log=log)
        axes.set_xlabel(ax_label + '1')
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
            axes[i,0].set_ylabel(ax_label + str(i), fontsize=fontsize)
            axes[latentDim-1,i].set_xlabel(ax_label + str(i),
                                           fontsize=fontsize)

        # >> removing axis
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        plt.subplots_adjust(hspace=0, wspace=0)
        
    if save:
        plt.savefig(out)
        plt.close(fig)
    
    return fig, axes
    # return fig, axes
    
    
def plot_tsne(bottleneck, labels, X=None, n_components=2, output_dir='./',
              prefix='', animate=False, elev=10, otypedict=None, alpha=0.5,
              class_marker='.', class_ms=5, debug=False):
    if type(X) == type(None):
        from sklearn.manifold import TSNE
        X = TSNE(n_components=n_components).fit_transform(bottleneck)
    unique_classes = np.unique(labels)
    # unique_classes = np.array(list(otypedict.keys()))
    colors = get_colors()

    # >> find 'center' of t-SNE
    centr = np.median(X, axis=0)
    maxlim = np.max(X, axis=0)
    minlim = np.min(X, axis=0)
    rad = 0.5 * np.sqrt(np.sum((maxlim - minlim)**2)) # >> radius of tSNE 

    fig = plt.figure()
    if X.shape[1] == 2:
        ax = fig.add_subplot()
    elif X.shape[1] == 3:
        ax = fig.add_subplot(projection='3d')
        
    for i in unique_classes:
        # >> find all light curves with this  class
        class_inds = np.nonzero(labels == i)

        # >> assign color
        if i < len(colors) - 1:
            color = colors[i]
            al = alpha
            marker = class_marker
            ms = class_ms
            # if debug:pdb.set_trace()
        else:
            color='black'
            marker = '.'
            ms = 5
        if type(otypedict) != type(None):
            if otypedict[i] == 'NONE':
                color = 'black'
                al = 0.01
                marker='.'
                ms = 5

        # >> plot all datapoints
        if X.shape[1] == 2:
            ax.plot(X[class_inds][:,0], X[class_inds][:,1], marker, color=color,
                    alpha=al, ms=ms)
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
        else:
            ax.scatter(X[class_inds][:,0], X[class_inds][:,1], X[class_inds][:,2],
                       marker='.', c=color, alpha=al)
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            ax.set_zlabel('t-SNE Component 3')

        # >> plot labels
        if type(otypedict) != type(None):
            cntpt = np.median(X[class_inds], axis=0) # >> center of cluster

            # >> find point furthest from t-SNE center
            maxind = np.argmax(np.sum((X[class_inds] - centr)**2, axis=1))
            maxpt = X[class_inds][maxind]            # >> furthest pt in clstr
            dst = np.sqrt(np.sum((cntpt - centr)**2))
            prjc = rad - dst                         # >> length of vector

            # >> vector pointing out
            if len(class_inds[0]) == 1:
                vect = maxpt - centr
            else:
                vect = maxpt - cntpt
            v_norm = vect / np.sqrt(np.sum(vect**2)) # >> norm of the vector
            endpt = cntpt + v_norm*prjc              # >> where label will be
            
            if X.shape[1] == 2:
                xp, yp = endpt
                ax.text(xp, yp,  otypedict[i], color=color, size='large')
                xv, yv = [cntpt[0], xp], [cntpt[1], yp]
                ax.plot(xv, yv, '-', color=color)
            else:
                xp, yp, zp = endpt
                ax.text(xp, yp, zp, otypedict[i], color=color, size='large')
                xv, yv, zv = [cntpt[0], xp], [cntpt[1], yp], [cntpt[2], zp]
                ax.plot(xv, yv, zv, '-', color=color)

    plt.savefig(output_dir + prefix + 't-sne.png', dpi=300)
    print('Saved '+output_dir+prefix+'t-sne.png')

    if animate:
        from matplotlib import animation
        def animate(i):
            ax.view_init(elev=elev, azim=i)
            return fig,
        anim = animation.FuncAnimation(fig, animate, frames=360, interval=20,
                                       blit=True)
        # anim.save(output_dir+prefix+'t-sne_animation.mp4', fps=30,
        #           extra_args=['-vcodec', 'libx264'])
        anim.save(output_dir+prefix+'t-sne_animation.mp4', fps=30)
        print('Saved '+output_dir+prefix+'t-sne_animation.mp4')
    plt.close()

def presentation_feature(features, ind, output_dir = './'):
    # >> assumes triangle shape
    ranked = np.argsort(features[ind])
    
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(features[:,ranked[-1]], features[:,ranked[-2]], '.', markersize=2)
    ax.plot([features[:,ranked[-1]][ind]], [features[:,ranked[-2]][ind]], 'Xg',
            markersize=30)
    ax.set_xlabel('\u03C6' + str(ranked[-1]))
    ax.set_ylabel('\u03C6' + str(ranked[-2]))
    fig.savefig(output_dir + 'feat'+str(ranked[-1])+'-'+str(ranked[-2])+'.png')
    plt.close()

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Novelty visualizations ::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def generate_novelty_scores(features, object_ids, output_dir, prefix='',
                            n_neighbors=20, p=2, metric='minkowski',
                            contamination=0.1, algorithm='auto'):
    """Calculates LOF based on feature vectors.
    * features : array of shape (n_objects, n_dimensions)
    * object_ids : array of shape (n_objects)
    * output_dir : string, output directory
    * prefix : string
    * n_neighbors, p, metric, contamination, algorithm : arguments of 
      sklearn.neighbors.LocalOutlierFactor()
    """
    print("Generating novelty scores...")

    clf = LocalOutlierFactor(n_neighbors=n_neighbors, p=p, metric=metric,
                             contamination=contamination, algorithm=algorithm)
    fit_predictor = clf.fit_predict(features)
    negative_factor = clf.negative_outlier_factor_
    
    lof = -1 * negative_factor
    with open(output_dir+'lof.txt', 'w') as f:
        f.write('OBJECT_ID LOF\n')
        for i in range(len(object_ids)):
            f.write('{} {}\n'.format(int(object_ids[i]), lof[i]))
    return lof

def load_novelty_scores(output_dir, ticid):
    print('Loading novelty scores...')
    fname = output_dir+'lof.txt'
    filo = np.loadtxt(fname, skiprows=1)
    ticid_filo, lof = [filo[:,0], filo[:,1]]

    # sorted_inds = np.argsort(ticid)
    # >> intersect1d returns sorted arrays, so
    # >> ticid == ticid[sorted_inds][np.argsort(sorted_inds)]
    # new_inds = np.argsort(sorted_inds)
    # _, comm1, comm2 = np.intersect1d(ticid, ticid_filo, return_indices=True)
    # learned_feature_vector = lof[comm2][new_inds]

    match = []
    for i in range(len(ticid)):
        match.append(ticid_filo[i] == ticid[i])
    if len(ticid) != np.count_nonzero(np.array(match)):
        print('!!! Missing '+str(len(ticid)-np.count_nonzero(np.array(match)))+\
             ' TICIDs')


    return lof

def plot_lof(savepath, lof, time, intensity, targets, features, n, n_tot=100,
             momentum_dump_csv = '../../Table_of_momentum_dumps.csv', spoc = False,
             target_info = False):
    """ Plots the most and least interesting light curves based on LOF.
    Most basic form of plotting - for more complex, see plot_lof_with_PSD()
    Parameters:
        * savepath: where you want the subfolder of these to go. assumed to end in /
        * lof: LOF values for your sample. calculate through run_LOF() in learn_utils.py
        * time : time axis for the sample
        * intensity: fluxes for all targets in sample
        * targets : list of identifiers (TICIDs/otherwise)
        * features
        * n : number of curves to plot in each figure
        * n_tot : total number of light curves to plots (number of figures =
                  n_tot / n)
        * momentum_dump_csv: filepath to the momt dump csv file for overplotting
        * spoc: if these are spoc light curves and the target id's are TICIDs, this is True
        * target_info: only required for spoc lc's you want labelled'
         
    Returns: Nothing
        
    modified [lcg 02272021 - revert to basics, added separate more complex version]
    """
    # make folder
    path = savepath + "lof/"
    try:
        os.makedirs(path)
    except OSError:
        print ("Directory %s already exists" % path)
    # -- Sort LOF vlaues
    ranked = np.argsort(lof)
    largest_indices = ranked[::-1][:n_tot] # >> outliers
    smallest_indices = ranked[:n_tot] # >> inliers
    random_inds = list(range(len(lof)))
    random.Random(4).shuffle(random_inds)
    random_inds = random_inds[:n_tot] # >> random
    ncols=1
      
    # >> make histogram of LOF values
    print('Make LOF histogram')
    lof_file = path+'lof-histogram.png'
    plot_histogram(lof, 20, "Local Outlier Factor (LOF)", lof_file)
        
    # -- momentum dumps ----------------------------------------------
    print('Loading momentum dump times')
    with open(momentum_dump_csv, 'r') as f:
        lines = f.readlines()
        mom_dumps = [ float(line.split()[3][:-1]) for line in lines[6:] ]
        inds = np.nonzero((mom_dumps >= np.min(time)) * \
                          (mom_dumps <= np.max(time)))
        mom_dumps = np.array(mom_dumps)[inds]
        
    
    # -- plot smallest and largest LOF light curves --------------------------
    print('Plot highest LOF and lowest LOF light curves')
    num_figs = int(n_tot/n) # >> number of figures to generate
    
    for j in range(num_figs):
        for i in range(3): # >> loop through smallest, largest, random LOF plots
            fig, ax = plt.subplots(n, ncols, sharex=False,
                                   figsize = (8*ncols, 3*n))
            for k in range(n): # >> loop through each row
                
                if i == 0: ind = largest_indices[j*n + k]
                elif i == 1: ind = smallest_indices[j*n + k]
                else: ind = random_inds[j*n + k]
                
                # >> plot momentum dumps
                for t in mom_dumps:
                    axis.axvline(t, color='g', linestyle='--')
                    
                # >> plot light curve
                axis.plot(time, intensity[ind], '.k')
                axis.text(0.98, 0.02, '%.3g'%lof[ind],
                           transform=axis.transAxes,
                           horizontalalignment='right',
                           verticalalignment='bottom',
                           fontsize='xx-small')
                format_axes(axis, ylabel=True)
                if spoc:
                    ticid_label(axis, targets[ind], target_info[ind],
                                title=True)                        
                if k != n - 1:
                    axis.set_xticklabels([])
                    
                ax[n-1].set_xlabel('time [BJD - 2457000]')
                
            # >> save figures
            if i == 0:
                fig.suptitle(str(n) + ' largest LOF targets', fontsize=16,
                                 y=0.9)
                fig.tight_layout()
                fig.savefig(path + 'lof-largest_' + str(j*n) + 'to' +\
                            str(j*n + n) + '.png',
                            bbox_inches='tight')
                plt.close(fig)
            elif i == 1:
                fig.suptitle(str(n) + ' smallest LOF targets', fontsize=16,
                                 y=0.9)
                fig.tight_layout()
                fig.savefig(path + 'lof-smallest' + str(j*n) + 'to' +\
                            str(j*n + n) + '.png',
                            bbox_inches='tight')
                plt.close(fig)
            else:
                fig.suptitle(str(n) + ' random LOF targets', fontsize=16, y=0.9)
                
                # >> save figure
                fig.tight_layout()
                fig.savefig(path + 'lof-random'+ str(j*n) + 'to' +\
                            str(j*n + n) +".png", bbox_inches='tight')
                plt.close(fig) 
    return

def plot_nvlty_lc(savepath, datapath, lof, objid, sector, feats, n=5, n_tot=50,
                  mdumpcsv='Table_of_momentum_dumps.csv',
                  datatype='SPOC', n_neighbors=20, n_freq=50000,
                  max_freq=1/(8/1440.), min_freq=1/27.):
    """ Plots the most and least interesting light curves based on LOF and their 
    Parameters:
        * savepath: where you want the subfolder of these to go. assumed to end in /
        * lof: LOF values for your sample. calculate through run_LOF() in learn_utils.py
        * objid : list of identifiers (TICIDs/otherwise)
        * feats
        * n : number of curves to plot in each figure
        * n_tot : total number of light curves to plots (number of figures =
                  n_tot / n)
        * mdumpcsv: filepath to the momt dump csv file for overplotting
                
    """
    from astropy.timeseries import LombScargle
    
    # >> make folder
    savepath = savepath + "nvlty/"
    dt.create_dir(savepath)

    # >> sort LOF values 
    ranked = np.argsort(lof)
    largest_indices = ranked[::-1][:n_tot] # >> outliers
    smallest_indices = ranked[:n_tot] # >> inliers
    random_inds = list(range(len(lof)))
    random.Random(4).shuffle(random_inds)
    random_inds = random_inds[:n_tot] # >> random
    ncols=2

    # >> make frequency grid
    freq = np.linspace(min_freq, max_freq, n_freq)

    # >> get momentum dump times
    print('Loading momentum dump times')
    with open(mdumpcsv, 'r') as f:
        lines = f.readlines()
        mom_dumps = [ float(line.split()[3][:-1]) for line in lines[6:] ]
        
    # -- plot smallest and largest LOF light curves --------------------------
    print('Plot highest LOF and lowest LOF light curves')
    num_figs = int(n_tot/n) # >> number of figures to generate
    for j in range(num_figs):
        for i in range(3): # >> loop through smallest, largest, random LOF plots
            fig, ax = plt.subplots(n, ncols, sharex=False,
                                   figsize = (8*ncols, 3*n))
            
            for k in range(n): # >> loop through each row
                axis = ax[k, 0]
                
                if i == 0: ind = largest_indices[j*n + k]
                elif i == 1: ind = smallest_indices[j*n + k]
                else: ind = random_inds[j*n + k]
                
                # >> load light curve
                lchdu_mask = fits.open(datapath+'clip/sector-%02d'%sector[ind]+\
                                       '/'+str(int(objid[ind]))+'.fits')
                time = lchdu_mask[1].data['TIME']
                flux = lchdu_mask[1].data['FLUX']
                target_info = [sector[ind], lchdu_mask[0].header['CAMERA'],
                               lchdu_mask[0].header['CCD'], 'SPOC', '2-min']

                # >> compute LS periodogram
                num_inds = np.nonzero(~np.isnan(flux))
                power = LombScargle(time[num_inds], flux[num_inds]).power(freq)

                # >> plot momentum dumps
                inds = np.nonzero((mom_dumps >= np.nanmin(time)) * \
                                  (mom_dumps <= np.nanmax(time)))
                mom_dumps_targ = np.array(mom_dumps)[inds]
                for t in mom_dumps_targ:
                    axis.axvline(t, color='g', linestyle='--')

                # >> plot light curve
                axis.plot(time, flux, '.k')
                axis.text(0.98, 0.02, '%.3g'%lof[ind],
                           transform=axis.transAxes,
                           horizontalalignment='right',
                           verticalalignment='bottom',
                           fontsize='xx-small')
                format_axes(axis, ylabel=True)
                if datatype == 'SPOC':
                    ticid_label(axis, objid[ind], target_info,
                                title=True)
                    # if cross_check_txt is not None:
                    #     if targets[ind] in ticid_classified:
                    #         classified_ind = np.nonzero(ticid_classified == targets[ind])[0][0]
                    #         classification_label(axis, targets[ind],
                    #                              class_info[classified_ind])       

                
                ax[k,1].plot(freq, power, '-k', linewidth=0.5)
                ax[k,1].set_ylabel('Power')
                # ax[k,2].set_xscale('log')
                ax[k,1].set_yscale('log')
                    
            # >> label axes
            ax[n-1,0].set_xlabel('Time [BJD - 2457000]')
            ax[n-1,1].set_xlabel('Frequency [days^-1]')
            
            # >> save figures
            if i == 0:
                # fig.suptitle(str(n) + ' largest LOF targets', fontsize=16,
                #                  y=0.9)
                fig.tight_layout()
                out = savepath+'lof-psd-largest_'+str(j*n)+'to'+str(j*n+n)+'.png'
                fig.savefig(out, bbox_inches='tight')
                print('Wrote '+out)
                plt.close(fig)
            elif i == 1:
                # fig.suptitle(str(n) + ' smallest LOF targets', fontsize=16,
                #                  y=0.9)
                fig.tight_layout()
                out = savepath+'lof-psd-smallest_'+str(j*n)+'to'+str(j*n+n)+'.png'
                fig.savefig(out, bbox_inches='tight')
                print('Wrote '+out)
                plt.close(fig)
            else:
                # fig.suptitle(str(n) + ' random LOF targets', fontsize=16, y=0.9)
                fig.tight_layout()
                out = savepath+'lof-psd-random_'+str(j*n)+'to'+str(j*n+n)+'.png'
                fig.savefig(out, bbox_inches='tight')
                print('Wrote '+out)
                plt.close(fig)

def isolate_plot_feature_outliers(path, sector, features, time, flux, ticids, target_info, sigma, version=0, plot=True):
    """ isolate features that are significantly out there and crazy
    plot those outliers, and remove them from the features going into the 
    main lof/plotting/
    also removes any TLS features which returned only nans
    parameters: 
        *path to save shit into
        * features (all)
        * time axis (1) (ALREADY PROCESSED)
        * flux (all) (must ALREADY BE PROCESSED)
        * ticids (all)
        
    returns: features_cropped, ticids_cropped, flux_cropped, outlier_indexes 
    modified [lcg 07272020 - changed plotting size issue]"""
    path = path + "clipped-feature-outliers/"
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s" % path)
    
    #rcParams['figure.figsize'] = 8,3
    if version==0:
        features_greek = [r'$\alpha$', 'B', r'$\Gamma$', r'$\Delta$', r'$\beta$', r'$\gamma$',r'$\delta$',
                  "E", r'$\epsilon$', "Z", "H", r'$\eta$', r'$\Theta$', "I", "K", r'$\Lambda$', "M", r'$\mu$'
                  ,"N", r'$\nu$']
    elif version==1: 
        features_greek = ["M", r'$\mu$',"N", r'$\nu$']

    outlier_indexes = []
    for i in range(len(features[0])):
        column = features[:,i]
        column_std = np.std(column)
        column_top = np.mean(column) + column_std * sigma
        column_bottom = np.mean(column) - (column_std * sigma)
        for n in range(len(column)):
            #find and note the position of any outliers
            if column[n] < column_bottom or column[n] > column_top or np.isnan(column[n]) ==True: 
                outlier_indexes.append((int(n), int(i)))
                
    print(np.asarray(outlier_indexes))
        
    outlier_indexes = np.asarray(outlier_indexes)
    target_indexes = outlier_indexes[:,0] #is the index of the target on the lists
    feature_indexes = outlier_indexes[:,1] #is the index of the feature that it triggered on
    if plot:
        for i in range(len(outlier_indexes)):
            target_index = target_indexes[i]
            feature_index = feature_indexes[i]
            plt.figure(figsize=(8,3))
            plt.scatter(time, flux[target_index], s=0.5)
            target = ticids[target_index]
            #print(features[target_index])
            
            if np.isnan(features[target_index][feature_index]) == True:
                feature_title = features_greek[feature_index] + "=nan"
            else: 
                feature_value = '%s' % float('%.2g' % features[target_index][feature_index])
                feature_title = features_greek[feature_index] + "=" + feature_value
            print(feature_title)
            
            plt.title("TIC " + str(int(target)) + " " + astroquery_pull_data(target, breaks=False) + 
                      "\n" + feature_title + "  STDEV limit: " + str(sigma), fontsize=8)
            plt.tight_layout()
            plt.savefig((path + "featureoutlier-SECTOR" + str(sector) +"-TICID" + str(int(target)) + ".png"))
            plt.show()
    else: 
        print("not plotting today!")
            
        
    features_cropped = np.delete(features, target_indexes, axis=0)
    ticids_cropped = np.delete(ticids, target_indexes)
    flux_cropped = np.delete(flux, target_indexes, axis=0)
    targetinfo_cropped = np.delete(target_info, target_indexes, axis=0)
        
    return features_cropped, ticids_cropped, flux_cropped, targetinfo_cropped,\
        outlier_indexes


def lof_and_insets_on_sector(pathtofolder, sector, numberofplots, momentumdumppath, sigma):
    """loads in a sector and plots lof +insets """
    

    flux, x, ticid, target_info = dt.load_data_from_metafiles(pathtofolder, sector, cams=[1,2,3,4],
                                 ccds=[1,2,3,4], DEBUG=False,
                                 output_dir=pathtofolder, debug_ind=10, nan_mask_check=True)
    featuresallpath = pathtofolder + "Sector" + str(sector) + "_features_v0_all.fits"
    f = fits.open(featuresallpath, mmap=False)
    
    features = f[0].data
    #targetsfits = f[1].data
    f.close()

    flux = dt.normalize(flux, axis=1)
    
    features, ticid, flux, outlier_indexes = isolate_plot_feature_outliers(pathtofolder, sector, features, x, flux, ticid, sigma)
    
    plot_lof(x, flux, ticid, features, sector, pathtofolder,
                 momentum_dump_csv = momentumdumppath,
                 n_neighbors=20, target_info=target_info,
                 prefix='', mock_data=False, feature_vector=True,
                 n_tot=numberofplots)

    features_insets(x, flux, features, ticid, pathtofolder)
    
    return features, x, flux, ticid, outlier_indexes    


def plot_lof_summary(time, intensity, targets, features, n, path,
             momentum_dump_csv = '../../Table_of_momentum_dumps.csv',
             n_neighbors=20, target_info=False, p=4, metric='minkowski',
             contamination=0.1, algorithm='auto', 
             prefix='', mock_data=False, feature_vector=False,
             log=False, database_dir=None, single_file=False,
             fontsize='xx-small', title=True, n_pgram=5000,
             nrows=5, ncols=4):
    import matplotlib as mpl
    mpl.rcParams['font.size'] = 11.
    
    # -- calculate LOF -------------------------------------------------------
    print('Calculating LOF')
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, p=p, metric=metric,
                             contamination=contamination, algorithm=algorithm)
    fit_predictor = clf.fit_predict(features)
    negative_factor = clf.negative_outlier_factor_
    
    lof = -1 * negative_factor
    ranked = np.argsort(lof)
    largest_indices = ranked[::-1] # >> outliers
    
    freq, tmp = LombScargle(time, intensity[0]).autopower()
    freq = np.linspace(np.min(freq), np.max(freq), n_pgram)       

    # -- momentum dumps ------------------------------------------------------
    # >> get momentum dump times
    print('Loading momentum dump times')
    with open(momentum_dump_csv, 'r') as f:
        lines = f.readlines()
        mom_dumps = [ float(line.split()[3][:-1]) for line in lines[6:] ]
        inds = np.nonzero((mom_dumps >= np.min(time)) * \
                          (mom_dumps <= np.max(time)))
        mom_dumps = np.array(mom_dumps)[inds]
        
    # -- plot cross-identifications -------------------------------------------
    if type(database_dir) != type(None):
        class_info = dt.get_true_classifications(targets, single_file=single_file,
                                                 database_dir=database_dir,
                                                 useless_classes=[])        
        ticid_classified = class_info[:,0].astype('int')

    # -- plot smallest and largest LOF light curves --------------------------
    print('Plot highest LOF and lowest LOF light curves')
        
    
    fig, ax = plt.subplots(nrows, ncols, sharex=False,
                           figsize = (8*ncols, 3*nrows))
    
    for i in range(nrows): # >> loop through each row
        for j in range(int(ncols/2)): # >> loop through 2 columns
            axis = ax[i, j*int(ncols/2)]
            ind = largest_indices[i*int(ncols/2) + j]
        
            # >> plot momentum dumps
            for t in mom_dumps:
                axis.axvline(t, color='g', linestyle='--')
                
            # >> plot light curve
            axis.plot(time, intensity[ind], '.k')
            axis.text(0.98, 0.02, '%.3g'%lof[ind], transform=axis.transAxes,
                       horizontalalignment='right', verticalalignment='bottom',
                       fontsize=fontsize)
            format_axes(axis, ylabel=True)
            if not mock_data:
                ticid_label(axis, targets[ind], target_info[ind],
                            title=True)
                if targets[ind] in ticid_classified:
                    classified_ind = np.nonzero(ticid_classified == targets[ind])[0][0]
                    classification_label(axis, targets[ind],
                                         class_info[classified_ind])                        
    
            # >> plot PSD
            axis = ax[i, j*int(ncols/2)+1]
            power = LombScargle(time, intensity[ind]).power(freq)
            axis.plot(freq, power, '-k')
            format_axes(axis)
            axis.set_ylabel('Power')
            # axis.set_yscale('log')
            
            # xlim, ylim = axis.get_xlim(), axis.get_ylim()
            # axis.set_aspect(abs((xlim[1]-xlim[0])/(ylim[1]-ylim[0])*(3./8.)))            
            
        
    # >> label axes
    for j in range(int(ncols/2)):
        ax[nrows-1,j*2].set_xlabel('time [BJD - 2457000]')
        ax[nrows-1,j*2+1].set_xlabel('Frequency [days^-1]')
        
    # >> save figures
    fig.tight_layout()
    fig.savefig(path + 'ensemble_summary_LOF.png',
                bbox_inches='tight')
    plt.close(fig)  

def paper_plot_lof(features=None, time=None, flux=None, target_info=None, 
                   targets=[192980481,  18783433], n_neighbors=20, nrows=4,
                   output_dir='./', lof=None,
                   momentum_dump_csv = '../../Table_of_momentum_dumps.csv',
                   fontsize=6, load_from_metafiles=True,
                   dat_dir = '/Users/studentadmin/Dropbox/TESS_UROP/data/' ,
                   sector=20, ccds=[1,2,3,4], cams=[1,2,3,4], custom_mask=[],
                   figsize=(11,6)):
    '''
    if load_from_metafiles, then the following must be provided:
        * lof, with len(lof = len(targets))
        * targets
        * dat_dir
        * sector
        * ccds
        * cams
        * custom_mask (optionally)
    if not load_from_metafiles, then the following must be provided:
        * time
        * flux
        * targets
        * target_info
        * lof (optionally)
    '''
    from astropy.timeseries import LombScargle
    
    # -- load from metafiles --------------------------------------------------
    if load_from_metafiles:
        flux, time, ticid, target_info = \
            dt.load_data_from_metafiles(dat_dir, sector, cams=cams, ccds=ccds,
                                        DEBUG=False, nan_mask_check=True,
                                        custom_mask=custom_mask)
        flux = dt.standardize(flux)
        inds = []
        for i in range(len(targets)):
            inds.append(np.nonzero(ticid==targets[i])[0][0])
        inds = np.array(inds)
        flux = flux[inds]
        target_info=target_info[inds]
        
    
    # -- calculate LOF --------------------------------------------------------
    if type(lof) == type(None):
        print('Calculating LOF')
        clf = LocalOutlierFactor(n_neighbors=n_neighbors)
        fit_predictor = clf.fit_predict(features)
        negative_factor = clf.negative_outlier_factor_
        lof = -1 * negative_factor
        
    ranked = np.argsort(lof)    
    largest_indices = ranked[::-1][:nrows] # >> outliers
    smallest_indices = ranked[:nrows] # >> inliers    
    
    # -- momentum dumps ------------------------------------------------------
    # >> get momentum dump times
    print('Loading momentum dump times')
    with open(momentum_dump_csv, 'r') as f:
        lines = f.readlines()
        mom_dumps = [ float(line.split()[3][:-1]) for line in lines[6:] ]
        inds = np.nonzero((mom_dumps >= np.min(time)) * \
                          (mom_dumps <= np.max(time)))
        mom_dumps = np.array(mom_dumps)[inds]

    # >> compute frequency    
    f, tmp = LombScargle(time, flux[0]).autopower() 
    
    # -- plot -----------------------------------------------------------------
    fig, ax= plt.subplots(nrows, 2, figsize=figsize)
    for i in range(nrows):
        # >> plot momentum dumps
        for t in mom_dumps:
            ax[i,0].axvline(t, color='g', linestyle='--')        
        
        ind = largest_indices[i]
        ax[i,0].plot(time, flux[ind], '.k')
        # format_axes(ax[i,0], ylabel=True)
        ticid_label(ax[i,0], targets[ind], target_info[ind], title=True,
                    fontsize=fontsize)        
        ax[i,0].set_ylabel('Relative flux')
        ax[i,0].text(0.98, 0.02, 'LOF %.3g'%lof[ind],
                     transform=ax[i,0].transAxes,
                     horizontalalignment='right',
                     verticalalignment='bottom', fontsize=fontsize)
        
        power = LombScargle(time, flux[ind]).power(f)
        ax[i,1].plot(f, power, 'k', lw=1)
        ax[i,1].set_xscale('log')
        ax[i,1].set_yscale('log')
        # format_axes(ax[i,1], ylabel=True)
        ticid_label(ax[i,1], targets[ind], target_info[ind], title=True,
                    fontsize=fontsize)        
        ax[i,1].set_ylabel('Power')
        ax[i,1].text(0.98, 0.02, 'LOF %.3g'%lof[ind],
                     transform=ax[i,1].transAxes,
                     horizontalalignment='right',
                     verticalalignment='bottom', fontsize=fontsize)        
        
    
    # >> more formatting
    for i in range(nrows-1):
        ax[i,0].set_xticklabels([])
        ax[i,1].set_xticklabels([])
    ax[-1,1].set_xlabel('Frequency [days$^{-1}$]')
    ax[-1,0].set_xlabel('Time [BJD - 2457000.0]')
    
    fig.tight_layout()
    # fig.subplots_adjust()
    fig.savefig(output_dir + 'LOF_paper_plot.png')

    
def presentation_LOF_by_feature(features, features_ticid, feature_lof=None,
                             n_neighbors=20, output_dir='./', bins=20):
    '''What is the LOF triggering on?'''
    
    # ind = np.nonzero(features_ticid == ticid)
    
    # >> compute the LOF for each feature
    if type(feature_lof) == type(None):
        feature_lof = []
        for i in range(features.shape[1]):
            clf = LocalOutlierFactor(n_neighbors=n_neighbors)
            fit_predictor = clf.fit_predict(features[:,i].reshape(-1,1))
            negative_factor = clf.negative_outlier_factor_
            lof = -1 * negative_factor    
            feature_lof.append(lof)
            
    for i in range(features.shape[1]):
        fig, ax = plt.subplots(figsize=(4,4))
        n_in, bins, patches = ax.hist(feature_lof[i], bins, log=True)        
        ax.set_ylabel('Number of light curves')
        if i == features.shape[1] - 1:
            ax.set_xlabel('LOF RMS (\u03C6'+str(i) + ')')
        else:
            ax.set_xlabel('LOF \u03C6'+str(i))
        fig.tight_layout()
        fig.savefig(output_dir+'lof_hist_'+str(i)+'.png')
        
        plt.close()
            
    return feature_lof

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Clustering Visualizations :::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def two_years_ensemble_summary(output_dir, data_dir='/nfs/blender/data/tdaylan/data/',
                               prefix='Mergen_Run_1-'):

    sectors=list(range(1,27))
    
    # >> get science label for every light curve in ensemble
    ticid_label = np.empty((0,2)) # >> list of [ticid, science label]
    for i in sectors:
        fname=output_dir+'Ensemble-Sector_'+str(i)+'/Sector'+str(i)+\
               '-ticid_to_label.txt'
        filo = np.loadtxt(fname, dtype='str', delimiter=',')
        ticid_label = np.append(ticid_label, filo, axis=0)
    ticid = ticid_label[:,0].astype('float')
    labels = ticid_label[:,1]

    # >> make ensemble summary plots
    ensemble_summary_plots(ticid, labels, output_dir, data_dir, sectors, prefix,
                           derive_assignments=False, merge_classes=True)

def ensemble_summary_plots(objid, numtot, numpot, otdict, datapath, output_dir,
                           metapath):

    pdb.set_trace()

    # >> before making a confusion matrix, assign each science label a number
    underlying_classes  = np.unique(labels)
    assignments = []
    for i in range(len(underlying_classes)):
        assignments.append([i, underlying_classes[i]])
    assignments = np.array(assignments)

    # >> get the true labels (looks for data_dir/databases/SectorX_true_labels.txt')
    if gcvs_only:
        ticid_true, label_true = dt.get_gcvs_classifications(data_dir+'databases/')
        class_info = np.concatenate([np.expand_dims(ticid_true,1),
                                     np.repeat(np.expand_dims(label_true, 1),
                                               2, 1)], 1)
        class_info =  dt.get_parents_only(class_info)

    else:
        class_info = get_classifications(ticid_pred, data_dir+'databases/')
    ticid_true = class_info[:,0].astype('float')
    y_true = class_info[:,1]

    class_info_pred = np.repeat(np.expand_dims(labels,1),3, axis=1)
    class_info_pred[:,0]  = ticid
    labels = class_info_pred[:,1]
    ticid = class_info_pred[:,0].astype('float')

    inter, comm1, comm2 = np.intersect1d(ticid_true, ticid,
                                         return_indices=True)
    ticid_true = ticid_true[comm1]
    y_true = y_true[comm1]
    y_pred = labels[comm2]

    row_labels = np.unique(np.concatenate([y_true, y_pred]))

    # >> create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=row_labels)
    plot_confusion_matrix(cm, row_labels, row_labels, output_dir, prefix)
    recall, false_discovery_rate, precision, accuracy, counts_true, counts_pred=\
        evaluate_classifications(cm, row_labels)

    # >> create summary pie charts
    ensemble_budget(ticid, labels, cm, assignments, row_labels, row_labels,
                        database_dir=data_dir+'./databases', output_dir=output_dir, 
                        prefix=prefix, data_dir=data_dir)
    recall=np.array(recall)
    # target_labels = row_labels[np.nonzero((recall > 0.5) * (recall <1.))]
    target_labels = ['CW', 'ELL|X', 'EW', 'RV']
    ensemble_summary_tables(row_labels, recall, false_discovery_rate,
                               precision, accuracy, counts_true, counts_pred,
                               output_dir+prefix, target_labels=target_labels)
    ensemble_summary_tables(row_labels, recall, false_discovery_rate,
                            precision, accuracy, counts_true, counts_pred,
                            output_dir+prefix, target_labels=[])

    # >> distribution plots
    inter, comm1, comm2 = np.intersect1d(ticid.astype('float'), ticid_true.astype('float'),
                                         return_indices=True)
    y_pred = labels[comm1]

    classes, counts = np.unique(y_true, return_counts=True)
    classes = classes[np.argsort(counts)]

    print('Plot feature ditsributions...')
    plot_class_dists(assignments, ticid_true, y_pred, y_true, data_dir, sectors,
                     label_list=classes[-20:], output_dir=output_dir+prefix)
    plot_class_dists(assignments, ticid_true, y_pred, y_true, data_dir, sectors,
                     label_list=target_labels, output_dir=output_dir+prefix)

def ensemble_budget(ticid_pred, y_pred, cm, assignments, y_true_labels,
                     columns, database_dir='databases/',
                   output_dir='./', prefix='', data_dir='./data/',
                   labels = [], merge_classes=False, class_info=None,
                   parents=None, fontsize=6., 
                    parent_dict = None, figsize=(15,15)):
    

    d = dt.get_otype_dict(data_dir=data_dir)
    
    orig_classes, counts = np.unique(y_pred, return_counts=True)
    orig_classes = orig_classes.astype('str')
    num_samples = len(ticid_pred)
        
    assigned_classes = {}
    for i in range(len(columns)):
        if columns[i] != 'NONE':
            if i < len(y_true_labels):
                if y_true_labels[i] != 'NONE':
                    if y_true_labels[i] in list(d.keys()):
                        assigned_classes[str(columns[i])] = y_true_labels[i]+' = '+\
                                         d[y_true_labels[i]]
                    else:
                        desc = []
                        for otype in y_true_labels[i].split('|'):
                            desc.append(d[otype])
                        assigned_classes[str(columns[i])] = str(columns[i])+' = '+\
                                                            '\nand '.join(desc)
                else:
                    assigned_classes[str(columns[i])] = str(columns[i])
            else:
                assigned_classes[str(columns[i])] = str(columns[i])



    for label in orig_classes:
        if label not in assigned_classes:
            assigned_classes[label] = label    
    # >> re-label class '-1' as '-1: outliers'
    if '-1' in orig_classes:
        num_classes = len(orig_classes) - 1
        assigned_classes['-1'] = '-1 = outliers'
    else:
        num_classes = len(orig_classes)
            
    # -- make pie charts ------------------------------------------------------
            
    import matplotlib as mpl
    mpl.rcParams['font.size'] = fontsize            

    # fig_labels = assignments[:,1]
    fig_labels = []
    for label in orig_classes:
        fig_labels.append(assigned_classes[label])

    # >> plot all classes
    print('Saving '+output_dir+prefix+'ensemble_budget_all.png')
    fig, ax = plt.subplots()
    fig.suptitle('Number of classes: '+str(num_classes) + \
                 '\nNumber of samples: '+str(num_samples))
    ax.pie(counts, labels=fig_labels)
    fig.savefig(output_dir+prefix+'ensemble_budget_all.png', dpi=300)
    plt.close(fig)
    
    print('Saving '+output_dir+prefix+'ensemble_budget_top5.png')
    explode = np.zeros(len(orig_classes))
    inds = np.argsort(counts)[:-5]
    explode[inds] = 0.1
    fig, ax = plt.subplots(ncols=2)
    fig.suptitle('Number of classes: '+str(num_classes) + \
                 '\nNumber of samples: '+str(num_samples))    
    ax[0].pie(counts, labels=fig_labels, explode=explode)
    ax[1].pie(counts[inds], labels=np.array(fig_labels)[inds])
    fig.tight_layout()
    fig.savefig(output_dir+prefix+'ensemble_budget_top5.png', dpi=300)  
    plt.close(fig)
    
    # >> split into 6 big buckets
    type_inds = get_gcvs_variability_types(y_pred)
    var_types = ['eruptive', 'pulsating', 'rotating', 'cataclysmic',
                 'eclipsing', 'xray', 'other', 'not classified']
    titles = ['Eruptive Variable Stars', 'Pulsating Variable Stars',
                 'Rotating variable Stars',
                 'Cataclysmic (Explosive and Novalike) Variables',
                 'Eclipsing binary systems', 'Intense Variable X-ray Sources',
                 'Other variable sources', 'Not classified']
    type_counts = [len(type_inds['eruptive']), len(type_inds['pulsating']),
                   len(type_inds['rotating']), len(type_inds['cataclysmic']),
                   len(type_inds['eclipsing']), len(type_inds['xray']),
                   len(type_inds['other']),
                   len(type_inds['not classified'])]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.pie(type_counts, labels=titles)
    fig.tight_layout()
    print('Saving '+output_dir+prefix+'ensemble_budget_pie.png')
    fig.savefig(output_dir+prefix+'ensemble_budget_pie.png', dpi=300)

    for i in range(len(var_types)):
        fig, ax = plt.subplots(figsize=(8,4))
        inter, comm1, comm2 = np.intersect1d(orig_classes,
                                             y_pred[type_inds[var_types[i]]],
                                             return_indices=True)
        ax.set_title(titles[i]+'\nNumber of samples: '+str(int(type_counts[i])))
        pie_labels = np.array(fig_labels)[comm1]
        inds = np.argsort(counts[comm1])[:-7]
        pie_labels[inds] = ''

        ax.pie(counts[comm1], labels=pie_labels)
        fig.tight_layout()
        print('Saving '+output_dir+prefix+'ensemble_budget_'+var_types[i]+'.png')
        fig.savefig(output_dir+prefix+'ensemble_budget_'+var_types[i]+'.png', dpi=300)
        

    fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(30,30))
    fig.suptitle('Number of classes: '+str(num_classes) + \
                 '\nNumber of samples: '+str(num_samples))
        
    for i in range(6):
        explode = np.zeros(len(orig_classes))
        inter, comm1, comm2 = np.intersect1d(orig_classes,
                                             y_pred[type_inds[var_types[i]]],
                                             return_indices=True)
        explode[comm1] = 0.1 
        ax[i,1].set_title(titles[i]+'\nNumber of samples: '+str(type_counts[i]))
        ax[i,0].pie(counts, labels=fig_labels, explode=explode)
        ax[i,1].pie(counts[comm1],
                    labels=np.array(fig_labels)[comm1])
    print('Saving '+output_dir+prefix+'ensemble_budget.png')
    fig.tight_layout()
    fig.savefig(output_dir+prefix+'ensemble_budget.png', dpi=300)
    plt.close(fig)

def plot_confusion_matrix(cm, rows, columns, output_dir='./', prefix='',
                          figsize=(30,30)):
    print('Plotting confusion matrix...')
    df_cm = pd.DataFrame(cm, index=rows, columns=columns)
    fig, ax = plt.subplots(figsize=figsize)
    sn.heatmap(df_cm, annot=True, annot_kws={'size':8})
    ax.set_aspect(1)
    fig.savefig(output_dir+prefix+'confusion_matrix.png')
    print('Saved '+output_dir+prefix+'confusion_matrix.png')
    plt.close()        

def classification_plots(features, time, flux_feat, ticid_feat, info_feat, labels,
                         x_predict, output_dir='./', prefix='', 
                         data_dir='./', do_diagnostic_plots=True, do_summary=True,
                         sectors=[1], true_label = 'EW'):

    # -- ensemble summary plots ------------------------------------------------
    if do_summary:
        print('Ensemble summary...')
        ensemble_summary_plots(ticid_feat, labels, sectors[0],
                               data_dir, output_dir, prefix)

    # -- plot light curves from each class -------------------------------------
    if do_diagnostic_plots:
        quick_plot_classification(time, flux_feat, ticid_feat, info_feat, 
                                     features, labels, path=output_dir,
                                     prefix=prefix+'learned_classes',
                                  database_dir=database_dir)
        plot_cross_identifications(time, flux_feat, ticid_feat, info_feat, features,
                                      labels, path=output_dir, database_dir=database_dir,
                                      data_dir=data_dir, prefix=prefix)


    # -- plot fail cases -------------------------------------------------------
    if do_diagnostic_plots:
        plot_fail_reconstructions(time, x_true, x_predict, ticid_true,
                                     y_true, y_pred, assignments,
                                     class_info, info_feat[comm1],
                                     output_dir=output_dir+prefix,
                                     true_label=true_label)
        plot_fail_cases(time, x_true, ticid_true, y_true, y_pred, assignments,
                        class_info, info_feat[comm1], output_dir+prefix)

def quick_plot_classification(savepath, time, intensity, targets, target_info,
                              labels, prefix='', title='', ncols=10, nrows=5):
    '''Plots first 5 light curves in each class to get a general sense of what they look like
    Parameters:
        * savepath
        * time
        * intensity
        * target (TICID's)
        * target_info
        * labels: from classifier
        * prefix: whatever you want everythign labelled as - probably name of classifier
        * title: main plot titles
        * ncols: default is 10 (classes per page)
        * nrows: default is 5, LC per class to plot
        
        
    Returns: Nothing'''
    classes, counts = np.unique(labels, return_counts=True)
    colors = get_colors()   
    
    num_figs = int(np.ceil(len(classes) / ncols))
    
    for i in range(num_figs): #
        fig, ax = plt.subplots(nrows, ncols, sharex=True,
                               figsize=(8*ncols*0.75, 3*nrows))
        fig.suptitle(title)
        
        if i == num_figs - 1 and len(classes) % ncols != 0:
            num_classes = len(classes) % ncols
        else:
            num_classes = ncols
        for j in range(num_classes): # >> loop through columns
            class_num = classes[ncols*i + j]
            
            # >> find all light curves with this  class
            class_inds = np.nonzero(labels == class_num)[0]
            
            if class_num == -1:
                color = 'black'
            elif class_num < len(colors) - 1:
                color = colors[class_num]
            else:
                color='black'
              
            for l in range(0, nrows):
                ind = class_inds[l]
                ax[l,j].plot(time, intensity[ind], '.k')
                ticid_label(ax[l,j], targets[ind], target_info[ind],
                            title=True, color=color)
                format_axes(ax[l,j], ylabel=False)
                
            ax[0, j].set_title('Class '+str(class_num)+ "# Curves:" + str(counts[j]),
                               color=color, fontsize='xx-small')
            ax[-1, j].set_xlabel('Time [BJD - 2457000]')   
                        
            if j == 0:
                for m in range(nrows):
                    ax[m, 0].set_ylabel('Relative flux')
                    
        fig.tight_layout()
        fig.savefig(path + prefix + '-' + str(i) + '.png')
        plt.close(fig)

def plot_paramscan_metrics(output_dir, parameter_sets, silhouette_scores, db_scores, ch_scores):
    """ For use in parameter searches for dbscan
    inputs:  
        * output_dir to save figure to, ends in /
        * parameter sets (only really need the len of them but yknow)
        * all associated scores - currently works for the 3 calculated
        
    returns: nothing. saves figure to folder specified
    requires: matplotlib
    
    modified [lcg 07282020 - created]"""

    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    
    
    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)
    
    par1 = host.twinx()
    par2 = host.twinx()
    
    x_axis = np.arange(0, len(parameter_sets), 1)
    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    par2.spines["right"].set_position(("axes", 1.2))
    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    make_patch_spines_invisible(par2)
    # Second, show the right spine.
    par2.spines["right"].set_visible(True)
    
    host.scatter(x_axis, db_scores, c='red', label="DB Scores")
    host.grid(True)
    par1.scatter(x_axis, silhouette_scores, c = 'green', label="Silhouette Scores")
    par2.scatter(x_axis, ch_scores, c='blue', label="CH Scores")
    
    host.set_xlabel("Parameter Set")
    host.set_ylabel("Davies-Boulin Score")
    par1.set_ylabel("Silhouette Score")
    par2.set_ylabel("Calinski-Harabasz Score")
    
    host.yaxis.label.set_color('red')
    par1.yaxis.label.set_color('green')
    par2.yaxis.label.set_color('blue')
    
    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors='red', **tkw)
    par1.tick_params(axis='y', colors='green', **tkw)
    par2.tick_params(axis='y', colors='blue', **tkw)
    host.tick_params(axis='x', **tkw)
    
    host.set_title("DBSCAN Parameter Scan Metric Results")
    
    plt.savefig(output_dir+"paramscan-metric-results.png")

    
def plot_paramscan_classes(output_dir, parameter_sets, num_classes, noise_points):
    """ For use in parameter searches for dbscan
    inputs: 
        * output_dir to save figure to, ends in /
        * parameter sets (only really need the len of them but yknow)
        * number of classes for each + number of point in the noise class
        
    returns: nothing. saves figure to folder specified
    requires: matplotlib
    
    modified [lcg 07292020 - created]"""
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    x_axis = np.arange(0, len(parameter_sets), 1)

    ax1.scatter(x_axis, num_classes, c='red', label="Number of Classes")
    ax1.grid(True)
    ax2.scatter(x_axis, noise_points, c = 'green', label="Noise Points")

    ax1.set_xlabel("Parameter Set")
    ax1.set_ylabel("Number of Classes")
    ax2.set_ylabel("Noise Points")
    
    ax1.yaxis.label.set_color('red')
    ax2.yaxis.label.set_color('green')

    tkw = dict(size=4, width=1.5)
    ax1.tick_params(axis='y', colors='red', **tkw)
    ax2.tick_params(axis='y', colors='green', **tkw)
    ax1.tick_params(axis='x', **tkw)
    
    ax1.set_title("DBSCAN Parameter Scan Class Results")
    
    plt.savefig(output_dir+"paramscan-class-results.png")
    
    plt.show()

    
def classification_diagnosis(features, labels_feat, output_dir, prefix='',
                             figsize=(15,15)):

    labels = np.unique(labels_feat)
    latentDim = np.shape(features)[1]       
    ax_label = '\u03C6'
    
    fig, ax = plt.subplots(nrows = latentDim, ncols = latentDim,
                             figsize = figsize)  
    for i in range(1, latentDim):
        for j in range(i):
            print(i, j)
            ax[i,j].plot(features[:,j], features[:,i], '.', ms=1, alpha=0.3)
            
    # >> remove axis frame of empty plots            
    for i in range(latentDim):
        for j in range(i):
            ax[latentDim-1-i, latentDim-1-j].axis('off')   
        # >> x and y labels
        ax[i,0].set_ylabel(ax_label + str(i), fontsize='xx-small')
        ax[latentDim-1,i].set_xlabel(ax_label + str(i), fontsize='xx-small')    
        
    for a in ax.flatten():
        # a.set_aspect(aspect=1)
        a.set_xticks([])
        a.set_yticks([])
        a.set_yticklabels([])
        a.set_xticklabels([])
    plt.subplots_adjust(hspace=0, wspace=0)        
    
    for k in range(len(labels)):

        label=labels[k]
        label_inds = np.nonzero(labels_feat == label)
                
        for i in range(1, latentDim):
            for j in range(i):
                X = features[label_inds,j].reshape(-1)
                Y = features[label_inds,i].reshape(-1)
                ax[i,j].plot(X, Y, 'Xr', ms=2)        
                
        fig.savefig(output_dir+prefix+'latent_space-'+str(labels[k])+'.png')
        
        for i in range(1, latentDim):
            for j in range(i):
                for l in range(len(label_inds[0])):
                    ax[i,j].lines.remove(ax[i,j].get_lines()[-1]) 
    


def plot_classification(time, intensity, targets, labels, path,
                        momentum_dump_csv = './Table_of_momentum_dumps.csv',
                        n=20, target_info=False,
                        prefix='', mock_data=False, addend=1.,
                        feature_vector=False):
    """ 
    """

    classes, counts = np.unique(labels, return_counts=True)
    colors = get_colors()
    
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
        if classes[i] == -1:
            color = 'black'
        elif classes[i] < len(colors) - 1:
            color = colors[i]
        else:
            color='black'
        
        for k in range(min(n, counts[i])): # >> loop through each row
            ind = class_inds[k]
            
            # >> plot momentum dumps
            for t in mom_dumps:
                ax[k].plot([t,t], [0, 1], '--g', alpha=0.5,
                           transform=ax[k].transAxes)            
            
            # >> plot light curve
            ax[k].plot(time, intensity[ind] + addend, '.k')
            ax[k].text(0.98, 0.02, str(labels[ind]), transform=ax[k].transAxes,
                       horizontalalignment='right', verticalalignment='bottom',
                       fontsize='xx-small')
            format_axes(ax[k], ylabel=True)
            if not mock_data:
                ticid_label(ax[k], targets[ind], target_info[ind], title=True)

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


def plot_cross_identifications(time, intensity, targets, target_info, features,
                               labels, path='./', prefix='', addend=0.,
                               database_dir='./databases/', ncols=10,
                               nrows=10, data_dir='./'):
    colors = get_colors()
       
    class_info = dt.get_true_classifications(targets,
                                             database_dir=database_dir,
                                             single_file=False)
    d = dt.get_otype_dict(data_dir=data_dir)
    
    classes = []
    for otype in class_info[:,1]:
        otype_list = otype.split('|')
        for o in otype_list:
            if o not in classes:
                classes.append(o)
    print('Num classes: '+str(len(classes)))
    print(classes)
    # classes, counts = np.unique(class_info[:,1], return_counts=True)
    num_figs = int(np.ceil(len(classes) / ncols))
    for i in range(num_figs): #
        print('Making figure '+str(i)+'/'+str(num_figs))
        fig, ax = plt.subplots(nrows, ncols, sharex=True,
                               figsize=(8*ncols*0.75, 3*nrows)) 
        if i == num_figs - 1 and len(classes) % ncols != 0:
            num_classes = len(classes) % ncols
        else:
            num_classes = ncols        
            
        for j in range(num_classes): # >> loop through columns
            class_label = classes[ncols*i + j]
            class_inds = np.nonzero([class_label in x for x in class_info[:,1]])[0]
            print('Plotting class '+class_label)
            
            for k in range(min(nrows, len(class_inds))):
                ind = class_inds[k]
                ticid = float(class_info[ind][0])
                flux_ind = np.nonzero(targets == ticid)[0][0]
                class_num = labels[flux_ind]
                if class_num < len(colors) - 1 and class_num != -1:
                    color = colors[class_num]
                else:
                    color='black'                
                ax[k, j].plot(time, intensity[flux_ind]+addend, '.k')
                classification_label(ax[k,j], ticid, class_info[ind])
                ticid_label(ax[k,j], ticid, target_info[flux_ind], title=True,
                            color=color)
                format_axes(ax[k,j], ylabel=True)
                ax[k,j].set_title('Class ' + str(class_num) + '\n' + \
                                  ax[k,j].get_title())
                if k == 0:
                    title = ax[k,j].get_title()
                    if class_label in list(d.keys()):
                        class_label = d[class_label]
                    ax[k,j].set_title(class_label+'\n'+title)
                    
        
        fig.tight_layout()
        fig.savefig(path + 'hdbscan-underlying-class-'+prefix + '-' + str(i) + '.png')
        # fig.savefig(path + prefix + '-' + str(i) + '.pdf')
        plt.close(fig)                
   

def plot_clusters(savepath, datapath, clstr, objid, sector, feats, 
                  mdumpcsv = 'Table_of_momentum_dumps.csv',
                  n=10, datatype='SPOC'):
    # etcnow
    classes, counts = np.unique(clstr, return_counts=True)
    # !!
    colors=['red', 'blue', 'green', 'purple', 'yellow', 'cyan', 'magenta',
            'skyblue', 'sienna', 'palegreen']*10
    
    # >> get momentum dump times
    with open(mdumpcsv, 'r') as f:
        lines = f.readlines()
        mom_dumps = [ float(line.split()[3][:-1]) for line in lines[6:] ]
        
    for i in range(len(classes)): # >> loop through each class
        fig, ax = plt.subplots(n, 1, sharex=True, figsize = (8, 3*n))
        class_inds = np.nonzero(labels == classes[i])[0]
        if classes[i] == -1:
            color = 'black'
        elif classes[i] < len(colors) - 1:
            color = colors[i]
        else:
            color='black'
        
        for k in range(min(n, counts[i])): # >> loop through each row
            ind = class_inds[k]
            
            # >> plot momentum dumps
            for t in mom_dumps:
                ax[k].plot([t,t], [0, 1], '--g', alpha=0.5,
                           transform=ax[k].transAxes)            
            
            # >> plot light curve
            ax[k].plot(time, intensity[ind], '.k')
            ax[k].text(0.98, 0.02, str(labels[ind]), transform=ax[k].transAxes,
                       horizontalalignment='right', verticalalignment='bottom',
                       fontsize='xx-small')
            format_axes(ax[k], ylabel=True)
            if not mock_data:
                ticid_label(ax[k], targets[ind], target_info[ind], title=True)

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

def evaluate_classifications(cm, row_labels):
    recall = []
    false_discovery_rate = []
    precision = []
    accuracy = []
    counts_true = []
    counts_pred = []
    for i in range(len(row_labels)):
        ind = i
        counts_true.append(np.sum(cm[ind]))
        counts_pred.append(np.sum(cm[:,ind]))
        
        TP = cm[ind, ind] # >> number of true positives
        FP = np.sum(cm[:,ind]) - cm[ind,ind] # >> number of false positives
        FN = np.sum(cm[ind]) - cm[ind,ind] # >> number of false negatives
        TN = np.sum(cm) - TP - FP - FN
        recall.append(TP/(TP+FN))
        false_discovery_rate.append(FP/(TP+FP))
        precision.append(TP/(TP+FP))
        accuracy.append((TP+TN)/np.sum(cm))    
    return recall, false_discovery_rate, precision, accuracy, counts_true,\
        counts_pred

def plot_fail_cases(time, flux, ticid, y_true, y_pred, assignments, class_info,
                    target_info, output_dir='./', nrows=10):
    
    colors = get_colors()
    for i in range(len(assignments)):
        fig, ax = plt.subplots(nrows, 3, sharex=True, figsize=(8*3*0.75, 3*nrows))
        
        ticid_pred = ticid[np.nonzero(y_pred == int(float(assignments[i][0])))]
        ticid_true = ticid[np.nonzero(y_true == assignments[i][1])]
        
        
        
        intersection = np.intersect1d(ticid_pred, ticid_true)
        for j in range(min(nrows, len(intersection))):
            ind = np.nonzero(ticid == intersection[j])[0][0]
            ax[j,0].plot(time, flux[ind], '.k')
            classification_label(ax[j,0], ticid[ind], class_info[ind])
            ticid_label(ax[j,0], ticid[ind], target_info[ind], title=True)
        ax[0,0].set_title('True positives\n'+ax[0,0].get_title())
        
        
        FP = []
        for j in range(len(ticid_pred)):
            if ticid_pred[j] not in intersection:
                FP.append(ticid_pred[j])
        for j in range(min(nrows, len(FP))):
            ind = np.nonzero(ticid == FP[j])[0][0]
            ax[j,1].plot(time, flux[ind], '.k')
            classification_label(ax[j,1], ticid[ind], class_info[ind])
            ticid_label(ax[j,1], ticid[ind], target_info[ind], title=True)                
        ax[0,1].set_title('False positives\n'+ax[0,1].get_title())
        
        FN = []
        for j in range(len(ticid_true)):
            if ticid_true[j] not in intersection:
                FN.append(ticid_true[j])
        for j in range(min(nrows, len(FN))):
            ind = np.nonzero(ticid == FN[j])[0][0]
            ax[j,2].plot(time, flux[ind], '.k')
            classification_label(ax[j,2], ticid[ind], class_info[ind])
            ticid_label(ax[j,2], ticid[ind], target_info[ind], title=True)
        ax[0,2].set_title('False negatives\n'+ax[0,2].get_title())
            
        fig.savefig(output_dir+'fail_analysis_'+\
                    assignments[i][1].replace('/', '-')+'.png')
        plt.close(fig)

def presentation_plot_classifications(x, flux, ticid, target_info, output_dir,
                                      ticid_list, classnum, 
                                      plot_psd=False, plot_mom_dump=False,
                                      momentum_dump_csv = 'Table_of_momentum_dumps.csv'):  
    from astropy.timeseries import LombScargle
    
    color = get_colors()[classnum+1]
    
    
    # -- momentum dumps ------------------------------------------------------
    # >> get momentum dump times
    if plot_mom_dump:
        print('Loading momentum dump times')
        with open(momentum_dump_csv, 'r') as f:
            lines = f.readlines()
            mom_dumps = [ float(line.split()[3][:-1]) for line in lines[6:] ]
            inds = np.nonzero((mom_dumps >= np.min(x)) * \
                              (mom_dumps <= np.max(x)))
            mom_dumps = np.array(mom_dumps)[inds]
    
    
    if not plot_psd:
        fig, ax = plt.subplots(len(ticid_list), figsize=(8, 3*len(ticid_list)))
        for i in range(len(ticid_list)):

            if plot_mom_dump:            
                # >> plot momentum dumps
                for t in mom_dumps:
                    ax[i].axvline(t, color='g', linestyle='--')                  
            
            ind = np.nonzero(ticid == ticid_list[i])
            ax[i].plot(x, flux[ind].reshape(-1), '.k')
            ticid_label(ax[i], ticid[ind], target_info[ind][0], title=True,
                        color=color, fontsize='small')
            format_axes(ax[i], xlabel=True, ylabel=True)
        fig.tight_layout()
        fig.savefig(output_dir+'class'+str(classnum)+'.png')
        plt.close()
        
    else:
        f, tmp = LombScargle(x, flux[0]).autopower()
        fig, ax = plt.subplots(len(ticid_list), 2, figsize=(8, 3*len(ticid_list)))
        for i in range(len(ticid_list)):
        
            if plot_mom_dump:
                # >> plot momentum dumps
                for t in mom_dumps:
                    ax[i,0].axvline(t, color='g', linestyle='--')            
            
            ind = np.nonzero(ticid == ticid_list[i])
            ax[i,0].plot(x, flux[ind].reshape(-1), '.k')
            ticid_label(ax[i,0], ticid[ind], target_info[ind][0], title=True,
                        color=color, fontsize='small')
            format_axes(ax[i,0], xlabel=True, ylabel=True)
            
            power = LombScargle(x, flux[ind].reshape(-1)).power(f)
            ax[i,1].plot(f, power, 'k', lw=1)
            ax[i,1].set_xscale('log')
            ax[i,1].set_yscale('log')
            ax[i,1].set_xlabel('Frequency [days$^{-1}$]')
            ax[i,1].set_ylabel('Power')
            ax[i,1].set_aspect(3./8)
            ticid_label(ax[i,1], ticid[ind], target_info[ind][0], title=True,
                        color=color, fontsize='small')            
        fig.tight_layout()
        fig.savefig(output_dir+'class'+str(classnum)+'-psd.png')
        plt.close()        

def plot_lof_2col(time, intensity, targets, features, n, path,
             momentum_dump_csv = '../../Table_of_momentum_dumps.csv',
             n_neighbors=20, target_info=False, p=2,
             prefix='', mock_data=False, feature_vector=False, log=False):
    """ lof plotting variant to specifically make a plot for the paper with
    two columns per plot
    """
    # -- calculate LOF -------------------------------------------------------
    print('Calculating LOF')
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, p=p)
    fit_predictor = clf.fit_predict(features)
    negative_factor = clf.negative_outlier_factor_
    
    lof = -1 * negative_factor
    ranked = np.argsort(lof)
    largest_indices = ranked[::-1][:n] # >> outliers
    smallest_indices = ranked[:n] # >> inliers
    
    # >> save LOF values in txt file
    print('Saving LOF values')
    with open(path+'lof-'+prefix+'.txt', 'w') as f:
        for i in range(len(targets)):
            f.write('{} {}\n'.format(int(targets[i]), lof[i]))
      

    # -- momentum dumps ------------------------------------------------------
    # >> get momentum dump times
    print('Loading momentum dump times')
    with open(momentum_dump_csv, 'r') as f:
        lines = f.readlines()
        mom_dumps = [ float(line.split()[3][:-1]) for line in lines[6:] ]
        inds = np.nonzero((mom_dumps >= np.min(time)) * \
                          (mom_dumps <= np.max(time)))
        mom_dumps = np.array(mom_dumps)[inds]

    # -- plot smallest and largest LOF light curves --------------------------
    print('Plot highest LOF and lowest LOF light curves')
    #num_figs = int(n_tot/n) # >> number of figures to generate
    
    rownumber = int(n/2) #number of rows per two column figure
    
    fig, ax = plt.subplots(rownumber, 2, sharex=True, figsize = (16, 3*rownumber))
    #just going to do largest first: 

    for i in range(n): #for each of the n plots
        ind = largest_indices[i] #get index
        if i < 10: #if in the first column
            col = 0 #column 0
            row = int(i) #row is row
        elif i >= 10: 
            col = 1
            row = int(i - rownumber)
            
        for t in mom_dumps:
                    ax[row, col].axvline(t, color='g', linestyle='--')
        
        ax[row, col].plot(time, intensity[ind], '.k')
        ax[row,col].text(0.98, 0.02, '%.3g'%lof[ind],
                           transform=ax[row, col].transAxes,
                           horizontalalignment='right',
                           verticalalignment='bottom',
                           fontsize='xx-small')
        format_axes(ax[row,col], ylabel=True)
        if not mock_data:
            ticid_label(ax[row, col], targets[ind], target_info[ind],
                                title=True)
            
        fig.suptitle(str(n) + ' largest LOF targets', fontsize=16,
                             y=0.9)
        fig.savefig(path + 'lof-' + prefix + 'kneigh' + \
                            str(n_neighbors) + '-largest.png',
                            bbox_inches='tight')
        plt.close(fig)
     
    fig, ax = plt.subplots(rownumber, 2, sharex=True, figsize = (16, 3*rownumber))
    for i in range(n): #for each of the n plots
        ind = smallest_indices[i] #get index
        if i < 10: #if in the first column
            col = 0 #column 0
            row = int(i) #row is row
        elif i >= 10: 
            col = 1
            row = int(i - rownumber)
            
        for t in mom_dumps:
                    ax[row, col].axvline(t, color='g', linestyle='--')
        
        ax[row, col].plot(time, intensity[ind], '.k')
        ax[row,col].text(0.98, 0.02, '%.3g'%lof[ind],
                           transform=ax[row, col].transAxes,
                           horizontalalignment='right',
                           verticalalignment='bottom',
                           fontsize='xx-small')
        format_axes(ax[row,col], ylabel=True)
        if not mock_data:
            ticid_label(ax[row, col], targets[ind], target_info[ind],
                                title=True)
            
        fig.suptitle(str(n) + ' smallest LOF targets', fontsize=16,
                             y=0.9)
        fig.savefig(path + 'lof-' + prefix + 'kneigh' + \
                            str(n_neighbors) + '-smallest.png',
                            bbox_inches='tight')
        plt.close(fig)

# -- Underlying distribution ---------------------------------------------------

def sector_dists(data_dir, sector, output_dir='./', figsize=(3,3)):
    tess_features = np.loadtxt(data_dir + 'Sector'+str(sector)+\
                               '/tess_features_sector'+str(sector)+'.txt',
                               delimiter=' ', usecols=[1,2,3,4,5,6])
    
    fig1, ax1 = plt.subplots(2,3)
    for i in range(5):
        fig, ax = plt.subplots(figsize=figsize)
        if i == 0:
            # ax.set_xlabel('log $T_{eff}$ [K]')
            ax.set_xlabel('$T_{eff}$ [K]')
            suffix='-Teff'
        elif i == 1:
            ax.set_xlabel('Radius [$R_{\odot}$]')
            suffix='-rad'
        elif i == 2:
            ax.set_xlabel('Mass [$M_{\odot}$]')
            suffix='-mass'
        elif i == 3:
            ax.set_xlabel('GAIA Mag')
            suffix='-GAIAmag'
        else:
            ax.set_xlabel('Distance [kpc]')
            suffix='-d'
        feat=tess_features[:,i+1]
        feat=feat[np.nonzero(~np.isnan(feat))]
        # feat=np.log(feat)
        ax.hist(feat, bins=30)
        ax.set_ylabel('Number of light curves')
        a = ax1[int(i/3),i//3]
        a.hist(feat, bins=30)
        ax.set_xscale('log')
        ax.set_yscale('log')
        a.set_xscale('log')
        a.set_yscale('log')
        fig.tight_layout()
        fig.savefig(output_dir+'Sector'+str(sector)+suffix+'.png')
    fig1.tight_layout()
    fig.savefig(output_dir+'Sector_dists.png')



# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Autoencoder Visualizations ::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
def hyperparam_opt_diagnosis(analyze_object, output_dir, supervised=False):
    import pandas as pd
    import matplotlib.pyplot as plt
    # analyze _object = talos.Analyze('talos_experiment.csv')
    
    print(analyze_object.data)
    print(analyze_object.low('val_loss'))
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    df = analyze_object.data
    print(dt.iloc[[np.argmin(df['val_loss'])]])
    
    with open(output_dir + 'best_params.txt', 'a') as f: 
        best_param_ind = np.argmin(df['val_loss'])
        f.write(str(dt.iloc[best_param_ind]) + '\n')
    
    if supervised:
        label_list = ['val_loss']
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
    # for col in ['round_epochs', 'val_loss', 'val_accuracy', 'val_precision_1',
    #         'val_recall_1', 'loss', 'accuracy', 'precision_1', 'recall_1']:
    for col in ['round_epochs', 'val_loss', 'loss']:    
        hyperparameters.remove(col)
        
    p = {}
    for key in hyperparameters:
        p[key] = dt.iloc[best_param_ind][key]
    
    return df, best_param_ind, p                
        
def plot_corr_matrix(data, savepath, cols=None, annot_kws=None):
    import seaborn as sn
    df = pd.DataFrame(data, columns=cols)
    corrMatrix = df.corr()
    plt.figure(figsize=(18,12))
    sn.heatmap(corrMatrix, annot=True, annot_kws=annot_kws)
    plt.tight_layout()
    plt.savefig(savepath+'corr_matrix.png')
    print('Saved '+savepath+'corr_matrix.png')

    

def plot_filter_vis(model, feat=0, layer_num=1,
                    output_dir='', prefix=''):
    from tf_keras_vis.activation_maximization import ActivationMaximization

    def model_modifier(m):
        target_layer = m.layers[layer_num]
        new_model = tf.keras.Model(inputs=m.inputs,
                                   outputs=target_layer.output)
        new_model.layers[-1].activation = tf.keras.activations.linear
        return new_model
    def loss(output):
        return output[:, feat]

    activation_maximization = ActivationMaximization(model, model_modifier,
                                                     clone=False)
    pdb.set_trace()
    activation = activation_maximization(loss)
    image = activation[0].astype(np.int8)
    fig, ax = plt.subplots()
    ax.imshow(image)
    fig.tight_layout()
    fig.savefig(output_dir+prefix+'dense_vis_feat'+str(feat)+'.png')
    plt.close(fig)


def plot_bottleneck_vis(model, feat=0, bottleneck_name='bottleneck',
                        output_dir='', prefix=''):
    from tf_keras_vis.activation_maximization import ActivationMaximization

    def model_modifier(m):
        bottleneck_layer = m.get_layer(name=bottleneck_name)
        new_model = tf.keras.Model(inputs=m.inputs,
                                   outputs=bottleneck_layer.output)
        new_model.layers[-1].activation = tf.keras.activations.linear
        return new_model
    def loss(output):
        pdb.set_trace()
        return output[:, feat]

    activation_maximization = ActivationMaximization(model, model_modifier,
                                                     clone=False)
    # pdb.set_trace()
    activation = activation_maximization(loss)
    image = activation[0].astype(np.int8)
    fig, ax = plt.subplots()
    ax.imshow(image)
    fig.tight_layout()
    fig.savefig(output_dir+prefix+'dense_vis_feat'+str(feat)+'.png')
    plt.close(fig)
        

def plot_saliency_map(savepath, objid, sector, objid_targ,
                      bottleneck_name='bottleneck', feat=0, smooth_samples=20,
                      smooth_noise=0.20, 
                      log=False):
    '''Uses Saliency from https://pypi.org/project/tf-keras-vis/
    Args:
    * model : Keras Model()
    * time, x_train, ticid_train
    * ticid_target : list of TICIDs to plot attention maps for
    * bottleneck_name : name of Keras layer (usually 'dense' or 'bottleneck')
    * feat : dimension of latent space (integer)
    * smooth_samples : number of calculating grdaients iterations
    * smooth_noise : noise spread level
    '''
    from tf_keras_vis.saliency import Saliency
    from tensorflow.keras.models import load_model
    import matplotlib.colors as colors

    # >> load model
    model = load_model(savepath+'model.hdf5')

    def model_modifier(current_model):
        target_layer = current_model.get_layer(name=bottleneck_name)
        new_model = tf.keras.Model(inputs=current_model.inputs,
                                   outputs=target_layer.output)
        new_model.layers[-1].activation = tf.keras.activations.linear
        return new_model

    # def loss(output):
    #     res = []
    #     for i in range(len(ticid_target)):
    #         res.append(output[i][feat])
    #     return res

    def loss(output):
        res = []
        res.append(output[0][feat])
        return res

    # etcnow
    x_train = np.load('/scratch/data/tess/lcur/spoc/dae/chunk00_train_lspm.npy')

    objid_targ, inds, _ = np.intersect1d(objid, objid_targ,
                                           return_indices=True)
    for i in range(len(objid_targ)):
        # >> find correct chunk
        X = np.expand_dims(x_train[inds[i]], 0)

        saliency = Saliency(model, model_modifier=model_modifier, clone=False)
        saliency_map = saliency(loss, X, smooth_samples=smooth_samples,
                                smooth_noise=smooth_noise, keepdims=True)

        saliency_map = saliency_map - np.min(saliency_map, axis=1, keepdims=True)
        saliency_map = saliency_map / np.max(saliency_map, axis=1, keepdims=True)

        # >> plot saliency map
        if log:
            cmap = colors.LogNorm(vmin=1e-4,
                                  vmax=np.max(saliency_map[0]))
        else:
            cmap = plt.cm.jet
        fig, ax = plt.subplots()
        for j in range(len(time)-1):
            ax.axvspan(time[j], time[j+1], alpha=0.2,
                          facecolor=cmap(saliency_map[0][j]))

        ax.plot(time, X[0], '.k', ms=1)
        format_axes(ax, xlabel=True, ylabel=True)
        fig.tight_layout()
        fig.savefig(output_dir+prefix+'saliency_overlay_feat'+str(feat)+\
                    '_TIC'+str(int(objid_targ[i]))+'.png')
        plt.close(fig)

        fig, ax = plt.subplots(2, figsize=(8, 6))
        for j in range(len(time)-1):
            ax[0].axvspan(time[j], time[j+1], alpha=0.2,
                          facecolor=cmap(saliency_map[0][j]))

        ax[0].plot(time, X[0], '.k', ms=1)
        if log:
            ax[1].plot(time, np.log(saliency_map[0]), '.k', ms=1)
            ax[1].set_ylabel('Log Attention')
        else:
            ax[1].plot(time, saliency_map[0], '.k', ms=1)
            ax[1].set_ylabel('Attention')

        format_axes(ax[0], xlabel=True, ylabel=True)

        fig.tight_layout()
        fig.savefig(output_dir+prefix+'saliency_feat'+str(feat)+'_TIC'+\
                    str(int(objid_targ[i]))+'.png')
        plt.close(fig)



    # # >> get light curves to plot
    # ticid_target, inds, _ = np.intersect1d(ticid_train, ticid_target,
    #                                        return_indices=True)
    # X = x_train[inds]

    # # >> make saliency map
    # saliency = Saliency(model, model_modifier=model_modifier, clone=False)
    # saliency_map = saliency(loss, X, smooth_samples=smooth_samples,
    #                         smooth_noise=smooth_noise, keepdims=True)

    # pdb.set_trace()
    # saliency_map = saliency_map - np.min(saliency_map, axis=1, keepdims=True)
    # saliency_map = saliency_map / np.max(saliency_map, axis=1, keepdims=True)

    # # >> plot saliency map
    # fig, ax = plt.subplots(1, len(ticid_target),
    #                        figsize=(6*len(ticid_target), 3))
    # for i in range(len(ticid_target)):
    #     for j in range(len(time)-1):
    #         ax[i].axvspan(time[j], time[j+1], alpha=0.2,
    #                    facecolor=plt.cm.jet(saliency_map[i][j]))

    #     ax[i].plot(time, X[i], '.k', ms=1)
    #     format_axes(ax[i], xlabel=True, ylabel=True)
  
    # fig.tight_layout()
    # fig.savefig(output_dir+prefix+'saliency_overlay_feat'+str(feat)+'.png')
    # plt.close(fig)
    
    # fig, ax = plt.subplots(2, len(ticid_target),
    #                        figsize=(6*len(ticid_target), 6))
    # for i in range(len(ticid_target)):
    #     ax[0][i].plot(time, X[i], '.k', ms=1)
    #     ax[1][i].plot(time, saliency_map[i], '.k', ms=1)
    #     format_axes(ax[0][i], xlabel=True, ylabel=True)
    #     format_axes(ax[1][i], xlabel=True, ylabel=False)
    #     ax[1][i].set_ylabel('Attention', fontsize='small')
  
    # fig.tight_layout()
    # fig.savefig(output_dir+prefix+'saliency_feat'+str(feat)+'.png')
    # plt.close(fig)
    
  
def diagnostic_plots(history, model, p, output_dir, 
                     x, x_train, x_test, x_predict, 
                     mock_data=False, target_info_test=False,
                     target_info_train=False, ticid_train=False,
                     ticid_test=False, sharey=False, prefix='',
                     supervised=False, y_true=False, y_predict=False,
                     y_train=False, y_test=False,
                     flux_test=False, flux_train=False, time=False,
                     rms_train=False, rms_test = False, input_rms = False,
                     input_psd=False,
                     inds = [-1,0,1,2,3,4,5,6,7,-2,-3,-4,-5,-6,-7],
                     intermed_inds = None,
                     input_bottle_inds = [0,1,2,-6,-7],
                     addend = 1., feature_vector=False, percentage=False,
                     input_features = False, load_bottleneck=False, n_tot=100,
                     DAE=False,
                     plot_epoch = False,
                     plot_in_out = False,
                     plot_in_bottle_out=False,
                     plot_latent_test = False,
                     plot_latent_train = False,
                     plot_kernel=False,
                     plot_intermed_act=False,
                     make_movie = False,
                     plot_lof_test=False,
                     plot_lof_train=False,
                     plot_lof_all=False,
                     plot_reconstruction_error_test=False,
                     plot_reconstruction_error_all=False):
    '''Produces all plots.
    Parameters:
        * history : Keras model.history
        * model : Keras Model()
        * p : parameter set given as a dictionary, e.g. {'latent_dim': 21, ...}
        * outout_dir : directory to save plots in
        * x : time array
        * x_train : training set, shape=(num light curves, num data points)
        * x_test : testing set, shape=(num light curves, num data points)
        * x_predict : autoencoder prediction, same shape as x_test
        
        * mock_data : if False, the following are required:
            * target_info_test : [sector, cam, ccd] for testing set,
                                 shape=(num light curves in test set, 3)
            * target_info_train
            * ticid_test
            * ticid_train
        
        * feature_vector : if True, the following are required:
            * flux_train
            * flux_test
            * time
        
        * supervised : if True, the following are required:
            * y_train
            * y_test
            * y_true
            * y_predict
            
        * input_rms
        '''

    # >> remove any plot settings
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams['lines.markersize'] = 2 
    
    if input_psd: # >> separate light curve from PSD
        psd_train = x_train[1]
        x_train = x_train[0]
        psd_test = x_test[1]
        x_test = x_test[0]
        f = x[1]
        x = x[0]
        psd_predict = x_predict[1]
        x_predict = x_predict[0]
        
    if not feature_vector:
        flux_test = x_test
        flux_train = x_train
        time = x
    
    # >> plot loss, accuracy, precision, recall vs. epochs
    if plot_epoch:
        print('Plotting loss vs. epoch')
        epoch_plots(history, p, output_dir+prefix+'epoch-',
                    supervised=supervised)   

    # -- unsupervised ---------------------------------------------------------
    # >> plot some decoded light curves
    if plot_in_out and not supervised:
        print('Plotting input, output and residual')
        
        if input_psd:
            fig, axes = input_output_plot(f, psd_test, psd_predict,
                                          output_dir+prefix+\
                                              'input_output_PSD.png',
                                          ticid_test=ticid_test,
                                          inds=inds, target_info=target_info_test,
                                          addend=addend, sharey=sharey,
                                          mock_data=mock_data,
                                          feature_vector=feature_vector,
                                          percentage=percentage) 
                
        fig, axes = input_output_plot(x, x_test, x_predict,
                                      output_dir+prefix+'input_output.png',
                                      ticid_test=ticid_test,
                                      inds=inds, target_info=target_info_test,
                                      addend=addend, sharey=sharey,
                                      mock_data=mock_data,
                                      feature_vector=feature_vector,
                                      percentage=percentage)
          
        
    # -- supervised -----------------------------------------------------------
    if supervised:
        print('Plotting classifications')
        y_train_classes = np.argmax(y_train, axis = 1)
        num_classes = len(np.unique(y_train_classes))
        training_test_plot(x,x_train,x_test,
                              y_train_classes,y_true,y_predict,num_classes,
                              output_dir+prefix+'lc-', ticid_train, ticid_test,
                              mock_data=mock_data)
        
    # -- latent space visualization -------------------------------------------
        
    # >> get bottleneck
    if input_features:
        features = []
        for ticid in ticid_test:
            res = dt.get_tess_features(ticid)
            features.append([res[1:6]])
        features = np.array(features)
    else: features=False
    if plot_in_bottle_out or plot_latent_test or plot_lof_test or plot_lof_all:
        if load_bottleneck:
            print('Loading bottleneck (testing set)')
            with fits.open(output_dir + 'bottleneck_test.fits', mmap=False) as hdul:
                bottleneck = hdul[0].data
        else:
            print('Getting bottleneck (testing set)')
            bottleneck = lt.get_bottleneck(model, x_test, p,
                                           input_features=input_features,
                                           features=features,
                                           input_rms=input_rms,
                                           rms=rms_test,
                                           DAE=DAE)
        
    # >> plot input, bottleneck, output
    if plot_in_bottle_out and not supervised:
        print('Plotting input, bottleneck, output')
        input_bottleneck_output_plot(x, x_test, x_predict,
                                     bottleneck, model, ticid_test,
                                     output_dir+prefix+\
                                     'input_bottleneck_output.png',
                                     addend=addend, inds = input_bottle_inds,
                                     sharey=False, mock_data=mock_data,
                                     feature_vector=feature_vector)

    # >> make corner plot of latent space
    if plot_latent_test:
        print('Plotting latent space for testing set')
        latent_space_plot(bottleneck, output_dir+prefix+'latent_space.png')
    
    # >> plot the 20 light curves with the highest LOF
    if plot_lof_test:
        print('Plotting LOF for testing set')
        for n in [20]: # [20, 50, 100]: loop through n_neighbors
            plot_lof(time, flux_test, ticid_test, bottleneck, 20, output_dir,
                     prefix='test-'+prefix, n_neighbors=n, mock_data=mock_data,
                     feature_vector=feature_vector, n_tot=n_tot,
                     target_info=target_info_test, log=True, addend=addend)

    
    if plot_latent_train or plot_lof_train or plot_lof_all:
        if load_bottleneck:
            print('Loading bottleneck (training set)')
            with fits.open(output_dir + 'bottleneck_train.fits', mmap=False) as hdul:
                bottleneck_train = hdul[0].data
        else:
            print('Getting bottleneck (training set)')
            bottleneck_train = lt.get_bottleneck(model, x_train, p,
                                                 input_features=input_features,
                                                 features=features,
                                                 input_rms=input_rms,
                                                 rms=rms_train,
                                                 DAE=DAE)
        
    if plot_latent_train:
        print('Plotting latent space for training set')
        latent_space_plot(bottleneck_train, output_dir+prefix+\
                          'latent_space-x_train.png')        
        
    if plot_lof_train:
        print('Plotting LOF for testing set')
        for n in [20]: # [20, 50, 100]:
            # if type(flux_train) != bool:
            plot_lof(time, flux_train, ticid_train, bottleneck_train, 20,
                     output_dir, prefix='train-'+prefix, n_neighbors=n,
                     mock_data=mock_data, feature_vector=feature_vector,
                     n_tot=n_tot, target_info=target_info_train,
                     log=True, addend=addend)

                
    if plot_lof_all:
        print('Plotting LOF for entire dataset')
        bottleneck_all = np.concatenate([bottleneck, bottleneck_train], axis=0)
        plot_lof(time, np.concatenate([flux_test, flux_train], axis=0),
                 np.concatenate([ticid_test, ticid_train]), bottleneck_all,
                 20, output_dir, prefix='all-'+prefix, n_neighbors=20,
                 mock_data=mock_data, feature_vector=feature_vector,
                 n_tot=n_tot, log=True, addend=addend,
                 target_info=np.concatenate([target_info_test,
                                             target_info_train], axis=0))   

    
    # -- plot reconstruction error (unsupervised) -----------------------------
    # >> plot light curves with highest, smallest and random reconstruction
    #    error
    if plot_reconstruction_error_test:
        print('Plotting reconstruction error for testing set')
        plot_reconstruction_error(x, x_test, x_test, x_predict, ticid_test,
                                  output_dir=output_dir,
                                  target_info=target_info_test,
                                  mock_data=mock_data, addend=addend)
    
    if plot_reconstruction_error_all:
        print('Plotting reconstruction error for entire dataset')
        # >> concatenate test and train sets
        tmp = np.concatenate([x_test, x_train], axis=0)
        
        if input_psd:
            tmp1 = np.concatenate([psd_test, psd_train], axis=0)
            tmp_predict = model.predict([tmp, tmp1])
        else:
            tmp_predict = model.predict(tmp)
        
        plot_reconstruction_error(x, tmp, tmp, tmp_predict, 
                                  np.concatenate([ticid_test, ticid_train],
                                                 axis=0),
                                  output_dir=output_dir, addend=addend,
                                  target_info=\
                                      np.concatenate([target_info_test,
                                                      target_info_train]))
        # >> remove x_train reconstructions from memory
        del tmp    
        
    # -- intermediate activations visualization -------------------------------
    if plot_intermed_act or make_movie:
        print('Calculating intermediate activations')
        if type(intermed_inds) == type(None):
            err = (x_test - x_predict)**2
            err = np.mean(err, axis=1)
            err = err.reshape(np.shape(err)[0])
            ranked = np.argsort(err)
            intermed_inds = [ranked[0], ranked[-1]]
        activations = lt.get_activations(model, x_test[intermed_inds]) 
    if plot_intermed_act:
        print('Plotting intermediate activations')
        intermed_act_plot(x, model, activations, x_test[intermed_inds],
                          output_dir+prefix+'intermed_act-', addend=addend,
                          inds=list(range(len(intermed_inds))),
                          feature_vector=feature_vector)
    
    if make_movie:
        print('Making movie of intermediate activations')
        movie(x, model, activations, x_test, p, output_dir+prefix+'movie-',
              ticid_test, addend=addend, inds=intermed_inds)        
        
    # >> plot kernel vs. filter
    if plot_kernel:
        print('Plotting kernel vs. filter')
        kernel_filter_plot(model, output_dir+prefix+'kernel-')    
        
    # return activations, bottleneck

def epoch_plots(history, p, out_dir):
    '''Plot metrics vs. epochs.
    Parameters:
        * history : dictionary, output from model.history
        * model = Keras Model()
        * activations
        * '''

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

    np.savetxt(out_dir+'loss.txt', history.history['loss'])

# == visualizations for unsupervised pipeline =================================

def input_output_plot(x, x_test, x_predict, out, ticid_test=False,
                      inds = [-1,0,1,2,3,4,5,6,7,-2,-3,-4,-5,-6,-7],
                      addend = 1., sharey=False,
                      mock_data=False, feature_vector=False,
                      percentage=False, target_info=False, psd=False):
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(15,12), sharey=False,
                             sharex=True)
    plt.subplots_adjust(hspace=0)
    for i in range(ncols):
        for ngroup in range(ngroups):
            ind = int(ngroup*ncols + i)
            if not mock_data:
                ticid_label(axes[ngroup*3,i], ticid_test[inds[ind]],
                            target_info[ind], title=True)
                
            # >> plot input
            axes[ngroup*3,i].plot(x,x_test[inds[ind]]+addend, '.k')
            
            # >> plot output
            axes[ngroup*3+1,i].plot(x,x_predict[inds[ind]]+addend, '.k')
            if sharey:
                bottom, top = axes[ngroup*3,i].get_ylim()
                axes[ngroup*3+1,i].set_ylim(bottom, top)
            # >> calculate residual
            residual = (x_test[inds[ind]] - x_predict[inds[ind]])
            if percentage:
                residual = residual / x_test[inds[ind]]
                
            # >> plot residual
            axes[ngroup*3+2, i].plot(x, residual, '.k')
            for j in range(3):
                format_axes(axes[ngroup*3+j,i])
            
        if feature_vector: # >> x-axis is latent dims
            axes[-1, i].set_xlabel('\u03C8', fontsize='small')
        elif psd:
            axes[-1, i].set_xlabel('Frequency [Hz]', fontsize='small')
        else: # >> x-axis is time
            axes[-1, i].set_xlabel('Time [BJD - 2457000]', fontsize='small')
            
    # >> change y-axis scale
    if psd:
        for a in axes.flatten():
            a.set_yscale('log')
        pdb.set_trace()

    # >> make y-axis labels
    for i in range(ngroups):
        if feature_vector:
            axes[3*i,   0].set_ylabel('input',  fontsize='small')
            axes[3*i+1, 0].set_ylabel('output', fontsize='small')
            axes[3*i+2, 0].set_ylabel('residual', fontsize='small')     
        elif psd:
            axes[3*i,   0].set_ylabel('input\nrelative PSD',  fontsize='small')
            axes[3*i+1, 0].set_ylabel('output\nrelative PSD', fontsize='small')
            axes[3*i+2, 0].set_ylabel('residual\nrelative PSD', fontsize='small')  
        else:            
            axes[3*i,   0].set_ylabel('input\nrelative flux',  fontsize='small')
            axes[3*i+1, 0].set_ylabel('output\nrelative flux', fontsize='small')
            axes[3*i+2, 0].set_ylabel('residual', fontsize='small') 
        
    fig.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    return fig, axes

def kernel_filter_plot(model, out_dir):
    '''Plots kernel against filters, i.e. an image with dimension
    (kernel_size, num_filters). Averages across 
    Parameters:
        * model : Keras Model()
        * out_dir : output directory (ending with '/')'''
    # >> get inds for plotting kernel and filters
    layer_inds = np.nonzero(['conv' in x.name for x in model.layers])[0]
    for a in layer_inds: # >> loop through conv layers
        filters, biases = model.layers[a].get_weights()
        # >> average over channels (assumes data_format='channels_last')
        filters = np.mean(filters, axis=-1)
        
        fig, ax = plt.subplots()
        ax.imshow(filters)
        # ax.imshow(np.reshape(filters, (np.shape(filters)[0],
        #                                np.shape(filters)[2])))
        ax.set_xlabel('filter')
        ax.set_ylabel('kernel')
        plt.savefig(out_dir + 'layer' + str(a) + '.png')
        plt.close(fig)

def intermed_act_plot(x, model, activations, x_test, out_dir, addend=1.,
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
                x_test[inds[c]] + addend, '.k')
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
            
            # >> reshape if image with image width = 1
            if len(np.shape(activation)) == 4:
                act_shape = np.array(activation.shape)
                new_shape = act_shape[np.nonzero(act_shape != 1)]
                activation = np.reshape(activation, new_shape)
            
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
                    ax.plot(x1, activation[inds[c]][:,b]+addend, '.k')
                else:
                    ax.plot(x1, activation[inds[c]]+addend, '.k')
                
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
    
def input_bottleneck_output_plot(x, x_test, x_predict, bottleneck, model,
                                 ticid_test, out, inds=[0,1,-1,-2,-3],
                                 addend = 1., sharey=False, mock_data=False,
                                 feature_vector=False):
    '''Can only handle len(inds) divisible by 3 or 5'''
    # bottleneck_ind = np.nonzero(['dense' in x.name for x in \
    #                              model.layers])[0][0]
    # bottleneck = activations[bottleneck_ind - 1]
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
            axes[ngroup*3,i].plot(x,x_test[inds[ind]]+addend, '.k')
            axes[ngroup*3+1,i].plot(np.linspace(np.min(x),np.max(x),
                                              len(bottleneck[inds[ind]])),
                                              bottleneck[inds[ind]], '.k')
            axes[ngroup*3+2,i].plot(x,x_predict[inds[ind]]+addend, '.k')
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
                  x_test[inds[c]] + addend, '.k')
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
                          activation[inds[c]] + addend, '.k')
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
                              y + addend, '.k')
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

# :: supervised pipeline ::::::::::::::::::::::::::::::::::::::::::::::::::::::

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
            ax[j,i].plot(x, x_train[inds[j]], '.'+colors[i])
            if not mock_data:
                ticid_label(ax[j,i], ticid_train[inds[j]])
        for j in range(min(7, len(inds1))):
            ax1[j,i].plot(x, x_test[inds1[j]], '.'+colors[y_predict[inds1[j]]])
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

def reconstruction_error_power(time, intensity, x_test, x_predict, ticid_test,
                               output_dir='./', target_info=False, n=20,
                              prefix='', powers=[1,2,3,4,5]):
    
    
    fig_large, ax_large = plt.subplots(n, len(powers), sharex=True,
                           figsize=(8*len(powers), 3*n))
    fig_small, ax_small = plt.subplots(n, len(powers), sharex=True,
                             figsize=(8*len(powers), 3*n))
    
    for col in range(len(powers)):
    
        print('Calculating reconstruction error ...')
        err = np.abs(x_test - x_predict)**powers[col]
        err = np.mean(err, axis=1)
        err = err.reshape(np.shape(err)[0])
    
        # >> get top n light curves
        ranked = np.argsort(err)
        largest_inds = np.copy(ranked[::-1][:n])
        smallest_inds = np.copy(ranked[:n])
    
        for i in range(2):
            if i == 0:
                fig, ax = fig_large, ax_large
                inds = largest_inds
            else:
                fig, ax = fig_small, ax_small
                inds =smallest_inds
            for k in range(n): # >> loop through each row
                ind = inds[k]
                
                # >> plot light curve
                ax[k, col].plot(time, intensity[ind], '.k')
                ax[k, col].plot(time, x_predict[ind], '.')
                # ax[k].text(0.98, 0.02, 'mse: ' +str(err[ind]),
                #            transform=ax[k].transAxes, horizontalalignment='right',
                #            verticalalignment='bottom', fontsize='xx-small')
                format_axes(ax[k, col], ylabel=True)
                ticid_label(ax[k, col], ticid_test[ind], target_info[ind],
                            title=True)
            ax[0, col].set_title('Power='+str(powers[col]))
            ax[n-1, col].set_xlabel('Time [BJD - 2457000]')
            
    fig_large.suptitle('largest reconstruction error', fontsize=16, y=0.9)
    fig_large.savefig(output_dir+prefix+'reconstruction_error-largest.png',
                      bbox_inches='tight')

    fig_small.suptitle('smallest reconstruction error', fontsize=16, y=0.9)
    fig_small.savefig(output_dir+prefix+'reconstruction_error-smallest.png',
       bbox_inches='tight') 

def plot_fail_reconstructions(x, x_test, x_predict, ticid, y_true, y_pred, assignments,
                              class_info, target_info, output_dir='./',
                              true_label=''):
    
    colors = get_colors()
    ind = np.nonzero(assignments[:,1] == true_label)[0][0]
    # >> get TICIDs of the true positives and false negatives (row of cm)
    inds = np.nonzero(y_true == true_label)[0]
    for i in range(len(inds)//15):
        fig, ax = input_output_plot(x, x_test, x_predict,
                                    output_dir+'input_output_'+true_label.replace('/', '-')+'.png',
                                    ticid_test=ticid, target_info=target_info,
                                    inds=inds[i*15:(i+1)*15])
        
        # >> change colors to reflect what class object was assigned
        # >> add text that says what object was assigned as
        for col in range(5):
            for row in range(3):
                ax[row*3,col].set_title(ax[row*3,col].get_title(),
                                        color=colors[y_pred[i*15+col*5+row]])
                classification_label(ax[row*3,col], ticid[i*15+col*5+row],
                                     class_info[i*15+col*5+row])
                
        fig.savefig(output_dir+'input_output_'+true_label.replace('/', '-')+'.png')

def presentation_act(activations, filter_nums=[3,9,15]):
    fig, ax = plt.subplots(3, figsize=(4,8))
    for i in range(len(filter_nums)):
        ax[i].plot(activations[1][0][:,filter_nums[i]], '.k')
    return fig, ax

def presentation_kernel_plots(model, x_test, x_predict, ind, output_dir='./'):
    from keras.models import Model
    layer_outputs = [layer.output for layer in model.layers][1:] 
    activation_model = Model(inputs=model.input, outputs=layer_outputs) 
    
    z1 = np.reshape(x_test[ind], (1, np.shape(x_test)[1]))
    activations = activation_model.predict(z1)
    
    conv_inds = np.nonzero(['conv' in x.name for x in model.layers])[0]
    conv_inds = conv_inds[:len(conv_inds)//2]

    conv_ind = 0
    filters, biases = model.layers[conv_inds[conv_ind]].get_weights()
    filters = np.reshape(filters, (np.shape(filters)[-1], np.shape(filters)[0]))
    
    # >> choose a filter
    i = len(conv_inds) -1
    plt.figure()
    plt.imshow(np.reshape(filters[i], (1,3)))
    plt.savefig(output_dir + str(i)+'.png')
    
    plt.figure()
    plt.plot(activations[conv_inds[conv_ind] + 1][0][:,i], '.k', markersize=2)
    plt.savefig(output_dir + str(i)+'-act.png')
    
    fig, ax = plt.subplots()
    ax.imshow(np.reshape(filters, (3, np.shape(filters)[0])))
    ax.set_yticklabels([])
    ax.set_ylabel('filter')
    fig.savefig(output_dir + str(i) + '-all.png')
    
    fig, ax = plt.subplots(8, 8)
    act = activations[conv_inds[conv_ind] + 1][0]
    lim = int(np.shape(act)[1]/8.)
    for k in range(lim):
        for j in range(8):
            ax[k,j].plot(act[:,8*k + j], '.k', markersize=1)
            ax[k,j].set_xticklabels([])
            ax[k,j].set_yticklabels([])
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(output_dir + str(i) + '-all_filt.png')

def presentation_validation(model, p, ind):
    '''Can only handle constant num_filters'''
    bottleneck_ind = np.nonzero(['dense' in x.name for x in model.layers])[0][0]
    filters, biases = model.layers[bottleneck_ind].get_weights()
    filters = filters.reshape(p['num_filters'],
                              int(np.shape(filters)[0]/p['num_filters']), -1)
    
    fig, ax = plt.subplots(4,4)
    for i in range(4):
        for j in range(4):
            ax[i,j].plot(filters[4*i+j,:,ind], '.') 
            ax[i,j].set_xticklabels([])
            ax[i,j].set_yticklabels([])

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Preprocessing Visualizations ::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
       
def sector_nan_mask_diag(custom_masks=[[]]*26,
                         output_dir='/nfs/blender/data/tdaylan/'):
    sectors=list(range(1,27))
    # Sector 1
    custom_masks[0] = list(range(800)) + list(range(15800, 17400)) + \
        list(range(19576, 20075))
    # Sector 4
    custom_masks[3] = list(range(9100, 9800))
    for i in range(26):
        fig, ax = plt.subplots(16, 2, sharex=True, figsize=(10, 30))
        sector = sectors[i]
        custom_mask = custom_masks[i]
        flux, time, ticid, target_info = \
            dt.load_data_from_metafiles('/nfs/blender/data/tdaylan/data/',
                                        sector, nan_mask_check=False)
        cams = target_info[:,1].astype('int')
        ccds = target_info [:,2].astype('int')
        masked_time = np.delete(time, custom_mask)
        for cam in [1,2,3,4]:
            for ccd in [1,2,3,4]:
                ind = np.nonzero((cams == cam)*(ccds == ccd))[0][0]
                a = ax[4*(cam-1)+(ccd-1),0]
                a.plot(time, flux[ind], ',k')
                ticid_label(a, ticid[ind], target_info[ind], title=True)
                format_axes(a, ylabel=True)
                masked_flux = np.delete(flux[ind], custom_mask)
                a = ax[4*(cam-1)+(ccd-1),1]
                a.plot(masked_time, masked_flux, ',k')
                ticid_label(a, ticid[ind], target_info[ind], title=True)
                format_axes(a, ylabel=True)
        ax[15,0].set_xlabel('Time [BJD - 2457000]')
        ax[15,1].set_xlabel('Time [BJD - 2457000]')
        ax[0,0].set_title('Unmasked'+'/n'+ax[0,0].get_title(), 
                          fontsize='xx-small')
        ax[0,1].set_title('Masked'+'/n'+ax[0,1].get_title(),
                          fontsize='xx-small')
        fig.tight_layout()
        fig.savefig(output_dir+'nan_mask_sector_'+str(sector)+'.png')
        plt.close(fig)

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Helper Functions ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def ENF_labels(version = 0):
        if version==0:
            graph_labels = ["Average", "Variance", "Skewness", "Kurtosis", "Log Variance",
                            "Log Skewness", "Log Kurtosis", "Maximum Power", "Log Maximum Power", 
                            "Period of Maximum Power (0.1 to 10 days)","Slope" , "Log Slope",
                            "P0", "P1", "P2", "Period of Maximum Power (0.001 to 0.1 days)"]
            fname_labels = ["Avg", "Var", "Skew", "Kurt", "LogVar", "LogSkew", "LogKurt",
                            "MaxPower", "LogMaxPower", "Period0_1to10", "Slope", "LogSlope",
                            "P0", "P1", "P2", "Period0to0_1"]
        elif version == 1: 
            graph_labels = ["TLS Best fit Period (days)", "TLS Best fit duration (days)", "TLS best fit depth (ppt from transit bottom",
                            "TLS Best fit Power"]
            fname_labels = ["TLSPeriod", "TLSDuration", "TLSDepth", "TLSPower"]
        elif version == 2:
            graph_labels = ["Average", "Variance", "Skewness", "Kurtosis", "Log Variance",
                            "Log Skewness", "Log Kurtosis", "Maximum Power", "Log Maximum Power", 
                            "Period of Maximum Power (0.1 to 10 days)","Slope" , "Log Slope",
                            "P0", "P1", "P2", "Period of Maximum Power (0.001 to 0.1 days)", "TLS Best fit Period (days)", "TLS Best fit duration (days)", "TLS best fit depth (ppt from transit bottom",
                            "TLS Best fit Power"]
            fname_labels = ["Avg", "Var", "Skew", "Kurt", "LogVar", "LogSkew", "LogKurt",
                            "MaxPower", "LogMaxPower", "Period0_1to10", "Slope", "LogSlope",
                            "P0", "P1", "P2", "Period0to0_1", "TLSPeriod", "TLSDuration", "TLSDepth", "TLSPower"]
        return graph_labels, fname_labels
    
def CAE_labels(num_features):
        graph_labels = []
        fname_labels = []
        for n in range(num_features):
            graph_labels.append('\u03C6' + str(n))
            fname_labels.append('phi'+str(n))
        return graph_labels, fname_labels
    
def get_colors():
    '''Returns list of 125 colors for plotting against white background1.
    now removes black from the list and tacks it onto the end (so that when
    indexing -1 will make the noise points black
    modified [lcg 08052020 changing pos. of black to end]'''
    from matplotlib import colors as mcolors
    import random
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    
    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]
    
    # >> get rid of light colors
    bad_colors = ['lightgray', 'lightgrey', 'gainsboro', 'whitesmoke', 'white',
                  'snow', 'mistyrose', 'seashell', 'peachpuff', 'linen',
                  'bisque', 'antiquewhite', 'blanchedalmond', 'papayawhip',
                  'moccasin', 'oldlace', 'floralwhite', 'cornsilk',
                  'lemonchiffon', 'ivory', 'beige', 'lightyellow',
                  'lightgoldenrodyellow', 'honeydew', 'mintcream',
                  'azure', 'lightcyan', 'aliceblue', 'ghostwhite', 'lavender',
                  'lavenderblush', 'black']
    for i in range(len(bad_colors)):
        ind = sorted_names.index(bad_colors[i])
        sorted_names.pop(ind)
        
    # >> now shuffle
    # random.Random(4).shuffle(sorted_names)
    random.Random(2).shuffle(sorted_names)
    sorted_names = sorted_names*20
    
    sorted_names.append('black')
    return sorted_names

def astroquery_pull_data(target, breaks=True):
    """Give a TIC ID - ID /only/, any format is fine, it'll get converted to str
    Searches the TIC catalog and pulls: 
        T_eff
        object type
        gaia magnitude
        radius
        mass
        distance
    returns a plot title string
    modified: [lcg 06302020]"""
    try: 
        catalog_data = Catalogs.query_object("TIC " + str(int(target)), radius=0.02, catalog="TIC")
        #https://arxiv.org/pdf/1905.10694.pdf
        Tmag = np.round(catalog_data[0]["Tmag"], 2)
        T_eff = np.round(catalog_data[0]["Teff"], 0)
        obj_type = catalog_data[0]["objType"]
        gaia_mag = np.round(catalog_data[0]["GAIAmag"], 2)
        radius = np.round(catalog_data[0]["rad"], 2)
        mass = np.round(catalog_data[0]["mass"], 2)
        distance = np.round(catalog_data[0]["d"], 1)
        if breaks:
            title = "T_eff:" + str(T_eff) + "," + str(obj_type) + ", G: " + str(gaia_mag) + ", Tmag: " + str(Tmag) + "\n Dist: " + str(distance) + ", R:" + str(radius) + " M:" + str(mass)
        else: 
             title = "T_eff:" + str(T_eff) + "," + str(obj_type) + ", G: " + str(gaia_mag) + ", Tmag: " + str(Tmag) + "Dist: " + str(distance) + ", R:" + str(radius) + " M:" + str(mass)
    except (ConnectionError, OSError, TimeoutError):
        print("there was a connection error!")
        title = "Connection error, no data"
    return title

def get_extrema(feature_vectors, feat1, feat2):
    """ Identifies the extrema in each direction for the pair of features given. 
    Eliminates any duplicate extrema (ie, the xmax that is also the ymax)
    Returns array of unique indexes of the extrema
    modified [lcg 07082020]"""
    indexes = []
    index_feat1 = np.argsort(feature_vectors[:,feat1])
    index_feat2 = np.argsort(feature_vectors[:,feat2])
    
    indexes.append(index_feat1[0]) #xmin
    indexes.append(index_feat2[-1]) #ymax
    indexes.append(index_feat2[-2]) #second ymax
    indexes.append(index_feat1[-2]) #second xmax
    
    indexes.append(index_feat1[1]) #second xmin
    indexes.append(index_feat2[1]) #second ymin
    indexes.append(index_feat2[0]) #ymin
    indexes.append(index_feat1[-1]) #xmax
    
    indexes.append(index_feat1[-3]) #third xmax
    indexes.append(index_feat2[-3]) #third ymax
    indexes.append(index_feat1[2]) #third xmin
    indexes.append(index_feat2[2]) #third ymin

    indexes_unique, ind_order = np.unique(np.asarray(indexes), return_index=True)
    #fixes the ordering of stuff
    indexes_unique = [np.asarray(indexes)[index] for index in sorted(ind_order)]
    
    return indexes_unique      

def plot_histogram(data, bins=40, x_label='', filename='./', insetx = None, insety = None, targets = None, 
                   insets=True, log=True, multix = False, skip_bins=2, figsize=(5,5),
                   inset_fontsize=6):
    """ 
    Plot a histogram with one light curve from each bin plotted on top
    * Data is the histogram data
    * Bins is bins for the histogram
    * x_label for the x-axis of the histogram
    * insetx is the xaxis to plot. 
        * if multix is False, assume that the xaxis is the same for all, 
        * if multix is True, each intensity has a specific time axis
    * insety is the full list of light curves
    * filename is the exact place you want it saved
    * insets is a true/false of if you want them
    * log is true/false if you want the histogram on a logarithmic scale
    modified [etc 210909]
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    # >> n_in : values of the histogram bins, bins : edges of the bins
    n_in, bins, patches = ax1.hist(data, bins, log=log, edgecolor='k')
    
    y_range = np.abs(n_in.max() - n_in.min())
    x_range = np.abs(data.max() - data.min())
    ax1.set_ylabel('Number of targets')
    ax1.set_xlabel(x_label)
    
    bin_width = bins[1] - bins[0]
    if insets == True:
        for n in np.nonzero(n_in)[0][::skip_bins]:
            axis_name = "axins" + str(n)

            # >> x location of lower-left corner of inset in data coords
            inset_width = 0.33 * x_range
            inset_x = bins[n] - (0.5*inset_width)
            inset_x = np.max([data.min(), inset_x]) # >> check within bounds of ax1

            # >> y location of lower-left corner inset in axes coords
            inset_height = 0.3 * inset_width/x_range 
            hoffset = 0.1
            if log:
                inset_y = np.log(n_in[n]) / np.log(y_range) + hoffset 
            else:
                inset_y = n_in[n] / y_range + hoffset
            inset_y = np.min([inset_y, 1-inset_height])
            inset_y = np.max([inset_y, 0])

            # >> want inset_x in data coords and inset_y in axes coordinates
            axis_name = ax1.inset_axes([inset_x, inset_y, inset_width, inset_height],
                                       transform = ax1.get_xaxis_transform())

            # >> create arrow from inset plot to bin
            xi, xf = inset_x + 0.5*inset_width, bins[n] + 0.5*bin_width
            yi, yf = inset_y, np.max([n_in.min(), inset_y-hoffset])

            ax1.arrow(xi, inset_y, xf-xi, yf-yi, width=0.01,
                      transform=ax1.get_xaxis_transform())

            # >> identify a light curve from relevant bin
            for m in range(len(data)):
                #print(bins[n], bins[n+1])
                if bins[n] <= data[m] <= bins[n+1]:
                    #print(data[m], m)
                    if multix:
                        lc_time_to_plot = insetx[m]
                    else:
                        lc_time_to_plot = insetx
                    lc_to_plot = insety[m]
                    lc_ticid = targets[m]
                    break
                else: 
                    continue

            axis_name.plot(lc_time_to_plot, lc_to_plot, '.k', ms=0.1,
                           rasterized=True, fillstyle='full')
            try:
                axis_name.text(0.5, 0.8, "TIC " + str(int(lc_ticid)),
                               fontsize=inset_fontsize, ha='center',
                               transform=axis_name.transAxes)
            except ValueError:
                axis_name.set_title(lc_ticid, fontsize=inset_fontsize)

            # if n == 0: pdb.set_trace()

            axis_name.set_xticklabels([])
            axis_name.tick_params('x', bottom=False)

            axis_name.set_ylabel('F', fontsize=inset_fontsize)               
            axis_name.tick_params('y', labelsize=inset_fontsize)
            # axis_name.set_yticklabels(axis_name.get_yticklabels(), c='k',
            #                           fontsize=inset_fontsize)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_lygos(t, intensity, error, title):
    mean = np.mean(intensity)
    mean_error = np.mean(error)
    print(mean, mean_error)
    
    sigclip = SigmaClip(sigma=5, maxiters=None, cenfunc='median')
    clipped_inds = np.nonzero(np.ma.getmask(sigclip(intensity)))
    intensity[clipped_inds] = mean
    
    plt.scatter(t, intensity)
    plt.title(title)
    plt.show()
    plt.errorbar(t, intensity, yerr = error, fmt = 'o', markersize = 0.1,
                 fillstyle='full')
    plt.show()
    
def ticid_label(ax, ticid, target_info, title=False, color='black',
                fontsize='xx-small'):
    '''Query catalog data and add text to axis.
    Parameters:
        * target_info : [sector, camera, ccd, data_type, cadence]
    TODO: use Simbad classifications and other cross-checking database
    classifications'''
    try:
        # >> query catalog data
        # target, Teff, rad, mass, GAIAmag, d, objType, Tmag = \
        #     dt.get_tess_features(ticid)
        cols = ['Teff', 'rad', 'mass', 'GAIAmag', 'd', 'Tmag']
        target, tfeats = dt.get_tess_features(ticid, cols=cols)
        Teff, rad, mass, GAIAmag, d, Tmag = tfeats
        # features = np.array(dt.get_tess_features(ticid))[1:]

        # >> change sigfigs for effective temperature
        if np.isnan(Teff):
            Teff  = 'nan'
        else: Teff = '%.4d'%Teff
        
        # >> query sector, camera, ccd
        sector, cam, ccd = target_info[:3]
        data_type = target_info[3]
        cadence = target_info[4]
        # obj_name = 'TIC ' + str(int(ticid))
        # obj_table = Tesscut.get_sectors(obj_name)
        # ind = np.nonzero(obj_table['sector']==sector)
        # cam = obj_table['camera'][ind][0]
        # ccd = obj_table['ccd'][ind][0]
    
        info = target+'\nTeff {}\nrad {}\nmass {}\nG {}\nd {}\nTmag {}'
        info1 = target+', Sector {}, Cam {}, CCD {}, {}, Cadence {},\n' +\
            'Teff {}, rad {}, mass {}, G {}, d {}, Tmag {}'
        
        
        # >> make text
        if title:
            ax.set_title(info1.format(sector, cam, ccd, data_type, cadence,
                                      Teff, '%.2g'%rad, '%.2g'%mass,
                                      '%.3g'%GAIAmag, '%.3g'%d,
                                      '%.3g'%Tmag),
                         fontsize=fontsize, color=color)
        else:
            ax.text(0.98, 0.98, info.format(Teff, '%.2g'%rad, '%.2g'%mass, 
                                            '%.3g'%GAIAmag, '%.3g'%d, 
                                            Tmag),
                      transform=ax.transAxes, horizontalalignment='right',
                      verticalalignment='top', fontsize=fontsize)
    except (ConnectionError, OSError, TimeoutError):
        print("there was a connection error!")
        ax.text(0.98, 0.98, "there was a connection error",
                      transform=ax.transAxes, horizontalalignment='right',
                      verticalalignment='top', fontsize=fontsize)
                
def classification_label(ax, ticid, classification_info, fontsize='xx-small'):
    '''classification_info = [ticid, otype, main id]
    TODO'''
    ticid, otype, bibcode = classification_info
    ax.text(0.98, 0.98, 'otype: '+otype+'\nmaind_id: '+bibcode,
            transform=ax.transAxes, fontsize=fontsize,
            horizontalalignment='right', verticalalignment='top')
    
def format_axes(ax, xlabel=True, ylabel=True):
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
    # ax.ticklabel_format(useOffset=False)
    if xlabel:
        ax.set_xlabel('Time [BJD - 2457000]')
    if ylabel:
        ax.set_ylabel('Relative flux')           
    
def plot_light_curves(targets, sector, output_dir='', prefix='', figsize=(8,8)):
    flux, time, ticid, target_info = \
        dt.load_data_from_metafiles('/nfs/blender/data/tdaylan/data/', sector,
                                    nan_mask_check=False)
    flux = dt.normalize(flux)
    inter, inds, comm2 = np.intersect1d(ticid, targets, return_indices=True)

    fig, ax = plt.subplots(len(targets), figsize=figsize)
    for i in range(len(targets)):
        ax[i].plot(time, flux[inds[i]], '.k', ms=1)
        ticid_label(ax[i], ticid[inds[i]], target_info[inds[i]], title=True,
                    fontsize='medium')
        if i == len(targets)-1:
            xlabel=True
        else:
            xlabel=False
        format_axes(ax[i], xlabel=xlabel, ylabel=True)

    fig.tight_layout()
    fig.savefig(output_dir+prefix+'lightcurves.png', dpi=300)    
             
def plot_lc(ticid, sector, output_dir='./',
            datapath='/scratch/data/tess/lcur/spoc/',
            mdumpcsv='/scratch/data/tess/meta/Table_of_momentum_dumps.csv',
            plot_mdump=True, plot_lspgram=True,
            max_freq=1/(8/1440.), min_freq=1/27., n_freq=50000,
            prefix='', suffix='', verbose=True):
                
    ticid=int(ticid)
    # >> load light curve
    lchdu = fits.open(datapath+'clip/sector-%02d'%sector+\
                           '/'+str(ticid)+'.fits')
    time = lchdu[1].data['TIME']
    flux = lchdu[1].data['FLUX']
    meta = lchdu[0].header

    # >> make title for plot
    target_info = [sector, meta['CAMERA'],
                   meta['CCD'], 'SPOC', '2-min']
    meta = dict(meta)
    for key in meta.keys():
        if type(meta[key]) == type(None):
            meta[key] = np.nan

    target_desc = ['TIC', 'Sector', 'Cam', 'CCD', 'DTYPE', 'Cadence\n',
                   'RA', 'DEC', 'Teff', 'rad', 'logG', 'Tmag']
    target_prop = [str(ticid), str(sector), str(meta['CAMERA']),
                   str(meta['CCD']), 'SPOC', '2-min',
                   '%.6g'%meta['RA_OBJ'], '%.6g'%meta['DEC_OBJ'], 
                   str(meta['TEFF']), '%.2g'%meta['RADIUS'], 
                   '%.2g'%meta['LOGG'], '%.3g'%meta['TESSMAG']]
    title = ''
    for i in range(len(target_desc)):
        title += target_desc[i]+' '+target_prop[i]+', '


    # >> make frequency grid
    if plot_lspgram:
        freq = np.linspace(min_freq, max_freq, n_freq)


    # -- momentum dumps ------------------------------------------------------
    # >> get momentum dump times
    print('Loading momentum dump times')
    with open(mdumpcsv, 'r') as f:
        lines = f.readlines()
        mom_dumps = [ float(line.split()[3][:-1]) for line in lines[6:] ]
        inds = np.nonzero((mom_dumps >= np.nanmin(time)) * \
                          (mom_dumps <= np.nanmax(time)))
        mom_dumps_targ = np.array(mom_dumps)[inds]
    
    # -- plot -----------------------------------------------------------------

    if plot_lspgram:
        figsize=(15, 5)
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax0 = ax[0]
        
    else:
        figsize=(9,4)
        fig, ax0 = plt.subplots(figsize=figsize)
    if plot_mdump:
        for t in mom_dumps_targ:
            ax0.axvline(t, color='g', linestyle='--')
        

    ax0.plot(time, flux, '.k', ms=0.5)
    # ax.set_title(str('TIC {} '.format(int(ticid))+str(title)))
    # ax0.set_title(title, fontsize='x-small')
    fig.suptitle(title, fontsize='small')
    # format_axes(ax0)
    # ticid_label(ax, ticid[ind], target_info[ind][0], title=True)      
    ax0.set_xlabel('Time [BJD-2457000]')
    ax0.set_ylabel('Relative flux')

    if plot_lspgram:
        # >> compute LS periodogram
        num_inds = np.nonzero(~np.isnan(flux))
        power = LombScargle(time[num_inds], flux[num_inds]).power(freq)
        ax[1].plot(freq, power, '-k', linewidth=0.5)
        ax[1].set_ylabel('Power')
        # ax[k,2].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].set_xlabel('Frequency [1/days]')
        # format_axes(ax[1], xlabel=False, ylabel=False)
        # ax[1].set_aspect(3./8., adjustable='box')


    fig.tight_layout()
    fname = output_dir + prefix + 'TIC' + str(ticid) + suffix + '.png'
    fig.savefig(fname, dpi=300)

    if verbose:
        print('Wrote '+fname)
    plt.close(fig)

