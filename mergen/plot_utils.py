# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 21:58:45 2020

Plotting functions only. 

@author: Lindsey Gordon @lcgordon and Emma Chickles @emmachickles

Last updated: Feb 26 2021

Custom feature plotting
    * features_plotting_2D() n-choose-2 plots of pairs of features against each
      other. 
        * can be color coded based on classification results. 
    * features2D_with_insets() same as features_plotting_2D but with insets of
      the extrema light curves
    * histo_features() produces histograms of all features
    * plot_lof() plots the n top, bottom, and random light curves ranked on
      their LOF scores
    * plot_lof_with_PSD() same as above with PSD plotted + indication of
      triggered feature in the LOF
    * quick_plot_classification() to plot the first n in each class

Helper functions
    * plot_lc()
    * ticid_label()
    * astroquery_pull_data (redundant)
    * format_axes()

Feature visualization
    * features_plotting_2d()
    * features_insets()
    * inset_plotting()
    * get_extrema()
    * features_insets_colored()
    * inset_plotting_colored()
    * inset_histogram()
    * features_2D_colorshape()
    * plot_lof()
    * plot_pca()
    * isolate_plot_feature_outliers
    * lof_and_insets_on_sector
    * plot_paramscan_metrics

Autoencoder visualizations
    * diagnostic_plots()
    * epoch_plots() 
    * input_output_plot()
    * kernel_filter_plot()
    * intermed_act_plot()
    * input_bottleneck_output_plot()
    * movie()
    * training_test_plot()
    * hyperparam_opt_diagnosis()
    * plot_reconstruction_error()
    * plot_classification()


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

import astropy
from astropy.io import fits
import scipy.signal as signal
from astropy.stats import SigmaClip
from astropy.utils import exceptions

from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import seaborn as sn

import astroquery
from astroquery.simbad import Simbad
from astroquery.mast import Catalogs
from astroquery.mast import Observations
from astroquery import exceptions
from astroquery.exceptions import RemoteServiceError
from astroquery.mast import Tesscut

# import data_functions as df
from . import data_utils as dt
import random


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Forward-Facing Visualizations :::::::::::::::::::::::::::::::::::::::::::::
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
                
def plot_lof_with_PSD(savepath, lof, time, intensity, targets, features, n, n_tot=100,
             momentum_dump_csv = '../../Table_of_momentum_dumps.csv', spoc = True,
             n_neighbors=20, target_info=False,cross_check_txt=None, single_file=False,
             n_pgram=1500):
    """ Plots the most and least interesting light curves based on LOF and their PSD
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
        
        
    Returns: Nothing
        
    modified [lcg 02272021 - condensing functionality]
    """
    # make folder
    path = savepath + "lof-with-psd/"
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
    ncols=3
    
    feature_lof = []
    for i in range(features.shape[1]):
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, p=p)
        clf.fit_predict(features[:,i].reshape(-1,1))
        negative_factor = clf.negative_outlier_factor_
        lof_tmp = -1 * negative_factor    
        feature_lof.append(lof_tmp) 
    feature_lof=np.array(feature_lof)
    # >> has shape (num_features, num_light_curves)
    freq, tmp = LombScargle(time, intensity[0]).autopower()
    freq = np.linspace(np.min(freq), np.max(freq), n_pgram) 
   
    # >> make histogram of LOF values
    print('Make LOF histogram')
    lof_file = path+'lof-histogram.png'
    plot_histogram(lof, 20, "Local Outlier Factor (LOF)", lof_file)

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
    if cross_check_txt is not None:
        class_info = dt.get_true_classifications(targets, single_file=single_file,
                                                 database_dir=cross_check_txt,
                                                 useless_classes=[])        
        ticid_classified = class_info[:,0].astype('int')

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
                    if cross_check_txt is not None:
                        if targets[ind] in ticid_classified:
                            classified_ind = np.nonzero(ticid_classified == targets[ind])[0][0]
                            classification_label(axis, targets[ind],
                                                 class_info[classified_ind])                        
                        
                if k != n - 1:
                    axis.set_xticklabels([])
                    
                
                # >> find the feature this light curve is triggering on
                feature_ranked = np.argsort(feature_lof[:,ind])
                ax[k,1].plot(features[:,feature_ranked[-1]],
                             features[:,feature_ranked[-2]], '.', ms=1)
                ax[k,1].plot([features[ind,feature_ranked[-1]]],
                              [features[ind,feature_ranked[-2]]], 'Xg',
                               ms=30)    
                ax[k,1].set_xlabel('\u03C6' + str(feature_ranked[-1]))
                ax[k,1].set_ylabel('\u03C6' + str(feature_ranked[-2])) 
                    # ax[k,1].set_adjustable("box")
                    # ax[k,1].set_aspect(1)    
                #psd
                power = LombScargle(time, intensity[ind]).power(freq)
                ax[k,2].plot(freq, power, '-k')
                ax[k,2].set_ylabel('Power')
                # ax[k,2].set_xscale('log')
                ax[k,2].set_yscale('log')
                    
            # >> label axes
            ax[n-1,0].set_xlabel('time [BJD - 2457000]')
            ax[n-1,2].set_xlabel('Frequency [days^-1]')
            
            # >> save figures
            if i == 0:
                fig.suptitle(str(n) + ' largest LOF targets', fontsize=16,
                                 y=0.9)
                fig.tight_layout()
                fig.savefig(path + 'lof-psd-largest_' + str(j*n) + 'to' +\
                            str(j*n + n) + '.png',
                            bbox_inches='tight')
                plt.close(fig)
            elif i == 1:
                fig.suptitle(str(n) + ' smallest LOF targets', fontsize=16,
                                 y=0.9)
                fig.tight_layout()
                fig.savefig(path + 'lof-psd-smallest' + str(j*n) + 'to' +\
                            str(j*n + n) + '.png',
                            bbox_inches='tight')
                plt.close(fig)
            else:
                fig.suptitle(str(n) + ' random LOF targets', fontsize=16, y=0.9)
                
                # >> save figure
                fig.tight_layout()
                fig.savefig(path + 'lof-psd-random' + str(j*n) + 'to' +\
                            str(j*n + n) +".png", bbox_inches='tight')
                plt.close(fig)

def quick_plot_classification(savepath, time, intensity, targets, target_info, labels,
                              prefix='', title='', ncols=10, nrows=5):
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
        
##### FEATURE PLOTS #####
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
        
    return features_cropped, ticids_cropped, flux_cropped, targetinfo_cropped, outlier_indexes


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


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Classification plots ::::::::::::::::::::::::::::::::::::::::::::::::::::::
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

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Machine Learning Visualizations :::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

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
        

def plot_saliency_map(model, time, x_train, ticid_train, ticid_target,
                      bottleneck_name='bottleneck', feat=0, smooth_samples=20,
                      smooth_noise=0.20, output_dir='./', prefix='',
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
    import matplotlib.colors as colors

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

    ticid_target, inds, _ = np.intersect1d(ticid_train, ticid_target,
                                           return_indices=True)
    for i in range(len(ticid_target)):
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
                    '_TIC'+str(int(ticid_target[i]))+'.png')
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
                    str(int(ticid_target[i]))+'.png')
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

def epoch_plots(history, p, out_dir, supervised=False, input_psd=False):
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
        
        if input_psd:
            fig, ax1 = plt.subplots()
            ax1.plot(history.history[list(history.history.keys())[-1]],
                     label='PSD loss')
            ax1.set_ylabel('loss')
            ax1.set_xlabel('epoch')
            ax1.set_xticks(np.arange(0, int(p['epochs']),
                                     max(int(p['epochs']/10),1)))
            ax1.tick_params('both', labelsize='x-small')
            fig.tight_layout()
            plt.savefig(out_dir + 'loss-PSD.png')
            plt.close(fig)

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

def plot_reconstruction_error(time, intensity, x_test, x_predict, ticid_test,
                              output_dir='./', addend=1., mock_data=False,
                              feature_vector=False, n=20, target_info=False):
    '''For autoencoder, intensity = x_test'''
    # >> calculate reconstruction error (mean squared error)
    err = (x_test - x_predict)**2
    err = np.mean(err, axis=1)
    err = err.reshape(np.shape(err)[0])
    
    # >> get top n light curves
    ranked = np.argsort(err)
    largest_inds = np.copy(ranked[::-1][:n])
    smallest_inds = np.copy(ranked[:n])
    random.Random(4).shuffle(ranked)
    random_inds = ranked[:n]
    
    # >> save in txt file
    if not mock_data:
        out = np.column_stack([ticid_test.astype('int'), err])
        np.savetxt(output_dir+'reconstruction_error.txt', out, fmt='%-16s')
        
        # with open(output_dir+'reconstruction_error.txt', 'w') as f:
        #     for i in range(len(ticid_test)):
        #         f.write('{}\t\t{}\n'.format(ticid_test[i], err[i]))
    
    for i in range(3):
        fig, ax = plt.subplots(n, 1, sharex=True, figsize = (8, 3*n))
        for k in range(n): # >> loop through each row
            if i == 0: ind = largest_inds[k]
            elif i == 1: ind = smallest_inds[k]
            else: ind = random_inds[k]
            
            # >> plot light curve
            ax[k].plot(time, intensity[ind]+addend, '.k')
            if not feature_vector:
                ax[k].plot(time, x_predict[ind]+addend, '.')
            ax[k].text(0.98, 0.02, 'mse: ' +str(err[ind]),
                       transform=ax[k].transAxes, horizontalalignment='right',
                       verticalalignment='bottom', fontsize='xx-small')
            format_axes(ax[k], ylabel=True)
            if not mock_data:
                ticid_label(ax[k], ticid_test[ind], target_info[ind],
                            title=True)
                
        if feature_vector:
            ax[n-1].set_xlabel('\u03C8')
        else:
            ax[n-1].set_xlabel('Time [BJD - 2457000]')
        if i == 0:
            fig.suptitle('largest reconstruction error', fontsize=16, y=0.9)
            fig.savefig(output_dir + 'reconstruction_error-largest.png',
                        bbox_inches='tight')
        elif i == 1:
            fig.suptitle('smallest reconstruction error', fontsize=16, y=0.9)
            fig.savefig(output_dir + 'reconstruction_error-smallest.png',
                        bbox_inches='tight')
        else:
            fig.suptitle('random reconstruction error', fontsize=16, y=0.9)
            fig.savefig(output_dir + 'reconstruction_error-random.png',
                        bbox_inches='tight')            
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
              prefix='', animate=False, elev=10):
    if type(X) == type(None):
        from sklearn.manifold import TSNE
        X = TSNE(n_components=n_components).fit_transform(bottleneck)
    unique_classes = np.unique(labels)
    colors = get_colors()

    fig = plt.figure()
    if X.shape[1] == 2:
        ax = fig.add_subplot()
    elif X.shape[1] == 3:
        ax = fig.add_subplot(projection='3d')

    for i in range(len(unique_classes)):
        # >> find all light curves with this  class
        class_inds = np.nonzero(labels == unique_classes[i])
        if unique_classes[i] == -1:
            color = 'black'
        elif unique_classes[i] < len(colors) - 1:
            color = colors[unique_classes[i]]
        else:
            color='black'
        if X.shape[1] == 2:
            ax.plot(X[class_inds][:,0], X[class_inds][:,1], '.', color=color)
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
        else:
            ax.scatter(X[class_inds][:,0], X[class_inds][:,1], X[class_inds][:,2],
                    marker='.', c=color)
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            ax.set_zlabel('t-SNE Component 3')

    plt.savefig(output_dir + prefix + 't-sne.png')
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
        anim.save(output_dir+prefix+'t-sne_animation.gif')
        print('Saved '+output_dir+prefix+'t-sne_animation.gif')
    plt.close()

def get_tsne(bottleneck, n_components=2):
    from sklearn.manifold import TSNE
    X = TSNE(n_components=n_components).fit_transform(bottleneck)
    return X
    
def plot_confusion_matrix(ticid_pred, y_pred, database_dir='./databases/',
                          output_dir='./', prefix='', single_file=False,
                          labels = [], merge_classes=False, class_info=None,
                          parents=['EB'],
                          parent_dict = None, figsize=(30,30)):
    from sklearn.metrics import confusion_matrix
    from scipy.optimize import linear_sum_assignment
    import seaborn as sn
    from itertools import permutations
    
    if type(parent_dict) == type(None):
        parent_dict= make_parent_dict()
    
    # >> get clusterer classes
    inds = np.nonzero(y_pred > -1)
    ticid_pred = ticid_pred[inds]
    y_pred = y_pred[inds]
    
    # >> get 'ground truth' classifications
    if type(class_info) == type(None):
        class_info = dt.get_true_classifications(ticid_pred,
                                                 database_dir=database_dir,
                                                 single_file=single_file)
    ticid_true = class_info[:,0].astype('int')

    if merge_classes:
        class_info = dt.get_parents_only(class_info, parents=parents,
                                         parent_dict=parent_dict)
        
    
    if len(labels) > 0:
        ticid_new = []
        class_info_new = []
        for i in range(len(ticid_true)):
            for j in range(len(labels)):
                if labels[j] in class_info[i][1] and \
                    ticid_true[i] not in ticid_new:
                    class_info_new.append([ticid_true[i], labels[j], class_info[i][2]])
                    ticid_new.append(ticid_true[i])
                    
        class_info = np.array(class_info_new)
        ticid_true = np.array(ticid_new)
     

    # >> find intersection
    intersection, comm1, comm2 = np.intersect1d(ticid_pred, ticid_true,
                                                return_indices=True)
    ticid_pred = ticid_pred[comm1]
    y_pred = y_pred[comm1]
    ticid_true = ticid_true[comm2]
    class_info = class_info[comm2]           
        
    columns = np.unique(y_pred).astype('str')
    y_true_labels = np.unique(class_info[:,1])

    y_true = []
    for i in range(len(ticid_true)):
        class_num = np.nonzero(y_true_labels == class_info[i][1])[0][0]
        y_true.append(class_num)
    y_true = np.array(y_true).astype('int')
    
    # -- make confusion matrix ------------------------------------------------       
    cm = confusion_matrix(y_true, y_pred)
    while len(columns) < len(cm):
        columns = np.append(columns, 'X')       
    while len(y_true_labels) < len(cm):
        y_true_labels = np.append(y_true_labels, 'X')     
    df_cm = pd.DataFrame(cm, index=y_true_labels, columns=columns)
    fig, ax = plt.subplots(figsize=figsize)
    sn.heatmap(df_cm, annot=True, annot_kws={'size':8})
    ax.set_aspect(1)
    fig.savefig(output_dir+prefix+'confusion_matrix_raw.png')
    plt.close()    
    # index = np.insert(labels, -1, 'Outlier')
    
    

    
    # >> find order of columns that gives the best accuracy using the
    # >> Hungarian algorithm (tries to minimize the diagonal)
    row_ind, col_ind = linear_sum_assignment(-1*cm)
    cm = cm[:,col_ind]
    # !! TODO need to reorder columns label
    
    df_cm = pd.DataFrame(cm, index=y_true_labels, columns=columns)
    fig, ax = plt.subplots(figsize=figsize)
    sn.heatmap(df_cm, annot=True, annot_kws={'size':8})
    ax.set_aspect(1)
    fig.savefig(output_dir+prefix+'confusion_matrix_ordered.png')
    plt.close()
    
    # >> remove rows and columns that are all zeros
    cm = np.delete(cm, np.nonzero(np.prod(cm == 0, axis=0)), axis=1)
    cm = np.delete(cm, np.nonzero(np.prod(cm == 0, axis=1)), axis=0)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    columns = np.delete(columns, np.nonzero(columns=='X'))
    y_true_labels = np.delete(y_true_labels, np.nonzero(y_true_labels=='X'))
    
    # >> plot
    df_cm = pd.DataFrame(cm, index=y_true_labels, columns=columns)
    fig, ax = plt.subplots(figsize=figsize)
    sn.heatmap(df_cm, annot=True, annot_kws={'size':6}, square=True, ax=ax)

    fig.savefig(output_dir+prefix+'confusion_matrix.png')
    fig.tight_layout()
    plt.close()
    
    return accuracy




        


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
        
##### HELPERS #######
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
    random.Random(4).shuffle(sorted_names)
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

def plot_histogram(data, bins, x_label, filename, insetx = None, insety = None, targets = None, 
                   insets=True, log=True, multix = False):
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
    modified [lcg 12302020]
    """
    fig, ax1 = plt.subplots()
    n_in, bins, patches = ax1.hist(data, bins, log=log)
    
    y_range = np.abs(n_in.max() - n_in.min())
    x_range = np.abs(data.max() - data.min())
    ax1.set_ylabel('Number of light curves')
    ax1.set_xlabel(x_label)
    
    if insets == True:
        for n in range(len(n_in)):
            if n_in[n] == 0: 
                continue
            else: 
                axis_name = "axins" + str(n)
                inset_width = 0.33 * x_range * 0.5
                inset_x = bins[n] - (0.5*inset_width)
                inset_y = n_in[n]
                inset_height = 0.125 * y_range * 0.5
                    
                axis_name = ax1.inset_axes([inset_x, inset_y, inset_width, inset_height], transform = ax1.transData) #x pos, y pos, width, height
                
                lc_to_plot = insetx
                
                #identify a light curve from that one
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
                
                axis_name.scatter(lc_time_to_plot, lc_to_plot, c='black', s = 0.1, rasterized=True)
                try:
                    axis_name.set_title("TIC " + str(int(lc_ticid)), fontsize=6)
                except ValueError:
                    axis_name.set_title(lc_ticid, fontsize=6)
    plt.savefig(filename)
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
    plt.errorbar(t, intensity, yerr = error, fmt = 'o', markersize = 0.1)
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
        target, Teff, rad, mass, GAIAmag, d, objType, Tmag = \
            dt.get_tess_features(ticid)
        # features = np.array(dt.get_tess_features(ticid))[1:]

        # >> change sigfigs for effective temperature
        if np.isnan(Teff):
            Teff = 'nan'
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
    
        info = target+'\nTeff {}\nrad {}\nmass {}\nG {}\nd {}\nO {}\nTmag {}'
        info1 = target+', Sector {}, Cam {}, CCD {}, {}, Cadence {},\n' +\
            'Teff {}, rad {}, mass {}, G {}, d {}, O {}, Tmag {}'
        
        
        # >> make text
        if title:
            ax.set_title(info1.format(sector, cam, ccd, data_type, cadence,
                                      Teff, '%.2g'%rad, '%.2g'%mass,
                                      '%.3g'%GAIAmag, '%.3g'%d, objType,
                                      '%.3g'%Tmag),
                         fontsize=fontsize, color=color)
        else:
            ax.text(0.98, 0.98, info.format(Teff, '%.2g'%rad, '%.2g'%mass, 
                                            '%.3g'%GAIAmag, '%.3g'%d, objType,
                                            Tmag),
                      transform=ax.transAxes, horizontalalignment='right',
                      verticalalignment='top', fontsize=fontsize)
    except (ConnectionError, OSError, TimeoutError):
        print("there was a connection error!")
        ax.text(0.98, 0.98, "there was a connection error",
                      transform=ax.transAxes, horizontalalignment='right',
                      verticalalignment='top', fontsize=fontsize)
            
def simbad_label(ax, ticid, simbad_info):
    '''simbad_info = [ticid, main_id, otype, bibcode]'''
    # ind = np.nonzero(np.array(simbad_info)[:,0].astype('int') == ticid)
    # if np.shape(ind)[1] != 0:
    #     ticid, main_id, otype, bibcode = simbad_info[ind[0][0]]
    ticid, main_id, otype, bibcode = simbad_info
    ax.text(0.98, 0.98, 'otype: '+otype+'\nmain_id: '+main_id+'\n'+bibcode,
            transform=ax.transAxes, fontsize='xx-small',
            horizontalalignment='right', verticalalignment='top')
    
def classification_label(ax, ticid, classification_info, fontsize='xx-small'):
    '''classification_info = [ticid, otype, main id]
    TODO'''
    ticid, otype, bibcode = classification_info
    ax.text(0.98, 0.98, 'otype: '+otype+'\nmaind_id: '+bibcode,
            transform=ax.transAxes, fontsize=fontsize,
            horizontalalignment='right', verticalalignment='top')
    
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
          
    
def plot_lc(time, intensity, target, sector):
    """plots a formatted light curve"""
    rcParams['figure.figsize'] = 8,3
    plt.scatter(time, intensity, c = 'black', s=0.5)
    plt.xlabel("BJD [-2457000]")
    plt.ylabel("relative flux")
    plt.title("TIC " + str(int(target)))
    
    data = pd.read_csv("/Users/conta/UROP_Spring_2020/Table_of_momentum_dumps.csv", header=5, skiprows=6)
    momdump = data.to_numpy()
    bjdcolumn = momdump[:,1]
    if sector == 20:
        dumppoints = bjdcolumn[1290:]
        for n in range(len(dumppoints)):
            plt.axvline(dumppoints[n], linewidth=0.5)
       
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
        
    return features_cropped, ticids_cropped, flux_cropped, targetinfo_cropped, outlier_indexes


def features_plotting_2D(feature_vectors, path, clustering,
                         time, intensity, targets, folder_suffix='',
                         feature_engineering=True, version=0, eps=0.5, min_samples=10,
                         metric='euclidean', algorithm='auto', leaf_size=30,
                         p=2, target_info=False, kmeans_clusters=4,
                         momentum_dump_csv='./Table_of_momentum_dumps.csv'):
    """plotting (n 2) features against each other
    parameters: 
        * feature_vectors - array of feature vectors
        * path to where you want everythigns aved - ends in a backslash
        * clustering - what you want to cluster as. options are 'dbscan', 'kmeans', or 
        any other keyword which will do no clustering
        * time axis
        * intensities
        *target ticids
        * folder suffix
        *feature_engineering - default is true
        * version - what version of engineered features, irrelevant integer if feature_engienering is false
        * eps, min_samples, metric, algorithm, leaf_size, p - dbscan parameters, comes with defaults
        * target_info - default is false
        *momentum dumps - not sure entirely why it's needed here tbh
        
    returns: only returns labels for dbscan/kmeans clustering. otherwise the only
    output is the files saved into the folder as given thru path
    """
    #detrmine which of the clustering algoirthms you're using: 
    rcParams['figure.figsize'] = 10,10
    folder_label = "blank"
    if clustering == 'dbscan':
        # !! TODO parameter optimization (eps, min_samples)
        db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
                    algorithm=algorithm, leaf_size=leaf_size,
                    p=p).fit(feature_vectors) #eps is NOT epochs
        classes_dbscan = db.labels_
        numclasses = str(len(set(classes_dbscan)))
        folder_label = "dbscan-colored"
        
        with open(path + 'dbscan_param_search.txt', 'a') as f:
            f.write('eps {} min samples {} metric {} algorithm {} \
                    leaf_size {} p {} # classes {} \n'.format(eps,min_samples,
                    metric,algorithm, leaf_size, p,numclasses))

    elif clustering == 'kmeans': 
        Kmean = KMeans(n_clusters=kmeans_clusters, max_iter=700, n_init = 20)
        x = Kmean.fit(feature_vectors)
        classes_kmeans = x.labels_
        folder_label = "kmeans-colored"
        
    elif clustering == 'hdbscan': 
        import hdbscan
        clusterer = hdbscan.HDBSCAN(metric=metric, min_samples=3,
                                    min_cluster_size=3)
        clusterer.fit(feature_vectors)
        classes_hdbscan = clusterer.labels_
        print(np.unique(classes_hdbscan, return_counts=True))
        folder_label = "hdbscan-colored"        
        
    else: 
        print("no clustering chosen")
        folder_label = "2DFeatures"
        
    #makes folder and saves to it    
    folder_path = path + folder_label
    try:
        os.makedirs(folder_path)
    except OSError:
        print ("Creation of the directory %s failed" % folder_path)
        print("New folder created will have -new at the end. Please rename.")
        folder_path = folder_path + "-new"
        os.makedirs(folder_path)
    else:
        print ("Successfully created the directory %s" % folder_path) 
 
    if clustering == 'dbscan':
        plot_classification(time, intensity, targets, db.labels_,
                            folder_path+'/', prefix='dbscan',
                            momentum_dump_csv=momentum_dump_csv,
                            target_info=target_info)
        plot_pca(feature_vectors, db.labels_,
                    output_dir=folder_path+'/')
    elif clustering == 'kmeans':
        plot_classification(time, intensity, targets, x.labels_,
                            path+folder_label+'/', prefix='kmeans',
                            momentum_dump_csv=momentum_dump_csv,
                            target_info=target_info)
 
    colors = get_colors()
    #creates labels based on if engineered features or not
    if feature_engineering:
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
            
        num_features = len(feature_vectors[0])
    else:
        # >> shape(feature_vectors) = [num_samples, num_features]
        num_features = np.shape(feature_vectors)[1]
        graph_labels = []
        fname_labels = []
        for n in range(num_features):
            graph_labels.append('\u03C6' + str(n))
            fname_labels.append('phi'+str(n))
    for n in range(num_features):
        feat1 = feature_vectors[:,n]
        graph_label1 = graph_labels[n]
        fname_label1 = fname_labels[n]
        for m in range(num_features):
            if m == n:
                continue
            graph_label2 = graph_labels[m]
            fname_label2 = fname_labels[m]                
            feat2 = feature_vectors[:,m]
 
            # >> fix all this repeated code
            if clustering == 'dbscan':
                plt.figure() # >> [etc 060520]
                plt.clf()
                for n in range(len(feature_vectors)):
                    plt.scatter(feat1[n], feat2[n], c=colors[classes_dbscan[n]], s=2)
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig((folder_path+'/' + fname_label1 + "-vs-" + fname_label2 + "-dbscan.png"))
                # plt.show()
                plt.close()
                 
            elif clustering == 'kmeans':
                plt.figure() # >> [etc 060520]
                plt.clf()
                for n in range(len(feature_vectors)):
                    plt.scatter(feat1[n], feat2[n], c=colors[classes_kmeans[n]], s=2)
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig(folder_path+'/' + fname_label1 + "-vs-" + fname_label2 + "-kmeans.png")
                # plt.show()
                plt.close()
                 
            elif clustering == 'hdbscan':
                plt.figure() # >> [etc 060520]
                plt.clf()
                for n in range(len(feature_vectors)):
                    plt.scatter(feat1[n], feat2[n], c=colors[classes_hdbscan[n]], s=2)
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig(folder_path+'/' + fname_label1 + "-vs-" + fname_label2 + "-hdbscan.png")
                # plt.show()
                plt.close()            
                
            elif clustering == 'none':
                plt.scatter(feat1, feat2, s = 2, color = 'black')
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig(folder_path+'/' + fname_label1 + "-vs-" + fname_label2 + ".png")
                # plt.show()
                plt.close()
                
    if clustering == 'dbscan':
        np.savetxt(folder_path+"/dbscan-classes.txt", classes_dbscan)
        return classes_dbscan
    if clustering == 'kmeans':
        return classes_kmeans
    if clustering == 'hdbscan':
        np.savetxt(folder_path+"/hdbscan-classes.txt", classes_hdbscan)
        return classes_hdbscan
                          
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


#inset plotting -------------------------------------------------------------------------------------------

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

# colored inset plotting -------------------------------------------------------
def features_insets_colored(time, intensity, feature_vectors, 
                            targets, path, realclasses,
                            eps = 2.4, min_samples=3):
    """Plots features in pairs against each other with inset plots. 
    Inset plots are colored based on the hand-identified classes, with the 
    lines connecting them to the feature point and the feature point colored by
    the predicted class. 
    currently only uses dbscan to sort them.
    Time, intensity, feature_vectors, targets are arrays
    path is the path into which you want the folder of plots ot be saved, it
    should not have a trailing /
    realclasses should be an array. 
    modified [lcg 06302020]"""   
    folderpath = path + "/2DFeatures-insets-colored"
    try:
        os.makedirs(folderpath)
    except OSError:
        print ("Creation of the directory %s failed" % folderpath)
        print("New folder created will have -new at the end. Please rename.")
        path = path + "-new"
        os.makedirs(folderpath)
    else:
        print ("Successfully created the directory %s" % folderpath) 
 
    graph_labels = ["Average", "Variance", "Skewness", "Kurtosis", "Log Variance",
                            "Log Skewness", "Log Kurtosis", "Maximum Power", "Log Maximum Power", 
                            "Period of Maximum Power (0.1 to 10 days)","Slope" , "Log Slope",
                            "P0", "P1", "P2", "Period of Maximum Power (0.001 to 0.1 days)", "TLS Best fit Period (days)", "TLS Best fit duration (days)", "TLS best fit depth (ppt from transit bottom",
                            "TLS Best fit Power"]
    fname_labels = ["Avg", "Var", "Skew", "Kurt", "LogVar", "LogSkew", "LogKurt",
                            "MaxPower", "LogMaxPower", "Period0_1to10", "Slope", "LogSlope",
                            "P0", "P1", "P2", "Period0to0_1", "TLSPeriod", "TLSDuration", "TLSDepth", "TLSPower"]
            
    db = DBSCAN(eps=eps, min_samples=min_samples,
                metric="minkowski", p=2).fit(feature_vectors) #eps is NOT epochs
    guessclasses = db.labels_
    
    for n in range(len(feature_vectors[0])):
        graph_label1 = graph_labels[n]
        fname_label1 = fname_labels[n]
        for m in range(len(feature_vectors[0])):
            if m == n:
                continue
            graph_label2 = graph_labels[m]
            fname_label2 = fname_labels[m]   
                      
            
            
            filename = folderpath + "/" + fname_label1 + "-vs-" + fname_label2 + ".png"     
            
            inset_indexes = get_extrema(feature_vectors, n,m)
            inset_plotting_colored(feature_vectors[:,n], feature_vectors[:,m], graph_label1, graph_label2, time, intensity, inset_indexes, targets, filename, realclasses, guessclasses)
            
            
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

def histo_features(features, bins, t, intensities, targets, path, insets=False, version=0):
    """ Plot histograms of each feature, both with and without inset light curves
    features (array of ALL), bins, time axis, intensities, path to put folder in
    modified [lcg 07202020]"""
    folderpath = path + "/feature_histograms_binning_" + str(bins) + "/"
    try:
        os.makedirs(folderpath)
    except OSError:
        print ("Directory %s already exists" % folderpath)
    
    if version==0:
        fname_labels = ["Avg", "Var", "Skew", "Kurt", "LogVar", "LogSkew", "LogKurt",
                            "MaxPower", "LogMaxPower", "Period0_1to10", "Slope", "LogSlope",
                            "P0", "P1", "P2", "Period0to0_1"]
    elif version == 1: 
        fname_labels = ["TLSPeriod", "TLSDuration", "TLSDepth", "TLSPower"]

    for n in range(len(features[0])):
        filename = folderpath + fname_labels[n] + "histogram.png"
        plot_histogram(features[:,n], bins, fname_labels[n], t, intensities, targets, filename, insets=False)
        if insets == True:
            filename = folderpath + fname_labels[n] + "histogram-insets.png"
            plot_histogram(features[:,n], bins, fname_labels[n], t, intensities, targets, filename, insets=True)

def plot_histogram(data, bins, x_label, insetx, insety,targets, filename,
                   insets=True, log=True):
    """ plot a histogram with one light curve from each bin plotted on top
    data is the histogram data
    bins is bins
    x-label is what you want the xaxis to be labelled as
    insetx is the SAME x-axis to plot
    insety is the full list of light curves
    filename is the exact place you want it saved
    insets is a true/false of if you want them
    modified [lcg 07012020]
    """
    fig, ax1 = plt.subplots()
    n_in, bins, patches = ax1.hist(data, bins, log=log)
    
    y_range = np.abs(n_in.max() - n_in.min())
    x_range = np.abs(data.max() - data.min())
    ax1.set_ylabel('Number of light curves')
    ax1.set_xlabel(x_label)
    
    if insets == True:
        for n in range(len(n_in)):
            if n_in[n] == 0: 
                continue
            else: 
                axis_name = "axins" + str(n)
                inset_width = 0.33 * x_range * 0.5
                inset_x = bins[n] - (0.5*inset_width)
                inset_y = n_in[n]
                inset_height = 0.125 * y_range * 0.5
                # if log: # >> use axes-relative coords
                #     inset_width = 0.33*0.5
                #     inset_height = 0.125*0.5
                #     inset_x = 
                    
                axis_name = ax1.inset_axes([inset_x, inset_y, inset_width, inset_height], transform = ax1.transData) #x pos, y pos, width, height
                
                lc_to_plot = insetx
                
                #identify a light curve from that one
                for m in range(len(data)):
                    #print(bins[n], bins[n+1])
                    if bins[n] <= data[m] <= bins[n+1]:
                        #print(data[m], m)
                        lc_to_plot = insety[m]
                        lc_ticid = targets[m]
                        break
                    else: 
                        continue
                
                axis_name.scatter(insetx, lc_to_plot, c='black', s = 0.1, rasterized=True)
                axis_name.set_title("TIC " + str(int(lc_ticid)), fontsize=6)
                # >> change aspect ratio
                # axis_name.set_adjustable("box")
                # axis_name.set_aspect(3./8)
    plt.savefig(filename)
    plt.close()


# plotting features by color and shape

                 

def diagnostic_plots(history, model, p, output_dir,
                     x, x_train, x_test, x_predict, x_predict_train=None,
                     mock_data=False, target_info_test=False,
                     target_info_train=False, ticid_train=False,
                     ticid_test=False, sharey=False, prefix='',
                     supervised=False, y_true=False, y_predict=False,
                     y_train=False, y_test=False,
                     rms_train=False, rms_test = False, input_rms = False,
                     input_psd=False,
                     inds = [-1,0,1,2,3,4,5,6,7,-2,-3,-4,-5,-6,-7],
                     intermed_inds = None,
                     input_bottle_inds = [0,1,2,-6,-7],
                     feature_vector=False, percentage=False,
                     input_features = False, load_bottleneck=False, n_tot=100,
                     DAE=False, bottleneck_train=None, bottleneck=None,
                     plot_epoch = True,
                     plot_in_out = False,
                     plot_in_out_train =True,
                     plot_in_bottle_out=False,
                     plot_latent_test = False,
                     plot_latent_train = True,
                     plot_kernel=False,
                     plot_intermed_act=False,
                     make_movie = False,
                     plot_lof_test=False,
                     plot_lof_train=False,
                     plot_lof_all=False,
                     plot_reconstruction_error_test=False,
                     plot_reconstruction_error_train=True,
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
        
    # >> plot loss, accuracy, precision, recall vs. epochs
    if plot_epoch:
        if type(history) != type(None):
            print('Plotting loss vs. epoch')
            epoch_plots(history, p, output_dir+prefix+'epoch-',
                        supervised=supervised)   

    # -- unsupervised ---------------------------------------------------------
    # >> plot some decoded light curves
    if plot_in_out and not supervised:
        print('Plotting input, output and residual from testing set')
        
        if input_psd:
            fig, axes = input_output_plot(f, psd_test, psd_predict,
                                          output_dir+prefix+\
                                              'input_output_test_PSD.png',
                                          ticid_test=ticid_test,
                                          inds=inds, target_info=target_info_test,
                                          sharey=sharey,
                                          mock_data=mock_data,
                                          feature_vector=feature_vector,
                                          percentage=percentage) 
                
        fig, axes = input_output_plot(x, x_test, x_predict,
                                      output_dir+prefix+'input_output_test.png',
                                      ticid_test=ticid_test,
                                      inds=inds, target_info=target_info_test,
                                      sharey=sharey,
                                      mock_data=mock_data,
                                      feature_vector=feature_vector,
                                      percentage=percentage)

    # >> load x_predict_train from fits file
    if (plot_in_out_train or plot_reconstruction_error_train) and \
       type(x_predict_train) == type(None):
        with fits.open(output_dir+prefix+'x_predict_train.fits') as hdul:
            x_predict_train = hdul[0].data

    if plot_in_out_train and not supervised:
        print('Plotting input, output and residual from training set')
        
        if input_psd:
            fig, axes = input_output_plot(f, psd_test, psd_predict,
                                          output_dir+prefix+\
                                              'input_output_train_PSD.png',
                                          ticid_test=ticid_test,
                                          inds=inds, target_info=target_info_test,
                                          sharey=sharey,
                                          mock_data=mock_data,
                                          feature_vector=feature_vector,
                                          percentage=percentage) 
                
        fig, axes = input_output_plot(x, x_train, x_predict_train,
                                      output_dir+prefix+'input_output_train.png',
                                      ticid_test=ticid_train,
                                      inds=inds, target_info=target_info_train,
                                      sharey=sharey,
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
        if type(bottleneck) == type(None):
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
                                     inds = input_bottle_inds,
                                     sharey=False, mock_data=mock_data,
                                     feature_vector=feature_vector)

    # >> make corner plot of latent space
    if plot_latent_test:
        print('Plotting latent space for testing set')
        latent_space_plot(bottleneck, output_dir+prefix+'latent_space.png')
    
    # [120920 - Now do novelty detection in lt.postprocessing]
    # # >> plot the 20 light curves with the highest LOF
    # if plot_lof_test:
    #     print('Plotting LOF for testing set')
    #     for n in [20]: # [20, 50, 100]: loop through n_neighbors
    #         plot_lof(time, flux_test, ticid_test, bottleneck, 20, output_dir,
    #                  prefix='test-'+prefix, n_neighbors=n, mock_data=mock_data,
    #                  feature_vector=feature_vector, n_tot=n_tot,
    #                  target_info=target_info_test, log=True)

    
    # if plot_latent_train or plot_lof_train or plot_lof_all:
    #     if type(bottleneck_train) == type(None):
    #         if load_bottleneck:
    #             print('Loading bottleneck (training set)')
    #             with fits.open(output_dir + 'bottleneck_train.fits', mmap=False) as hdul:
    #                 bottleneck_train = hdul[0].data
    #         else:
    #             print('Getting bottleneck (training set)')
    #             bottleneck_train = lt.get_bottleneck(model, x_train, p,
    #                                                  input_features=input_features,
    #                                                  features=features,
    #                                                  input_rms=input_rms,
    #                                                  rms=rms_train,
    #                                                  DAE=DAE)

    # if plot_latent_train:
    #     print('Plotting latent space for training set')
    #     latent_space_plot(bottleneck_train, output_dir+prefix+\
    #                       'latent_space-x_train.png')        
        
    # if plot_lof_train:
    #     print('Plotting LOF for testing set')
    #     for n in [20]: # [20, 50, 100]:
    #         # if type(flux_train) != bool:
    #         plot_lof(time, flux_train, ticid_train, bottleneck_train, 20,
    #                  output_dir, prefix='train-'+prefix, n_neighbors=n,
    #                  mock_data=mock_data, feature_vector=feature_vector,
    #                  n_tot=n_tot, target_info=target_info_train,
    #                  log=True)

                
    # if plot_lof_all:
    #     print('Plotting LOF for entire dataset')
    #     bottleneck_all = np.concatenate([bottleneck, bottleneck_train], axis=0)
    #     plot_lof(time, np.concatenate([flux_test, flux_train], axis=0),
    #              np.concatenate([ticid_test, ticid_train]), bottleneck_all,
    #              20, output_dir, prefix='all-'+prefix, n_neighbors=20,
    #              mock_data=mock_data, feature_vector=feature_vector,
    #              n_tot=n_tot, log=True,
    #              target_info=np.concatenate([target_info_test,
    #                                          target_info_train], axis=0))   

    
    # -- plot reconstruction error (unsupervised) -----------------------------
    # >> plot light curves with highest, smallest and random reconstruction
    #    error
    if plot_reconstruction_error_test:
        print('Plotting reconstruction error for testing set')
        plot_reconstruction_error(x, x_test, x_test, x_predict, ticid_test,
                                  output_dir=output_dir, prefix=prefix,
                                  target_info=target_info_test,
                                  mock_data=mock_data)

    if plot_reconstruction_error_train:
        print('Plotting reconstruction error for training set')
        plot_reconstruction_error(x, x_train, x_train, x_predict_train, ticid_train,
                                  output_dir=output_dir, prefix=prefix,
                                  target_info=target_info_train,
                                  mock_data=mock_data)
    
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
                                                 axis=0), prefix=prefix,
                                  output_dir=output_dir,
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
                          output_dir+prefix+'intermed_act-',
                          inds=list(range(len(intermed_inds))),
                          feature_vector=feature_vector)
    
    if make_movie:
        print('Making movie of intermediate activations')
        movie(x, model, activations, x_test, p, output_dir+prefix+'movie-',
              ticid_test, inds=intermed_inds)        
        
    # >> plot kernel vs. filter
    if plot_kernel:
        print('Plotting kernel vs. filter')
        kernel_filter_plot(model, output_dir+prefix+'kernel-')    

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



def epoch_plots(history, p, out_dir, supervised=False, input_psd=False):
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
        
        if input_psd:
            fig, ax1 = plt.subplots()
            ax1.plot(history.history[list(history.history.keys())[-1]],
                     label='PSD loss')
            ax1.set_ylabel('loss')
            ax1.set_xlabel('epoch')
            ax1.set_xticks(np.arange(0, int(p['epochs']),
                                     max(int(p['epochs']/10),1)))
            ax1.tick_params('both', labelsize='x-small')
            fig.tight_layout()
            plt.savefig(out_dir + 'loss-PSD.png')
            plt.close(fig)

# == visualizations for unsupervised pipeline =================================

def input_output_plot(x, x_test, x_predict, out, ticid_test=False,
                      inds = [-1,0,1,2,3,4,5,6,7,-2,-3,-4,-5,-6,-7],
                      sharey=False,
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
            axes[ngroup*3,i].plot(x,x_test[inds[ind]], '.k')
            
            # >> plot output
            axes[ngroup*3+1,i].plot(x,x_predict[inds[ind]], '.k')
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

def intermed_act_plot(x, model, activations, x_test, out_dir,
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
        axes.plot(np.linspace(np.min(x), np.max(x), np.shape(x_test)[1]),
                x_test[inds[c]], '.k')
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
                    ax.plot(x1, activation[inds[c]][:,b], '.k')
                else:
                    ax.plot(x1, activation[inds[c]], '.k')
                
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
                                 sharey=False, mock_data=False,
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
            axes[ngroup*3,i].plot(x,x_test[inds[ind]], '.k')
            axes[ngroup*3+1,i].plot(np.linspace(np.min(x),np.max(x),
                                              len(bottleneck[inds[ind]])),
                                              bottleneck[inds[ind]], '.k')
            axes[ngroup*3+2,i].plot(x,x_predict[inds[ind]], '.k')
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
    

def movie(x, model, activations, x_test, p, out_dir, ticid_test, inds = [0, -1]):
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
        ymin = np.min(ymin) + 0.3*np.median(x_test[inds[c]])
        ymax = np.max(ymax) - 0.3*np.median(x_test[inds[c]])

        # >> plot input
        axes.plot(np.linspace(np.min(x), np.max(x), np.shape(x_test)[1]),
                  x_test[inds[c]], '.k')
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
                          activation[inds[c]], '.k')
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
                              y, '.k')
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

def plot_lof(time, intensity, targets, features, n, path,
             momentum_dump_csv = '../../Table_of_momentum_dumps.csv',
             n_neighbors=20, target_info=False, p=2, metric='minkowski',
             contamination=0.1, algorithm='auto',
             prefix='', mock_data=False, feature_vector=False,
             n_tot=100, log=False, debug=False, feature_lof=None,
             bins=50, database_dir=None, single_file=False,
             fontsize='xx-small', title=True, plot_psd=True, n_pgram=1500):
    """ Plots the 20 most and least interesting light curves based on LOF.
    Parameters:
        * time : array with shape 
        * intensity
        * targets : list of TICIDs
        * n : number of curves to plot in each figure
        * path : output directory
        * n_tot : total number of light curves to plots (number of figures =
                  n_tot / n)
        * feature vector : assumes x axis is latent dimensions, not time  
        * mock_data : if True, will not plot TICID label
        * target_input : [sector, camera, ccd]
    Outputs:
        * Text file with TICID in column 1, and LOF in column 2 (lof-*.txt)
        * Log histogram of LOF (lof-histogram.png)
        * Top 20 light curves with highest and lowest LOF
        * Random 20 light curves
    modified [lcg 07012020 - includes inset histogram plotting]
    """
    # -- calculate LOF -------------------------------------------------------
    print('Calculating LOF')
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, p=p, metric=metric,
                             contamination=contamination, algorithm=algorithm)
    fit_predictor = clf.fit_predict(features)
    negative_factor = clf.negative_outlier_factor_
    
    lof = -1 * negative_factor
    ranked = np.argsort(lof)
    largest_indices = ranked[::-1][:n_tot] # >> outliers
    smallest_indices = ranked[:n_tot] # >> inliers
    random_inds = list(range(len(lof)))
    random.Random(4).shuffle(random_inds)
    random_inds = random_inds[:n_tot] # >> random
    ncols=1
    
    if debug:
        # >> figure out what the LOF is triggering on
        ncols=2
        if type(feature_lof) == type(None):
            feature_lof = []
            for i in range(features.shape[1]):
                clf = LocalOutlierFactor(n_neighbors=n_neighbors, p=p)
                fit_predictor = clf.fit_predict(features[:,i].reshape(-1,1))
                negative_factor = clf.negative_outlier_factor_
                lof_tmp = -1 * negative_factor    
                feature_lof.append(lof_tmp) 
            feature_lof=np.array(feature_lof)
            # >> has shape (num_features, num_light_curves)
            
        if plot_psd:
            ncols=3
            freq, tmp = LombScargle(time, intensity[0]).autopower()
            freq = np.linspace(np.min(freq), np.max(freq), n_pgram)            
    
    # >> save LOF values in txt file
    print('Saving LOF values')
    with open(path+'lof-'+prefix+'kneigh' + str(n_neighbors)+'.txt', 'w') as f:
        for i in range(len(targets)):
            f.write('{} {}\n'.format(int(targets[i]), lof[i]))
        f.write("Ten highest LOF\n")
        for k in range(n):
            ind = largest_indices[k]
            f.write(str(targets[ind]) + " " + str(features[ind]) + "\n")
      
    # >> make histogram of LOF values
    print('Make LOF histogram')
    #plot_histogram(lof, 20, "Local Outlier Factor (LOF)", time, intensity,
     #              targets, path+'lof-'+prefix+'histogram-insets.png',
      #             insets=True, log=log)
    plot_histogram(lof, 20, "Local Outlier Factor (LOF)", time, intensity,
                   targets, path+'lof-'+prefix+'histogram.png', insets=False,
                   log=log)

    if not mock_data:
        print('Saving LOF values')
        with open(path+'lof-'+prefix+'kneigh' + str(n_neighbors)+'.txt', 'w') as f:
            for i in range(len(targets)):
                f.write('{} {}\n'.format(int(targets[i]), lof[i]))
          
        # >> make histogram of LOF values
        print('Make LOF histogram')
        plot_histogram(lof, 20, "Local Outlier Factor (LOF)", time, intensity,
                       targets, path+'lof-'+prefix+'kneigh' + str(n_neighbors)+\
                           'histogram-insets.png',
                       insets=True, log=log)
        plot_histogram(lof, 20, "Local Outlier Factor (LOF)", time, intensity,
                       targets, path+'lof-'+prefix+'kneigh' + str(n_neighbors)+\
                           'histogram.png', insets=False,
                       log=log)

        
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
    num_figs = int(n_tot/n) # >> number of figures to generate
    
    for j in range(num_figs):
        
        for i in range(3): # >> loop through smallest, largest, random LOF plots
            fig, ax = plt.subplots(n, ncols, sharex=False,
                                   figsize = (8*ncols, 3*n))
            
            for k in range(n): # >> loop through each row
                if debug:
                    axis = ax[k, 0]
                else:
                    axis = ax[k]
                
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
                           fontsize=fontsize)
                format_axes(axis, ylabel=True)
                if not mock_data:
                    ticid_label(axis, targets[ind], target_info[ind],
                                title=True)
                    if targets[ind] in ticid_classified:
                        classified_ind = np.nonzero(ticid_classified == targets[ind])[0][0]
                        classification_label(axis, targets[ind],
                                             class_info[classified_ind])                        
                    
                if k != n - 1:
                    axis.set_xticklabels([])
                    
                if debug:
                    # >> find the feature this light curve is triggering on
                    feature_ranked = np.argsort(feature_lof[:,ind])
                    ax[k,1].plot(features[:,feature_ranked[-1]],
                                 features[:,feature_ranked[-2]], '.', ms=1)
                    ax[k,1].plot([features[ind,feature_ranked[-1]]],
                                  [features[ind,feature_ranked[-2]]], 'Xg',
                                   ms=30)    
                    ax[k,1].set_xlabel('\u03C6' + str(feature_ranked[-1]))
                    ax[k,1].set_ylabel('\u03C6' + str(feature_ranked[-2])) 
                    # ax[k,1].set_adjustable("box")
                    # ax[k,1].set_aspect(1)    

                if plot_psd:
                    power = LombScargle(time, intensity[ind]).power(freq)
                    ax[k,2].plot(freq, power, '-k')
                    ax[k,2].set_ylabel('Power')
                    # ax[k,2].set_xscale('log')
                    ax[k,2].set_yscale('log')
                    
                    
    
            # >> label axes
            if debug:
                ax[n-1,0].set_xlabel('time [BJD - 2457000]')
                if plot_psd:
                    ax[n-1,2].set_xlabel('Frequency [days^-1]')
            else:
                ax[n-1].set_xlabel('time [BJD - 2457000]')
                
            # >> save figures
            if i == 0:
                if title:
                    fig.suptitle(str(n) + ' largest LOF targets', fontsize=16,
                                 y=0.9)
                fig.tight_layout()
                fig.savefig(path + 'lof-' + prefix + 'kneigh' + \
                            str(n_neighbors) + '-largest_' + str(j*n) + 'to' +\
                            str(j*n + n) + '.png',
                            bbox_inches='tight')
                plt.close(fig)
            elif i == 1:
                if title:
                    fig.suptitle(str(n) + ' smallest LOF targets', fontsize=16,
                                 y=0.9)
                fig.tight_layout()
                fig.savefig(path + 'lof-' + prefix + 'kneigh' + \
                            str(n_neighbors) + '-smallest' + str(j*n) + 'to' +\
                            str(j*n + n) + '.png',
                            bbox_inches='tight')
                plt.close(fig)
            else:
                if title:
                    fig.suptitle(str(n) + ' random LOF targets', fontsize=16, y=0.9)
                
                # >> save figure
                fig.tight_layout()
                fig.savefig(path + 'lof-' + prefix + 'kneigh' + str(n_neighbors) \
                            + "-random"+ str(j*n) + 'to' +\
                            str(j*n + n) +".png", bbox_inches='tight')
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

    # plt.show()
    
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


def plot_reconstruction_error(time, intensity, x_test, x_predict, ticid_test,
                              output_dir='./', err=None, mock_data=False,
                              feature_vector=False, n=20, target_info=False,
                              prefix=''):
    if type(err) == type(None):
        print('Calculating reconstruction error ...')
        err = (x_test - x_predict)**2
        err = np.mean(err, axis=1)
        err = err.reshape(np.shape(err)[0])
    
    # >> get top n light curves
    ranked = np.argsort(err)
    largest_inds = np.copy(ranked[::-1][:n])
    smallest_inds = np.copy(ranked[:n])
    random.Random(4).shuffle(ranked)
    random_inds = ranked[:n]
    
    # >> save in txt file
    if not mock_data:
        out = np.column_stack([ticid_test.astype('int'), err])
        np.savetxt(output_dir+prefix+'reconstruction_error.txt', out, fmt='%-16s')
    
    
    plt.figure()
    plt.hist(err, bins=50)
    plt.savefig(output_dir + prefix + 'reconstruction_error_histogram.png')
    plt.close()
    
    for i in range(3):
        fig, ax = plt.subplots(n, 1, sharex=True, figsize = (8, 3*n))
        for k in range(n): # >> loop through each row
            if i == 0: ind = largest_inds[k]
            elif i == 1: ind = smallest_inds[k]
            else: ind = random_inds[k]
            
            # >> plot light curve
            ax[k].plot(time, intensity[ind], '.k')
            if not feature_vector:
                ax[k].plot(time, x_predict[ind], '.')
            text='mae: '+str(np.mean(np.abs(intensity[ind]-x_predict[ind])))+\
                '\nmse: '+str(err[ind])+\
                '\nmce: '+str(np.mean(np.abs((intensity[ind]-x_predict[ind])**3)))
            ax[k].text(0.98, 0.02, 'mse: ' +str(err[ind]),
                       transform=ax[k].transAxes, horizontalalignment='right',
                       verticalalignment='bottom', fontsize='xx-small')
            format_axes(ax[k], ylabel=True)
            if not mock_data:
                ticid_label(ax[k], ticid_test[ind], target_info[ind],
                            title=True)
                
        if feature_vector:
            ax[n-1].set_xlabel('\u03C8')
        else:
            ax[n-1].set_xlabel('Time [BJD - 2457000]')
        if i == 0:
            fig.suptitle('largest reconstruction error', fontsize=16, y=0.9)
            fig.savefig(output_dir+prefix+'reconstruction_error-largest.png',
                        bbox_inches='tight')
        elif i == 1:
            fig.suptitle('smallest reconstruction error', fontsize=16, y=0.9)
            fig.savefig(output_dir+prefix+'reconstruction_error-smallest.png',
                        bbox_inches='tight')
        else:
            fig.suptitle('random reconstruction error', fontsize=16, y=0.9)
            fig.savefig(output_dir+prefix+'reconstruction_error-random.png',
                        bbox_inches='tight')            
        plt.close(fig)

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
    
def quick_plot_classification(time, intensity, targets, target_info, features, labels,
                              path='./', prefix='', 
                              simbad_database_txt='./simbad_database.txt',
                              title='', ncols=10, nrows=5,
                              database_dir='./databases/', single_file=False):
    '''Unfinished. Aim is to give an overview of the classifications, by
    plotting the first 5 light curves of each class. Any light curves
    classified by Simbad will be plotted first in their respective classes.'''
    classes, counts = np.unique(labels, return_counts=True)
    # colors=['red', 'blue', 'green', 'purple', 'yellow', 'cyan', 'magenta',
    #         'skyblue', 'sienna', 'palegreen', 'darksalmon', 'sandybrown',
    #         'lightsalmon', 'lightslategray', 'fuchsia', 'deeppink', 'crimson']*10
    colors = get_colors()
    
    # class_info = dt.get_simbad_classifications(targets, simbad_database_txt)
    # ticid_classified = np.array(simbad_info)[:,0].astype('int')    
    class_info = dt.get_true_classifications(targets,
                                             database_dir=database_dir,
                                             single_file=single_file)
    ticid_classified = class_info[:,0].astype('int')
    
    num_figs = int(np.ceil(len(classes) / ncols))
    features_greek = [r'$\alpha$', 'B', r'$\Gamma$', r'$\Delta$', r'$\beta$', r'$\gamma$',r'$\delta$',
                  "E", r'$\epsilon$', "Z", "H", r'$\eta$', r'$\Theta$', "I", "K", r'$\Lambda$', "M", r'$\mu$'
                  ,"N", r'$\nu$']
    
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
            
            # >> find light curves with this class and with true classifications
            # inds = np.isin(ticid_simbad, targets[class_inds])
            inds = np.isin(targets[class_inds], ticid_classified)
            classified_inds = class_inds[np.nonzero(inds)]
            not_classified_inds = class_inds[np.nonzero(~inds)]
            
            if class_num == -1:
                color = 'black'
            elif class_num < len(colors) - 1:
                color = colors[class_num]
            else:
                color='black'
                
            k=-1
            # >> first plot any Simbad classified light curves
            for k in range(min(nrows, len(classified_inds))): 
                ind = classified_inds[k] # >> to index targets
                classified_ind = np.nonzero(ticid_classified == targets[ind])[0][0]
                ax[k, j].plot(time, intensity[ind], '.k')
                # simbad_label(ax[k,j], targets[ind], simbad_info[simbad_ind])
                classification_label(ax[k,j], targets[ind],
                                     class_info[classified_ind])
                ticid_label(ax[k, j], targets[ind], target_info[ind],
                            title=True, color=color)
                format_axes(ax[k, j], ylabel=True)
            
            # >> now plot non-classified light curves
            for l in range(k+1, min(nrows, len(not_classified_inds))):
                ind = not_classified_inds[l]
                ax[l,j].plot(time, intensity[ind], '.k')
                ticid_label(ax[l,j], targets[ind], target_info[ind],
                            title=True, color=color)
                format_axes(ax[l,j], ylabel=False)
                
            # ax[0, j].set_title('Class ' + str(class_num), color=color)   
                
            #get median features for the class
                #which feature vectors do i need
                #get only those feature vectors
                #take median and convert to string
            relevant_feats  =[]
            for k in range(len(labels)):
                if labels[k] == class_num:
                    relevant_feats.append(int(k))
            #index just this list 
            features_byclass = features[relevant_feats]
            med_features = np.median(features_byclass, axis=0)
            med_string = str(med_features)
            ax[0, j].set_title('Class '+str(class_num)+ "# Curves:" + str(counts[j]) +
                               '\n Median Features:' + med_string + 
                               "\n"+ax[0,j].get_title(),
                               color=color, fontsize='xx-small')
            ax[-1, j].set_xlabel('Time [BJD - 2457000]')   
                        
            if j == 0:
                for m in range(nrows):
                    ax[m, 0].set_ylabel('Relative flux')
                    
        fig.tight_layout()
        fig.savefig(path + prefix + '-' + str(i) + '.png')
        # fig.savefig(path + prefix + '-' + str(i) + '.pdf')
        plt.close(fig)
                
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
    random.Random(4).shuffle(sorted_names)
    sorted_names = sorted_names*20
    
    sorted_names.append('black')
    return sorted_names
    
def plot_classification(time, intensity, targets, labels, path,
                        momentum_dump_csv = './Table_of_momentum_dumps.csv',
                        n=20, target_info=False,
                        prefix='', mock_data=False,
                        feature_vector=False):
    """ 
    """

    classes, counts = np.unique(labels, return_counts=True)
    # !!
    colors=['red', 'blue', 'green', 'purple', 'yellow', 'cyan', 'magenta',
            'skyblue', 'sienna', 'palegreen']*10
    
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
        
def plot_pca(bottleneck, classes, n_components=2, output_dir='./', prefix=''):
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
    # colors=['red', 'blue', 'green', 'purple', 'yellow', 'cyan', 'magenta',
    #     'skyblue', 'sienna', 'palegreen']*10
    colors = get_colors() 
    # >> loop through classes
    class_labels = np.unique(classes)
    for i in range(len(class_labels)):
        inds = np.nonzero(classes == class_labels[i])
        # if class_labels[i] == 0:
        #     color='r'
        # elif class_labels[i] == 1:
        #     color = 'b'
        # elif class_labels[i] == 2:
        #     color='g'
        # elif class_labels[i] == 3:
        #     color='m'
        # else:
        #     color='k'
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

# == helper functions =========================================================

def ticid_label(ax, ticid, target_info, title=False, color='black',
                fontsize='xx-small'):
    '''Query catalog data and add text to axis.
    Parameters:
        * target_info : [sector, camera, ccd, data_type, cadence]
    TODO: use Simbad classifications and other cross-checking database
    classifications'''
    try:
        # >> query catalog data
        target, Teff, rad, mass, GAIAmag, d, objType, Tmag = \
            dt.get_tess_features(ticid)
        # features = np.array(dt.get_tess_features(ticid))[1:]

        # >> change sigfigs for effective temperature
        if np.isnan(Teff):
            Teff = 'nan'
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
    
        info = target+'\nTeff {}\nrad {}\nmass {}\nG {}\nd {}\nO {}\nTmag {}'
        info1 = target+', Sector {}, Cam {}, CCD {}, {}, Cadence {},\n' +\
            'Teff {}, rad {}, mass {}, G {}, d {}, O {}, Tmag {}'
        
        
        # >> make text
        if title:
            ax.set_title(info1.format(sector, cam, ccd, data_type, cadence,
                                      Teff, '%.2g'%rad, '%.2g'%mass,
                                      '%.3g'%GAIAmag, '%.3g'%d, objType,
                                      '%.3g'%Tmag),
                         fontsize=fontsize, color=color)
        else:
            ax.text(0.98, 0.98, info.format(Teff, '%.2g'%rad, '%.2g'%mass, 
                                            '%.3g'%GAIAmag, '%.3g'%d, objType,
                                            Tmag),
                      transform=ax.transAxes, horizontalalignment='right',
                      verticalalignment='top', fontsize=fontsize)
    except (ConnectionError, OSError, TimeoutError):
        print("there was a connection error!")
        ax.text(0.98, 0.98, "there was a connection error",
                      transform=ax.transAxes, horizontalalignment='right',
                      verticalalignment='top', fontsize=fontsize)
            
def simbad_label(ax, ticid, simbad_info):
    '''simbad_info = [ticid, main_id, otype, bibcode]'''
    # ind = np.nonzero(np.array(simbad_info)[:,0].astype('int') == ticid)
    # if np.shape(ind)[1] != 0:
    #     ticid, main_id, otype, bibcode = simbad_info[ind[0][0]]
    ticid, main_id, otype, bibcode = simbad_info
    ax.text(0.98, 0.98, 'otype: '+otype+'\nmain_id: '+main_id+'\n'+bibcode,
            transform=ax.transAxes, fontsize='xx-small',
            horizontalalignment='right', verticalalignment='top')
    
def classification_label(ax, ticid, classification_info, fontsize='xx-small'):
    '''classification_info = [ticid, otype, main id]'''
    ticid, otype, bibcode = classification_info
    ax.text(0.98, 0.98, 'otype: '+otype+'\nmaind_id: '+bibcode,
            transform=ax.transAxes, fontsize=fontsize,
            horizontalalignment='left', verticalalignment='top')
    
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


def latent_space_plot(activation, out='./latent_space.png', n_bins = 20,
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
            # f1, a1 = plt.subplots()
            # a1.set_xlabel(ax_label+str(i))
            # a1.hist(activation[:,i], n_bins, log=log)
            # f1.savefig(out.split('.')[0]+'-phi'+str(i)+'.png')

            # axes[i,i].set_aspect(aspect='equal', adjustable='box')
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
        fig.savefig(out)
        plt.close(fig)
    
    return fig, axes
    # return fig, axes
    
    
# def plot_tsne(bottleneck, labels, X=None, n_components=2, output_dir='./',
#               prefix=''):
#     if type(X) == type(None):
#         from sklearn.manifold import TSNE
#         X = TSNE(n_components=n_components).fit_transform(bottleneck)
#     unique_classes = np.unique(labels)
#     colors = get_colors()
    
#     plt.figure()
#     for i in range(len(unique_classes)):
#         # >> find all light curves with this  class
#         class_inds = np.nonzero(labels == unique_classes[i])
        
#         if unique_classes[i] == -1:
#             color = 'black'
#         elif unique_classes[i] < len(colors) - 1:
#             color = colors[unique_classes[i]]
#         else:
#             color='black'
        
#         plt.plot(X[class_inds][:,0], X[class_inds][:,1], '.', color=color)
        
#     plt.savefig(output_dir + prefix + 't-sne.png')
#     plt.close()
    
def get_tsne(bottleneck, n_components=2):
    from sklearn.manifold import TSNE
    X = TSNE(n_components=n_components).fit_transform(bottleneck)
    return X
    

def get_gcvs_variability_types(labels):
    var_d = {'eruptive':
         ['Fl', 'BE', 'FU', 'GCAS', 'I', 'IA', 'IB', 'IN', 'INA', 'INB', 'INT,IT',
          'IN(YY)', 'IS', 'ISA', 'ISB', 'RCB', 'RS', 'SDOR', 'UV', 'UV', 'UVN',
          'WR', 'INTIT', 'GCAS'],
         'pulsating':
             ['Pu', 'ACYG', 'BCEP', 'BCEPS', 'BLBOO', 'CEP', 'CEP(B)', 'CW', 'CWA',
              'CWB', 'DCEP', 'DCEPS', 'DSCT', 'DSCTC', 'GDOR', 'L', 'LB', 'LC',
              'LPB', 'M', 'PVTEL', 'RPHS', 'RR', 'RR(B)', 'RRAB', 'RRC', 'RV',
              'RVA', 'RVB', 'SR', 'SRA', 'SRB' 'SRC', 'SRD', 'SRS', 'SXPHE',
              'ZZ', 'ZZA', 'ZZB', 'ZZO'],
         'rotating': ['ACV', 'ACVO', 'BY', 'ELL', 'FKCOM', 'PSR',
                      'R', 'SXARI'],
         'cataclysmic':
             ['N', 'NA', 'NB', 'NC', 'NL', 'NR', 'SN', 'SNI', 'SNII', 'UG',
              'UGSS', 'UGSU', 'UGZ', 'ZAND'],
         'eclipsing':
             ['E', 'EA', 'EB', 'EP', 'EW', 'GS', 'PN', 'RS', 'WD', 'WR', 'AR',
              'D', 'DM', 'DS', 'DW', 'K', 'KE', 'KW', 'SD'],
             'xray':
             ['AM', 'X', 'XB', 'XF', 'XI', 'XJ', 'XND', 'XNG', 'XP', 'XPR', 'XPRM',
              'XM'],
             'other': ['VAR']} 

    subclasses = []
    for var_type in list(var_d.keys()):
        subclasses.extend(var_d[var_type])

    # >> initialize counts dictionary
    type_inds = {}
    for var_type in ['eruptive', 'pulsating', 'rotating', 'cataclysmic',
                     'eclipsing', 'xray', 'other', 'not classified']:
        type_inds[var_type] = []

    for i in range(len(labels)):
        if labels[i] == 'NONE':
            type_inds['not classified'].append(i)
        else:
            otypes = labels[i].split('|')
            for var_type in list(var_d.keys()):
                inter = np.intersect1d(otypes, var_d[var_type])
                if len(inter) > 0:
                    type_inds[var_type].append(i)

            if np.intersect1d(otypes, subclasses) == 0:

                type_inds['other'].append(i)



    return type_inds
            

def get_classifications(ticid_pred, database_dir, class_info=None, parents=None,
                             parent_dict=None, merge_classes=True):
    if type(class_info) == type(None):
        class_info = dt.get_true_classifications(ticid_pred,
                                                 database_dir=database_dir)

    if merge_classes:
        if type(parent_dict) == type(None):
            parent_dict=dt.make_parent_dict()
        if type(parents) == type(None):
            parents = list(parent_dict.keys())

        remove_classes=['V', 'VAR', '**', '*i', '*iC', '*iA','*iN', 'Em']

        class_info = dt.get_parents_only(class_info, parents=parents,
                                         parent_dict=parent_dict,
                                         remove_classes=remove_classes,
                                         remove_flags=['+', '/', ':'])


    return class_info

def ensemble_summary_plots(ticid_pred, y_pred, sector='all',
                       data_dir='./data/', output_dir='./',
                       prefix=''):
    
    '''
    Outputs:
        * An array of tuples [label_pred, label_true].
        * A dictionary with keys of real_labels and values of English
          descriptions.'''

    orig_ticid = ticid_pred
    orig_y = y_pred

    database_dir = data_dir+'databases/'
    num_samples = len(ticid_pred)
    
    # -- create confusion matrix -----------------------------------------------

    # >> get learned classifications
    class_list_pred, counts = np.unique(y_pred, return_counts=True)
    class_list_pred = class_list_pred.astype('str')
        
    # >> get 'ground truth' classifications
    ticid_true, label_true = dt.get_true_classifications(ticid_pred, data_dir,
                                                         sector=sector)
    ticid_true, label_true = dt.get_parent_otypes(ticid_true, label_true)
        
    # >> get classified objects
    _, comm1, comm2 = np.intersect1d(ticid_pred,ticid_true,return_indices=True)
    ticid_pred = ticid_pred[comm1]
    y_pred = y_pred[comm1]
    ticid_true = ticid_true[comm2]
    label_true = label_true[comm2]
    class_list_true = np.unique(label_true)
    
    # >> create number labels for true labels so we can make confusion matrix
    y_true = []
    for i in range(len(ticid_true)):
        class_num = np.nonzero(class_list_true == label_true[i])[0][0]
        y_true.append(class_num)
    y_true = np.array(y_true).astype('int')
    
    # >> create a confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # >> make the matrix square so that we can apply linear_sum_assignment
    while len(class_list_pred) < len(cm):
        class_list_pred = np.append(class_list_pred, 'X')     
    while len(class_list_true) < len(cm):
        class_list_true = np.append(class_list_true, 'X')
        
    # >> make confusion matrix diagonal by re-ordering columns
    row_ind, col_ind = linear_sum_assignment(-1*cm)
    cm = cm[:,col_ind]
    class_list_pred = class_list_pred[col_ind]
    
    # -- make assignment dictionary --------------------------------------------

    # >> create a list of tuples [label_pred, label_true]
    print('Saving assignment dictionary...')
    assignments = []
    f = open(output_dir+prefix+'assignments.txt', 'a')
    for i in range(len(class_list_pred)):
        # >> check if there is a real label assigned
        if class_list_pred[i] != 'X' and class_list_true[i] != 'X':
            assignments.append([class_list_pred[i], class_list_true[i]])
            f.write(class_list_pred[i]+','+class_list_true[i]+'\n')
    assignments = np.array(assignments)

    # -- save results ----------------------------------------------------------
    y_pred = orig_y
    ticid = orig_ticid
    ticid_label = []
    with open(output_dir+prefix+'ticid_to_label.txt', 'w') as f:
        for i in range(len(orig_ticid)):
            ind = np.nonzero(assignments[:,0].astype('float') == y_pred[i])
            if len(ind[0]) == 0:
                ticid_label.append('NONE')
                f.write(str(ticid[i])+',NONE\n')
            else:
                ticid_label.append(str(assignments[:,1][ind][0]))
                f.write(str(ticid[i])+','+str(assignments[:,1][ind][0])+'\n')
    print('Saved '+output_dir+prefix+'ticid_to_label.txt')

    # -- plotting --------------------------------------------------------------
    plot_confusion_matrix(cm, class_list_pred, class_list_true, output_dir, prefix)
    recall, false_discovery_rate, precision, accuracy, counts_true, counts_pred=\
        evaluate_classifications(cm, class_list_true)

    # >> create summary pie charts
    ensemble_budget(ticid, y_pred, cm, assignments, class_list_true, class_list_true,
                    database_dir=data_dir+'databases/', output_dir=output_dir, 
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

    return assignments, ticid_label


    

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
    

def ensemble_summary_tables(labels, recalls, false_discovery_rates, precisions,
                            accuracy, counts_true, counts_pred,  output_dir='./',
                            target_labels = ['E', 'EA', 'EW', 'EB', 'L', 'LB', 'S', 'V', 'SB', 'HS']):
    
    if len(target_labels) > 0:
        target_inds = []
        for i in range(len(labels)):
            if labels[i] in target_labels:
                target_inds.append(i)
        labels = np.array(labels)[target_inds]
        recalls = np.array(recalls)[target_inds]
        false_discovery_rates = np.array(false_discovery_rates)[target_inds]
        precisions = np.array(precisions)[target_inds]
        accuracy = np.array(accuracy)[target_inds]
        counts_true = np.array(counts_true)[target_inds]
        counts_pred = np.array(counts_pred)[target_inds]
        
    
    # -- prediction summary table ---------------------------------------------
    table_columns = ['false discovery rate', 'number of predictions']
    table_data = np.array([false_discovery_rates, counts_pred]).T
    table_data = np.round(table_data, 2)
    fig, ax = plt.subplots(1,1, figsize=(15,0.25*len(labels)))
    ax.table(cellText=table_data, rowLabels=labels, fontsize=12.,
             colLabels=table_columns, loc='center', colWidths=[0.2]*len(table_columns))
    ax.axis('off')
    fig.tight_layout()
    plt.subplots_adjust(left=0.2)
    if len(target_labels) > 0:
        fig.savefig(output_dir+'ensemble_summary_target_prediction.png',dpi=300)
        print('Saved '+output_dir+'ensemble_summary_target_prediction.png')
    else:
        fig.savefig(output_dir+'ensemble_summary_table_all_prediction.png')       
        print('Saved '+output_dir+'ensemble_summary_table_all_prediction.png')
    
    # -- target summary table -------------------------------------------------    
    table_columns = ['recall', 'accuracy', 'number of targets']
    table_data = np.array([recalls, accuracy, counts_pred]).T
    df = pd.DataFrame(table_data)
    df[0] = np.round(table_data[:,0], 2)
    df[2] = df[2].astype('int')
    fig, ax = plt.subplots(1,1, figsize=(15,0.25*len(labels)))
    ax.table(cellText=dt.values, rowLabels=labels, fontsize=12.,
             colLabels=table_columns, loc='center', colWidths=[0.2]*len(table_columns))
    ax.axis('off')
    fig.tight_layout()

    # plt.subplots_adjust(left=0.2)
    if len(target_labels) > 0:
        fig.savefig(output_dir+'ensemble_summary_table_target_true.png',dpi=300)
        print('Saved '+output_dir+'ensemble_summary_table_target_true.png')
    else:
        fig.savefig(output_dir+'ensemble_summary_table_all_true.png')    
        print('Saved '+output_dir+'ensemble_summary_table_all_true.png')

    table_columns = ['recall', 'number of targets']
    table_data = np.array([recalls, counts_pred]).T
    df = pd.DataFrame(table_data)
    df[0] = np.round(table_data[:,0], 2)
    df[1] = df[1].astype('int')
    fig, ax = plt.subplots(1,1, figsize=(15,0.25*len(labels)))
    # pd.plotting.table(ax, df, rowLabels=labels, colLabels=table_columns)
    ax.table(cellText=dt.values, rowLabels=labels, fontsize=12.,
             colLabels=table_columns, loc='center', colWidths=[0.2]*len(table_columns))
    ax.axis('off')
    fig.tight_layout()

    # plt.subplots_adjust(left=0.2)
    if len(target_labels) > 0:
        fig.savefig(output_dir+'ensemble_summary_table_target_recall.png',dpi=300)
        print('Saved '+output_dir+'ensemble_summary_table_target_recall.png')
    else:
        fig.savefig(output_dir+'ensemble_summary_table_all_recall.png')    
        print('Saved '+output_dir+'ensemble_summary_table_all_recall.png')


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
             
# def plot_light_curves(ticid, target_info):                       

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

# def ensemble_summary_plots(ticid, labels, output_dir, data_dir, sectors,
#                            prefix='', merge_classes=True):
#     # >> before making a confusion matrix, assign each science label a number
#     underlying_classes  = np.unique(labels)
#     assignments = []
#     for i in range(len(underlying_classes)):
#         assignments.append([i, underlying_classes[i]])
#     assignments = np.array(assignments)

#     # >> get the true labels (looks for data_dir/databases/SectorX_true_labels.txt')
#     if gcvs_only:
#         ticid_true, label_true = dt.get_gcvs_classifications(data_dir+'databases/')
#         class_info = np.concatenate([np.expand_dims(ticid_true,1),
#                                      np.repeat(np.expand_dims(label_true, 1),
#                                                2, 1)], 1)
#         class_info =  dt.get_parents_only(class_info)

#     else:
#         class_info = get_classifications(ticid_pred, data_dir+'databases/')
#     ticid_true = class_info[:,0].astype('float')
#     y_true = class_info[:,1]

#     class_info_pred = np.repeat(np.expand_dims(labels,1),3, axis=1)
#     class_info_pred[:,0]  = ticid
#     labels = class_info_pred[:,1]
#     ticid = class_info_pred[:,0].astype('float')

#     inter, comm1, comm2 = np.intersect1d(ticid_true, ticid,
#                                          return_indices=True)
#     ticid_true = ticid_true[comm1]
#     y_true = y_true[comm1]
#     y_pred = labels[comm2]

#     row_labels = np.unique(np.concatenate([y_true, y_pred]))

#     # >> create confusion matrix
#     cm = confusion_matrix(y_true, y_pred, labels=row_labels)
#     plot_confusion_matrix(cm, row_labels, row_labels, output_dir, prefix)
#     recall, false_discovery_rate, precision, accuracy, counts_true, counts_pred=\
#         evaluate_classifications(cm, row_labels)

#     # >> create summary pie charts
#     ensemble_budget(ticid, labels, cm, assignments, row_labels, row_labels,
#                         database_dir=data_dir+'./databases', output_dir=output_dir, 
#                         prefix=prefix, data_dir=data_dir)
#     recall=np.array(recall)
#     # target_labels = row_labels[np.nonzero((recall > 0.5) * (recall <1.))]
#     target_labels = ['CW', 'ELL|X', 'EW', 'RV']
#     ensemble_summary_tables(row_labels, recall, false_discovery_rate,
#                                precision, accuracy, counts_true, counts_pred,
#                                output_dir+prefix, target_labels=target_labels)
#     ensemble_summary_tables(row_labels, recall, false_discovery_rate,
#                             precision, accuracy, counts_true, counts_pred,
#                             output_dir+prefix, target_labels=[])

#     # >> distribution plots
#     inter, comm1, comm2 = np.intersect1d(ticid.astype('float'), ticid_true.astype('float'),
#                                          return_indices=True)
#     y_pred = labels[comm1]

#     classes, counts = np.unique(y_true, return_counts=True)
#     classes = classes[np.argsort(counts)]

#     print('Plot feature ditsributions...')
#     plot_class_dists(assignments, ticid_true, y_pred, y_true, data_dir, sectors,
#                      label_list=classes[-20:], output_dir=output_dir+prefix)
#     plot_class_dists(assignments, ticid_true, y_pred, y_true, data_dir, sectors,
#                      label_list=target_labels, output_dir=output_dir+prefix)

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
    
        
def get_lof(ticid_feat, features, database_dir='./databases/',
            n_neighbors=20, output_dir='./'):
    class_info = dt.get_true_classifications(ticid_feat,
                                             database_dir=database_dir)
    ticid_classified = class_info[:,0].astype('int')     
    
    print('Calculating LOF')
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)
    fit_predictor = clf.fit_predict(features)
    negative_factor = clf.negative_outlier_factor_
    lof = -1 * negative_factor   
    lof = np.array(lof) 
    
    intersection, comm1, comm2 = np.intersect1d(ticid_feat, ticid_classified,
                                                return_indices=True)
    
    sorted_lof = np.argsort(lof[comm1])
    
    with open(output_dir+'lof_classified.txt', 'w') as f:
        for i in range(len(intersection)):
            intersection_ind = sorted_lof[-1*i - 1]
            lof_ind = comm1[intersection_ind]
            class_ind = comm2[intersection_ind]
            f.write('{} {} {} {} \n'.format(ticid_feat[lof_ind],
                                              lof[lof_ind],
                                              class_info[class_ind][1],
                                              class_info[class_ind][2]))    
    
    return lof
    
    
def paper_schematic(x_test, x_predict, output_dir='./'):
    x_predict = np.reshape(x_predict, (np.shape(x_predict)[0], 
                                       np.shape(x_predict)[1]))
    err = (x_test - x_predict)**2
    err = np.mean(err, axis=1)
    err = err.reshape(np.shape(err)[0])
    ranked = np.argsort(err)  
    ind = ranked[0]
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.plot(x_test[ind], '.k', markersize=1)
    
    fig1, ax = plt.subplots(figsize=(8, 3))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.plot(x_predict[ind], '.k', markersize=1)    

def plot_lc(time, flux, target_info, ticid, ind, output_dir='./',
            momentum_dump_csv = '../../Table_of_momentum_dumps.csv',
            plot_mom_dump=False):
    # -- momentum dumps ------------------------------------------------------
    # >> get momentum dump times
    print('Loading momentum dump times')
    with open(momentum_dump_csv, 'r') as f:
        lines = f.readlines()
        mom_dumps = [ float(line.split()[3][:-1]) for line in lines[6:] ]
        inds = np.nonzero((mom_dumps >= np.min(time)) * \
                          (mom_dumps <= np.max(time)))
        mom_dumps = np.array(mom_dumps)[inds]    
    
    # -- plot -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8,3))
    if plot_mom_dump:
        for t in mom_dumps:
            ax.axvline(t, color='g', linestyle='--')
        
    ax.plot(time, flux[ind][0], '.k', ms=2)
    format_axes(ax, xlabel=True)
    ticid_label(ax, ticid[ind], target_info[ind][0], title=True)      
    ax.set_ylabel('Flux')
    
    fig.tight_layout()
    fig.savefig(output_dir + 'TICID' + str(int(ticid[ind])) + '.png')
    
    return fig, ax
    
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
    

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
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


# def plot_fail_cases(assigned_labels, ticid_pred, y_pred,
#                     ticid_true, y_true, class_info):
    
#     true_label = 'Nl'
#     assigned_class = int(assigned_labels[true_label])
#     nrows = 10
#     ncols = 2
    
#     predicted_ticid = ticid_pred[np.nonzero(y_pred == assigned_class)]
#     true_ticid = class_info[:,0][np.nonzero(y_true == assigned_class)].astype('int')
#     intersection, comm1, comm2 = np.intersect1d(predicted_ticid, true_ticid,
#                                                 return_indices=True)
    
#     fig, ax = plt.subplots(nrows, ncols, sharex=True, figsize=(8*ncols*0.75, 3*nrows))
#     ax[0,0].set_title('Success')
#     ax[0,1].set_title('Fail')
#     for i in range(nrows):
#         ax[i,0].plot()
    
    
    
def plot_cross_identifications(time, intensity, targets, target_info, features,
                               labels, path='./', prefix='', 
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
                ax[k, j].plot(time, intensity[flux_ind], '.k')
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
        fig.savefig(path + prefix + '-underlying-class-'+ str(i) + '.png')
        # fig.savefig(path + prefix + '-' + str(i) + '.pdf')
        plt.close(fig)                
   
def plot_class_dists(assignments, ticid, y_pred, y_true, data_dir, sectors, 
                     label_list=[], output_dir='./'):
    
    data = []
    var_data = []
    var_tic = []
    for sector in sectors:
        print('Loading TICv8 catalog data: Sector '+str(sector))
        sector_data = pd.read_csv(data_dir+'Sector'+str(sector)+\
                                  '/Sector'+str(sector)+'tic_cat_all.csv')
        data.append(sector_data)

        tic, var = np.loadtxt(data_dir+'Sector'+str(sector)+\
                              '/Sector'+str(sector)+'variability_statistics.txt')
        var_tic.append(tic)
        var_data.append(var)
    data = pd.concat(data)
    var_data = np.concatenate(var_data, axis=0)
    var_tic = np.concatenate(var_tic)

    _, _, inds = np.intersect1d(ticid, data['ID'],
                                return_indices=True)
    data = data.iloc[inds]

    _, _, inds = np.intersect1d(ticid.astype('float'), var_tic, return_indices=True)
    var_data = var_data[inds]

    # sector_dists(data, var_data, ticid,
    #              output_dir, prefix='all-')


    if len(label_list) == 0:
        # >> create plot for every class
        for i in range(len(assignments)):
            # >> find TICIDs of the true positives
            ticid_pred = ticid[np.nonzero(y_pred == int(float(assignments[i][0])))]
            ticid_true = ticid[np.nonzero(y_true == assignments[i][1])]
            
            sector_dists(data,  var_data, ticid_pred, output_dir,
                         prefix=assignments[i][1]+'_PRED_')

            sector_dists(data, var_data, ticid_true, output_dir,
                         prefix=assignments[i][1]+'_TRUE_')
            
    else:
        for true_label in label_list:
            if len( np.nonzero(assignments[:,1] == true_label)[0]) > 0:
                # ind = np.nonzero(assignments[:,1] == true_label)[0][0]
                # # >> find TICIDs of the true positives
                # inds = np.nonzero(y_pred == int(float(assignments[ind][0])))
                inds = np.nonzero(y_pred == true_label)
                ticid_pred = ticid[inds]
                
                # if len(inds[0]) > 0:
                sector_dists(data.iloc[inds], var_data[inds], ticid_pred,
                             output_dir, prefix=true_label+'_PRED_')

                inds = np.nonzero(y_true == true_label)
                ticid_true = ticid[inds]
                if len(inds[0]) > 0:
                    sector_dists(data.iloc[inds], var_data[inds], ticid_true,
                                 output_dir, prefix=true_label+'_TRUE_')
    
    



def sector_dists(data, var_data, ticid_list=[], output_dir='./',
                 figsize=(2,2), prefix='', bins=40):
    
    # >> make sure prefix doesn't have '/'
    prefix = prefix.replace('/', '-')
    
    # >> get features for TICIDs in ticid_list
    columns = ['Teff', 'rad', 'mass', 'GAIAmag', 'd']
    xlabels = ['$T_{eff}$ [K]', 'Radius [$R_{\odot}$]', 'Mass [$M_{\odot}$]',
               'GAIA Mag', 'Distance [kpc]']


    for i in range(len(columns)):
        if np.count_nonzero(np.isnan(data[columns[i]])) != len(data[columns[i]]):
            fig, ax = plt.subplots(figsize=figsize)
            ax.hist(data[columns[i]], bins=bins)
            ax.set_xlabel(xlabels[i])
            # ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title('Mean '+columns[i]+': '+\
                         str(round(np.mean(data[columns[i]]), 2)))
            fig.tight_layout()
            fig.savefig(output_dir+prefix+columns[i]+'.png', dpi=300)
            plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(var_data, bins=bins)
    ax.set_xlabel('$R_{var}$')
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Mean $R_{var}$: '+str(round(np.mean(var_data),2)))
    fig.tight_layout()
    fig.savefig(output_dir+prefix+'Rvar.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data['Teff'], var_data, '.k', markersize=1)
    ax.set_xlabel('$T_{eff}$ [K]')
    ax.set_ylabel('$R_{var}$')
    # ax.set_xscale('log')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(output_dir+prefix+'Rvar_Teff.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data['GAIAmag'], var_data, '.k', markersize=1)
    ax.set_xlabel('GAIA Mag')
    ax.set_ylabel('$R_{var}$')
    # ax.set_xscale('log')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(output_dir+prefix+'Rvar_GAIAmag.png', dpi=300)
    plt.close(fig)

    print('Saved '+output_dir+prefix)
