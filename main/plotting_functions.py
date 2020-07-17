# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 21:58:45 2020

Plotting functions only. 

@author: Lindsey Gordon @lcgordon and Emma Chickles @emmachickles

Last updated: June 15 2020

Helper functions
* plot_lc()
* ticid_label()
* astroquery_pull_data !! combine with get_tess_features() and ticid_label()
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

"""
import numpy as np
import numpy.ma as ma 
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)                
import pdb # >> debugging tool
import model as ml


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

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

import astroquery
from astroquery.simbad import Simbad
from astroquery.mast import Catalogs
from astroquery.mast import Observations
from astroquery import exceptions
from astroquery.exceptions import RemoteServiceError
from astroquery.mast import Tesscut

import data_functions as df


def test_plotting():
    print("Plotting functions loaded in")
    
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
       
def lof_and_insets_on_sector(pathtofolder, sector, numberofplots, momentumdumppath, sigma):
    """loads in a sector and plots lof +insets """
    

    flux, x, ticid, target_info = df.load_data_from_metafiles(pathtofolder, sector, cams=[1,2,3,4],
                                 ccds=[1,2,3,4], DEBUG=False,
                                 output_dir=pathtofolder, debug_ind=10, nan_mask_check=True)
    featuresallpath = pathtofolder + "Sector" + str(sector) + "_features_v0_all.fits"
    f = fits.open(featuresallpath, mmap=False)
    
    features = f[0].data
    #targetsfits = f[1].data
    f.close()

    flux = df.normalize(flux, axis=1)
    
    features, ticid, flux, outlier_indexes = isolate_plot_feature_outliers(pathtofolder, sector, features, x, flux, ticid, sigma)
    
    plot_lof(x, flux, ticid, features, sector, pathtofolder,
                 momentum_dump_csv = momentumdumppath,
                 n_neighbors=20, target_info=target_info,
                 prefix='', mock_data=False, addend=1., feature_vector=True,
                 n_tot=numberofplots)

    features_insets(x, flux, features, ticid, pathtofolder)
    
    return features, x, flux, ticid, outlier_indexes    
        
def isolate_plot_feature_outliers(path, sector, features, time, flux, ticids, sigma):
    """ isolate features that are significantly out there and crazy
    plot those outliers, and remove them from the features going into the 
    main lof/plotting/
    parameters: 
        *path to save shit into
        * features (all)
        * time axis (1) (ALREADY PROCESSED)
        * flux (all) (must ALREADY BE PROCESSED)
        * ticids (all)
        
    returns: features_cropped, ticids_cropped, flux_cropped 
    modified [lcg 07172020]"""
    rcParams['figure.figsize'] = 8,3
    features_greek = [r'$\alpha$', 'B', r'$\Gamma$', r'$\Delta$', r'$\beta$', r'$\gamma$',r'$\delta$',
                  "E", r'$\epsilon$', "Z", "E", r'$\eta$', r'$\Theta$', "I", "K", r'$\Lambda$']
    outlier_indexes = []
    for i in range(len(features[0])):
        column = features[:,i]
        column_std = np.std(column)
        column_top = np.mean(column) + column_std * sigma
        column_bottom = np.mean(column) - (column_std * sigma)
        for n in range(len(column)):
            #find and note the position of any outliers
            if column[n] < column_bottom or column[n] > column_top: 
                outlier_indexes.append((int(n), int(i)))
                
    print(np.asarray(outlier_indexes))
        
    outlier_indexes = np.asarray(outlier_indexes)
    
    for i in range(len(outlier_indexes)):
        plt.scatter(time, flux[outlier_indexes[i][0]], s=0.5)
        target = ticids[outlier_indexes[i][0]]
        
        feature_value = '%s' % float('%.2g' % features[i][outlier_indexes[i][1]])
        feature_title = features_greek[outlier_indexes[i][1]] + "=" + feature_value
        
        plt.title("TIC " + str(int(target)) + " " + astroquery_pull_data(target, breaks=False) + 
                  "\n" + feature_title + "  Sigma cap: " + str(sigma), fontsize=8)
        plt.tight_layout()
        plt.savefig((path + "featureoutlier-SECTOR" + str(sector) +"-TICID" + str(int(target)) + ".png"))
        plt.show()
        
        
    features_cropped = np.delete(features, outlier_indexes, axis=0)
    ticids_cropped = np.delete(ticids, outlier_indexes)
    flux_cropped = np.delete(flux, outlier_indexes, axis=0)
        
    return features_cropped, ticids_cropped, flux_cropped, outlier_indexes


def features_plotting_2D(feature_vectors, cluster_columns, path, clustering,
                         time, intensity, targets, folder_suffix='',
                         feature_engineering=True, eps=0.5, min_samples=10,
                         metric='euclidean', algorithm='auto', leaf_size=30,
                         p=2, target_info=False,
                         momentum_dump_csv='./Table_of_momentum_dumps.csv'):
    """plotting (n 2) features against each other
    feature_vectors is the list of ALL feature_vectors
    cluster_columns is the vectors that you want to use to do the clustering based on
        this can be the same as feature_vectors
        path needs to end in a backslash
    clustering must equal 'dbscan', 'kmeans', or 'empty'
    """
    #detrmine which of the clustering algoirthms you're using: 
    folder_label = "blank"
    if clustering == 'dbscan':
        # !! TODO parameter optimization (eps, min_samples)
        # cluster_columns = ml.standardize(cluster_columns, ax=0)
        db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
                    algorithm=algorithm, leaf_size=leaf_size,
                    p=p).fit(cluster_columns) #eps is NOT epochs
        classes_dbscan = db.labels_
        numclasses = str(len(set(classes_dbscan)))
        folder_label = "dbscan-colored" + folder_suffix

    elif clustering == 'kmeans': 
        Kmean = KMeans(n_clusters=4, max_iter=700, n_init = 20)
        x = Kmean.fit(cluster_columns)
        classes_kmeans = x.labels_
        folder_label = "kmeans-colored"
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
        os.makedirs(folder_path + "-new")
    else:
        print ("Successfully created the directory %s" % folder_path) 
 
    if clustering == 'dbscan':
        plot_classification(time, intensity, targets, db.labels_,
                            path+folder_label+'/', prefix='dbscan',
                            momentum_dump_csv=momentum_dump_csv,
                            target_info=target_info)
        plot_pca(cluster_columns, db.labels_,
                    output_dir=path+folder_label+'/')
    elif clustering == 'kmeans':
        plot_classification(time, intensity, targets, x.labels_,
                            path+folder_label+'/', prefix='kmeans',
                            momentum_dump_csv=momentum_dump_csv,
                            target_info=target_info)
    # >> [etc 060620]
    colors = ["red","blue", "green", "purple" ,"yellow", "magenta", 'black']
    colors=['red', 'blue', 'green', 'purple', 'yellow', 'cyan', 'magenta',
            'skyblue', 'sienna', 'palegreen', 'black']
    #creates labels based on if engineered features or not
    if feature_engineering:
        graph_labels = ["Average", "Variance", "Skewness", "Kurtosis", "Log Variance",
                        "Log Skewness", "Log Kurtosis", "Maximum Power", "Log Maximum Power", 
                        "Period of Maximum Power (0.1 to 10 days)","Slope" , "Log Slope",
                        "P0", "P1", "P2", "Period of Maximum Power (0.001 to 0.1 days)"]
        fname_labels = ["Avg", "Var", "Skew", "Kurt", "LogVar", "LogSkew", "LogKurt",
                        "MaxPower", "LogMaxPower", "Period0_1to10", "Slope", "LogSlope",
                        "P0", "P1", "P2", "Period0to0_1"]
        num_features = 16
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
            
    #actually plotting the damn things
 
            if clustering == 'dbscan':
                plt.figure() # >> [etc 060520]
                plt.clf()
                for n in range(len(feature_vectors)):
                    plt.scatter(feat1[n], feat2[n], c=colors[classes_dbscan[n]], s=2)
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig((folder_path + "/" + fname_label1 + "-vs-" + fname_label2 + "-dbscan.png"))

                 
            elif clustering == 'kmeans':
                plt.figure() # >> [etc 060520]
                plt.clf()
                for n in range(len(feature_vectors)):
                    plt.scatter(feat1[n], feat2[n], c=colors[classes_kmeans[n]], s=2)
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig(folder_path + "/" + fname_label1 + "-vs-" + fname_label2 + "-kmeans.png")
                  
            elif clustering == 'none':
                plt.scatter(feat1, feat2, s = 2, color = 'black')
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig(folder_path + "/" + fname_label1 + "-vs-" + fname_label2 + ".png")
                # plt.show()
                
    if clustering == 'dbscan':
        return db.labels_
    if clustering == 'kmeans':
        return x.labels
                          
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
        T_eff = np.round(catalog_data[0]["Teff"], 0)
        obj_type = catalog_data[0]["objType"]
        gaia_mag = np.round(catalog_data[0]["GAIAmag"], 2)
        radius = np.round(catalog_data[0]["rad"], 2)
        mass = np.round(catalog_data[0]["mass"], 2)
        distance = np.round(catalog_data[0]["d"], 1)
        if breaks:
            title = "T_eff:" + str(T_eff) + "," + str(obj_type) + ", G: " + str(gaia_mag) + "\n Dist: " + str(distance) + ", R:" + str(radius) + " M:" + str(mass)
        else: 
             title = "T_eff:" + str(T_eff) + "," + str(obj_type) + ", G: " + str(gaia_mag) + "Dist: " + str(distance) + ", R:" + str(radius) + " M:" + str(mass)
    except (ConnectionError, OSError, TimeoutError):
        print("there was a connection error!")
        title = "Connection error, no data"
    return title


#inset plotting -------------------------------------------------------------------------------------------

def features_insets(time, intensity, feature_vectors, targets, path):
    """ Plots 2 features against each other with the extrema points' associated
    light curves plotted as insets along the top and bottom of the plot. 
    
    time is the time axis for the group
    intensity is the full list of intensities
    feature_vectors is the complete list of feature vectors
    targets is the complete list of targets
    folder is the folder into which you wish to save the folder of plots. it 
    should be formatted as a string, with no trailing /
    modified [lcg 06292020]
    """   
    path = path + "/2DFeatures-insets"
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
        print("New folder created will have -new at the end. Please rename.")
        path = path + "-new"
        os.makedirs(path)
    else:
        print ("Successfully created the directory %s" % path) 
 
    graph_labels = ["Average", "Variance", "Skewness", "Kurtosis", "Log Variance",
                    "Log Skewness", "Log Kurtosis", "Maximum Power", "Log Maximum Power", 
                    "Period of Maximum Power (0.1 to 10 days)","Slope" , "Log Slope",
                    "P0", "P1", "P2", "Period of Maximum Power (0.001 to 0.1 days)"]
    fname_labels = ["Avg", "Var", "Skew", "Kurt", "LogVar", "LogSkew", "LogKurt",
                    "MaxPower", "LogMaxPower", "Period0_1to10", "Slope", "LogSlope",
                    "P0", "P1", "P2", "Period0to0_1"]
    for n in range(16):
        graph_label1 = graph_labels[n]
        fname_label1 = fname_labels[n]
        for m in range(16):
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
        axis_name.set_title(astroquery_pull_data(targets[inset_indexes[n]]), fontsize=6)
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

    indexes_unique = np.unique(np.asarray(indexes))
    
    return indexes_unique      

# colored inset plotting -------------------------------------------------------
def features_insets_colored(time, intensity, feature_vectors, targets, path, realclasses):
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
                    "P0", "P1", "P2", "Period of Maximum Power (0.001 to 0.1 days)"]
    fname_labels = ["Avg", "Var", "Skew", "Kurt", "LogVar", "LogSkew", "LogKurt",
                    "MaxPower", "LogMaxPower", "Period0_1to10", "Slope", "LogSlope",
                    "P0", "P1", "P2", "Period0to0_1"]
    
    db = DBSCAN(eps=2.2, min_samples=18).fit(feature_vectors) #eps is NOT epochs
    guessclasses = db.labels_
    
    for n in range(16):
        graph_label1 = graph_labels[n]
        fname_label1 = fname_labels[n]
        for m in range(16):
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
    colors = ["red","blue", "green", "purple" ,"yellow", "magenta", 'black']
    
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

def histo_features(features, bins, t, intensities, targets, path, insets=False):
    """ Plot histograms of each feature, both with and without inset light curves
    features (array of ALL), bins, time axis, intensities, path to put folder in
    modified [lcg 07012020]"""
    folderpath = path + "/feature_histograms/"
    try:
        os.makedirs(folderpath)
    except OSError:
        print ("Directory %s already exists" % folderpath)
        
    fname_labels = ["Avg", "Var", "Skew", "Kurt", "LogVar", "LogSkew", "LogKurt",
                        "MaxPower", "LogMaxPower", "Period0_1to10", "Slope", "LogSlope",
                        "P0", "P1", "P2", "Period0to0_1"]
    for n in range(16):
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
    plt.savefig(filename)
    plt.close()


# plotting features by color and shape
def features_2D_colorshape(feature_vectors, path, clusteralg, hand_classes):
    """ plots features against each other
    COLORING based on the given hand classes. 
    SHAPE based on the assigned class by the given cluster algorithm
    folderpath and clusteralg should be strings
    """
    if clusteralg == 'dbscan':
        db = DBSCAN(eps=2.2, min_samples=18).fit(feature_vectors) #eps is NOT epochs
        classes_dbscan = db.labels_

    elif clusteralg == 'kmeans': 
        Kmean = KMeans(n_clusters=4, max_iter=700, n_init = 20)
        x = Kmean.fit(feature_vectors)
        classes_kmeans = x.labels_
    else: 
        print("please enter a valid clustering algorithm")
 
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
        print("New folder created will have -new at the end. Please rename.")
        os.makedirs(path + "-new")
    else:
        print ("Successfully created the directory %s" % path) 
 
    graph_labels = ["Average", "Variance", "Skewness", "Kurtosis", "Log Variance",
                    "Log Skewness", "Log Kurtosis", "Maximum Power", "Log Maximum Power", 
                    "Period of Maximum Power (0.1 to 10 days)","Slope" , "Log Slope",
                    "P0", "P1", "P2", "Period of Maximum Power (0.001 to 0.1 days)"]
    fname_labels = ["Avg", "Var", "Skew", "Kurt", "LogVar", "LogSkew", "LogKurt",
                    "MaxPower", "LogMaxPower", "Period0_1to10", "Slope", "LogSlope",
                    "P0", "P1", "P2", "Period0to0_1"]
    for n in range(16):
        feat1 = feature_vectors[:,n]
        graph_label1 = graph_labels[n]
        fname_label1 = fname_labels[n]
        for m in range(16):
            if m == n:
                continue
            graph_label2 = graph_labels[m]
            fname_label2 = fname_labels[m]                
            feat2 = feature_vectors[:,m]
            
            colors = ["red", "blue", "green", "purple", "yellow", "magenta", "black"]
            shapes = ['.', 'P', 'h', '+', 'x']
            
            if clusteralg == 'dbscan':
                for p in range(len(feature_vectors)):
                    #assign a color
                    c = colors[classes_dbscan[p]]
                    
                    if classes_dbscan[p] == hand_classes[p]:
                        s = '^' #if they match the arrow goes up
                    else:
                        s = 'v' #if they do not match the arrow goes down
                    
                    plt.scatter(feat1[p], feat2[p], c = c, s = 1, marker=s)
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig((path + "/" + fname_label1 + "-vs-" + fname_label2 + "-dbscan.pdf"))
                plt.show()
            elif clusteralg == 'kmeans':
                for p in range(len(feature_vectors)):
                    #assign color
                    c = colors[classes_kmeans[p]]
                    if classes_kmeans[p] == hand_classes[p]:
                        s = '^'
                    else:
                        s = 'v'
                    plt.scatter(feat1[p], feat2[p], c = c,s = 1, marker=s)
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig(path + "/" + fname_label1 + "-vs-" + fname_label2 + "-kmeans.pdf")
                plt.show()
                 

def diagnostic_plots(history, model, p, output_dir, 
                     x, x_train, x_test, x_predict, 
                     target_info_test=False, target_info_train=[],
                     sharey=False, prefix='',
                     mock_data=False, ticid_train=False, ticid_test=False,
                     supervised=False, y_true=False, y_predict=False,
                     y_train=False, y_test=False,
                     flux_test=False, flux_train=False, time=False,
                     rms_train=False, rms_test = False, input_rms = False,
                     inds = [-1,0,1,2,3,4,5,6,7,-2,-3,-4,-5,-6,-7],
                     intermed_inds = [6,0],
                     input_bottle_inds = [0,1,2,-6,-7],
                     addend = 1., feature_vector=False, percentage=False,
                     input_features = False, load_bottleneck=False, n_tot=100,
                     plot_epoch = True,
                     plot_in_out = True,
                     plot_in_bottle_out=False,
                     plot_latent_test = False,
                     plot_latent_train = False,
                     plot_kernel=False,
                     plot_intermed_act=False,
                     plot_clustering=False,
                     make_movie = False,
                     plot_lof_test=True,
                     plot_lof_train=True,
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
    
    # >> plot loss, accuracy, precision, recall vs. epochs
    if plot_epoch:
        print('Plotting loss vs. epoch')
        epoch_plots(history, p, output_dir+prefix+'epoch-',
                    supervised=supervised)   

    # -- unsupervised ---------------------------------------------------------
    # >> plot some decoded light curves
    if plot_in_out and not supervised:
        print('Plotting input, output and residual')
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
        
    # -- intermediate activations visualization -------------------------------
    if plot_intermed_act or make_movie:
        print('Calculating intermediate activations')
        activations = ml.get_activations(model, x_test) 
    if plot_intermed_act:
        print('Plotting intermediate activations')
        intermed_act_plot(x, model, activations, x_test,
                          output_dir+prefix+'intermed_act-', addend=addend,
                          inds=intermed_inds, feature_vector=feature_vector)
    
    if make_movie:
        print('Making movie of intermediate activations')
        movie(x, model, activations, x_test, p, output_dir+prefix+'movie-',
              ticid_test, addend=addend, inds=intermed_inds)        
        
    # >> plot kernel vs. filter
    if plot_kernel:
        print('Plotting kernel vs. filter')
        kernel_filter_plot(model, output_dir+prefix+'kernel-')
        
    # -- latent space visualization -------------------------------------------
        
    # >> get bottleneck
    if input_features:
        features = []
        for ticid in ticid_test:
            res = df.get_tess_features(ticid)
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
            bottleneck = ml.get_bottleneck(model, x_test,
                                           input_features=input_features,
                                           features=features,
                                           input_rms=input_rms,
                                           rms=rms_test)
            # activations = ml.get_activations(model, x_test)
            # bottleneck = ml.get_bottleneck(model, activations, p,
            #                             input_features=input_features,
            #                             features=features,
            #                             input_rms=input_rms, rms=rms_test)
            # >> remove activations from memory
            # del activations
        
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
        latent_space_plot(bottleneck, p, output_dir+prefix+'latent_space.png')
    
    # >> plot the 20 light curves with the highest LOF
    if plot_lof_test:
        print('Plotting LOF for testing set')
        for n in [20]: # [20, 50, 100]: loop through n_neighbors
            if feature_vector:
                plot_lof(time, flux_test, ticid_test, bottleneck, 20,
                         output_dir, prefix='test-'+prefix, n_neighbors=n,
                         mock_data=mock_data, feature_vector=feature_vector,
                         n_tot=n_tot, target_info=target_info_test)
            else:
                plot_lof(x, x_test, ticid_test, bottleneck, 20, output_dir,
                         prefix = 'test-'+prefix, n_neighbors=n,
                         mock_data=mock_data, feature_vector=feature_vector,
                         n_tot=n_tot, target_info=target_info_test)
    
    if plot_latent_train or plot_lof_train or plot_lof_all:
        if load_bottleneck:
            print('Loading bottleneck (training set)')
            with fits.open(output_dir + 'bottleneck_train.fits', mmap=False) as hdul:
                bottleneck_train = hdul[0].data
        else:
            print('Getting bottleneck (training set)')
            bottleneck_train = ml.get_bottleneck(model, x_train,
                                                 input_features=input_features,
                                                 features=features,
                                                 input_rms=input_rms,
                                                 rms=rms_train)
            # activations_train = get_activations(model, x_train)        
            # bottleneck_train = get_bottleneck(model, activations_train, p,
            #                                   input_features=input_features,
            #                                   features=features,
            #                                   input_rms=input_rms,
            #                                   rms=rms_train)
            # # >> remove activations from memory
            # del activations_train
        
    if plot_latent_train:
        print('Plotting latent space for training set')
        latent_space_plot(bottleneck_train, p, output_dir+prefix+\
                          'latent_space-x_train.png')        
        
    if plot_lof_train:
        print('Plotting LOF for testing set')
        for n in [20]: # [20, 50, 100]:
            # if type(flux_train) != bool:
            if feature_vector: # >> x_train is features, not fluxes
                plot_lof(time, flux_train, ticid_train, bottleneck_train, 20,
                         output_dir, prefix='train-'+prefix, n_neighbors=n,
                         mock_data=mock_data, feature_vector=feature_vector,
                         n_tot=n_tot, target_info=target_info_train)
            else:
                plot_lof(x, x_train, ticid_train, bottleneck_train, 20,
                         output_dir, prefix = 'train-'+prefix, n_neighbors=n,
                         mock_data=mock_data, feature_vector=feature_vector,
                         n_tot=n_tot, target_info=target_info_train)   
                
    if plot_lof_all:
        print('Plotting LOF for entire dataset')
        bottleneck_all = np.concatenate([bottleneck, bottleneck_train], axis=0)
        # # >> save to fits file
        # hdr = fits.Header()
        # hdu=fits.PrimaryHDU(bottleneck_all, header=hdr)
        # hdu.writeto(output_dir+'bottleneck.fits')
        if feature_vector:
            plot_lof(time, np.concatenate([flux_test, flux_train], axis=0),
                     np.concatenate([ticid_test, ticid_train]), bottleneck_all,
                     20, output_dir, prefix='all-'+prefix, n_neighbors=n,
                     mock_data=mock_data, feature_vector=feature_vector,
                     n_tot=n_tot,
                     target_info=np.concatenate([target_info_test,
                                                 target_info_train], axis=0))   
        else:
            plot_lof(x, np.concatenate([x_test, x_train], axis=0),
                     np.concatenate([ticid_test, ticid_train], axis=0),
                     bottleneck_all, 20, output_dir, prefix='all-'+prefix,
                     n_neighbors=20, n_tot=n_tot,
                     mock_data=mock_data, feature_vector=feature_vector,
                     target_info=np.concatenate([target_info_test,
                                                 target_info_train], axis=0))
    
    # -- plot reconstruction error (unsupervised) -----------------------------
    # >> plot light curves with highest, smallest and random reconstruction
    #    error
    if plot_reconstruction_error_test:
        print('Plotting reconstruction error for testing set')
        plot_reconstruction_error(x, x_test, x_test, x_predict, ticid_test,
                                  output_dir=output_dir,
                                  target_info=target_info_test)
    
    if plot_reconstruction_error_all:
        print('Plotting reconstruction error for entire dataset')
        # >> concatenate test and train sets
        tmp = np.concatenate([x_test, x_train], axis=0)
        tmp_predict = model.predict(tmp)
        plot_reconstruction_error(x, tmp, tmp, tmp_predict, 
                                  np.concatenate([ticid_test, ticid_train],
                                                 axis=0),
                                  output_dir=output_dir,
                                  target_info=\
                                      np.concatenate([target_info_test,
                                                      target_info_train]))
        # >> remove x_train reconstructions from memory
        del tmp    
        
    # return activations, bottleneck

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
                      percentage=False, target_info=False):
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
                ticid_label(axes[ngroup*3,i], ticid_test[inds[ind]],
                            target_info[ind], title=True)
                
            # >> plot input
            axes[ngroup*3,i].plot(x,x_test[inds[ind]]+addend, '.k')
            
            # >> plot output
            axes[ngroup*3+1,i].plot(x,x_predict[inds[ind]]+addend, '.k')
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

def plot_lof(time, intensity, targets, features, n, path,
             momentum_dump_csv = '../../Table_of_momentum_dumps.csv',
             n_neighbors=20, target_info=False,
             prefix='', mock_data=False, addend=1., feature_vector=False,
             n_tot=100):
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
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)
    fit_predictor = clf.fit_predict(features)
    negative_factor = clf.negative_outlier_factor_
    
    lof = -1 * negative_factor
    ranked = np.argsort(lof)
    largest_indices = ranked[::-1][:n_tot] # >> outliers
    smallest_indices = ranked[:n_tot] # >> inliers
    
    # >> save LOF values in txt file
    print('Saving LOF values')
    with open(path+'lof-'+prefix+'.txt', 'w') as f:
        for i in range(len(targets)):
            f.write('{} {}\n'.format(int(targets[i]), lof[i]))
      
    # >> make histogram of LOF values
    print('Make LOF histogram')
    plot_histogram(lof, 20, "Local Outlier Factor (LOF)", time, intensity,
                   targets, path+'lof-'+prefix+'histogram-insets.png',
                   insets=True, log=False)
    plot_histogram(lof, 20, "Local Outlier Factor (LOF)", time, intensity,
                   targets, path+'lof-'+prefix+'histogram.png', insets=False,
                   log=False)
        
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
    num_figs = int(n_tot/n) # >> number of figures to generate
    
    for j in range(num_figs):
        
        for i in range(2): # >> loop through smallest and largest LOF plots
            fig, ax = plt.subplots(n, 1, sharex=True, figsize = (8, 3*n))
            
            for k in range(n): # >> loop through each row
                if i == 0: ind = largest_indices[j*n + k]
                elif i == 1: ind = smallest_indices[j*n + k]\
                
                # >> plot momentum dumps
                for t in mom_dumps:
                    ax[k].axvline(t, color='g', linestyle='--')
                    
                # >> plot light curve
                ax[k].plot(time, intensity[ind] + addend, '.k')
                ax[k].text(0.98, 0.02, '%.3g'%lof[ind],
                           transform=ax[k].transAxes,
                           horizontalalignment='right',
                           verticalalignment='bottom',
                           fontsize='xx-small')
                format_axes(ax[k], ylabel=True)
                if not mock_data:
                    ticid_label(ax[k], targets[ind], target_info[ind],
                                title=True)
    
            # >> label axes
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
            ax[k].axvline(t, color='g', linestyle='--')
            
        # >> plot light curve
        ax[k].plot(time, intensity[ind] + addend, '.k')
        ax[k].text(0.98, 0.02, '%.3g'%lof[ind], transform=ax[k].transAxes,
                   horizontalalignment='right', verticalalignment='bottom',
                   fontsize='xx-small')
        
        # >> formatting
        format_axes(ax[k], ylabel=True)
        if not mock_data:
            ticid_label(ax[k], targets[ind], target_info[ind], title=True)
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
                              feature_vector=False, n=20, target_info=False):
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
            ax[k].plot(time, intensity[ind]+addend, '.k')
            if not feature_vector:
                ax[k].plot(time, x_predict[ind]+addend, '-')
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
            ax[n-1].set_xlabel('time [BJD - 2457000]')
        if i == 0:
            fig.suptitle('largest reconstruction error', fontsize=16, y=0.9)
            fig.savefig(output_dir + 'reconstruction_error-largest.png',
                        bbox_inches='tight')
        else:
            fig.suptitle('smallest reconstruction error', fontsize=16, y=0.9)
            fig.savefig(output_dir + 'reconstruction_error-smallest.png',
                        bbox_inches='tight')
        plt.close(fig)
    
def quick_plot_classification(time, intensity, targets, target_info, labels,
                              path='./', prefix='', addend=1.,
                              simbad_database_txt='./simbad_database.txt',
                              title='', ncols=10, nrows=5):
    '''Unfinished. Aim is to give an overview of the classifications, by
    plotting the first 5 light curves of each class. Any light curves
    classified by Simbad will be plotted first in their respective classes.'''
    classes, counts = np.unique(labels, return_counts=True)
    # colors=['red', 'blue', 'green', 'purple', 'yellow', 'cyan', 'magenta',
    #         'skyblue', 'sienna', 'palegreen', 'darksalmon', 'sandybrown',
    #         'lightsalmon', 'lightslategray', 'fuchsia', 'deeppink', 'crimson']*10
    colors = get_colors()
    
    simbad_info = df.get_simbad_classifications(targets, simbad_database_txt)
    ticid_simbad = np.array(simbad_info)[:,0].astype('int')
    
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
            
            # >> find light curves with this class and classified in Simbad
            # inds = np.isin(ticid_simbad, targets[class_inds])
            inds = np.isin(targets[class_inds], ticid_simbad)
            simbad_inds = class_inds[np.nonzero(inds)]
            not_simbad_inds = class_inds[np.nonzero(~inds)]
            
            if class_num == -1:
                color = 'black'
            elif class_num < len(colors) - 1:
                color = colors[class_num]
            else:
                color='black'
                
            k=-1
            # >> first plot any Simbad classified light curves
            for k in range(min(nrows, len(simbad_inds))): 
                ind = simbad_inds[k] # >> to index targets
                simbad_ind = np.nonzero(ticid_simbad == targets[ind])[0][0]
                ax[k, j].plot(time, intensity[ind]+addend, '.k')
                simbad_label(ax[k,j], targets[ind], simbad_info[simbad_ind])
                ticid_label(ax[k, j], targets[ind], target_info[ind],
                            title=True, color=color)
                format_axes(ax[k, j], ylabel=True)
            
            # >> now plot non-classified light curves
            for l in range(k+1, min(nrows, len(not_simbad_inds))):
                ind = not_simbad_inds[l]
                ax[l,j].plot(time, intensity[ind]+addend, '.k')
                ticid_label(ax[l,j], targets[ind], target_info[ind],
                            title=True, color=color)
                format_axes(ax[l,j], ylabel=False)
                
            # ax[0, j].set_title('Class ' + str(class_num), color=color)            
            ax[0, j].set_title('Class '+str(class_num)+'\n'+ax[0,j].get_title(),
                               color=color, fontsize='xx-small')
            ax[-1, j].set_xlabel('Time [BJD - 2457000]')   
                        
            if j == 0:
                for m in range(nrows):
                    ax[m, 0].set_ylabel('Relative flux')
                    
        # fig.tight_layout()
        fig.savefig(path + prefix + '-' + str(i) + '.png')
        plt.close(fig)
                
def get_colors():
    '''Returns list of 125 colors for plotting against white background1.'''
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
                  'lavenderblush']
    for i in range(len(bad_colors)):
        ind = sorted_names.index(bad_colors[i])
        sorted_names.pop(ind)
        
    # >> now shuffle
    random.Random(4).shuffle(sorted_names)
    
    return sorted_names
    
def plot_classification(time, intensity, targets, labels, path,
                        momentum_dump_csv = './Table_of_momentum_dumps.csv',
                        n=20, target_info=False,
                        prefix='', mock_data=False, addend=1.,
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

# == helper functions =========================================================

def ticid_label(ax, ticid, target_info, title=False, color='black'):
    '''Query catalog data and add text to axis.
    Parameters:
        * target_info : [sector, camera, ccd, data_type, cadence]'''
    try:
        # >> query catalog data
        target, Teff, rad, mass, GAIAmag, d, objType = \
            df.get_tess_features(ticid)

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
    
        info = target+'\nTeff {}\nrad {}\nmass {}\nG {}\nd {}\nO {}'
        info1 = target+', Sector {}, Cam {}, CCD {}, {}, Cadence {},\n' +\
            'Teff {}, rad {}, mass {}, G {}, d {}, O {}'
        
        
        # >> make text
        if title:
            ax.set_title(info1.format(sector, cam, ccd, data_type, cadence,
                                      Teff, '%.2g'%rad, '%.2g'%mass,
                                      '%.3g'%GAIAmag, '%.3g'%d, objType),
                         fontsize='xx-small', color=color)
        else:
            ax.text(0.98, 0.98, info.format(Teff, '%.2g'%rad, '%.2g'%mass, 
                                            '%.3g'%GAIAmag, '%.3g'%d, objType),
                      transform=ax.transAxes, horizontalalignment='right',
                      verticalalignment='top', fontsize='xx-small')
    except (ConnectionError, OSError, TimeoutError):
        print("there was a connection error!")
        ax.text(0.98, 0.98, "there was a connection error",
                      transform=ax.transAxes, horizontalalignment='right',
                      verticalalignment='top', fontsize='xx-small')
            
def simbad_label(ax, ticid, simbad_info):
    '''simbad_info = [ticid, main_id, otype, bibcode]'''
    # ind = np.nonzero(np.array(simbad_info)[:,0].astype('int') == ticid)
    # if np.shape(ind)[1] != 0:
    #     ticid, main_id, otype, bibcode = simbad_info[ind[0][0]]
    ticid, main_id, otype, bibcode = simbad_info
    ax.text(0.98, 0.98, 'otype: '+otype+'\nmain_id: '+main_id+'\n'+bibcode,
            transform=ax.transAxes, fontsize='xx-small',
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
    
def latent_space_plot(activation, p, out, n_bins = 50, log = True,
                      units='phi'):
    '''Creates corner plot of latent space.
        Parameters:
        * bottleneck : bottleneck layer, shape=(num light curves, num features)
        * params : dictionary, with 'latent_dim' key
        * out : output filename
        * n_bins : number of bins in histogram (int)
        * log : if True, plots log histogram
        * units : either 'phi' (learned features), or 'psi'
          (engineered features)
    '''
        
    from matplotlib.colors import LogNorm
    
    latentDim = p['latent_dim']

    fig, axes = plt.subplots(nrows = latentDim, ncols = latentDim,
                             figsize = (10, 10))

    if units == 'phi':
        ax_label = '\u03C61'
    elif units == 'psi':
        ax_label = '\u03C8'

    # >> deal with 1 latent dimension case
    if latentDim == 1:
        axes.hist(np.reshape(activation, np.shape(activation)[0]), n_bins,
                  log=log)
        axes.set_xlabel(ax_label)
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
            axes[i,0].set_ylabel(ax_label + str(i))
            axes[latentDim-1,i].set_xlabel(ax_label + str(i))

        # >> removing axis
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        plt.subplots_adjust(hspace=0, wspace=0)
        
    plt.savefig(out)
    plt.close(fig)
    # return fig, axes
    
def plot_tsne(bottleneck, labels, n_components=2, output_dir='./', prefix=''):
    from sklearn.manifold import TSNE
    X = TSNE(n_components=n_components).fit_transform(bottleneck)
    unique_classes = np.unique(labels)
    colors = get_colors()
    
    plt.figure()
    for i in range(len(unique_classes)):
        # >> find all light curves with this  class
        class_inds = np.nonzero(labels == unique_classes[i])
        
        if unique_classes[i] == -1:
            color = 'black'
        elif unique_classes[i] < len(colors) - 1:
            color = colors[unique_classes[i]]
        else:
            color='black'
        
        plt.plot(X[class_inds][:,0], X[class_inds][:,1], '.', color=colors[i])
        
    plt.savefig(output_dir + prefix + 't-sne.png')
    plt.close()
    
    
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    

# if plot_clustering:
#     bottleneck_ind = np.nonzero(['dense' in x.name for x in \
#                                  model.layers])[0][0]
#     bottleneck = activations[bottleneck_ind - 1]        
#     latent_space_clustering(bottleneck, x_test, x, ticid_test,
#                             out=output_dir+prefix+\
#                                 'clustering-x_test-', addend=addend)

# def features_plotting_2D(feature_vectors, cluster_columns, path, clustering):
#     """plotting (n 2) features against each other
#     feature_vectors is the list of ALL feature_vectors
#     cluster_columns is the vectors that you want to use to do the clustering based on
#         this can be the same as feature_vectors
#     date must be a string in the format of the folder you are saving into ie "4-13"
#     clustering must equal 'dbscan', 'kmeans', or 'empty'
#     """
#     clustering = "empty"
#     folder_label = "blank"
#     if clustering == 'dbscan':
#         db = DBSCAN(eps=2.2, min_samples=18).fit(cluster_columns) #eps is NOT epochs
#         classes_dbscan = db.labels_
#         numclasses = str(len(set(classes_dbscan)))
#         folder_label = "dbscan-colored"
#     elif clustering == 'kmeans': 
#         Kmean = KMeans(n_clusters=4, max_iter=700, n_init = 20)
#         x = Kmean.fit(cluster_columns)
#         classes_kmeans = x.labels_
#         folder_label = "kmeans-colored"
#     else: 
#         print("no clustering chosen")
#         folder_label = "2DFeatures-NoCluster"
#     #makes folder and saves to it    
#     folder_path = path + "/" + folder_label
#     try:
#         os.makedirs(folder_path)
#     except OSError:
#         print ("Creation of the directory %s failed" % folder_path)
#         print("New folder created will have -new at the end. Please rename.")
#         os.makedirs(folder_path + "-new")
#     else:
#         print ("Successfully created the directory %s" % folder_path) 
 
#     graph_labels = ["Average", "Variance", "Skewness", "Kurtosis", "Log Variance",
#                     "Log Skewness", "Log Kurtosis", "Maximum Power", "Log Maximum Power", 
#                     "Period of Maximum Power (0.1 to 10 days)","Slope" , "Log Slope",
#                     "P0", "P1", "P2", "Period of Maximum Power (0.001 to 0.1 days)"]
#     fname_labels = ["Avg", "Var", "Skew", "Kurt", "LogVar", "LogSkew", "LogKurt",
#                     "MaxPower", "LogMaxPower", "Period0_1to10", "Slope", "LogSlope",
#                     "P0", "P1", "P2", "Period0to0_1"]
#     color = ["red", "blue", "green", "purple", "black"]
#     for n in range(16):
#         feat1 = feature_vectors[:,n]
#         graph_label1 = graph_labels[n]
#         fname_label1 = fname_labels[n]
#         for m in range(16):
#             if m == n:
#                 continue
#             graph_label2 = graph_labels[m]
#             fname_label2 = fname_labels[m]                
#             feat2 = feature_vectors[:,m]
            
#             if clustering == 'dbscan':
#                 for p in range(len(feature_vectors)):
#                     plt.scatter(feat1[p], feat2[p], c = color[classes_dbscan[p]], s = 5)
#                 plt.xlabel(graph_label1)
#                 plt.ylabel(graph_label2)
#                 plt.savefig((folder_path + "/" + fname_label1 + "-vs-" + fname_label2 + "-dbscan.pdf"))
#                 plt.show()
#             elif clustering == 'kmeans':
#                 for p in range(len(feature_vectors)):
#                     plt.scatter(feat1[p], feat2[p], c = color[classes_kmeans[p]])
#                 plt.xlabel(graph_label1)
#                 plt.ylabel(graph_label2)
#                 plt.savefig(folder_path + "/" + fname_label1 + "-vs-" + fname_label2 + "-kmeans.pdf")
#                 plt.show()
#             elif clustering == 'none':
#                 plt.scatter(feat1, feat2, s = 2, color = 'black')
#                 #plt.autoscale(enable=True, axis='both', tight=True)
#                 plt.xlabel(graph_label1)
#                 plt.ylabel(graph_label2)
#                 plt.savefig(folder_path + "/" + fname_label1 + "-vs-" + fname_label2 + ".pdf")
#                 plt.show()
                

     
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

