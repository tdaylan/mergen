# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 21:58:45 2020

Plotting functions only. 

@author: Lindsey Gordon @lcgordon

Last updated: June 4 2020
"""
import numpy as np
import numpy.ma as ma 
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)

import scipy.signal as signal
from scipy.stats import moment
from scipy import stats
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
rcParams["lines.markersize"] = 2
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

import shapely
from shapely import geometry
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def test_plotting():
    print("plotting loaded in successfully")
    
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

def post_process_plotting(time, intensity, features_all, features_using, targets, path):
    """plotting all the things"""
    features_plotting_2D(features_all, features_using, path, "none")
    features_plotting_2D(features_all, features_using, path, "kmeans")
    features_plotting_2D(features_all, features_using, path, "dbscan")
    
    plot_lof(time, intensity, targets, features_all, 20, path)

    features_insets2D(time, intensity, features_all, targets, path)


def features_plotting_2D(feature_vectors, cluster_columns, path, clustering):
    """plotting (n 2) features against each other
    feature_vectors is the list of ALL feature_vectors
    cluster_columns is the vectors that you want to use to do the clustering based on
        this can be the same as feature_vectors
    date must be a string in the format of the folder you are saving into ie "4-13"
    clustering must equal 'dbscan', 'kmeans', or 'empty'
    """
    cluster = "empty"
    folder_label = "blank"
    if clustering == 'dbscan':
        db = DBSCAN(eps=2.2, min_samples=18).fit(cluster_columns) #eps is NOT epochs
        classes_dbscan = db.labels_
        numclasses = str(len(set(classes_dbscan)))
        cluster = 'dbscan'
        folder_label = "dbscan-colored"
    elif clustering == 'kmeans': 
        Kmean = KMeans(n_clusters=4, max_iter=700, n_init = 20)
        x = Kmean.fit(cluster_columns)
        classes_kmeans = x.labels_
        cluster = 'kmeans'
        folder_label = "kmeans-colored"
    else: 
        print("no clustering chosen")
        cluster = 'none'
        folder_label = "2DFeatures"
    #makes folder and saves to it    
    folder_path = path + "/" + folder_label
    try:
        os.makedirs(folder_path)
    except OSError:
        print ("Creation of the directory %s failed" % folder_path)
        print("New folder created will have -new at the end. Please rename.")
        os.makedirs(folder_path + "-new")
    else:
        print ("Successfully created the directory %s" % folder_path) 
 
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
            
            if cluster == 'dbscan':
                for p in range(len(feature_vectors)):
                    if classes_dbscan[p] == 0:
                        color = "red"
                    elif classes_dbscan[p] == -1:
                        color = "black"
                    elif classes_dbscan[p] == 1:
                        color = "blue"
                    elif classes_dbscan[p] == 2:
                        color = "green"
                    elif classes_dbscan[p] == 3:
                        color = "purple"
                    plt.scatter(feat1[p], feat2[p], c = color, s = 5)
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig((folder_path + "/" + fname_label1 + "-vs-" + fname_label2 + "-dbscan.pdf"))
                plt.show()
            elif cluster == 'kmeans':
                for p in range(len(feature_vectors)):
                    if classes_kmeans[p] == 0:
                        color = "red"
                    elif classes_kmeans[p] == 1:
                        color = "blue"
                    elif classes_kmeans[p] == 2:
                        color = "green"
                    elif classes_kmeans[p] == 3:
                        color = "purple"
                    plt.scatter(feat1[p], feat2[p], c = color)
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig(folder_path + "/" + fname_label1 + "-vs-" + fname_label2 + "-kmeans.pdf")
                plt.show()
            elif cluster == 'none':
                plt.scatter(feat1, feat2, s = 2, color = 'black')
                #plt.autoscale(enable=True, axis='both', tight=True)
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig(folder_path + "/" + fname_label1 + "-vs-" + fname_label2 + ".pdf")
                plt.show()
                
def plot_lof(time, intensity, targets, features, n, path):
    """plots the 20 most and least interesting light curves based on LOF
    takes input: time, intensity, targets, featurelist, n number of curves you want, path to where you want it
    saved (no end slash)
    modified [lcg 06292020]"""
    fname_lof = path + "/LOF-features.txt"
    from sklearn.neighbors import LocalOutlierFactor

    clf = LocalOutlierFactor(n_neighbors=50)
    
    fit_predictor = clf.fit_predict(features)
    negative_factor = clf.negative_outlier_factor_
    
    lof = -1 * negative_factor
    ranked = np.argsort(lof)
    largest_indices = ranked[::-1][:n]
    smallest_indices = ranked[:n]

    #plot a histogram of the lof values
    fig1, ax1 = plt.subplots()
    n, bins, patches = ax1.hist(lof, 50, density=1)
    ax1.title("LOF Histogram")
    plt.savefig(path+"/LOF-histogram.png")
    plt.close()

    with open(fname_lof, 'a') as file_object:
        file_object.write("Largest LOF's features: \n")
        np.savetxt(file_object, features[largest_indices])
        file_object.write("\n Smallest LOF's features: \n")
        np.savetxt(file_object, features[smallest_indices])
    #plot just the largest indices
    #rows, columns
    fig, axs = plt.subplots(n, 1, sharex = True, figsize = (8,n*3), constrained_layout=False)
    fig.subplots_adjust(hspace=0)
    dumppoints = [1842.5, 1847.9, 1853.3, 1856.4, 1861.9, 1867.4]
    for k in range(n):
        ind = largest_indices[k]
        axs[k].plot(time, intensity[ind], '.k', label="TIC " + str(int(targets[ind])) + ", LOF:" + str(np.round(lof[ind], 2)))
        for a in range(len(dumppoints)):
            axs[k].axvline(dumppoints[a], linewidth=0.5)
        axs[k].legend(loc="upper left")
        axs[k].set_ylabel("relative flux")
        title = astroquery_pull_data(targets[ind])
        axs[k].set_title(title)
        axs[-1].set_xlabel("BJD [-2457000]")
    fig.suptitle(str(n) + ' largest LOF targets', fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.96)
    fig.savefig(path + "/largest-lof.png")

    #plot the smallest indices
    fig1, axs1 = plt.subplots(n, 1, sharex = True, figsize = (8,n*3), constrained_layout=False)
    fig1.subplots_adjust(hspace=0)
    
    for m in range(n):
        ind = smallest_indices[m]
        axs1[m].plot(time, intensity[ind], '.k', label="TIC " + str(int(targets[ind])) + ", LOF:" + str(np.round(lof[ind], 2)))
        axs1[m].legend(loc="upper left")
        for a in range(len(dumppoints)):
            axs1[m].axvline(dumppoints[a], linewidth=0.5)
        axs1[m].set_ylabel("relative flux")
        title = astroquery_pull_data(targets[ind])
        axs1[m].set_title(title)
        axs1[-1].set_xlabel("BJD [-2457000]")
    fig1.suptitle(str(n) + ' smallest LOF targets', fontsize=16)
    fig1.tight_layout()
    fig1.subplots_adjust(top=0.96)
    fig1.savefig(path +  "/smallest-lof.png")
                
def astroquery_pull_data(target):
    """pulls data on object from astroquery
    target needs to be JUST the number- can be any format, will be reformatted appropriately"""
    try: 
        catalog_data = Catalogs.query_object("TIC " + str(int(target)), radius=0.02, catalog="TIC")
        #https://arxiv.org/pdf/1905.10694.pdf
        T_eff = np.round(catalog_data[0]["Teff"], 0)
        obj_type = catalog_data[0]["objType"]
        gaia_mag = np.round(catalog_data[0]["GAIAmag"], 2)
        radius = np.round(catalog_data[0]["rad"], 2)
        mass = np.round(catalog_data[0]["mass"], 2)
        distance = np.round(catalog_data[0]["d"], 1)
        title = "\nT_eff:" + str(T_eff) + "," + str(obj_type) + ", G: " + str(gaia_mag) + "\n Dist: " + str(distance) + ", R:" + str(radius) + " M:" + str(mass)
    except (ConnectionError, OSError, TimeoutError):
        print("there was a connection error!")
        title = "connection error, no data"
    return title

#plotting DBSCAN classification color coded inset plots:
def features_insets2D_colored(time, intensity, feature_vectors, targets, hand_classes, folder):
    """plotting (n 2) features against each other w/ inset plots
    inset plots are colored based on the hand classification value given to them
    feature_vectors is the list of ALL feature_vectors
    folder in string format, usually plot_output/date"
    """   
    path = "/Users/conta/UROP_Spring_2020/" + folder + "/2DFeatures-insets-colored"
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
    
    db = DBSCAN(eps=2.2, min_samples=18).fit(feats) #eps is NOT epochs
    classes_dbscan = db.labels_
    
    for n in range(16):
        graph_label1 = graph_labels[n]
        fname_label1 = fname_labels[n]
        feat1 = feature_vectors[:,n]
        for m in range(16):
            if m == n:
                continue
            graph_label2 = graph_labels[m]
            fname_label2 = fname_labels[m]   
            feat2 = feature_vectors[:,m]             
            
            
            fig, ax1 = plt.subplots()
            
            for p in range(len(feature_vectors)):
                if hand_classes[p] == 0:
                    color = "red"
                elif hand_classes[p] == 1:
                    color = "blue"
                elif hand_classes[p] == 2:
                    color = "green"
                elif hand_classes[p] == 3:
                    color = "purple"
                elif hand_classes[p] == 4:
                    color = "yellow"
                elif hand_classes[p] == 5:
                    color = "magenta"
                
                ax1.scatter(feat1[p], feat2[p], c = color, s = 2)
            
            
            
            #ax1.scatter(feature_vectors[:,n], feature_vectors[:,m], c = "black")
            ax1.set_xlabel(graph_label1)
            ax1.set_ylabel(graph_label2)
            
            plot_inset_color(ax1, "axins1", targets, intensity, time, feature_vectors, hand_classes, classes_dbscan, n, m)
    
            plt.savefig(path + "/" + fname_label1 + "-vs-" + fname_label2 + ".png")
            plt.show()
            

    
def plot_inset_color(ax1, axis_name, targets, intensity, time, feature_vectors, realclasses, guessclasses, feat1, feat2):
    """ plots the inset plots. colored to match the hand classification assigned to it 
    ax1 is the name of the axis being used. it is ESSENTIAL to getting this to run
    axis_name should be axins + a number as a STRING
    feat1 is x axis, feat2 is yaxis (n and m)
    colors the inset plot with what the predictor thinks it is
    colors the points and lines connecting with what the actual class is"""
    range_x = feature_vectors[:,feat1].max() - feature_vectors[:,feat1].min()
    range_y = feature_vectors[:,feat2].max() - feature_vectors[:,feat2].min()
    x_offset = range_x * 0.001
    y_offset = range_y * 0.001
    inset_positions = np.zeros((12,2))
    
    indexes_unique, targets_to_plot, tuples_plotting, titles = get_extrema(feature_vectors, targets, feat1, feat2)
    #print(indexes_unique)
    for n in range(len(indexes_unique)):
        x_shift = np.random.randint(0,2)
        y_shift = np.random.randint(0,2)
        index = indexes_unique[n]
        thetuple = tuples_plotting[n]
        title = titles[n]
        real_class = int(realclasses[index])
        guessed_class = int(guessclasses[index])
        colors = ["red","blue", "green", "purple" ,"yellow", "magenta", 'black']
        
        inset_x, inset_y, inset_width, inset_height = check_box_location(feature_vectors, thetuple, feat1, feat2, range_x, range_y, x_shift, y_shift, inset_positions)
        inset_positions[n] = (inset_x, inset_y)
        #print(inset_positions)
        axis_name = ax1.inset_axes([inset_x, inset_y, inset_width, inset_height], transform = ax1.transData) #x pos, y pos, width, height
        axis_name.scatter(time, intensity[index], c=colors[guessed_class], s = 0.01, rasterized=True)
            
        x1, x2, y1, y2 =  feature_vectors[index][feat1], feature_vectors[index][feat1] + x_offset, feature_vectors[index][feat2], feature_vectors[index][feat2] + y_offset
        axis_name.set_xlim(x1, x2)
        axis_name.set_ylim(y1, y2)
        ax1.indicate_inset_zoom(axis_name, edgecolor=colors[real_class])
            
        axis_name.set_xlim(time[0], time[-1])
        axis_name.set_ylim(intensity[index].min(), intensity[index].max())
        axis_name.set_title("TIC " + str(int(targets[index])) + title, fontsize=6)

#PLOTTING INSET PLOTS (x/y max/min points per feature)

def features_insets2D(time, intensity, feature_vectors, targets, path):
    """ Plots 2 features against each other with the extrema points' associated light curves plotted as insets
    along the top and bottom of the plot. 
    time is the time axis for the group
    intensity is the full list of intensities
    feature_vectors is the complete list of feature vectors
    targets is the complete list of targets
    folder is the folder into which you wish to save the folder of plots. it should be formatted as a string, with no trailing /
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
            
            inset_plotting(feature_vectors[n], feature_vectors[m], graph_label1, graph_label2, time, intensity, inset_indexes, targets, filename)
            

def inset_plotting(datax, datay, label1, label2, insetx, insety, inset_indexes, targets, filename):
    """ Plots the extrema of a 2D feature plot as insets on the top and bottom border
    datax and datay are the features being plotted as a scatter plot beneath it
    label1 and label2 are the x and y labels
    insetx is the time axis for the insets
    insety is the complete list of intensities 
    inset_indexes are the identified extrema to be plotted
    targets is the complete list of target TICs
    filename is the exact path that the plot is to be saved to.
    modified [lcg 06292020]"""
    
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
        axis_name.scatter(insetx, insety[inset_indexes[n]], c='black', s = 0.3, rasterized=True)
        
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

def get_extrema(feature_vectors, feat1, feat2):
    """ Identifies the extrema in each direction for the pair of features given. 
    Eliminates any duplicate extrema (ie, the xmax that is also the ymax)
    Returns array of unique indexes of the extrema
    modified [lcg 06292020]"""
    indexes = []
    index_feat1 = np.argsort(feature_vectors[:,feat1])
    index_feat2 = np.argsort(feature_vectors[:,feat2])
    
    indexes.append(index_feat1[-1]) #largest
    indexes.append(index_feat1[-2]) #second largest
    indexes.append(index_feat1[-3]) #third largest
    indexes.append(index_feat1[0]) #smallest
    indexes.append(index_feat1[1]) #second smallest
    indexes.append(index_feat2[-1]) #largest
    indexes.append(index_feat2[-2]) #second largest
    indexes.append(index_feat2[0]) #smallest
    indexes.append(index_feat2[1]) #second smallest

    indexes_unique = np.unique(np.asarray(indexes))
    
    return indexes_unique      


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