# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 00:18:52 2020

@author: Lindsey Gordon 

Functions used across files. Updated April 2020.
"""

#Imports ---------------------------------------
import numpy as np
import numpy.ma as ma 
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
from scipy.stats import moment, sigmaclip

import astropy
from astropy.io import fits
import scipy.signal as signal
from astropy.stats import SigmaClip

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor

import astroquery
from astroquery.simbad import Simbad
from astroquery.mast import Catalogs

import shapely
from shapely import geometry
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


#Testing that this file imported correctly ------


def test(num):
    print(num * 4)
    
# For running on the current data/ feature vectors (as of 5/4/20)
def get_from_files():
    """pulls time, intensity, and feature vectors from text files that they are saved in
    currently pulling the 5/4 version of it"""
    intensity = np.loadtxt("/Users/conta/UROP_Spring_2020/intensities.txt", delimiter = " ")

    time = np.loadtxt("/Users/conta/UROP_Spring_2020/timeindex.txt", delimiter = " ")
    
    targets = np.loadtxt("/Users/conta/UROP_Spring_2020/targets.txt", dtype = str, delimiter = ",")
    
    lc_feat = np.loadtxt("/Users/conta/UROP_Spring_2020/featvecs-5-4-20.txt", delimiter = " ")
    return time, intensity, targets, lc_feat
    
    
#Pulling data from files and processing it ---------
    
def print_header(index):
    fitspath = '/Users/conta/UROP_Spring_2020/tessdata_lc_sector20_1000/'
    fnames_all = os.listdir(fitspath)
    fnames = fnmatch.filter(fnames_all, '*fits*')
    
    #print(fnames[index])
    f = fits.open(fitspath + fnames[index])
    hdr = f[0].header
    print(hdr)
    return hdr

def get_data_from_fits():
    """ imports data from fits files. 
        based on emma's code"""    

    fitspath = '/Users/conta/UROP_Spring_2020/tessdata_lc_sector20_1000/'
    fnames_all = os.listdir(fitspath)
    fnames = fnmatch.filter(fnames_all, '*fits*')
    
    interp_tol = 20. / (24*60) # >> interpolate small gaps (less than 20 minutes)
    
    intensity = []
    targets = []
    #coordinates = []
    for file in fnames:
        # -- open file -------------------------------------------------------------
        f = fits.open(fitspath + file)
    
        # >> get data
        time = f[1].data['TIME']
        i = f[1].data['PDCSAP_FLUX']
        tic = f[1].header["OBJECT"]
        targets.append(tic)

        #ra = f[1].header["RA_OBJ"]
        #dec = f[1].header["DEC_OBJ"]
        #coords = str(ra) + " " + str(dec)
        #coordinates.append(coords)
        # -- find small nan gaps ---------------------------------------------------
        # >> adapted from https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
        # >> find run starts
        n = np.shape(i)[0]
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(np.isnan(i)[:-1], np.isnan(i)[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]
    
        # >> find run lengths
        run_lengths = np.diff(np.append(run_starts, n))
    
        tdim = time[1] - time[0]
        interp_inds = run_starts[np.nonzero((run_lengths * tdim <= interp_tol) * \
                                            np.isnan(i[run_starts]))]
        interp_lens = run_lengths[np.nonzero((run_lengths * tdim <= interp_tol) * \
                                             np.isnan(i[run_starts]))]
    
        # -- interpolation ---------------------------------------------------------
        # >> interpolate small gaps
        i_interp = np.copy(i)
        for a in range(np.shape(interp_inds)[0]):
            start_ind = interp_inds[a]
            end_ind = interp_inds[a] + interp_lens[a]
            i_interp[start_ind:end_ind] = np.interp(time[start_ind:end_ind],
                                                    time[np.nonzero(~np.isnan(i))],
                                                    i[np.nonzero(~np.isnan(i))])
        intensity.append(i_interp)
    
    # -- remove orbit nan gap ------------------------------------------------------
    intensity = np.array(intensity)
    # nan_inds = np.nonzero(np.prod(np.isnan(intensity)==False), axis = 0))
    nan_inds = np.nonzero(np.prod(np.isnan(intensity)==False, axis = 0) == False)
    time = np.delete(time, nan_inds)
    intensity = np.delete(intensity, nan_inds, 1) #each row of intensity is one interpolated light curve.
    return time, intensity, targets


#normalizing each light curve
def normalize(intensity):
    """normalizes the intensity from the median value 
    by dividing out. then sigmaclips using astropy
    returns a masked array"""
    sigclip = SigmaClip(sigma=5, maxiters=None, cenfunc='median')
    intense = []
    for i in np.arange(len(intensity)):
        intensity[i] = intensity[i] / np.median(intensity[i])
        inte = sigclip(intensity[i], masked=True, copy = False)
        intense.append(inte)
    intensity = np.ma.asarray(intense)
    print("Normalization and sigma clipping complete")
    return intensity
    

#producing the feature vector list -----------------------------
    
    
def create_list_featvec(time_axis, datasets):
    """input: all of the datasets being turned into feature vectors (ie, intensity)
        num_features is the number of features currently being worked on. 
    you just changed to a range, if it hates you it's because of that
    returns a list of featurevectors, one for each input . """
    num_data = len(datasets) #how many datasets
    x = time_axis #creates the x axis
    feature_list = np.zeros((num_data, 16)) #MANUALLY UPDATE WHEN CHANGING NUM FEATURES
    for n in range(num_data):
        feature_list[n] = featvec(x, datasets[n])
        if n % 50 == 0: print(str(n) + " completed")
    return feature_list

def featvec(x_axis, sampledata): 
    """calculates the feature vector of the single set of data (ie, intensity[0])
    currently returns 16: 
        0 - Average
        1 - Variance
        2 - Skewness
        3 - Kurtosis
        
        4 - ln variance
        5 - ln skewness
        6 - ln kurtosis
        
        (over 0.1 to 10 days)
        7 - maximum power
        8 - ln maximum power
        9 - period of maximum power
        
        10 - slope
        11 - ln slope
        
        (integration of periodogram over time frame)
        12 - P0 - 0.1-1 days
        13 - P1 - 1-3 days
        14 - P2 - 3-10 days
        
        (over 0-0.1 days, for moving objects)
        15 - Period of max power
        
        
        ***if you update the number of features, 
        you have to update the number of features in create_list_featvec!!!!"""
    featvec = moments(sampledata) #produces moments and log moments
    
    
    f = np.linspace(0.6, 62.8, 5000)  #period range converted to frequencies
    periods = np.linspace(0.1, 10, 5000)#0.1 to 10 day period
    pg = signal.lombscargle(x_axis, sampledata, f, normalize = True)
    rel_maxes = argrelextrema(pg, np.greater)
    
    powers = []
    indexes = []
    for n in range(len(rel_maxes[0])):
        #print(rel_maxes[0][n]) #accessing each index
        index = rel_maxes[0][n]
        indexes.append(index)
        power_level_at_rel_max = pg[index]
        #print(power_level_at_rel_max)
        powers.append(power_level_at_rel_max)
    
    max_power = np.max(powers)
    #print("the max power is" + str(max_power))
    index_of_max_power = np.argmax(powers)
    #print("the index of that power is" + str(index_of_max_power))
    index_of_f_max = rel_maxes[0][index_of_max_power]
    f_max_power = f[index_of_f_max]
    #print("the frequency at that index is" + str(f_max_power))
    period_max_power = 2*np.pi / f_max_power
    featvec.append(max_power)
    featvec.append(np.log(np.abs(max_power)))
    featvec.append(period_max_power)
    
    slope = stats.linregress(x_axis, sampledata)[0]
    featvec.append(slope)
    featvec.append(np.log(np.abs(slope)))
    
    
    #integrating = np.trapz(pg, periods) #integrates the whole 0.1-10 day range
    integrating1 = np.trapz(pg[457:5000], periods[457:5000]) #0.1 days to 1 days
    integrating2 = np.trapz(pg[121:457], periods[121:457])#1-3 days
    integrating3 = np.trapz(pg[0:121], periods[0:121]) #3-10 days
    
    featvec.append(integrating1)
    featvec.append(integrating2)
    featvec.append(integrating3)
    
    #for 0.001 to 1 day periods
    f2 = np.linspace(62.8, 6283.2, 20)  #period range converted to frequencies
    p2 = np.linspace(0.001, 0.1, 20)#0.001 to 1 day periods
    pg2 = signal.lombscargle(x_axis, sampledata, f2, normalize = True)
    rel_maxes2 = argrelextrema(pg2, np.greater)
    powers2 = []
    indexes2 = []
    for n in range(len(rel_maxes2[0])):
        index2 = rel_maxes2[0][n]
        indexes2.append(index2)
        power_level_at_rel_max2 = pg2[index2]
        powers2.append(power_level_at_rel_max2)
    max_power2 = np.max(powers2)
    index_of_max_power2 = np.argmax(powers2)
    index_of_f_max2 = rel_maxes2[0][index_of_max_power2]
    f_max_power2 = f2[index_of_f_max2]
    period_max_power2 = 2*np.pi / f_max_power2
    featvec.append(period_max_power2)
    #print("done")
    return(featvec) 

def moments(dataset): 
    """calculates the 1st through 4th moment of a single row of data (ie, intensity[0])"""
    moments = []
    moments.append(np.mean(dataset)) #mean (don't use moment, always gives 0)
    moments.append(moment(dataset, moment = 2)) #variance
    moments.append(moment(dataset, moment = 3)) #skew
    moments.append(moment(dataset, moment = 4)) #kurtosis
    moments.append(np.log(np.abs(moment(dataset, moment = 2)))) #ln variance
    moments.append(np.log(np.abs(moment(dataset, moment = 3)))) #ln skew
    moments.append(np.log(np.abs(moment(dataset, moment = 4)))) #ln kurtosis
    return(moments)

#Plotting functions ------------------------------------------------------
    
def n_choose_2_features_plotting(feature_vectors, cluster_columns, date, clustering):
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
        db = DBSCAN(eps=0.5, min_samples=10).fit(cluster_columns) #eps is NOT epochs
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
        folder_label = "nchoose2"
    #makes folder and saves to it    
    path = "/Users/conta/UROP_Spring_2020/plot_output/" + date + "/" + folder_label
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
                plt.savefig(("/Users/conta/UROP_Spring_2020/plot_output/" + date + "/dbscan-colored/" + date + "-" + fname_label1 + "-vs-" + fname_label2 + "-dbscan.png"))
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
                plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/" + date + "/kmeans-colored/" + date + "-" + fname_label1 + "-vs-" + fname_label2 + "-kmeans.png")
                plt.show()
            elif cluster == 'none':
                plt.scatter(feat1, feat2, s = 2, color = 'black')
                #plt.autoscale(enable=True, axis='both', tight=True)
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/" + date + "/nchoose2/" + date + "-" + fname_label1 + "-vs-" + fname_label2 + ".png")
                plt.show()
                



def plot_lof(time, intensity, targets, features, n, date):
    """plots the 20 most and least interesting light curves based on LOF
    takes input: time, intensity, targets, featurelist, n number of curves you want, date as a string """
    from sklearn.neighbors import LocalOutlierFactor

    clf = LocalOutlierFactor(n_neighbors=2)
    
    fit_predictor = clf.fit_predict(features)
    negative_factor = clf.negative_outlier_factor_
    
    lof = -1 * negative_factor
    ranked = np.argsort(lof)
    largest_indices = ranked[::-1][:n]
    smallest_indices = ranked[:n]

    #plot just the largest indices
    #rows, columns
    fig, axs = plt.subplots(n, 1, sharex = True, figsize = (8,n*3), constrained_layout=False)
    fig.subplots_adjust(hspace=0)
    
    for k in range(n):
        ind = largest_indices[k]
        axs[k].plot(time, intensity[ind], '.k', label=targets[ind])
        axs[k].legend(loc="upper left")
        axs[k].set_ylabel("relative flux")
        axs[-1].set_xlabel("BJD [-2457000]")
    fig.suptitle(str(n) + ' largest LOF targets', fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.96)
    fig.savefig("/Users/conta/UROP_Spring_2020/plot_output/" + date + "/" + date + "-"+ str(n) + "-largest-lof.png")

    #plot the smallest indices
    fig1, axs1 = plt.subplots(n, 1, sharex = True, figsize = (8,n*3), constrained_layout=False)
    fig1.subplots_adjust(hspace=0)
    
    for m in range(n):
        ind = smallest_indices[m]
        axs1[m].plot(time, intensity[ind], '.k', label=targets[ind])
        axs1[m].legend(loc="upper left")
        axs1[m].set_ylabel("relative flux")
        axs1[-1].set_xlabel("BJD [-2457000]")
    fig1.suptitle(str(n) + ' smallest LOF targets', fontsize=16)
    fig1.tight_layout()
    fig1.subplots_adjust(top=0.96)
    fig1.savefig("/Users/conta/UROP_Spring_2020/plot_output/" + date + "/" + date + "-"+ str(n) + "-smallest-lof.png")
                
def astroquery_pull_data(target):
    """pulls data on object from astroquery
    target needs to be a string"""
    try: 
        catalog_data = Catalogs.query_object(target, radius=0.02, catalog="TIC")
        #https://arxiv.org/pdf/1905.10694.pdf
        T_eff = np.round(catalog_data[0]["Teff"], 0)
        obj_type = catalog_data[0]["objType"]
        gaia_mag = np.round(catalog_data[0]["GAIAmag"], 2)
        radius = np.round(catalog_data[0]["rad"], 2)
        mass = np.round(catalog_data[0]["mass"], 2)
        distance = np.round(catalog_data[0]["d"], 1)
        title = "\nT_eff:" + str(T_eff) + ", ObjType: " + str(obj_type) + ", GAIA mag: " + str(gaia_mag) + "\n Dist: " + str(distance) + ", Radius:" + str(radius) + " Mass:" + str(mass)
    except (ConnectionError, OSError):
        print("there was a connection error!")
        title = "connection error, no data"
    return title

def inset_labelling(axis_name, time, intensity, targets, index, title):
    """formatting the labels for the inset plots"""
    axis_name.set_xlim(time[0], time[-1])
    axis_name.set_ylim(intensity[index].min(), intensity[index].max())
    axis_name.set_xlabel("BJD [2457000]")
    axis_name.set_ylabel("relative flux")
    axis_name.set_title(targets[index] + title, fontsize=8)
    
#PLOTTING INSET PLOTS (x/y max/min points per feature)
    
def n_choose_2_insets(time, intensity, feature_vectors, targets, date):
    """plotting (n 2) features against each other w/ 4 extremes inset plotted
    feature_vectors is the list of ALL feature_vectors
    date must be a string in the format of the folder you are saving into ie "4-13"
    """   
    path = "/Users/conta/UROP_Spring_2020/plot_output/" + date + "/nchoose2"
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
        #feat1 = feature_vectors[:,n]
        graph_label1 = graph_labels[n]
        fname_label1 = fname_labels[n]
        for m in range(16):
            if m == n:
                continue
            graph_label2 = graph_labels[m]
            fname_label2 = fname_labels[m]                
            #feat2 = feature_vectors[:,m]
            
            plot_all_insets(feature_vectors, targets, intensity, time, n, m, graph_label1, graph_label2)
 
            plt.savefig(path + date + "-" + fname_label1 + "-vs-" + fname_label2 + ".png")
            plt.show()
def plot_all_insets(feature_vectors,targets, intensity, time, feat1, feat2, graph_label1, graph_label2):
    """plots the x/y min/max points' associated light curve on the plot"""
    fig, ax1 = plt.subplots()
    ax1.scatter(feature_vectors[:,feat1], feature_vectors[:,feat2], c = "black")
    ax1.set_xlabel(graph_label1)
    ax1.set_ylabel(graph_label2)
    
    plot_inset(ax1, "axins1", targets, intensity, time,feature_vectors, feat1, feat2)

    
def plot_inset(ax1, axis_name, targets, intensity, time, feature_vectors, feat1, feat2):
    """ plots the inset plots. 
    ax1 is the name of the axis being used. it is ESSENTIAL to getting this to run
    axis_name should be axins + a number as a STRING
    feat1 is x axis, feat2 is yaxis
    extreme is which point you want to plot- options are 'ymax', 'ymin'
    if no extreme is given, prints that you fucked up."""
    range_x = feature_vectors[:,feat1].max() - feature_vectors[:,feat1].min()
    range_y = feature_vectors[:,feat2].max() - feature_vectors[:,feat2].min()
    x_offset = range_x * 0.001
    y_offset = range_y * 0.001
    inset_positions = np.zeros((8,4))
    
    indexes_unique, targets_to_plot, tuples_plotting, titles = get_extrema(feature_vectors, targets, feat1, feat2)
    #print(indexes_unique)
    for n in range(len(indexes_unique)):
        x_shift = np.random.randint(0,2)
        y_shift = np.random.randint(0,2)
        index = indexes_unique[n]
        thetuple = tuples_plotting[n]
        title = titles[n]
        
        inset_x, inset_y, inset_width, inset_height = check_box_location(feature_vectors, thetuple, feat1, feat2, range_x, range_y, x_shift, y_shift, inset_positions)
        #inset_width, inset_height, inset_x, inset_y = box_locate_no_repositioning(thetuple, range_x, range_y, x_shift, y_shift)
        inset_positions[n] = (inset_x, inset_y, inset_width, inset_height)
        
        axis_name = ax1.inset_axes([inset_x, inset_y, inset_width, inset_height], transform = ax1.transData) #x pos, y pos, width, height
        axis_name.scatter(time, intensity[index], c='black', s = 0.01)
            
        x1, x2, y1, y2 =  feature_vectors[index][feat1], feature_vectors[index][feat1] + x_offset, feature_vectors[index][feat2], feature_vectors[index][feat2] + y_offset
        axis_name.set_xlim(x1, x2)
        axis_name.set_ylim(y1, y2)
        ax1.indicate_inset_zoom(axis_name)
            
        inset_labelling(axis_name, time, intensity, targets, index, title)

        

def get_extrema(feature_vectors, targets, feat1, feat2):
    """ identifies the 8 extrema in each direction and 
    pulls the data needed on each
    eliminates any duplicate extrema (ie, the xmax is also the ymax)"""
    indexes = []
    index_feat1 = np.argsort(feature_vectors[:,feat1])
    index_feat2 = np.argsort(feature_vectors[:,feat2])
    indexes.append(index_feat1[-1]) #largest
    indexes.append(index_feat1[-2]) #second largest
    indexes.append(index_feat1[0]) #smallest
    indexes.append(index_feat1[1]) #second smallest

    indexes.append(index_feat2[-1]) #largest
    indexes.append(index_feat2[-2]) #second largest
    indexes.append(index_feat2[0]) #smallest
    indexes.append(index_feat2[1]) #second smallest

    indexes_unique = np.unique(np.asarray(indexes))
    
    targets_to_plot = []
    tuples_plotting = []
    titles = []
    
    for n in range(len(indexes_unique)):
        targets_to_plot.append(targets[indexes_unique[n]])
        tuples_plotting.append( (feature_vectors[:,feat1][indexes_unique[n]], feature_vectors[:,feat2][indexes_unique[n]]) )
        title = astroquery_pull_data(targets[indexes_unique[n]])
        titles.append(title)
    return indexes_unique, targets_to_plot, tuples_plotting, titles


def check_box_location(feature_vectors, coordtuple, feat1, feat2, range_x, range_y, x, y, inset_positions):
    """ checks if data points lie within the area of the inset plot
    coordtuple is the (x,y) point in feature space
    feat1, feat2 are the number for the features being used
    range_x and range_y are ranges for each feature
    x is whether it will be left or right of the point
    y is whether it will be above/below the point
    inset_positions is a list  from a diff. function that holds the pos of insets"""
    #position of box - needs to be dependent on location
    inset_width = range_x / 3
    inset_height = range_y /8
    if x == 0:
        inset_x = coordtuple[0] - (inset_width * 1.2) #move left
    elif x == 1:
        inset_x = coordtuple[0] + (inset_width * 1.2) #move right
    if y == 0:
        inset_y = coordtuple[1] + (inset_height) #move up
    elif y == 1:
        inset_y = coordtuple[1] - (inset_height) #move down
    
    inset_BL = (inset_x, inset_y)
    inset_BR = (inset_x + inset_width, inset_y)
    inset_TL = (inset_x, inset_y + inset_height)
    inset_TR = (inset_x + inset_width, inset_y + inset_height)
    
    conc = np.column_stack((feature_vectors[:,feat1], feature_vectors[:,feat2]))
    polygon = Polygon([inset_BL, inset_BR, inset_TL, inset_TR])
    
    i = 0
    n = len(conc)
    
    while i < n:
        point = Point(conc[i])
        #is it on top of a point?
        if polygon.contains(point) == True:
            if x == 0: 
                inset_x = inset_x - (0.01 * range_x)
            elif x == 1:
                inset_x = inset_x + (0.01 * range_x)
            if y == 0:
                inset_y = inset_y + (0.01 * range_y)
            elif y == 1:
                inset_y = inset_y - (0.01 * range_y)
            
            inset_BL = (inset_x, inset_y)
            inset_BR = (inset_x + inset_width, inset_y)
            inset_TL = (inset_x, inset_y + inset_height)
            inset_TR = (inset_x + inset_width, inset_y + inset_height)
            polygon = Polygon([inset_BL, inset_BR, inset_TL, inset_TR])
            i = 0
            #print("moving")
        elif polygon.contains(point) == False:
            i = i + 1
        #is it on top of another plot
        for n in range(8):
            bx, by, bw, bh = inset_positions[n]
            bBL = (bx, by)
            bBR = (bx + bw, by)
            bTL = (bx, by+bh)
            bTR = (bx + bw, by + bh)
            p1 = Polygon([bBL, bBR, bTL, bTR])
            
            if p1.intersects(polygon):
                inset_x = inset_x + (0.5*inset_width)
                inset_y = inset_y + (0.5*inset_height)
                inset_BL = (inset_x, inset_y)
                inset_BR = (inset_x + inset_width, inset_y)
                inset_TL = (inset_x, inset_y + inset_height)
                inset_TR = (inset_x + inset_width, inset_y + inset_height)
                polygon = Polygon([inset_BL, inset_BR, inset_TL, inset_TR])
                i = 0
    print("position determined")
    
    return inset_x, inset_y, inset_width, inset_height
#Other functions (mostly unfinished or no longer used) ---------------------

def check_diagonalized(c_matrix):
    """Metric for optimization of diagonal of confusion matrix"""
    num_labels = len(c_matrix)
    total = np.sum(c_matrix, axis=None)
    diagonal = 0
    n = 0
    while n < num_labels:
        diagonal = diagonal + c_matrix[n][n]
        n = n+1
    fraction_diagonal = diagonal/total
    return fraction_diagonal


def gaussian(datapoints, a, b, c):
    """Produces a gaussian function"""
    x = np.linspace(0, xmax, datapoints)
    return  a * np.exp(-(x-b)**2 / 2*c**2) + np.random.normal(size=(datapoints))