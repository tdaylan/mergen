# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:00:17 2020

@author: conta
"""


import numpy as np
import numpy.ma as ma 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)

from pylab import rcParams
rcParams['figure.figsize'] = 10,10
rcParams["lines.markersize"] = 2
from scipy.signal import argrelextrema

import astropy
from astropy.io import fits
import scipy.signal as signal
from astropy.stats import SigmaClip

from datetime import datetime
import os
from scipy.stats import moment, sigmaclip

import sklearn
from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import fnmatch

from sklearn.metrics import confusion_matrix
import feature_functions
from feature_functions import *
from sklearn.neighbors import LocalOutlierFactor

import astroquery
from astroquery.simbad import Simbad
from astroquery.mast import Catalogs

import shapely
from shapely import geometry
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

test(8) #should return 8 * 4


#%%

time, intensity, targets, lc_feat = get_from_files()

#%%
intensity = normalize(intensity)

lc_feat = create_list_featvec(time, intensity)

n_choose_2_features_plotting(lc_feat, "4-29", "none")
#%%
plot_lof(time, intensity, targets, lc_feat, 10, "4-29")

#%%

#1st-4th moments (0-3), natural log variance (4), skew (5), kurtosis (6), 
  # power, natural log power, period of max power (0.1 to 10 days) (7-9), 
   # slope, natural log of slope (10-11)
    # integration of periodogram over: period of 0.1-10, period of 0.1-1, period of 1-3,
     #   period of 3-10 days, (12-16)
      #  period of max power for 0.01-0.1 days (for moving objects) (17)

#%%
#use to dig up header info
print_header(266)

#%%
all_outliers = []
#period of 0.1-1 (integrated) vs log of max power

#five largest points (outliers) are colored separately
period_01_1_outliers = np.argsort(lc_feat[:,13])[-5:]

plt.scatter(lc_feat[:,13][period_01_1_outliers], lc_feat[:,8][period_01_1_outliers])
plt.show()

for i in range(len(period_01_1_outliers)):
    all_outliers.append(int(period_01_1_outliers[i]))

#for log of max power outliers
logmaxpoweroutlier = np.argmax(lc_feat[:,8])
print(logmaxpoweroutlier)
plt.scatter(lc_feat[:,13],lc_feat[:,8]) 
plt.scatter(lc_feat[:,13][logmaxpoweroutlier], lc_feat[:,8][logmaxpoweroutlier])
plt.show()


all_outliers.append(logmaxpoweroutlier)


#removing average outliers 
average_max = np.argsort(lc_feat[:,0])[-4:]
average_min = np.argsort(lc_feat[:,0])[:4]

print(average_max, average_min)

for i in range(4):
    all_outliers.append(average_max[i])
for i in range(4):
    all_outliers.append(average_min[i])

outliers = np.asarray(all_outliers)
print(outliers, type(outliers))
outliers = np.unique(outliers)
print(outliers)

#plot all the outliers

for i in range(len(outliers)):
    plt.scatter(time, intensity[outliers[i]])
    plt.title(targets[outliers[i]])
    plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-29/feature-outliers/4-29" + targets[outliers[i]] +".png")
    plt.show()


featvec_reduced = np.delete(lc_feat, outliers, 0)   
print(len(featvec_reduced), len(lc_feat))

#%%
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
    
    indexes_unique, targets_to_plot, tuples_plotting, titles = get_extrema(feature_vectors, targets, feat1, feat2)
    #print(indexes_unique)
    for n in range(len(indexes_unique)):
        x_shift = np.random.randint(0,2)
        y_shift = np.random.randint(0,2)
        index = indexes_unique[n]
        thetuple = tuples_plotting[n]
        title = titles[n]
        
        inset_x, inset_y, inset_width, inset_height = check_box_location(feature_vectors, thetuple, feat1, feat2, range_x, range_y, x_shift, y_shift)
        #inset_width, inset_height, inset_x, inset_y = box_locate_no_repositioning(thetuple, range_x, range_y, x_shift, y_shift)
        
        axis_name = ax1.inset_axes([inset_x, inset_y, inset_width, inset_height], transform = ax1.transData) #x pos, y pos, width, height
        axis_name.scatter(time, intensity[index], c='black', s = 0.01)
            
        x1, x2, y1, y2 =  feature_vectors[index][feat1], feature_vectors[index][feat1] + x_offset, feature_vectors[index][feat2], feature_vectors[index][feat2] + y_offset
        axis_name.set_xlim(x1, x2)
        axis_name.set_ylim(y1, y2)
        ax1.indicate_inset_zoom(axis_name)
            
        inset_labelling(axis_name, time, intensity, targets, index, title)

        

def get_extrema(feature_vectors, targets, feat1, feat2):
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

def box_locate_no_repositioning(coordtuple, range_x, range_y, x_shift, y_shift):
    inset_width = range_x / 3
    inset_height = range_y / 8
    if x_shift == 0:
        inset_x = coordtuple[0] - (inset_width * 4) #move left
    elif x_shift == 1:
        inset_x = coordtuple[0] + (inset_width * 4) #move right
    if y_shift == 0:
        inset_y = coordtuple[1] + (inset_height) #move up
    elif y_shift == 1:
        inset_y = coordtuple[1] - (inset_height) #move down
    
    return inset_width, inset_height, inset_x, inset_y

def check_box_location(feature_vectors, coordtuple, feat1, feat2, range_x, range_y, x, y):
    """ checks if data points lie within the area of the inset plot
    coordtuple is the (x,y) point in feature space
    feat1, feat2 are the number for the features being used
    range_x and range_y are ranges for each feature
    x is whether it will be left or right of the point
    y is whether it will be above/below the point"""
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
    
    inset_positions = []
    
    while i < n:
        point = Point(conc[i])
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
    print("position determined")
    return inset_x, inset_y, inset_width, inset_height

#%%
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
    for n in range(3):
        #feat1 = feature_vectors[:,n]
        graph_label1 = graph_labels[n]
        fname_label1 = fname_labels[n]
        for m in range(3):
            if m == n:
                continue
            graph_label2 = graph_labels[m]
            fname_label2 = fname_labels[m]                
            #feat2 = feature_vectors[:,m]
            
            plot_all_insets(feature_vectors, targets, intensity, time, n, m, graph_label1, graph_label2)
 
            plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/" + date + "/nchoose2/" + date + "-" + fname_label1 + "-vs-" + fname_label2 + ".png")
            plt.show()


