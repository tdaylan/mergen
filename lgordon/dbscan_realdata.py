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
        
def check_box_location(coordtuple, feat1, feat2, range_x, range_y, x, y):
    """ checks if data points lie within the area of the inset plot"""
    #position of box - needs to be dependent on location
    inset_width = range_x / 2
    inset_height = range_y /5 
    if x == 'left':
        inset_x = coordtuple[0] - inset_width * 1.2
    elif x == 'right':
        inset_x = coordtuple[0] + 0.01 * range_x
    if y == 'up':
        inset_y = coordtuple[1] + 0.01 * range_y
    elif y == 'down':
        inset_y = coordtuple[1] - inset_height * 1.2
    
    
    inset_BL = (inset_x, inset_y)
    inset_BR = (inset_x + inset_width, inset_y)
    inset_TL = (inset_x, inset_y + inset_height)
    inset_TR = (inset_x + inset_width, inset_y + inset_height)
    
    conc = np.column_stack((lc_feat[:,feat1], lc_feat[:,feat2]))
    polygon = Polygon([inset_BL, inset_BR, inset_TL, inset_TR])
    
    i = 0
    n = len(conc)
    
    while i < n:
        point = Point(conc[i])
        if polygon.contains(point) == True:
            inset_x += 0.001 * range_x
            inset_y += 0.001 * range_y
            i = 0
            print("moving")
        elif polygon.contains(point) == False:
            i = i + 1
    return inset_x, inset_y, inset_width, inset_height

def astroquery_pull_data(target):
    """pulls data on object from astroquery
    target needs to be a string"""
    try: 
        catalog_data = Catalogs.query_object(target, radius=0.02, catalog="TIC")
        #https://arxiv.org/pdf/1905.10694.pdf
        T_eff = catalog_data[0]["Teff"]
        obj_type = catalog_data[0]["objType"]
        gaia_mag = catalog_data[0]["GAIAmag"]
        radius = catalog_data[0]["rad"]
        mass = catalog_data[0]["mass"]
        distance = catalog_data[0]["d"]
        title = "\nT_eff:" + str(T_eff) + ", ObjType: " + str(obj_type) + ", GAIA mag: " + str(gaia_mag) + "\n Dist: " + str(distance) + ", Radius:" + str(radius) + " Mass:" + str(mass)
    except (ConnectionError, OSError):
        print("there was a connection error!")
        title = "connection error, no data"
    return title

def inset_labelling(axis_name, index, title):
    axis_name.set_xlim(time[0], time[-1])
    axis_name.set_ylim(intensity[index].min(), intensity[index].max())
    axis_name.set_xlabel("BJD [2457000]")
    axis_name.set_ylabel("relative flux")
    axis_name.set_title(targets[index] + title, fontsize=8)

#%%
def max_y(feat1, feat2, range_x, range_y):
    """produce maximum y information"""
    y_max_index = np.argmax(lc_feat[:,feat2])
    targ_y_max = targets[y_max_index]
    y_max_tuple = (lc_feat[:,feat1][y_max_index], lc_feat[:,feat2][y_max_index])
    inset_x, inset_y, inset_width, inset_height = check_box_location(y_max_tuple, feat1, feat2, range_x, range_y, "left", 'up')
    title = astroquery_pull_data(targ_y_max)
    return y_max_index, title, inset_x, inset_y, inset_width, inset_height

def min_y(feat1, feat2, range_x, range_y):
    """produce minimum y information"""
    y_min_index = np.argmin(lc_feat[:,feat2])
    targ_y_min = targets[y_min_index]
    y_min_tuple = (lc_feat[:,feat1][y_min_index], lc_feat[:,feat2][y_min_index])
    inset_x, inset_y, inset_width, inset_height = check_box_location(y_min_tuple, feat1, feat2, range_x, range_y, "left", "down")
    title = astroquery_pull_data(targ_y_min)
    return y_min_index, title, inset_x, inset_y, inset_width, inset_height

def max_x(feat1, feat2, range_x, range_y):
    """maximum x point inset"""
    x_max_index = np.argmax(lc_feat[:,feat1])
    targ_x_max = targets[x_max_index]
    x_max_tuple = (lc_feat[:,feat1][x_max_index], lc_feat[:,feat2][x_max_index])
    inset_x, inset_y, inset_width, inset_height = check_box_location(x_max_tuple, feat1, feat2, range_x, range_y, 'right', 'down')
    title = astroquery_pull_data(targ_x_max)
    return x_max_index, title, inset_x, inset_y, inset_width, inset_height
    
def min_x(feat1, feat2, range_x, range_y):
    """produce minimum y information"""
    x_min_index = np.argmin(lc_feat[:,feat1])
    targ_x_min = targets[x_min_index]
    y_min_tuple = (lc_feat[:,feat1][x_min_index], lc_feat[:,feat2][x_min_index])
    inset_x, inset_y, inset_width, inset_height = check_box_location(y_min_tuple, feat1, feat2, range_x, range_y, "left", "up")
    title = astroquery_pull_data(targ_x_min)
    return x_min_index, title, inset_x, inset_y, inset_width, inset_height
    
    
def plot_inset(ax1, axis_name, feat1, feat2, extreme, index_list):
    """ plots the inset plots. axis_name should be axins + a number as a STRING
    feat1 is x axis, feat2 is yaxis
    extreme is which point you want to plot- options are 'ymax', 'ymin'
    if no extreme is given, prints that you fucked up."""
    range_x = lc_feat[:,feat1].max() - lc_feat[:,feat1].min()
    range_y = lc_feat[:,feat2].max() - lc_feat[:,feat2].min()
    x_offset = range_x * 0.001
    y_offset = range_y * 0.001
    if extreme == "ymax":
        index, title, inset_x, inset_y, inset_width, inset_height = max_y(feat1,feat2, range_x, range_y)
        #print(index, title, inset_x, inset_y, inset_width, inset_height)
        if index in index_list: 
            print("this has already been plotted")
        else: 
            index_list.append(index)
            axis_name = ax1.inset_axes([inset_x, inset_y, inset_width, inset_height], transform = ax1.transData) #x pos, y pos, width, height
            axis_name.scatter(time, intensity[index], c='black', s = 0.01)
            
            x1, x2, y1, y2 =  lc_feat[index][feat1], lc_feat[index][feat1] + x_offset, lc_feat[index][feat2], lc_feat[index][feat2] + y_offset
            axis_name.set_xlim(x1, x2)
            axis_name.set_ylim(y1, y2)
            ax1.indicate_inset_zoom(axis_name)
            
            inset_labelling(axis_name, index, title)
    elif extreme == 'ymin':
        index, title, inset_x, inset_y, inset_width, inset_height = min_y(feat1,feat2, range_x, range_y)
        if index in index_list: 
            print("this has already been plotted")
        else:
            index_list.append(index)
            axis_name = ax1.inset_axes([inset_x, inset_y, inset_width, inset_height], transform = ax1.transData) #x pos, y pos, width, height
            axis_name.scatter(time, intensity[index], c='black', s = 0.01)
            
            x1, x2, y1, y2 =  lc_feat[index][feat1], lc_feat[index][feat1] + x_offset, lc_feat[index][feat2], lc_feat[index][feat2] + y_offset
            axis_name.set_xlim(x1, x2)
            axis_name.set_ylim(y1, y2)
            ax1.indicate_inset_zoom(axis_name)
            
            inset_labelling(axis_name, index, title)
    elif extreme == 'xmax':
        index, title, inset_x, inset_y, inset_width, inset_height = max_x(feat1,feat2, range_x, range_y)
        if index in index_list: 
            print("this has already been plotted")
        else:
            index_list.append(index)
            axis_name = ax1.inset_axes([inset_x, inset_y, inset_width, inset_height], transform = ax1.transData) #x pos, y pos, width, height
            axis_name.scatter(time, intensity[index], c='black', s = 0.01)
            
            x1, x2, y1, y2 =  lc_feat[index][feat1], lc_feat[index][feat1] + x_offset, lc_feat[index][feat2], lc_feat[index][feat2] + y_offset
            axis_name.set_xlim(x1, x2)
            axis_name.set_ylim(y1, y2)
            ax1.indicate_inset_zoom(axis_name)
            
            inset_labelling(axis_name, index, title)
    elif extreme == 'xmin':
        index, title, inset_x, inset_y, inset_width, inset_height = min_x(feat1,feat2, range_x, range_y)
        if index in index_list: 
            print("this has already been plotted")
        else:
            index_list.append(index)
            axis_name = ax1.inset_axes([inset_x, inset_y, inset_width, inset_height], transform = ax1.transData) #x pos, y pos, width, height
            axis_name.scatter(time, intensity[index], c='black', s = 0.01)
            
            x1, x2, y1, y2 =  lc_feat[index][feat1], lc_feat[index][feat1] + x_offset, lc_feat[index][feat2], lc_feat[index][feat2] + y_offset
            axis_name.set_xlim(x1, x2)
            axis_name.set_ylim(y1, y2)
            ax1.indicate_inset_zoom(axis_name)
            
            inset_labelling(axis_name, index, title)
    else: 
        print(axis_name + ": this extreme does not exist and cannot be plotted")
#%%

def plot_4_insets(feat1, feat2, graph_label1, graph_label2):
        
    fig, ax1 = plt.subplots()
    ax1.scatter(lc_feat[:,feat1], lc_feat[:,feat2], c = "black")
    ax1.set_xlabel(graph_label1)
    ax1.set_ylabel(graph_label2)
    
    index_list = []
    
    plot_inset(ax1, "axins1", feat1, feat2, 'ymax', index_list)
    plot_inset(ax1, "axins2", feat1, feat2, 'ymin', index_list)
    plot_inset(ax1, "axins3", feat1, feat2, 'xmax', index_list)
    plot_inset(ax1,"axins4", feat1, feat2, 'xmin', index_list)
    plt.tight_layout()

    plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/5-13/inset-plot-15.png")
    
#%%
#make this bad boy a function
def n_choose_2_insets(feature_vectors, date):
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
    for n in range(8):
        #feat1 = feature_vectors[:,n]
        graph_label1 = graph_labels[n]
        fname_label1 = fname_labels[n]
        for m in range(8):
            if m == n:
                continue
            graph_label2 = graph_labels[m]
            fname_label2 = fname_labels[m]                
            #feat2 = feature_vectors[:,m]
            
            plot_4_insets(n, m, graph_label1, graph_label2)
            #plt.scatter(feat1, feat2, s = 2, color = 'black')
            #plt.xlabel(graph_label1)
            #plt.ylabel(graph_label2)
 
            plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/" + date + "/nchoose2/" + date + "-" + fname_label1 + "-vs-" + fname_label2 + ".png")
            plt.show()

#%%
n_choose_2_insets(lc_feat, "5-13")
