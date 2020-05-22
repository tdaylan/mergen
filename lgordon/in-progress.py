# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:01:26 2020

@author: Lindsey Gordon 

Updated: May 2020
"""

import numpy as np
import numpy.ma as ma 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)

from pylab import rcParams
rcParams['figure.figsize'] = 16, 6
rcParams["lines.markersize"] = 2
from scipy.signal import argrelextrema

import astropy
from astropy.io import fits
import scipy.signal as signal
from astropy.stats import SigmaClip
from astropy.utils import exceptions

from datetime import datetime
import os
import shutil
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
from astroquery.mast import Observations

import shapely
from shapely import geometry
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

test(8) #should return 8 * 4

#%%
n_choose_2_insets(t[0], intensities, features, targets_strings, "plot_output/5-22")

#%%
def inset_labelling(axis_name, time, intensity, targets, index, title):
    """formatting the labels for the inset plots"""
    axis_name.set_xlim(time[0], time[-1])
    axis_name.set_ylim(intensity[index].min(), intensity[index].max())
    #axis_name.set_xlabel("BJD [2457000]")
    #axis_name.set_ylabel("relative flux")
    axis_name.set_title(targets[index] + title, fontsize=8)

def n_choose_2_insets(time, intensity, feature_vectors, targets, folder):
    """plotting (n 2) features against each other w/ 4 extremes inset plotted
    feature_vectors is the list of ALL feature_vectors
    date must be a string in the format of the folder you are saving into ie "4-13"
    """   
    path = "/Users/conta/UROP_Spring_2020/" + folder + "/nchoose2-insets"
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
 
            plt.savefig(path + "/" + fname_label1 + "-vs-" + fname_label2 + ".png")
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
    xmax = feature_vectors[:,feat1].max() 
    xmin = feature_vectors[:,feat1].min()
    
    ymax = feature_vectors[:,feat2].max() 
    ymin = feature_vectors[:,feat2].min()
    
    inset_width = range_x / 6
    inset_height = range_y /16
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
    
    m = 0
    i = 0
    n = len(conc)
    k = 0
    
    while i < n:
        point = Point(conc[i])
        #is it on top of a point?
        if inset_x >= xmax:
            inset_x = inset_x - inset_width
            i = 0
        if inset_x < xmin:
            inset_x = inset_x + inset_width
            i = 0
            
        if inset_y >= ymax:
            inset_y = inset_y - inset_height
            i = 0
        if inset_y < ymin:
            inset_y = inset_y + inset_height
            i = 0
            
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
        #is it on top of another plot?
            while k < 8:
                bx, by, bw, bh = inset_positions[k]
                if bx <= inset_x <= (bx + bw):
                    inset_x = inset_x + (0.5 * inset_width)
                    k = 0
                    i = 0
                elif by <= inset_y <= (by + bh):
                    inset_y = inset_y + (0.5 * inset_height)
                    k = 0
                    i = 0
                else:
                    k = k + 1
            i = i + 1
                #this is the old way i ran it and it was okay but this is better
                #bBL = (bx, by)
                #bBR = (bx + bw, by)
                #bTL = (bx, by+bh)
                #bTR = (bx + bw, by + bh)
                #p1 = Polygon([bBL, bBR, bTL, bTR])
                
                #if p1.intersects(polygon):
                 #   inset_x = inset_x + (0.5*inset_width)
                  #  inset_y = inset_y + (0.5*inset_height)
                   # inset_BL = (inset_x, inset_y)
                   # inset_BR = (inset_x + inset_width, inset_y)
                   # inset_TL = (inset_x, inset_y + inset_height)
                   # inset_TR = (inset_x + inset_width, inset_y + inset_height)
                   # polygon = Polygon([inset_BL, inset_BR, inset_TL, inset_TR])
                   # k = 0
                #else: 
               #     k = k + 1
                    
            
    print("position determined")
    
    return inset_x, inset_y, inset_width, inset_height

#%%
np.savetxt("/Users/conta/UROP_Spring_2020/Sector20Cam1CCD2/GroupNotes.txt", failed_to_get)
#%%
t = np.loadtxt("/Users/conta/UROP_Spring_2020/Sector20Cam1CCD2/Sector20Cam1CCD2_times_raw.txt")
intensities = np.loadtxt("/Users/conta/UROP_Spring_2020/Sector20Cam1CCD2/Sector20Cam1CCD2_intensities_raw.txt")
targets = np.loadtxt("/Users/conta/UROP_Spring_2020/Sector20Cam1CCD2/Sector20Cam1CCD2_targets.txt")

#%%


#%%
mypath = "/Users/conta/UROP_Spring_2020/"
sectorfile = "/Users/conta/UROP_Spring_2020/all_targets_S020_v1.txt"
sector = 20
camera = 1
ccd = 2
position = 313

times, intensities, failed_to_get, targets = interrupted_start_in_middle(position, mypath, sectorfile, sector, camera, ccd)

#%%
#feature optimizing for dbscan

    predict_on_100 =  lc_feat[0:100][:,[3,11,12]]
    db_100 = DBSCAN(eps=0.5, min_samples=10).fit(predict_on_100)
    predicted_100 = db_100.labels_
    
    #producing the confusion matrix
    labelled_100 = np.loadtxt("/Users/conta/UROP_Spring_2020/100-labelled/labelled_100.txt", delimiter=',', usecols=1, skiprows=1, unpack=True)
    print("predicted 100:", predicted_100, "\nlabelled 100:", labelled_100)
    
    dbscan_matrix = confusion_matrix(labelled_100, predicted_100)
    
    print(dbscan_matrix)
    
    dbscan_diagonal = check_diagonalized(dbscan_matrix)
    
    with open("/Users/conta/UROP_Spring_2020/plot_output/5-11/dbscan-confusion-matrices.txt", 'a') as file_object:
        # Append 'hello' at the end of file
        file_object.write("\n")
        file_object.write("kurtosis, ln slope, P0\n" + str(dbscan_matrix) + "\n" + str( dbscan_diagonal))

#%%
lc_cropped = lc_feat[0:100][:,[0,3,8,9,11,12,13,14,15]] 

n_choose_2_features_plotting(lc_feat[0:100], lc_cropped[0:100], "5-11", "dbscan")
   

with open("/Users/conta/UROP_Spring_2020/plot_output/5-11/dbscan-matrices-plotted.txt", 'a') as file_object:
        # Append 'hello' at the end of file
        file_object.write("\n")
        file_object.write("kurtosis, ln slope, P0\n" + str(dbscan_matrix) + "\n" + str( dbscan_diagonal))
#%%
        
from sklearn.decomposition import PCA


pca = PCA(n_components=1, whiten=True)
pca_feat = pca.fit_transform(lc_feat[0:100])
print(pca_feat)

db_100 = DBSCAN(eps=0.5, min_samples=10).fit(pca_feat)
predicted_100 = db_100.labels_

dbscan_matrix = confusion_matrix(labelled_100, predicted_100)
dbscan_matrix
dbscan_diagonal = check_diagonalized(dbscan_matrix)


with open("/Users/conta/UROP_Spring_2020/plot_output/5-11/PCA-confusion-matrices.txt", 'a') as file_object:
        # Append 'hello' at the end of file
        file_object.write("\n")
        file_object.write("n components: 1, whiten = True" + str(dbscan_matrix) + "\n" + str( dbscan_diagonal))
#%%

number_targets = len(targets)
sector_number = np.zeros((number_targets, 1))
camera_number = np.zeros((number_targets, 1))
ccd_number = np.zeros((number_targets, 1))
for n in range(number_targets):
    head = print_header(n)
    sector_number[n] = head["SECTOR"]
    camera_number[n] = head["CAMERA"]
    ccd_number[n] = np.round(head["CCD"], 0)
   
sectorcameraccd = np.column_stack((sector_number, camera_number, ccd_number))    
#%%    
np.savetxt("/Users/conta/UROP_Spring_2020/sector-cam-ccd.txt", sectorcameraccd, header = "sector-camera-ccd numbers for each value") 