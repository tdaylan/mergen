# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:01:26 2020

@author: Lindsey Gordon 

Updated: May 29 2020
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

def load_in_a_group(sector, camera, ccd, path):
    """loads in a given group's data
    path needs to be a string, ending with a forward slash"""
    folder = "Sector"+str(sector)+"Cam"+str(camera)+"CCD"+str(ccd)
    time_path = path + folder + "/" + folder + "_times_processed.txt"
    intensities_path = path + folder + "/" + folder + "_intensities_processed.txt"
    features_path = path + folder + "/" + folder + "_features.txt"
    targets_path = path + folder + "/" + folder + "_targets.txt"
    notes_path = path + folder + "/" + folder + "_group_notes.txt"
    
    t = np.loadtxt(time_path)
    intensities = np.loadtxt(intensities_path)
    targets = np.loadtxt(targets_path)
    features = np.loadtxt(features_path, skiprows=1)
    notes = np.loadtxt(notes_path, skiprows=1)
    
    return t, intensities, targets, features, notes

t, inty, targ, feats, notes = load_in_a_group(20,1,1,"/Users/conta/UROP_Spring_2020/")

#%%
n_choose_2_insets(t[0], intensities, features, targets_strings, "plot_output/5-26")
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
    inset_positions = np.zeros((8,2))
    
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
        inset_positions[n] = (inset_x, inset_y)
        #print(inset_positions)
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

def calculate_polygon(inset_x, inset_y, inset_width, inset_height):
    """ calculates the polygon of the inset plot"""
    inset_BL = (inset_x, inset_y)
    inset_BR = (inset_x + inset_width, inset_y)
    inset_TL = (inset_x, inset_y + inset_height)
    inset_TR = (inset_x + inset_width, inset_y + inset_height)
    polygon = Polygon([inset_BL, inset_BR, inset_TL, inset_TR])
    return polygon
        
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
    
    conc = np.column_stack((feature_vectors[:,feat1], feature_vectors[:,feat2]))
    polygon = calculate_polygon(inset_x, inset_y, inset_width, inset_height)
    
    points_good = 0
    insets_good = 0
    borders_good = 0
    m = 0
    i = 0
    n = len(conc)
    k = 0
    
    while i < n:
        #is it on the graph? if it is not, move it, recalculate, and go back to the beginning
        while m == 0:
            if inset_x >= xmax:
                inset_x = inset_x - inset_width
                polygon = calculate_polygon(inset_x, inset_y, inset_width, inset_height)
                i = 0
                k = 0
            elif inset_x < xmin:
                inset_x = inset_x + inset_width
                polygon = calculate_polygon(inset_x, inset_y, inset_width, inset_height)
                i = 0
                k = 0
            elif inset_y >= ymax:
                inset_y = inset_y - inset_height
                polygon = calculate_polygon(inset_x, inset_y, inset_width, inset_height)
                i = 0
                k = 0
            elif inset_y < ymin:
                inset_y = inset_y + inset_height
                polygon = calculate_polygon(inset_x, inset_y, inset_width, inset_height)
                i = 0
                k = 0
            else: 
                m = 1 #it is on the graph
            
        #is it on top of another plot? if it is, move it, recalculate, and go back to the 
        #absolute beginning to double check if it's in the borders still
        while k < 8:
            bx, by = inset_positions[k]
            p1 = calculate_polygon(bx, by, inset_width, inset_height)
            if polygon.intersects(p1):
                if x == 0: 
                    inset_x = inset_x - (0.1 * range_x)
                elif x == 1:
                    inset_x = inset_x + (0.1 * range_x)
                if y == 0:
                    inset_y = inset_y + (0.1 * range_y)
                elif y == 1:
                    inset_y = inset_y - (0.1 * range_y) 
                
                polygon = calculate_polygon(inset_x, inset_y, inset_width, inset_height)
                m = 0
                k = 0
                i = 0
            else: 
                k = k + 1 #check next inset in list
            
        #is it on top of a point? if it is, move it, recalculate, and go back to beginning
        point = Point(conc[i])
        if polygon.contains(point) == True:
            if x == 0: 
                inset_x = inset_x - (0.01 * range_x)
            else:
                inset_x = inset_x + (0.01 * range_x)
                
            if y == 0:
                inset_y = inset_y + (0.01 * range_y)
            else:
                inset_y = inset_y - (0.01 * range_y)
            
            polygon = calculate_polygon(inset_x, inset_y, inset_width, inset_height)
            
            m = 0
            k = 0
            i = 0
            #print("moving")
        elif polygon.contains(point) == False:
        #if not on top of a point, move to the next one
            i = i + 1
            
    print("position determined")
    
    return inset_x, inset_y, inset_width, inset_height

#%%
    
classifications = np.loadtxt("/Users/conta/UROP_Spring_2020/Sector20Cam1CCD1/classified_Sector20Cam1CCD1.txt", delimiter = ' ')

#%%
classes = classifications[:,1]
rcParams['figure.figsize'] = 10,10
graph_labels = ["Average", "Variance", "Skewness", "Kurtosis", "Log Variance",
                    "Log Skewness", "Log Kurtosis", "Maximum Power", "Log Maximum Power", 
                    "Period of Maximum Power (0.1 to 10 days)","Slope" , "Log Slope",
                    "P0", "P1", "P2", "Period of Maximum Power (0.001 to 0.1 days)"]
fname_labels = ["Avg", "Var", "Skew", "Kurt", "LogVar", "LogSkew", "LogKurt",
                    "MaxPower", "LogMaxPower", "Period0_1to10", "Slope", "LogSlope",
                    "P0", "P1", "P2", "Period0to0_1"]
for n in range(16):
    feat1 = feats[:,n]
    graph_label1 = graph_labels[n]
    fname_label1 = fname_labels[n]
    for m in range(16):
        if m == n:
            continue
        graph_label2 = graph_labels[m]
        fname_label2 = fname_labels[m]                
        feat2 = feats[:,m]
            
        for p in range(len(feats)):
            if classes[p] == 0:
                color = "red"
            elif classes[p] == 1:
                color = "blue"
            elif classes[p] == 2:
                color = "green"
            elif classes[p] == 3:
                color = "purple"
            elif classes[p] == 4:
                color = "yellow"
            elif classes[p] == 5:
                color = "magenta"
            plt.scatter(feat1[p], feat2[p], c = color, s = 5)
        plt.xlabel(graph_label1)
        plt.ylabel(graph_label2)
        plt.savefig(("/Users/conta/UROP_Spring_2020/plot_output/5-29/2DFeatures-Colored/" + fname_label1 + "-vs-" + fname_label2 + "-handclassed.png"))
        plt.show()


#%%
filedb = "/Users/conta/UROP_Spring_2020/plot_output/5-29/dbscan-confusion-matrices-6.txt"
#feature optimizing for dbscan
#0 flat 1 sine 2 multiple transits 3 flares 4 single transits 5 not sure
hand_classes = classifications[:,1] #there are no class 5s for this group!!

with open(filedb, 'a') as file_object:
        file_object.write("This file contains the confusion matrices for Group 20-1-1, undergoing DBSCAN optimization")
        file_object.write("\n Min samples 6. Changing eps value")

eps_values = np.arange(0.2, 3, 0.2)
#min_samps = np.arange(2,60,4)
for n in range(len(eps_values)):
    #dbscan predicting on features
    #feature vectors -> feats
    db_run = DBSCAN(eps=eps_values[n], min_samples=6).fit(feats) #run dbscan on all features
    predicted_classes = db_run.labels_
            
    #produce a confusion matrix
    db_matrix = confusion_matrix(hand_classes, predicted_classes)
    #print(db_matrix)
    noise_true = IsItIdentifyingNoise(predicted_classes)
    #check main diagonal
    db_accuracy = matrix_accuracy(db_matrix)     
    #print(db_accuracy)
    
    db_precision = matrix_precision(db_matrix)
    #print(db_precision)
    
    db_recall = matrix_recall(db_matrix)
    
    with open(filedb, 'a') as file_object:
        #file_object.write("\n")
        file_object.write("\n eps value:" + str(eps_values[n]))
        if noise_true == 'True':
            file_object.write("\n The 0th row and column represent a noise class (-1)")
        #file_object.write("\n")
        file_object.write("\n" + str(db_matrix) + "\n Accuracy:" + str(db_accuracy) + "\n Precisions:" + str(db_precision) + "\n Recalls:" + str(db_recall) + "\n")


#%%

def IsItIdentifyingNoise(predicted_classes):
    noise = "False"
    for n in range(len(predicted_classes)):
        if predicted_classes[n] == -1:
            noise = "True"
    return noise
        
        
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
#