# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:01:26 2020

@author: Lindsey Gordon 

Updated: May 31 2020
"""

import numpy as np
import numpy.ma as ma 
import pandas as pd 
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
from astropy.utils import exceptions

from datetime import datetime
import os
import shutil
from scipy.stats import moment, sigmaclip

import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
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
filedb = "/Users/conta/UROP_Spring_2020/plot_output/6-1/dbscan-confusion-matrices-scan.txt"
#feature optimizing for dbscan
#0 flat 1 sine 2 multiple transits 3 flares 4 single transits 5 not sure
hand_classes = classifications[:,1] #there are no class 5s for this group!!

with open(filedb, 'a') as file_object:
        file_object.write("This file contains the confusion matrices for Group 20-1-1, undergoing DBSCAN optimization")
        file_object.write("\n Eps values between 0.2 and 3. Min samples between 2 and 100")

eps_values = np.arange(0.2, 3, 0.2)
min_samps = np.arange(2,100,4)
avg_precision = []
avg_recall = []
accuracies = []
for i in range(len(min_samps)):
    for n in range(len(eps_values)):
        #dbscan predicting on features
        #feature vectors -> feats
        db_run = DBSCAN(eps=eps_values[n], min_samples=min_samps[i]).fit(feats) #run dbscan on all features
        predicted_classes = db_run.labels_
                
        #produce a confusion matrix
        db_matrix = confusion_matrix(hand_classes, predicted_classes)
        #print(db_matrix)
        noise_true = IsItIdentifyingNoise(predicted_classes)
        #check main diagonal
        db_accuracy = matrix_accuracy(db_matrix)     
        accuracies.append(db_accuracy)
        
        db_precision = matrix_precision(db_matrix)
        avg_precision.append(np.average(db_precision))
        #print(db_precision)
        
        db_recall = matrix_recall(db_matrix)
        avg_recall.append(np.average(db_recall))
        
        with open(filedb, 'a') as file_object:
            #file_object.write("\n")
            file_object.write("\n eps value:" + str(eps_values[n]) + " min samples: " + str(min_samps[i]))
            if noise_true == 'True':
                file_object.write("\n The 0th row and column represent a noise class (-1)")
            #file_object.write("\n")
            file_object.write("\n" + str(db_matrix) + "\n Accuracy:" + str(db_accuracy) + "\n Precisions:" + str(db_precision) + "\n Recalls:" + str(db_recall) + "\n")

#%%

fig, ax1 = plt.subplots()
    
ax1.set_ylabel('eps value')
ax1.set_xlabel('accuracy')
ax2 = ax1.twinx()
ax2.set_ylabel('min samples')

for n in range(len(accuracies)):
    k = n % 14
    ax1.scatter(accuracies[n], eps_values[k], c='black')
 
p = 0
for n in range(len(accuracies)):
    #k = n % 25 #for min samples plots
    if n == 0:
        ax2.scatter(accuracies[n], min_samps[0], c = 'red')
    else:
        if n % 14 == 0:
            p = p + 1
        ax2.scatter(accuracies[n], min_samps[p], c = 'red')       
    
    #ax1.scatter(accuracies[n], eps_values[l], c = 'black')
    #ax2.scatter(accuracies[n], min_samps[k], c = 'red')
    
#%%

#plotting for min samples
p = 0
for n in range(len(accuracies)):
    k = n % 14 #tells you what eps value you're on
    if 0 <= k <= 3:
        color = 'red'
    elif 3 < k <=6:
        color = 'pink'
    elif 6 < k <= 9:
        color = 'green'
    elif 9 < k <= 13:
        color = 'blue'
    else:
        color = 'black'
    
    if n == 0:
        plt.scatter(min_samps[0], avg_precision[n], c = color)
    else:
        if n % 14 == 0:
            p = p + 1
        
        plt.scatter(min_samps[p], avg_precision[n], c = color)

plt.xlabel("min samples")
plt.ylabel("avg precision")
plt.title("red: eps 0.2-0.8, pink 1-1.4, green 1.6-2, blue 2.2-2.8")

plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/6-1/dbscan-paramscan-avg-precision-samples-colored.pdf")

#%%
#plotting for eps value ranges

for n in range(len(accuracies)):
    k = n % 14 #what eps value is it
    
    if n <= 70: 
        color = 'red'
    elif 70 <n<= 140:
        color = 'pink'
    elif 140 < n <= 210:
        color = 'green'
    elif 210 < n <= 280:
        color = 'blue'
    elif n > 280:
        color = 'purple'
    
    #color = 'blue'
    plt.scatter(eps_values[k], avg_precision[n], c = color)

plt.xlabel("eps value")
plt.ylabel('avg precision')
plt.title("sample range by color: red: 2-18, pink 22-38, green 42-58, blue 62-78, purple 82-98")

plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/6-1/dbscan-paramscan-avg-precision-eps-colored.pdf")


#%%
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
        
    
plot_lc(t[0], inty[0], targ[0], 20)
plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/6-1/9154354-momentumdumps.png")
#%%

        

        