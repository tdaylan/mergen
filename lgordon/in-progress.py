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
from sklearn.neighbors import LocalOutlierFactor

import astroquery
from astroquery.simbad import Simbad
from astroquery.mast import Catalogs
from astroquery.mast import Observations

import shapely
from shapely import geometry
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import classification_functions
from classification_functions import *
import data_functions
from data_functions import *
import plotting_functions
from plotting_functions import *


test_data() #should return 8 * 4
test_plotting()
#%%

t, inty, targ, feats, notes = load_in_a_group(20,1,1,"/Users/conta/UROP_Spring_2020/")
  
classifications = np.loadtxt("/Users/conta/UROP_Spring_2020/Sector20Cam1CCD1/classified_Sector20Cam1CCD1.txt", delimiter = ' ')
classes = classifications[:,1]


#%%
filedb = "/Users/conta/UROP_Spring_2020/plot_output/6-5/dbscan-confusion-matrices-scan.txt"
hand_classes = classifications[:,1] #there are no class 5s for this group!!



path = "/Users/conta/UROP_Spring_2020/plot_output/6-5"

def dbscan_param_scan(path, features, epsmin, epsmax, epsstep, sampmin, sampmax, sampstep, hand_classes):
    """run parameter scan for dbscan over given range of eps and samples,
    knowing the hand classified values"""
    filedb = path + "/dbscan-confusion-matrices-scan.txt"
    #feature optimizing for dbscan
    #0 flat 1 sine 2 multiple transits 3 flares 4 single transits 5 not sure
    text1 = "\n Eps values between " + str(epsmin) + " and " + str(epsmax) + ". Min samples between " + str(sampmin) + " and " + str(sampmax)
    with open(filedb, 'a') as file_object:
            file_object.write("This file contains the confusion matrices for the given features undergoing DBSCAN optimization")
            file_object.write(text1)
    
    eps_values = np.arange(epsmin, epsmax, epsstep)
    min_samps = np.arange(sampmin,sampmax,sampstep)
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
        #then do color-coded plotting for the different ranges: 
        
    #plotting eps value ranges: 
    num_eps = len(eps_values)
    num_samps = len(min_samps)
    num_combos = num_eps*num_samps
    color_division = num_combos / 5
    y_axes = [accuracies, avg_precision, avg_recall]
    y_labels = ["accuracies", "average-precision", "average-recall"]
    for m in range(3):
        y_axis = y_axes[m]
        y_label = y_labels[m]
        for n in range(num_combos):
            k = n % num_eps #what eps value is it
            
            if n <= color_division: 
                color = 'red'
            elif color_division <n<= 2*color_division:
                color = 'pink'
            elif 2*color_division < n <= 3*color_division:
                color = 'green'
            elif 3*color_division < n <= 4*color_division:
                color = 'blue'
            elif n > 4*color_division:
                color = 'purple'
            
            plt.scatter(eps_values[k], y_axis[n], c = color)
        
        plt.xlabel("eps value")
        plt.ylabel(y_label)
        plt.title("sample range by color: red, pink, green, blue, purple, are increasing by # of samples")
        
        plt.savefig(path + "/dbscan-paramscan-" + y_label +"-eps-colored.pdf")
        plt.show()


        
    return accuracies, avg_precision, avg_recall

acc, avgp, avgr = dbscan_param_scan(path, feats, 0.2, 3, 0.2, 2, 50, 4, hand_classes)

#%%
#color coded plotting of eps/min samples
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
#plotting to indicate both actual class and the class predicted by dbscan
#eps 2.2, min samples 18
path = "/Users/conta/UROP_Spring_2020/plot_output/6-5/color-shape-2D-plots-kmeans"
hand_classes = classifications[:,1] #there are no class 5s for this group!!


        
features_2D_colorshape(feats, path, 'kmeans', hand_classes)

   
#%%
    
def data_access_by_group(yourpath, sectorfile, sector, camera, ccd):
    """you will need:
        your path into the main folder you're working in - must end with /
        the file for your sector from TESS (full path)
        sector number (as int)
        camera number you want (as int/float)
        ccd number you want (as int/float)
        this ONLY """
    # produce the folder to save everything into and set up file names
    folder_name = "Sector" + str(sector) + "Cam" + str(camera) + "CCD" + str(ccd)
    path = yourpath + folder_name
    fname_time = path + "/" + folder_name + "_times_raw.txt"
    fname_int = path + "/" + folder_name + "_intensities_raw.txt"
    fname_targets = path + "/" + folder_name + "_targets.txt"
    fname_notes = path + "/" + folder_name + "_group_notes.txt"
    
    try:
        os.makedirs(path)
        print ("Successfully created the directory %s" % path) 
        with open(fname_time, 'a') as file_object:
            file_object.write("This file contains the raw time indices for this group")
        with open(fname_int, 'a') as file_object:
            file_object.write("This file contains the raw intensities for this group")
        with open(fname_targets, 'a') as file_object:
            file_object.write("This file contains the target TICs for this group")
        with open(fname_notes, 'a') as file_object:
            file_object.write("This file contains group notes, including any TICs that could not be accessed.")
        # get just the list of targets for the specified sector, camera, ccd --------
        target_list = lc_by_camera_ccd(sectorfile, camera, ccd)
        print("there are ", len(target_list), "targets")
    # get the light curve for each target on the list, and save into a text file
        confirmation = lc_from_target_list(yourpath, target_list, fname_time, fname_int, fname_targets, fname_notes)
        print(confirmation)
        #print("failed to get", len(failed_to_get), "targets")
    # import the files you just created
        times = np.loadtxt(fname_time, skiprows=1)
        intensities = np.loadtxt(fname_int, skiprows=1)
        targets = np.loadtxt(fname_targets, skiprows=1)
        print("found data for ", len(targets), " targets")
    #check to be sure all have the same size, if not, report back an error
        
    except OSError: #if there is an error creating the folder
        print("There was an OS Error trying to create the folder. Checking to see if data is already saved there")
        times = intensities = targets = path = "empty"
        
    return times, intensities, targets, path

def follow_up_on_missed_targets(yourpath, sector, camera, ccd):
    """ function to follow up on rejected TIC ids"""
    folder_name = "Sector" + str(sector) + "Cam" + str(camera) + "CCD" + str(ccd)
    path = yourpath + folder_name
    fname_time = path + "/" + folder_name + "_times_raw.txt"
    fname_int = path + "/" + folder_name + "_intensities_raw.txt"
    fname_targets = path + "/" + folder_name + "_targets.txt"
    fname_notes = path + "/" + folder_name + "_group_notes.txt"
    fname_notes_followed_up = path + "/" + folder_name + "_targets_still_no_data.txt"
    
    
    retry_targets = np.loadtxt(fname_notes, skiprows=1)
    
    with open(fname_notes, 'a') as file_object:
        file_object.write("Data could not be found for the following TICs after two attempts")
    
    
    
    confirmation = lc_from_target_list(folderpath, retry_targets, fname_time, fname_int, fname_targets, fname_notes_followed_up)
    print(confirmation)
    
    times = np.loadtxt(fname_time, skiprows=1)
    intensities = np.loadtxt(fname_int, skiprows=1)
    targets = np.loadtxt(fname_targets, skiprows=1)
    print("after following up, found data for ", len(targets), " targets")
    
    return times, intensities, targets, path

def interp_norm_sigmaclip_features(yourpath, times, intensities, targets):
    """interpolates, normalizes, and sigma clips all light curves
    then produces feature vectors for them"""
    folder_name = "Sector" + str(sector) + "Cam" + str(camera) + "CCD" + str(ccd)
    path = yourpath + folder_name
    fname_times_interp = path + "/" + folder_name + "_times_processed.txt"
    fname_ints_processed = path + "/" + folder_name + "_intensities_processed.txt"
    fname_features = path + "/"+ folder_name + "_features.txt"
    
    with open(fname_times_interp, 'a') as file_object:
        file_object.write("This file contains the processed time indices for this group")
    with open(fname_ints_processed, 'a') as file_object:
        file_object.write("This file contains the processed intensities for this group")
    with open(fname_features, 'a') as file_object:
        file_object.write("This file contains the feature vectors for each target in this group. ")
    
    
    if len(intensities) == len(targets):
    #interpolate and normalize/sigma clip
        interp_times, interp_intensities = interpolate_lc(times, intensities)
        normalized_intensities = normalize(interp_intensities)
    #save these into their own files, and report these arrays back
        np.savetxt(fname_times_interp, interp_times)
        times = np.loadtxt(fname_times_interp, skiprows=1)
        np.savetxt(fname_ints_processed, normalized_intensities)
        intensities = np.loadtxt(fname_ints_processed, skiprows=1)
        print("You can now access time arrays, processed intensities, targets, and an array of TICs you could not get")
        
        features = create_list_featvec(times[0], intensities)
        np.savetxt(fname_features, features)
        print("Feature vector creation complete")
            
    else: #if there is an error with the number of lines in times vs ints vs targets
        print("There is a disagreement between the number of lines saved in intensities and targets, cannot process data")
        
    return times, intensities, features
        