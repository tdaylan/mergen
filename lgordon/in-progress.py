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
from astroquery import exceptions
from astroquery.exceptions import RemoteServiceError
#from astropy.utils.exceptions import AstropyWarning, RemoteServiceError

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
#convert into fits files
#add TLS
t, inty, targ, feats, notes = load_in_a_group(20,1,1,"/Users/conta/UROP_Spring_2020/")


#%%

hdr = fits.Header()
hdr['SECTOR'] = 20
hdr['CAM'] = 1
hdr['CCD'] = 1


hdu = fits.PrimaryHDU(t, header=hdr)
hdu.writeto('/Users/conta/UROP_Spring_2020/fitstesting3.fits')
n2 = inty[0]
fits.append('/Users/conta/UROP_Spring_2020/fitstesting3.fits', n2, header=hdr)

#%%

### timeout faster in script!

def get_lc_file_and_data(yourpath, target):
    """ goes in, grabs the data for the target, gets the time index, intensity,
    etc. for the image. if connection error w/ MAST, skips it"""
    fitspath = yourpath + 'mastDownload/TESS/'
    targ = "TIC " + str(int(target))
    print(targ)
    try:
        #find and download data products for your target
        obs_table = Observations.query_object(targ, radius=".02 deg")
        data_products_by_obs = Observations.get_product_list(obs_table[0:4])
            
        #in theory, filter_products should let you sort out the non fits files but i 
        #simply could not get it to accept it despite following the API guidelines
        filter_products = Observations.filter_products(data_products_by_obs, dataproduct_type = 'timeseries')
        manifest = Observations.download_products(filter_products)
            
        #get all the paths to lc.fits files
        filepaths = []
        for root, dirs, files in os.walk(fitspath):
            for name in files:
                print(name)
                if name.endswith(("lc.fits")):
                    filepaths.append(root + "/" + name)
                    #print("appended", name, "to filepaths")
        
        print(len(filepaths))
        
        if len(filepaths) == 0: #if no lc.fits were downloaded, move on
            print(targ, "no light curve available")
            time1 = 0
            i1 = 0
        else: #if there are lc.fits files, open them and get the goods
                #get the goods and then close it #!!!!!!!!!!!!!!!!!!!! GET THE TIC FROM THE F I L E 
            f = fits.open(filepaths[0], memmap=False)
            time1 = f[1].data['TIME']
            i1 = f[1].data['PDCSAP_FLUX']                
            f.close()
                  
        #then delete all downloads in the folder, no matter what type
        if os.path.isdir("mastDownload") == True:
            shutil.rmtree("mastDownload")
            print("folder deleted")
            
        #corrects for connnection errors
    except (ConnectionError, OSError, TimeoutError, RemoteServiceError):
        print(targ + "could not be accessed due to an error")
        i1 = 0
        time1 = 0
    
    return time1, i1


#%%
#how to put a time limit on this bitch
    



#%%
data_access_by_group_fits("/Users/conta/UROP_Spring_2020/", "/Users/conta/UROP_Spring_2020/all_targets_S020_v1.txt", 20, 2, 1)
#%%

def get_lc_file_and_data(yourpath, target):
    """ goes in, grabs the data for the target, gets the time index, intensity,
    etc. for the image. if connection error w/ MAST, skips it"""
    fitspath = yourpath + 'mastDownload/TESS/'
    targ = "TIC " + str(int(target))
    print(targ)
    try:
        #find and download data products for your target
        obs_table = Observations.query_object(targ, radius=".02 deg")
        data_products_by_obs = Observations.get_product_list(obs_table[0:5])
            
        #in theory, filter_products should let you sort out the non fits files but i 
        #simply could not get it to accept it despite following the API guidelines
        filter_products = Observations.filter_products(data_products_by_obs, dataproduct_type = 'timeseries')
        manifest = Observations.download_products(filter_products)
            
        #get all the paths to lc.fits files
        filepaths = []
        for root, dirs, files in os.walk(fitspath):
            for name in files:
                print(name)
                if name.endswith(("lc.fits")):
                    filepaths.append(root + "/" + name)
                    #print("appended", name, "to filepaths")
        
        print(len(filepaths))
        
        if len(filepaths) == 0: #if no lc.fits were downloaded, move on
            print(targ, "no light curve available")
            time1 = 0
            i1 = 0
        else: #if there are lc.fits files, open them and get the goods
                #get the goods and then close it
            f = fits.open(filepaths[0], memmap=False)
            time1 = f[1].data['TIME']
            i1 = f[1].data['PDCSAP_FLUX']                
            f.close()
                  
        #then delete all downloads in the folder, no matter what type
        if os.path.isdir("mastDownload") == True:
            shutil.rmtree("mastDownload")
            print("folder deleted")
            
        #corrects for connnection errors
    except (ConnectionError, OSError, TimeoutError, RemoteServiceError):
        print(targ + "could not be accessed due to an error")
        i1 = 0
        time1 = 0
    
    return time1, i1



t1, i1 = get_lc_file_and_data(mypath, 71560002)
#%%


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


   
#%%  

features_insets2D_colored(t[0], inty, feats, targ, hand_classes, 'plot_output/6-12')

#%%
features_insets2D(t[0], inty, feats, targ, 'plot_output/6-12')

#%%
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
    inset_positions is a list  from a diff. function that holds the pos of insets
    last updated 5/29/20"""
    #position of box - needs to be dependent on location
    #max and min of the x axis features
    xmax = feature_vectors[:,feat1].max() 
    xmin = feature_vectors[:,feat1].min()
    
    ymax = feature_vectors[:,feat2].max() #min/max of y axis feature vectors
    ymin = feature_vectors[:,feat2].min()
    
    inset_width = range_x / 6 #this needs to be normalized to prevent weird stretching on graphs w/ bad proportions
    inset_height = range_y /16
    if x == 0:
        inset_x = coordtuple[0] - (inset_width * 1.5) #move left of point
    elif x == 1:
        inset_x = coordtuple[0] + (inset_width * 1.5) #move right of point
    if y == 0:
        inset_y = coordtuple[1] + (inset_height) #move up from point
    elif y == 1:
        inset_y = coordtuple[1] - (inset_height) #move down from point
    
    conc = np.column_stack((feature_vectors[:,feat1], feature_vectors[:,feat2]))
    polygon = calculate_polygon(inset_x, inset_y, inset_width, inset_height)
    
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
                inset_x = inset_x + (0.1*range_x)
                inset_y = inset_y + (0.1*range_y) 
            
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
    
times, intensities, features = interp_norm_sigmaclip_features("/Users/conta/UROP_Spring_2020/", raw_time, raw_int, targets, 20, 1, 3)
    
#%%
raw_time = np.loadtxt("/Users/conta/UROP_Spring_2020/Sector20Cam1CCD3/Sector20Cam1CCD3_times_raw.txt", skiprows=1)
raw_int = np.loadtxt("/Users/conta/UROP_Spring_2020/Sector20Cam1CCD3/Sector20Cam1CCD3_intensities_raw.txt", skiprows=1)

targets = np.loadtxt("/Users/conta/UROP_Spring_2020/Sector20Cam1CCD3/Sector20Cam1CCD3_targets.txt", skiprows=1)

#%%
plot_lof(times, intensities, targets, features, 20, "/Users/conta/UROP_Spring_2020/Sector20Cam1CCD3")