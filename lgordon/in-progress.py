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