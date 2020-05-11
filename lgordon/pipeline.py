# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:48:56 2020

@author: Lindsey Gordon

Last updated: April 2020
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

test(8) #should return 8 * 4

#if doing from scratch, run these: 
#time, intensity, targets = get_data_from_fits()
#intensity = normalize(intensity)
#lc_feat = create_list_featvec(time, intensity)

#if just running on intensity files you already have, run this:
#%%
time, intensity, targets, lc_feat = get_from_files()



#%%
n_choose_2_features_plotting(lc_feat, lc_feat, "5-4", "none")
n_choose_2_features_plotting(lc_feat, lc_feat, "5-4", "kmeans")
n_choose_2_features_plotting(lc_feat, lc_feat, "5-4", "dbscan")

plot_lof(time, intensity, targets, lc_feat, 10, "5-4")

#%%

targets_sector20 = np.loadtxt("/Users/conta/UROP_Spring_2020/all_targets_S020_v1.txt", usecols = 0)

print(targets_sector20)

targets20_5000 = targets_sector20[0:5000]

#targets20_10000

#%%

def lc_from_target_list(targetList):
    fitspath = '/Users/conta/UROP_Spring_2020/mastDownload/TESS/'
    ints = []
    times = []
    targets_TICS = []
    for target in targetList:
        #print(target, type(target), int(target))
        targ = "TIC " + str(int(target))
        print(targ)
        obs_table = Observations.query_object(targ, radius=".02 deg")
        data_products_by_obs = Observations.get_product_list(obs_table[0:2])
        
        filter_products = Observations.filter_products(data_products_by_obs, dataproduct_type = 'timeseries')
        manifest = Observations.download_products(filter_products)
        print(manifest)
        
        filepaths = []
        for root, dirs, files in os.walk(fitspath):
            for name in files:
                if name.endswith(("lc.fits")):
                    filepaths.append(root + "/" + name)
            
        print(filepaths)
        
        for file in filepaths:
                # -- open file -------------------------------------------------------------
            f = fits.open(file, memmap=False)

            time1 = f[1].data['TIME']
            i1 = f[1].data['PDCSAP_FLUX']
            tic1 = f[1].header["OBJECT"]
            times.append(time1)
            ints.append(i1)
            targets_TICS.append(tic1)
            f.close()
            
        if os.path.isdir("mastDownload") == True:
            shutil.rmtree("mastDownload")               #deletes ALL data to conserve space
            print("folder deleted")
        
        #print(ints)
        
    return times, ints, targets_TICS
        
        
times, ints, targets_TICS = lc_from_target_list(targets20_5000)


#%%
file_object = open('/Users/conta/UROP_Spring_2020/plot_output/5-11/dbscan-confusion-matrices.txt', 'a')
 
# Append 'hello' at the end of file
file_object.write('This file contains the confusion matrices and diagonal score for different combinations of features')
 
# Close the file
file_object.close()
#%%


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
