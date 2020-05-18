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

import shapely
from shapely import geometry
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

test(8) #should return 8 * 4

#if doing from scratch, run these: 
#time, intensity, targets = get_data_from_fits()
#intensity = normalize(intensity)
#lc_feat = create_list_featvec(time, intensity)

#if just running on intensity files you already have, run this:
#%%
time, intensity, targets, lc_feat = get_from_files()
#%%

n_choose_2_insets(time, intensity, lc_feat, targets, "5-15")

n_choose_2_features_plotting(lc_feat, lc_feat, "5-4", "none")
n_choose_2_features_plotting(lc_feat, lc_feat, "5-4", "kmeans")
n_choose_2_features_plotting(lc_feat, lc_feat, "5-4", "dbscan")

plot_lof(time, intensity, targets, lc_feat, 10, "5-4")

#%%

def lc_by_camera_ccd(file, camera, ccd):
    target_list = np.loadtxt(file)
    indexes = []
    for n in range(len(target_list)):
        if target_list[n][1] == camera and target_list[n][2] == ccd:
            indexes.append(n)
    matching_targets = target_list[indexes]
    return matching_targets

targets_20_1_1 = lc_by_camera_ccd("/Users/conta/UROP_Spring_2020/all_targets_S020_v1.txt", 1, 1)
#%%
targets_sector20 = np.loadtxt("/Users/conta/UROP_Spring_2020/all_targets_S020_v1.txt")

print(targets_sector20)

#getting just the targets on camera 1 ccd 1
indexes_20_1_1 = []
for n in range(len(targets_sector20)):
    if targets_sector20[n][1] == 1 and targets_sector20[n][2] == 1:
        indexes_20_1_1.append(n)

#print(indexes_20_1_1) #there are 925 of these

targets_20_1_1 = targets_sector20[indexes_20_1_1]

#%%

def get_lc_file_and_data(target):
    """ goes in, grabs the data for the target, gets the time index, intensity,
    etc. for the image. if connection error w/ MAST, adds to a list of failed targets
    to go get manually later."""
    fitspath = '/Users/conta/UROP_Spring_2020/mastDownload/TESS/'
    targ = "TIC " + str(int(target))
    print(targ)
    try:
        obs_table = Observations.query_object(targ, radius=".02 deg")
        data_products_by_obs = Observations.get_product_list(obs_table[0:2])
            
        filter_products = Observations.filter_products(data_products_by_obs, dataproduct_type = 'timeseries')
        manifest = Observations.download_products(filter_products)
        #print(manifest)
            
        filepaths = []
        for root, dirs, files in os.walk(fitspath):
            for name in files:
                if name.endswith(("lc.fits")):
                    filepaths.append(root + "/" + name)
                
        #print(filepaths)
            
        for file in filepaths:
                    # -- open file -------------------------------------------------------------
            f = fits.open(file, memmap=False)
            time1 = f[1].data['TIME']
            i1 = f[1].data['PDCSAP_FLUX']
            tic1 = f[1].header["OBJECT"]
                
            f.close()
                
        if os.path.isdir("mastDownload") == True:
            shutil.rmtree("mastDownload")               #deletes ALL data to conserve space
            print("folder deleted")
            
    except (ConnectionError, OSError, TimeoutError):
        print(targ + "could not be accessed due to a connection error")
        i1 = "Target failed, retry"
        time1 = "target failed, retry"
    
    return time1, i1



def lc_from_target_list(targetList):
    """ runs getting the file and data for all targets on the list
    then appends the time & intensity arrays and the TIC number into text files
    that can later be accessed
    also if it crashes in the night you just have to len the rows in the file and can
    pick up appending where you left off originally"""
    for n in range(len(targetList)):
        target = targetList[n][0]
        time1, i1 = get_lc_file_and_data(target)
        #storinglist[n][0] = target
        with open("/Users/conta/UROP_Spring_2020/plot_output/5-18/sector20_cam1_ccd1_targets.txt", 'a') as file_object:
            file_object.write("\n")
            file_object.write(str(target))
        with open("/Users/conta/UROP_Spring_2020/plot_output/5-18/sector20_cam1_ccd1_time.txt", 'a') as file_object:
            file_object.write("\n")
            np.savetxt(file_object, time1, delimiter = ',', newline = ' ')
        with open("/Users/conta/UROP_Spring_2020/plot_output/5-18/sector20_cam1_ccd1_intensities.txt", 'a') as file_object:   
            file_object.write("\n")
            np.savetxt(file_object, i1, delimiter = ',', newline = ' ')
        
        if n %30 == 0:
            print("30 completed")
       
#%%
lc_from_target_list(targets_20_1_1[502:])


#%%
t = np.loadtxt("/Users/conta/UROP_Spring_2020/plot_output/5-18/sector20_cam1_ccd1_time.txt")
inty = np.loadtxt("/Users/conta/UROP_Spring_2020/plot_output/5-18/sector20_cam1_ccd1_intensities.txt")
targy = np.loadtxt("/Users/conta/UROP_Spring_2020/plot_output/5-18/sector20_cam1_ccd1_targets.txt")
#%%

def interpolate_lc(time_indexes, intensities):
    """ interpolates all light curves in an array of all light curves""""
    
    interp_tol = 20. / (24*60) # >> interpolate small gaps (less than 20 minutes)
    
    interpolated_intensities = []
    interpolated_time = []
    for p in range(len(times)):

        time = time_indexes[p]
        i = intensities[p]
        
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
        interpolated_intensities.append(i_interp)
    
    # -- remove orbit nan gap ------------------------------------------------------
    interpolated_intensities = np.array(interpolated_intensities)
    # nan_inds = np.nonzero(np.prod(np.isnan(intensity)==False), axis = 0))
    nan_inds = np.nonzero(np.prod(np.isnan(interpolated_intensities)==False, axis = 0) == False)
    intensity = np.delete(intensity, nan_inds, 1) #each row of intensity is one interpolated light curve.
    for p in range(len(time_indexes)):
        time = time_indexes[p]
        time_corrected = np.delete(time, nan_inds)
        interpolated_times.append(time_corrected)
    
    interpolated_times = np.array(interpolated_times)
    return interpolated_times, interpolated_intensities



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