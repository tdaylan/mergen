# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:48:56 2020

@author: Lindsey Gordon

Pipeline to produce all files for a given group of data.

Last updated: May 2020
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
#add a yourpath argument to make it universalized
mypath = "/Users/conta/UROP_Spring_2020/"
def data_process_a_group(yourpath, sectorfile, sector, camera, ccd):
    """you will need:
        the file for your sector from TESS
        sector number (as int)
        camera number you want (as int/float)
        ccd number you want (as int/float)"""
    # produce the folder to save everything into and set up file names
    folder_name = "Sector" + str(sector) + "Cam" + str(camera) + "CCD" + str(ccd)
    path = yourpath + folder_name
    fname_time = path + "/" + folder_name + "_times_raw.txt"
    fname_int = path + "/" + folder_name + "_intensities_raw.txt"
    fname_targets = path + "/" + folder_name + "_targets.txt"
    fname_times_interp = path + "/" + folder_name + "_interp_times.txt"
    fname_ints_processed = path + "/" + folder_name + "_ints_processed.txt"
    try:
        os.makedirs(path)
        print ("Successfully created the directory %s" % path) 
        
    # get just the list of targets for the specified sector, camera, ccd --------
        target_list = lc_by_camera_ccd(sectorfile, camera, ccd)
        print("there are ", len(target_list), "targets")
    # get the light curve for each target on the list, and save into a text file
        confirmation, failed_to_get = lc_from_target_list(yourpath, target_list, fname_time, fname_int, fname_targets)
        print(confirmation)
        print("failed to get", len(failed_to_get), "targets, go back in and investigate")
    # import the files you just created
        times = np.loadtxt(fname_time)
        intensities = np.loadtxt(fname_int)
        targets = np.loadtxt(fname_targets)
    #check to be sure all have the same size, if not, report back an error
        if len(times) == len(intensities) == len(targets):
    #interpolate and normalize/sigma clip
            interp_times, interp_intensities = interpolate_lc(times, intensities)
            normalized_intensities = normalize(interp_intensities)
    #save these into their own files, and report these arrays back
            np.savetxt(fname_times_interp, interp_times)
            times = np.loadtxt(fname_times_interp)
            np.savetxt(fname_ints_processed, normalized_intensities)
            intensities = np.loadtxt(fname_ints_processed)
     
        else: #if there is an error with the number of lines in times vs ints vs targets
            print("There is a disagreement between the number of lines saved in each text file")
            times = "Does not exist"
            intensities = "Does not exist"
            failed_to_get = "all"
    
    except OSError: #if there is an error creating the folder
        print ("This directory already exists, or there is some other OS error")
        times = "Does not exist" 
        intensities = "does not exist"
        failed_to_get = "all"
        
    return times, intensities, failed_to_get

#%%
def lc_by_camera_ccd(sectorfile, camera, ccd):
    """gets all the targets for a given sector, camera, ccd"""
    target_list = np.loadtxt(sectorfile)     #load in the target file
    indexes = [] #empty array to save indexes into
    for n in range(len(target_list)): #for each item in the list of targets
        if target_list[n][1] == camera and target_list[n][2] == ccd: #be sure it matches
            indexes.append(n) #if it does, append to index list
    matching_targets = target_list[indexes] #just grab those indexes
    return matching_targets #return list of only targets on that specific ccd


def get_lc_file_and_data(yourpath, target):
    """ goes in, grabs the data for the target, gets the time index, intensity,
    etc. for the image. if connection error w/ MAST, skips it"""
    fitspath = yourpath + 'mastDownload/TESS/'
    targ = "TIC " + str(int(target))
    print(targ)
    try:
        #find and download data products for your target
        obs_table = Observations.query_object(targ, radius=".02 deg")
        data_products_by_obs = Observations.get_product_list(obs_table[0:2])
            
        #in theory, filter_products should let you sort out the non fits files but i 
        #simply could not get it to accept it despite followin the API guidelines
        filter_products = Observations.filter_products(data_products_by_obs, dataproduct_type = 'timeseries')
        manifest = Observations.download_products(filter_products)
        #print(manifest)
            
        #get all the paths to lc.fits files
        filepaths = []
        for root, dirs, files in os.walk(fitspath):
            for name in files:
                if name.endswith(("lc.fits")):
                    filepaths.append(root + "/" + name)
                 
        if len(filepaths) == 0: #if no lc.fits were downloaded, move on
            print(targ, "no light curve available")
            time1 = 0
            i1 = 0
        else: #if there are lc.fits files, open them and get the goods
            for file in filepaths:
                #get the goods and then close it
                f = fits.open(file, memmap=False)
                time1 = f[1].data['TIME']
                i1 = f[1].data['PDCSAP_FLUX']                
                f.close()
                  
        #then delete all downloads in the folder, no matter what type
        if os.path.isdir("mastDownload") == True:
            shutil.rmtree("mastDownload")
            print("folder deleted")
            
        #corrects for connnection errors
    except (ConnectionError, OSError, TimeoutError):
        print(targ + "could not be accessed due to an error")
        i1 = 0
        time1 = 0
    
    return time1, i1

def lc_from_target_list(yourpath, targetList, fname_time, fname_int, fname_targets):
    """ runs getting the file and data for all targets on the list
    then appends the time & intensity arrays and the TIC number into text files
    that can later be accessed
    also if it crashes in the night you just have to len the rows in the file and can
    pick up appending where you left off originally"""
    failed_to_get = [] #empty array for all failures
    for n in range(len(targetList)): #for each item on the list
        target = targetList[n][0] #get that target number
        time1, i1 = get_lc_file_and_data(yourpath, target) #go in and get the time and int
        if time1 == 0 or i1 == 0: #if there was an error, add it to the list and continue
            failed_to_get.append(target)
            continue
    # add data to the files
        with open(fname_targets, 'a') as file_object:
            file_object.write("\n")
            file_object.write(str(target))
        with open(fname_time, 'a') as file_object:
            file_object.write("\n")
            np.savetxt(file_object, time1, delimiter = ',', newline = ' ')
        with open(fname_int, 'a') as file_object:   
            file_object.write("\n")
            np.savetxt(file_object, i1, delimiter = ',', newline = ' ')
        
        if n %30 == 0: #every 30, print how many have been done
            print(str(n), "completed")
    confirmation = "lc_from_target_list has finished running"
    return confirmation, failed_to_get


#%%
t6 = np.loadtxt("/Users/conta/UROP_Spring_2020/Sector20Cam1CCD1/sector20_cam1_ccd1_interp_times.txt")
inty7 = np.loadtxt("/Users/conta/UROP_Spring_2020/Sector20Cam1CCD1/sector20_cam1_ccd1_processed_intensities.txt")



#%%
    
missing_curves = ["241167417", "453404919", "453406071", "458409287", "741653758", "80201562"]

fitspath = '/Users/conta/UROP_Spring_2020/mastDownload/TESS/'
for n in range(len(missing_curves)):
    obs_table = Observations.query_object("TIC " + missing_curves[n], radius=".02 deg")
    data_products_by_obs = Observations.get_product_list(obs_table[0:2])
            
    filter_products = Observations.filter_products(data_products_by_obs, dataproduct_type = 'timeseries')
    manifest = Observations.download_products(filter_products)
        #print(manifest)         
#%%
    filepaths = []
    for root, dirs, files in os.walk(fitspath):
        for name in files:
            if name.endswith(("lc.fits")):
                filepaths.append(root + "/" + name)
                    
print(filepaths)
                #%%
        for file in filepaths:
                        # -- open file -------------------------------------------------------------
            f = fits.open(file, memmap=False)
            time1 = f[1].data['TIME']
            i1 = f[1].data['PDCSAP_FLUX']
            tic1 = f[1].header["OBJECT"]
                    
            f.close()
       
          
                
                
                