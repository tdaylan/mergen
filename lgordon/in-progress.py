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

def data_process_a_group(yourpath, sectorfile, sector, camera, ccd):
    """you will need:
        your path into the main folder you're working in
        the file for your sector from TESS (full path)
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
        print("failed to get", len(failed_to_get), "targets")
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
        targets = "none"
        
    return times, intensities, failed_to_get, targets

def lc_by_camera_ccd(sectorfile, camera, ccd):
    """gets all the targets for a given sector, camera, ccd
    from the master list for that sector"""
    target_list = np.loadtxt(sectorfile)     #load in the target file
    indexes = [] #empty array to save indexes into
    for n in range(len(target_list)): #for each item in the list of targets
        if target_list[n][1] == camera and target_list[n][2] == ccd: #be sure it matches
            indexes.append(n) #if it does, append to index list
    matching_targets = target_list[indexes] #just grab those indexes
    return matching_targets #return list of only targets on that specific ccd

def lc_from_target_list(yourpath, targetList, fname_time, fname_int, fname_targets):
    """ runs getting the files and data for all targets on the list
    then appends the time & intensity arrays and the TIC number into text files
    that can later be accessed
    also if it crashes in the night you just have to len the rows in the file and can
    pick up appending where you left off originally"""
    failed_to_get = [] #empty array for all failures
    for n in range(len(targetList)): #for each item on the list
        target = targetList[n][0] #get that target number
        time1, i1 = get_lc_file_and_data(yourpath, target) #go in and get the time and int
        if type(time1) != np.ndarray: #if there was an error, add it to the list and continue
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
        
        if n %50 == 0: #every 50, print how many have been done
            print(str(n), "completed")
    confirmation = "lc_from_target_list has finished running"
    return confirmation, failed_to_get

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
    except (ConnectionError, OSError, TimeoutError, RemoteServiceError):
        print(targ + "could not be accessed due to an error")
        i1 = 0
        time1 = 0
    
    return time1, i1

#%%
mypath = "/Users/conta/UROP_Spring_2020/"
sectorfile = "/Users/conta/UROP_Spring_2020/all_targets_S020_v1.txt"
sector = 20
camera = 1
ccd = 2

times, intensities, failed_to_get, targets = data_process_a_group(mypath, sectorfile, sector, camera, ccd)

#%%
t = np.loadtxt("/Users/conta/UROP_Spring_2020/Sector20Cam1CCD2/Sector20Cam1CCD2_times_raw.txt")
intensities = np.loadtxt("/Users/conta/UROP_Spring_2020/Sector20Cam1CCD2/Sector20Cam1CCD2_intensities_raw.txt")
targets = np.loadtxt("/Users/conta/UROP_Spring_2020/Sector20Cam1CCD2/Sector20Cam1CCD2_targets.txt")

#%%

def interrupted_start_in_middle(position, yourpath, sectorfile, sector, camera, ccd):
    """ for cases where running the main list got fucked up somehow but you know
    where in the list you need to pick up from"""
    """you will need:
        your path into the main folder you're working in
        the file for your sector from TESS (full path)
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
        #os.makedirs(path)
        #print ("Successfully created the directory %s" % path) 
        
    # get just the list of targets for the specified sector, camera, ccd --------
        target_list_raw = lc_by_camera_ccd(sectorfile, camera, ccd)
        print("there are ", len(target_list_raw), "total targets")
        target_list = target_list_raw[position:]
        print("picking up at index", position)
    # get the light curve for each target on the list, and save into a text file
        confirmation, failed_to_get = lc_from_target_list(yourpath, target_list, fname_time, fname_int, fname_targets)
        print(confirmation)
        print("failed to get", len(failed_to_get), "targets")
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
        targets = "none"
        
    return times, intensities, failed_to_get, targets
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