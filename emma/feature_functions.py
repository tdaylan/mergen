# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 00:18:52 2020

@author: Lindsey Gordon 

Functions used across files. Last updated May 31th 2020.
"""

#Imports ---------------------------------------
import numpy as np
import numpy.ma as ma 
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)

import scipy.signal as signal
from scipy.stats import moment
from scipy import stats
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
rcParams["lines.markersize"] = 2
from scipy.signal import argrelextrema

import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import fnmatch

from datetime import datetime
import os
import shutil
from scipy.stats import moment, sigmaclip

import astropy
from astropy.io import fits
import scipy.signal as signal
from astropy.stats import SigmaClip
from astropy.utils import exceptions

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

import astroquery
from astroquery.simbad import Simbad
from astroquery.mast import Catalogs
from astroquery.mast import Observations

import shapely
from shapely import geometry
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


#Testing that this file imported correctly ------


def test(num):
    print(num * 4)
    
    
 
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

#Pulling data from files and processing it ---------
    
def print_header(index):
    fitspath = '/Users/conta/UROP_Spring_2020/tessdata_lc_sector20_1000/'
    fnames_all = os.listdir(fitspath)
    fnames = fnmatch.filter(fnames_all, '*fits*')
    
    #print(fnames[index])
    f = fits.open(fitspath + fnames[index])
    hdr = f[0].header
    #print(hdr)
    return hdr

def get_data_from_fits():
    """ imports data from fits files. based on emma's code"""    

    fitspath = '/Users/conta/UROP_Spring_2020/tessdata_lc_sector20_1000/'
    fnames_all = os.listdir(fitspath)
    fnames = fnmatch.filter(fnames_all, '*fits*')
    
    interp_tol = 20. / (24*60) # >> interpolate small gaps (less than 20 minutes)
    
    intensity = []
    targets = []
    #coordinates = []
    for file in fnames:
        # -- open file -------------------------------------------------------------
        f = fits.open(fitspath + file)
    
        # >> get data
        time = f[1].data['TIME']
        i = f[1].data['PDCSAP_FLUX']
        tic = f[1].header["OBJECT"]
        targets.append(tic)
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
        intensity.append(i_interp)
    
    # -- remove orbit nan gap ------------------------------------------------------
    intensity = np.array(intensity)
    # nan_inds = np.nonzero(np.prod(np.isnan(intensity)==False), axis = 0))
    nan_inds = np.nonzero(np.prod(np.isnan(intensity)==False, axis = 0) == False)
    time = np.delete(time, nan_inds)
    intensity = np.delete(intensity, nan_inds, 1) #each row of intensity is one interpolated light curve.
    return time, intensity, targets

#data process an entire group of TICs
    

def data_process_a_group(yourpath, sectorfile, sector, camera, ccd):
    """you will need:
        your path into the main folder you're working in - must end with /
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
    fname_times_interp = path + "/" + folder_name + "_times_processed.txt"
    fname_ints_processed = path + "/" + folder_name + "_intensities_processed.txt"
    fname_notes = path + "/" + folder_name + "_group_notes.txt"
    fname_features = path + "/"+ folder_name + "_features.txt"
    
    try:
        os.makedirs(path)
        print ("Successfully created the directory %s" % path) 
        with open(fname_time, 'a') as file_object:
            file_object.write("This file contains the raw time indices for this group")
        with open(fname_int, 'a') as file_object:
            file_object.write("This file contains the raw intensities for this group")
        with open(fname_targets, 'a') as file_object:
            file_object.write("This file contains the target TICs for this group")
        with open(fname_times_interp, 'a') as file_object:
            file_object.write("This file contains the processed time indices for this group")
        with open(fname_ints_processed, 'a') as file_object:
            file_object.write("This file contains the processed intensities for this group")
        with open(fname_notes, 'a') as file_object:
            file_object.write("This file contains group notes, including any TICs that could not be accessed.")
        with open(fname_features, 'a') as file_object:
            file_object.write("This file contains the feature vectors for each target in this group. ")
    # get just the list of targets for the specified sector, camera, ccd --------
        target_list = lc_by_camera_ccd(sectorfile, camera, ccd)
        print("there are ", len(target_list), "targets")
    # get the light curve for each target on the list, and save into a text file
        confirmation, failed_to_get = lc_from_target_list(yourpath, target_list, fname_time, fname_int, fname_targets, fname_notes)
        print(confirmation)
        print("failed to get", len(failed_to_get), "targets")
    # import the files you just created
        times = np.loadtxt(fname_time, skiprows=1)
        intensities = np.loadtxt(fname_int, skiprows=1)
        targets = np.loadtxt(fname_targets, skiprows=1)
    #check to be sure all have the same size, if not, report back an error
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
            print("There is a disagreement between the number of lines saved in intensities and targets, cannot run feature creation")
            failed_to_get = "something"
            features = "empty"
    
    except OSError: #if there is an error creating the folder
        print("There was an OS Error trying to create the folder. Checking to see if data is already saved there")
        #try to load in and process the data anyways
        with open(fname_times_interp, 'a') as file_object:
            file_object.write("This file contains the processed time indices for this group")
        with open(fname_ints_processed, 'a') as file_object:
            file_object.write("This file contains the processed intensities for this group")
        with open(fname_notes, 'a') as file_object:
            file_object.write("This file contains the notes for this group")
       
        try: 
            times = np.loadtxt(fname_time, skiprows=1)
            intensities = np.loadtxt(fname_int, skiprows=1)
            targets = np.loadtxt(fname_targets, skiprows=1)
            print("files loaded in")
        #check to be sure all have the same size, if not, report back an error
            if len(intensities) == len(targets):
                print("number of lines matches, running data processing")
                
        #interpolate and normalize/sigma clip
                interp_times, interp_intensities = interpolate_lc(times, intensities)
                #print("interpolation complete")
                normalized_intensities = normalize(interp_intensities)
                #print("normalization complete")
        #save these into their own files, and report these arrays back
                np.savetxt(fname_times_interp, interp_times)
                #times = np.loadtxt(fname_times_interp, skiprows=1)
                np.savetxt(fname_ints_processed, normalized_intensities)
                #intensities = np.loadtxt(fname_ints_processed, skiprows=1)
                times = interp_times
                intensities = normalized_intensities
                print("You can now access time arrays, processed intensities, targets, and an array of TICs you could not get")
            
                features = create_list_featvec(times, intensities)
                print("feature vectors created")
                np.savetxt(fname_features, features)
                print("Feature vector creation complete")
                failed_to_get = np.loadtxt(fname_notes, skiprows=1)
                
            else: #if there is an error with the number of lines in times vs ints vs targets
                print("There is still an error loading things in")
                failed_to_get = "something"
                features = "empty"
        except: 
            print ("This directory already exists, or there is some other OS error")
            failed_to_get = "Error"
            features = "Error"
        
    return times, intensities, failed_to_get, targets, path, features

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

def lc_from_target_list(yourpath, targetList, fname_time, fname_int, fname_targets, fname_notes):
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
            with open(fname_notes, 'a') as file_object:
                file_object.write("\n")
                file_object.write(str(int(target)))
            continue
    # add data to the files
        with open(fname_targets, 'a') as file_object:
            file_object.write("\n")
            file_object.write(str(target))
        with open(fname_int, 'a') as file_object:   
            file_object.write("\n")
            np.savetxt(file_object, i1, delimiter = ',', newline = ' ')
            
        if n == 1:
            with open(fname_time, 'a') as file_object:
                file_object.write("\n")
                np.savetxt(file_object, time1, delimiter = ',', newline = ' ')
        
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
        #simply could not get it to accept it despite following the API guidelines
        filter_products = Observations.filter_products(data_products_by_obs, dataproduct_type = 'timeseries')
        manifest = Observations.download_products(filter_products)
            
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


#normalizing each light curve
def normalize(intensity):
    """normalizes the intensity from the median value 
    by dividing out. then sigmaclips using astropy
    returns a masked array"""
    sigclip = SigmaClip(sigma=4, maxiters=None, cenfunc='median')
    intense = []
    for i in np.arange(len(intensity)):
        intensity[i] = intensity[i] / np.median(intensity[i])
        inte = sigclip(intensity[i], masked=True, copy = False)
        intense.append(inte)
    intensity = np.ma.asarray(intense)
    print("Normalization and sigma clipping complete")
    return intensity
    
def interpolate_lc(time, intensities):
    """interpolates all light curves in an array of all light curves"""
    
    interp_tol = 20. / (24*60) # >> interpolate small gaps (less than 20 minutes)
    
    interpolated_intensities = []
    for p in range(len(intensities)):

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
    len(nan_inds)
    interpolated_intensities = np.delete(interpolated_intensities, nan_inds, 1) #each row of intensity is one interpolated light curve.
    
    time_corrected = np.delete(time, nan_inds)
    print(len(time_corrected), len(interpolated_intensities[0]))
    #interpolated_times = np.array(interpolated_times)
    return time_corrected, interpolated_intensities
#producing the feature vector list -----------------------------

    
def create_list_featvec(time_axis, datasets):
    """input: all of the datasets being turned into feature vectors (ie, intensity)
        num_features is the number of features currently being worked on. 
    you just changed to a range, if it hates you it's because of that
    returns a list of featurevectors, one for each input . """
    num_data = len(datasets) #how many datasets
    x = time_axis #creates the x axis
    feature_list = np.zeros((num_data, 16)) #MANUALLY UPDATE WHEN CHANGING NUM FEATURES
    print("creating feature vectors about to begin")
    for n in range(num_data):
        feature_list[n] = featvec(x, datasets[n])
        if n % 50 == 0: print(str(n) + " completed")
    return feature_list

def featvec(x_axis, sampledata): 
    """calculates the feature vector of the single set of data (ie, intensity[0])
    currently returns 16: 
        0 - Average
        1 - Variance
        2 - Skewness
        3 - Kurtosis
        
        4 - ln variance
        5 - ln skewness
        6 - ln kurtosis
        
        (over 0.1 to 10 days)
        7 - maximum power
        8 - ln maximum power
        9 - period of maximum power
        
        10 - slope
        11 - ln slope
        
        (integration of periodogram over time frame)
        12 - P0 - 0.1-1 days
        13 - P1 - 1-3 days
        14 - P2 - 3-10 days
        
        (over 0-0.1 days, for moving objects)
        15 - Period of max power
        
        
        ***if you update the number of features, 
        you have to update the number of features in create_list_featvec!!!!"""
    featvec = moments(sampledata) #produces moments and log moments
    
    f = np.linspace(0.6, 62.8, 5000)  #period range converted to frequencies
    periods = np.linspace(0.1, 10, 5000)#0.1 to 10 day period
    pg = signal.lombscargle(x_axis, sampledata, f, normalize = True)
    rel_maxes = argrelextrema(pg, np.greater)
    
    powers = []
    indexes = []
    for n in range(len(rel_maxes[0])):
        index = rel_maxes[0][n]
        indexes.append(index)
        power_level_at_rel_max = pg[index]
        powers.append(power_level_at_rel_max)
    
    max_power = np.max(powers)
    index_of_max_power = np.argmax(powers)
    index_of_f_max = rel_maxes[0][index_of_max_power]
    f_max_power = f[index_of_f_max]
    period_max_power = 2*np.pi / f_max_power
    
    featvec.append(max_power)
    featvec.append(np.log(np.abs(max_power)))
    featvec.append(period_max_power)
    
    slope = stats.linregress(x_axis, sampledata)[0]
    featvec.append(slope)
    featvec.append(np.log(np.abs(slope)))
    
    #integrates the whole 0.1-10 day range
    integrating1 = np.trapz(pg[457:5000], periods[457:5000]) #0.1 days to 1 days
    integrating2 = np.trapz(pg[121:457], periods[121:457])#1-3 days
    integrating3 = np.trapz(pg[0:121], periods[0:121]) #3-10 days
    
    featvec.append(integrating1)
    featvec.append(integrating2)
    featvec.append(integrating3)
    
    #for 0.001 to 1 day periods
    f2 = np.linspace(62.8, 6283.2, 20)  #period range converted to frequencies
    p2 = np.linspace(0.001, 0.1, 20)#0.001 to 1 day periods
    pg2 = signal.lombscargle(x_axis, sampledata, f2, normalize = True)
    rel_maxes2 = argrelextrema(pg2, np.greater)
    powers2 = []
    indexes2 = []
    for n in range(len(rel_maxes2[0])):
        index2 = rel_maxes2[0][n]
        indexes2.append(index2)
        power_level_at_rel_max2 = pg2[index2]
        powers2.append(power_level_at_rel_max2)
    max_power2 = np.max(powers2)
    index_of_max_power2 = np.argmax(powers2)
    index_of_f_max2 = rel_maxes2[0][index_of_max_power2]
    f_max_power2 = f2[index_of_f_max2]
    period_max_power2 = 2*np.pi / f_max_power2
    featvec.append(period_max_power2)
    #print("done")
    return(featvec) 

def moments(dataset): 
    """calculates the 1st through 4th moment of a single row of data (ie, intensity[0])"""
    moments = []
    moments.append(np.mean(dataset)) #mean (don't use moment, always gives 0)
    moments.append(moment(dataset, moment = 2)) #variance
    moments.append(moment(dataset, moment = 3)) #skew
    moments.append(moment(dataset, moment = 4)) #kurtosis
    moments.append(np.log(np.abs(moment(dataset, moment = 2)))) #ln variance
    moments.append(np.log(np.abs(moment(dataset, moment = 3)))) #ln skew
    moments.append(np.log(np.abs(moment(dataset, moment = 4)))) #ln kurtosis
    return(moments)

#Plotting functions ------------------------------------------------------
def post_process_plotting(time, intensity, features_all, features_using, targets, path):
    """plotting all the things"""
    features_plotting_2D(features_all, features_using, path, "none")
    features_plotting_2D(features_all, features_using, path, "kmeans")
    features_plotting_2D(features_all, features_using, path, "dbscan")
    
    plot_lof(time, intensity, targets, features_all, 10, path)

    features_insets2D(time, intensity, features_all, targets, path)


def features_plotting_2D(feature_vectors, cluster_columns, path, clustering,
                         time, intensity, targets, folder_suffix='',
                         feature_engineering=True, eps=0.5, min_samples=10,
                         metric='euclidean', algorithm='auto', leaf_size=30,
                         p=2,
                         momentum_dump_csv='./Table_of_momentum_dumps.csv'):
    """plotting (n 2) features against each other
    feature_vectors is the list of ALL feature_vectors
    cluster_columns is the vectors that you want to use to do the clustering based on
        this can be the same as feature_vectors
    date must be a string in the format of the folder you are saving into ie "4-13"
    clustering must equal 'dbscan', 'kmeans', or 'empty'
    """
    import pdb # >> [etc 060620]
    import modellibrary as ml
    import plots_lib as pl
    cluster = "empty"
    folder_label = "blank"
    if clustering == 'dbscan':
        # !! TODO parameter optimization (eps, min_samples)
        # cluster_columns = ml.standardize(cluster_columns, ax=0)
        db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
                    algorithm=algorithm, leaf_size=leaf_size,
                    p=p).fit(cluster_columns) #eps is NOT epochs
        classes_dbscan = db.labels_
        numclasses = str(len(set(classes_dbscan)))
        cluster = 'dbscan'
        folder_label = "dbscan-colored" + folder_suffix
        # pdb.set_trace()

    elif clustering == 'kmeans': 
        Kmean = KMeans(n_clusters=4, max_iter=700, n_init = 20)
        x = Kmean.fit(cluster_columns)
        classes_kmeans = x.labels_
        cluster = 'kmeans'
        folder_label = "kmeans-colored"
    else: 
        print("no clustering chosen")
        cluster = 'none'
        folder_label = "2DFeatures"
    #makes folder and saves to it    
    folder_path = path + "/" + folder_label
    try:
        os.makedirs(folder_path)
    except OSError:
        print ("Creation of the directory %s failed" % folder_path)
        print("New folder created will have -new at the end. Please rename.")
        os.makedirs(folder_path + "-new")
    else:
        print ("Successfully created the directory %s" % folder_path) 
 
    if clustering == 'dbscan':
        pl.plot_classification(time, intensity, targets, db.labels_,
                               path+folder_label+'/', prefix='dbscan',
                               momentum_dump_csv=momentum_dump_csv)
        pl.plot_pca(cluster_columns, db.labels_,
                    output_dir=path+folder_label+'/')
    elif clustering == 'kmeans':
        pl.plot_classification(time, intensity, targets, x.labels_,
                               path+folder_label+'/', prefix='kmeans')
    # >> [etc 060620]
    if feature_engineering:
        graph_labels = ["Average", "Variance", "Skewness", "Kurtosis", "Log Variance",
                        "Log Skewness", "Log Kurtosis", "Maximum Power", "Log Maximum Power", 
                        "Period of Maximum Power (0.1 to 10 days)","Slope" , "Log Slope",
                        "P0", "P1", "P2", "Period of Maximum Power (0.001 to 0.1 days)"]
        fname_labels = ["Avg", "Var", "Skew", "Kurt", "LogVar", "LogSkew", "LogKurt",
                        "MaxPower", "LogMaxPower", "Period0_1to10", "Slope", "LogSlope",
                        "P0", "P1", "P2", "Period0to0_1"]
        num_features = 16
    else:
        # >> shape(feature_vectors) = [num_samples, num_features]
        num_features = np.shape(feature_vectors)[1]
        graph_labels = []
        fname_labels = []
        for n in range(num_features):
            graph_labels.append('\u03C6' + str(n))
            fname_labels.append('phi'+str(n))
    for n in range(num_features):
        feat1 = feature_vectors[:,n]
        graph_label1 = graph_labels[n]
        fname_label1 = fname_labels[n]
        for m in range(num_features):
            if m == n:
                continue
            graph_label2 = graph_labels[m]
            fname_label2 = fname_labels[m]                
            feat2 = feature_vectors[:,m]
            
            if cluster == 'dbscan':
                plt.figure() # >> [etc 060520]
                plt.clf()
                for p in range(len(feature_vectors)):
                    if classes_dbscan[p] == 0:
                        color = "red"
                    elif classes_dbscan[p] == -1:
                        color = "black"
                    elif classes_dbscan[p] == 1:
                        color = "blue"
                    elif classes_dbscan[p] == 2:
                        color = "green"
                    elif classes_dbscan[p] == 3:
                        color = "purple"
                    plt.scatter(feat1[p], feat2[p], c = color, s = 2)
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig((folder_path + "/" + fname_label1 + "-vs-" + fname_label2 + "-dbscan.png"))

                # plt.show()
            elif cluster == 'kmeans':
                for p in range(len(feature_vectors)):
                    if classes_kmeans[p] == 0:
                        color = "red"
                    elif classes_kmeans[p] == 1:
                        color = "blue"
                    elif classes_kmeans[p] == 2:
                        color = "green"
                    elif classes_kmeans[p] == 3:
                        color = "purple"
                    plt.scatter(feat1[p], feat2[p], c = color, s=2)
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig(folder_path + "/" + fname_label1 + "-vs-" + fname_label2 + "-kmeans.png")
                # plt.show()
            elif cluster == 'none':
                plt.scatter(feat1, feat2, s = 2, color = 'black')
                #plt.autoscale(enable=True, axis='both', tight=True)
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig(folder_path + "/" + fname_label1 + "-vs-" + fname_label2 + ".png")
                # plt.show()
                
    if cluster == 'dbscan':
        return db.labels_
    if cluster == 'kmeans':
        return x.labels
        
                
def plot_lof(time, intensity, targets, features, path):
    """plots the 20 most and least interesting light curves based on LOF
    takes input: time, intensity, targets, featurelist, n number of curves you want, date as a string """
    fname_lof = path + "/LOF_features.txt"
    n = 10
    from sklearn.neighbors import LocalOutlierFactor

    clf = LocalOutlierFactor(n_neighbors=50)
    
    fit_predictor = clf.fit_predict(features)
    negative_factor = clf.negative_outlier_factor_
    
    lof = -1 * negative_factor
    ranked = np.argsort(lof)
    largest_indices = ranked[::-1][:10]
    smallest_indices = ranked[:10]

    with open(fname_lof, 'a') as file_object:
        file_object.write("Ten largest LOF's features: \n")
        np.savetxt(file_object, features[largest_indices])
        file_object.write("\n Ten smallest LOF's features: \n")
        np.savetxt(file_object, features[smallest_indices])
    #plot just the largest indices
    #rows, columns
    fig, axs = plt.subplots(10, 1, sharex = True, figsize = (8,30), constrained_layout=False)
    fig.subplots_adjust(hspace=0)
    
    for k in range(10):
        ind = largest_indices[k]
        axs[k].plot(time, intensity[ind], '.k', label=targets[ind] + ", " + str(np.round(lof[ind], 2)))
        axs[k].legend(loc="upper left")
        axs[k].set_ylabel("relative flux")
        axs[-1].set_xlabel("BJD [-2457000]")
    fig.suptitle(str(10) + ' largest LOF targets', fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.96)
    fig.savefig(path + "/largest-lof.pdf")

    #plot the smallest indices
    fig1, axs1 = plt.subplots(10, 1, sharex = True, figsize = (8,30), constrained_layout=False)
    fig1.subplots_adjust(hspace=0)
    
    for m in range(10):
        ind = smallest_indices[m]
        axs1[m].plot(time, intensity[ind], '.k', label=targets[ind] + ", " + str(np.round(lof[ind], 2)))
        axs1[m].legend(loc="upper left")
        axs1[m].set_ylabel("relative flux")
        axs1[-1].set_xlabel("BJD [-2457000]")
    fig1.suptitle(str(10) + ' smallest LOF targets', fontsize=16)
    fig1.tight_layout()
    fig1.subplots_adjust(top=0.96)
    fig1.savefig(path +  "/smallest-lof.pdf")
                
def astroquery_pull_data(target):
    """pulls data on object from astroquery
    target needs to be a string"""
    try: 
        catalog_data = Catalogs.query_object(target, radius=0.02, catalog="TIC")
        #https://arxiv.org/pdf/1905.10694.pdf
        T_eff = np.round(catalog_data[0]["Teff"], 0)
        obj_type = catalog_data[0]["objType"]
        gaia_mag = np.round(catalog_data[0]["GAIAmag"], 2)
        radius = np.round(catalog_data[0]["rad"], 2)
        mass = np.round(catalog_data[0]["mass"], 2)
        distance = np.round(catalog_data[0]["d"], 1)
        title = "\nT_eff:" + str(T_eff) + "," + str(obj_type) + ", G: " + str(gaia_mag) + "\n Dist: " + str(distance) + ", R:" + str(radius) + " M:" + str(mass)
    except (ConnectionError, OSError, TimeoutError):
        print("there was a connection error!")
        title = "connection error, no data"
    return title

#PLOTTING INSET PLOTS (x/y max/min points per feature)
    
def inset_labelling(axis_name, time, intensity, targets, index, title):
    """formatting the labels for the inset plots"""
    axis_name.set_xlim(time[0], time[-1])
    axis_name.set_ylim(intensity[index].min(), intensity[index].max())
    #axis_name.set_xlabel("BJD [2457000]")
    #axis_name.set_ylabel("relative flux")
    axis_name.set_title(targets[index] + title, fontsize=6)

def features_insets2D(time, intensity, feature_vectors, targets, folder):
    """plotting (n 2) features against each other w/ 4 extremes inset plotted
    feature_vectors is the list of ALL feature_vectors
    date must be a string in the format of the folder you are saving into ie "4-13"
    """   
    # path = "/Users/conta/UROP_Spring_2020/" + folder + "/2DFeatures-insets"
    path = folder + "/2DFeatures-insets" # >> [etc 060520]
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
 
            plt.savefig(path + "/" + fname_label1 + "-vs-" + fname_label2 + ".pdf")
            # plt.show()
            
            
def plot_all_insets(feature_vectors,targets, intensity, time, feat1, feat2, graph_label1, graph_label2):
    """plots the x/y min/max points' associated light curve on the plot"""
    fig, ax1 = plt.subplots()
    ax1.scatter(feature_vectors[:,feat1], feature_vectors[:,feat2], c = "black",
                s=2)
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
    inset_positions = np.zeros((9,2))
    
    indexes_unique, targets_to_plot, tuples_plotting, titles = get_extrema(feature_vectors, targets, feat1, feat2)
    #print(indexes_unique)
    for n in range(min(len(indexes_unique), 9)):# >> [etc 060520]
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
        axis_name.scatter(time, intensity[index], c='black', s = 0.01, rasterized=True)
            
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
    indexes.append(index_feat1[-3]) #third largest
    indexes.append(index_feat1[0]) #smallest
    indexes.append(index_feat1[1]) #second smallest
    indexes.append(index_feat1[2]) #third smallest

    indexes.append(index_feat2[-1]) #largest
    indexes.append(index_feat2[-2]) #second largest
    indexes.append(index_feat2[-3]) #third largest
    indexes.append(index_feat2[0]) #smallest
    indexes.append(index_feat2[1]) #second smallest
    indexes.append(index_feat2[2]) #third smallest

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
    inset_positions is a list  from a diff. function that holds the pos of insets
    last updated 5/29/20"""
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

#confusion matrix functions ------------------------------------------------

def matrix_accuracy(c_matrix):
    """calculate the accuracy of the matrix"""
    num_labels = len(c_matrix)
    total = np.sum(c_matrix, axis=None)
    diagonal = 0
    n = 0
    while n < num_labels:
        diagonal = diagonal + c_matrix[n][n]
        n = n+1
    accuracy = diagonal/total
    return accuracy

def IsItIdentifyingNoise(predicted_classes):
    """check if it identified any as a noise category """
    noise = "False"
    for n in range(len(predicted_classes)):
        if predicted_classes[n] == -1:
            noise = "True"
    return noise

def matrix_precision(matrix):
    """calculates the precision of each class"""
    precisions = []
    for n in range(len(matrix)):
        column = matrix[:,n]
        column_total = np.sum(column)
        
        correct = matrix[n][n]
        if column_total == 0:
            prec = 0
        else:
            prec = correct/column_total
        
        precisions.append(prec)
    
    return np.asarray(precisions)

def matrix_recall(matrix):
    """calculates the recall of each class"""
    recalls = []
    for n in range(len(matrix)):
        row = matrix[n]
        row_total = np.sum(row)
        
        correct = matrix[n][n]
        if row_total == 0:
            rec = 0
        else:
            rec = correct/row_total
        
        recalls.append(rec)
    
    return np.asarray(recalls)

#Other functions (old/rarely used) ---------------------
def get_pdcsap_and_sap(yourpath, target):
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
        #simply could not get it to accept it despite following the API guidelines
        filter_products = Observations.filter_products(data_products_by_obs, dataproduct_type = 'timeseries')
        manifest = Observations.download_products(filter_products)
            
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
                i2 = f[1].data['SAP_FLUX']
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
    
    return time1, i1, i2 

def gaussian(datapoints, a, b, c):
    """Produces a gaussian function"""
    x = np.linspace(0, xmax, datapoints)
    return  a * np.exp(-(x-b)**2 / 2*c**2) + np.random.normal(size=(datapoints))

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
    with open(fname_int, 'a') as file_object:
        file_object.write("This file contains the raw time indices for this group")
    fname_targets = path + "/" + folder_name + "_targets.txt"
    fname_times_interp = path + "/" + folder_name + "_interp_times.txt"
    fname_ints_processed = path + "/" + folder_name + "_ints_processed.txt"
    fname_notes = path + "/" + folder_name + "_group_notes.txt"
    fname_features = path + "/"+ folder_name + "_features.txt"
    try:
        #os.makedirs(path)
        #print ("Successfully created the directory %s" % path) 
        
    # get just the list of targets for the specified sector, camera, ccd --------
        target_list_raw = lc_by_camera_ccd(sectorfile, camera, ccd)
        print("there are ", len(target_list_raw), "total targets")
        target_list = target_list_raw[position:]
        print("picking up at index", position)
    # get the light curve for each target on the list, and save into a text file
        confirmation, failed_to_get = lc_from_target_list(yourpath, target_list, fname_time, fname_int, fname_targets, fname_notes)
        print(confirmation)
        print("failed to get", len(failed_to_get), "targets")
    # import the files you just created
        times = np.loadtxt(fname_time)
        intensities = np.loadtxt(fname_int)
        targets = np.loadtxt(fname_targets)
        
    #turn all targets into integer strings
        targets_strings = []
        for n in range(len(targets)):
            targets_strings.append(("TIC " + str(int(targets[n]))))
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
        
    #then produce and save the feature vectors into a file
            features = create_list_featvec(times[0], intensities)
            np.savetxt(fname_features, features)
            
        else: #if there is an error with the number of lines in times vs ints vs targets
            print("There is a disagreement between the number of lines saved in each text file")
            times = "Does not exist"
            intensities = "Does not exist"
            failed_to_get = "all"
            features = "empty"
    
    except OSError: #if there is an error creating the folder
        print ("This directory already exists, or there is some other OS error")
        times = "Does not exist" 
        intensities = "does not exist"
        failed_to_get = "all"
        targets = "none"
        features = "empty"
        
    return times, intensities, failed_to_get, targets, path, features

# For running on the current data/ feature vectors (as of 5/4/20)
def get_from_files_1200():
    """pulls time, intensity, and feature vectors from text files that they are saved in
    currently pulling the 5/4 version of it"""
    intensity = np.loadtxt("/Users/conta/UROP_Spring_2020/intensities.txt", delimiter = " ")

    time = np.loadtxt("/Users/conta/UROP_Spring_2020/timeindex.txt", delimiter = " ")
    
    targets = np.loadtxt("/Users/conta/UROP_Spring_2020/targets.txt", dtype = str, delimiter = ",")
    
    lc_feat = np.loadtxt("/Users/conta/UROP_Spring_2020/featvecs-5-4-20.txt", delimiter = " ")
    return time, intensity, targets, lc_feat