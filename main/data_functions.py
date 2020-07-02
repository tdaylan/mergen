# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 21:54:56 2020

Data access, data processing, feature vector creation functions.

@author: Lindsey Gordon @lcgordon and Emma Chickles (@??)

Updated: June 26 2020

Data access
* test_data()
* load_group_from_txt()
* data_access_by_group_fits()
* follow_up_on_missed_targets_fits()
* interp_norm_sigmaclip_features()
* lc_by_camera_ccd()
* lc_from_target_list_fits()
* get_lc_file_and_data()

Data processing
* normalize()       : median normalization
* interpolate_all() : sigma clip and interpolate flux array
* interpolate_lc()  : sigma clip and interpolate one light curve
* nan_mask()        : apply NaN mask to flux array

Engineered features
* create_save_featvec
* featvec


"""

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

import plotting_functions as pf

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
from astroquery import exceptions
from astroquery.exceptions import RemoteServiceError

#import model as ml # >> autoencoder functions

# import batman # >> I don't have this library yet [etc 063020]
import numba


def test_data():
    """make sure the module loads in"""
    print("Data functions loaded in.")
    
def load_group_from_txt(sector, camera, ccd, path):
    """loads in a given group's data provided you have it saved in TEXT metafiles already
    path needs to be a string, ending with a forward slash
    camera, ccd, secotr all should be integers
    """
    folder = "Sector"+str(sector)+"Cam"+str(camera)+"CCD"+str(ccd)
    time_path = path + folder + "/" + folder + "_times_processed.txt"
    intensities_path = path + folder + "/" + folder + "_intensities_processed.txt"
    features_path = path + folder + "/" + folder + "_features.txt"
    targets_path = path + folder + "/" + folder + "_targets.txt"
    notes_path = path + folder + "/" + folder + "_group_notes.txt"
    
    t = np.loadtxt(time_path)
    intensities = np.loadtxt(intensities_path)
    try: 
        targets = np.loadtxt(targets_path)
    except ValueError:
        targets = np.loadtxt(targets_path, skiprows=1)
        
    targets.astype(int)
    features = np.loadtxt(features_path, skiprows=1)
    notes = np.loadtxt(notes_path, skiprows=1)
    
    return t, intensities, targets, features, notes 


def data_access_sector_by_bulk(yourpath, sectorfile, sector,
                               bulk_download_dir):
    '''Get interpolated flux array for each group, if you already have all the
    _lc.fits files downloaded in bulk_download_dir.
    Parameters:
        * yourpath : directory to store outputs in
        * sectorfile : txt file containing the camera and ccd of each light
          curve in the sector, from
          https://tess.mit.edu/observations/target-lists/
        * sector : int
        * bulk_download_dir : directory containing all the _lc.fits files,
          can be downloaded from 
          http://archive.stsci.edu/tess/bulk_downloads.html
    '''
    
    for cam in [1,2,3,4]:
        for ccd in [1,2,3,4]:
            data_access_by_group_fits(yourpath, sectorfile, sector, cam,
                                      ccd, bulk_download=True,
                                      bulk_download_dir=bulk_download_dir)

#data process an entire group of TICs
def data_access_by_group_fits(yourpath, sectorfile, sector, camera, ccd,
                              bulk_download=False, bulk_download_dir='./'):
    """you will need:
        your path into the main folder you're working in - must end with /
        the file for your sector from TESS (full path)
        sector number (as int)
        camera number you want (as int/float)
        ccd number you want (as int/float)
        this ONLY returns the target list and folderpath for the group
        
        Saves a .fits file with primaryHDU=f[0]=time,
        f[1]=raw intensity array, f[2] = interpolated intensity array, f[3]=TICIDs
        """
    # produce the folder to save everything into and set up file names
    folder_name = "Sector" + str(sector) + "Cam" + str(camera) + "CCD" + str(ccd)
    path = yourpath + folder_name
    # fname_time_intensities_raw = path + "/" + folder_name + "_raw_lightcurves.fits"
    fname_time_intensities_raw = path + "/" + folder_name + "_lightcurves.fits"
    fname_targets = path + "/" + folder_name + "_targets.txt"
    fname_notes = path + "/" + folder_name + "_group_notes.txt"
    
    try:
        os.makedirs(path)
        print ("Successfully created the directory %s" % path) 
        with open(fname_targets, 'a') as file_object:
            file_object.write("This file contains the target TICs for this group. Fits light curves are 1-indexed, so first target is all zeroes \n 00000000")
        with open(fname_notes, 'a') as file_object:
            file_object.write("This file contains group notes, including any TICs that could not be accessed.\n")
        # get just the list of targets for the specified sector, camera, ccd --------
        target_list = lc_by_camera_ccd(sectorfile, camera, ccd)
        print("there are ", len(target_list), "targets")
        
        # >> get the light curve for each target on the list, and save into a
        # >> fits file
        if bulk_download:
            confirmation = lc_from_bulk_download(bulk_download_dir,
                                                 target_list,
                                                 fname_time_intensities_raw,
                                                 fname_targets,
                                                 fname_notes, path)
        else: # >> download each light curve
            confirmation = lc_from_target_list_fits(yourpath, target_list,
                                                    fname_time_intensities_raw,
                                                    fname_targets, fname_notes,
                                                    sector, path=path)
        print(confirmation)
        #print("failed to get", len(failed_to_get), "targets")
        targets = np.loadtxt(fname_targets, skiprows=1)
        # print("found data for ", len(targets), " targets")
    #check to be sure all have the same size, if not, report back an error
        
    except OSError: #if there is an error creating the folder
        print("There was an OS Error trying to create the folder. Checking to see if data is already saved there")
        targets = "empty"
        
    return targets, path

def follow_up_on_missed_targets_fits(yourpath, sector, camera, ccd):
    """ function to follow up on rejected TIC ids"""
    folder_name = "Sector" + str(sector) + "Cam" + str(camera) + "CCD" + str(ccd)
    path = yourpath + folder_name
    fname_time_intensities_raw = path + "/" + folder_name + "_raw_lightcurves.fits"
    fname_targets = path + "/" + folder_name + "_targets.txt"
    fname_notes = path + "/" + folder_name + "_group_notes.txt"
    fname_notes_followed_up = path + "/" + folder_name + "_targets_still_no_data.txt"
    
    
    retry_targets = np.loadtxt(fname_notes, skiprows=1)
    retry_targets = np.column_stack((retry_targets, retry_targets)) #loak this is a crude way to get around some indexing issues and we're working with it
    
    with open(fname_notes_followed_up, 'a') as file_object:
        file_object.write("Data could not be found for the following TICs after two attempts")
    
    
    for n in range(len(retry_targets)):
        target = retry_targets[n][0] #get that target number
        time1, i1 = get_lc_file_and_data(yourpath, target, sector)
        if type(i1) == np.ndarray: #IF THE DATA IS FORMATTED LKE DATA
            fits.append(fname_time_intensities_raw, i1, header=hdr)
            with open(fname_targets, 'a') as file_object:
                file_object.write("\n")
                file_object.write(str(int(target)))
        else: #IF THE DATA IS NOT DATA
            print("File failed to return targets")
            with open(fname_notes_followed_up, 'a') as file_object:
                file_object.write("\n")
                file_object.write(str(int(target)))
    
    
    targets = np.loadtxt(fname_targets, skiprows=1)
    print("after following up, found data for ", len(targets), " targets")
    
    return targets, path


def interp_norm_sigmaclip_features(yourpath, times, intensities, targets, sector, camera, ccd):
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
        interp_times, interp_intensities = interpolate_lc(intensities, times, flux_err=False, interp_tol=20./(24*60),num_sigma=5, orbit_gap_len = 3, DEBUG=False,spline_interpolate=True)
        normalized_intensities = normalize(interp_intensities)
    #save these into their own files, and report these arrays back
        with open(fname_times_interp, 'a') as file_object:
            np.savetxt(fname_times_interp, interp_times)
        with open(fname_ints_processed, 'a') as file_object:
            np.savetxt(fname_ints_processed, normalized_intensities)
        times = np.loadtxt(fname_times_interp)
        intensities = np.loadtxt(fname_ints_processed)
        print("You can now access time arrays, processed intensities, targets, and an array of TICs you could not get")
        
        features = create_list_featvec(times, intensities)
        with open(fname_features, 'a') as file_object:
            np.savetxt(fname_features, features)
        print("Feature vector creation complete")
            
    else: #if there is an error with the number of lines in times vs ints vs targets
        print("There is a disagreement between the number of lines saved in intensities and targets, cannot process data")
        
    return times, intensities, features


######
    

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


def lc_from_target_list_fits(yourpath, targetList, fname_time_intensities_raw,
                             fname_targets, fname_notes, sector, path='./'):
    """ runs getting the files and data for all targets on the list
    then appends the time & intensity arrays and the TIC number into text files
    that can later be accessed
    modified [lcg 062620]
    """
    intensity = []
    i_interpolated = []
    ticids = []
    for n in range(len(targetList)): #for each item on the list
        
        if n == 0: #for the first target only do you need to get the time index
            target = targetList[n][0] #get that target number
            time1, i1, tic = get_lc_file_and_data(yourpath, target, sector) #grab that data
            
            if type(i1) == np.ndarray: #if the data IS data
                # i_interp = interpolate_lc(i1, time1)
                # i_interpolated.append(i_interp)
                intensity.append(i1)
                # hdr = fits.Header() #make-a the header
                # hdu = fits.PrimaryHDU(time1, header=hdr)
                # hdu.writeto(fname_time_intensities_raw) #make the fits file
                ticids.append(tic)
            else: #if the data is NOT a data
                print("First target failed, no time index was saved")
                with open(fname_notes, 'a') as file_object:
                    file_object.write("\n")
                    file_object.write(str(int(target)))
        else: #only saving the light curve into the fits file because it's all you need
            target = targetList[n][0] #get that target number
            time1, i1, tic = get_lc_file_and_data(yourpath, target, sector)
            if type(i1) == np.ndarray:
                # i_interp = interpolate_lc(i1, time1)
                # i_interpolated.append(i_interp)
                intensity.append(i1)    
                ticids.append(tic)
            else: #IF THE DATA IS NOT DATA
                print("File failed to return targets")
                with open(fname_notes, 'a') as file_object:
                    file_object.write("\n")
                    file_object.write(str(int(target)))
        
        #if n %10 == 0: #every 50, print how many have been done
        print(str(n), "completed")
            
    intensity = np.array(intensity)
    ticids = np.array(ticids)
    
    # >> interpolate and nan mask
    print('Interpolating and applying nan mask')
    intensity_interp, time = interpolate_all(intensity, time1)
    
    print('Saving to fits file')
    # i_interp = np.array(i_interp)
    hdr = fits.Header() # >> make the header
    hdu = fits.PrimaryHDU(time, header=hdr)
    hdu.writeto(fname_time_intensities_raw)
    fits.append(fname_time_intensities_raw, intensity_interp)
    fits.append(fname_time_intensities_raw, ticids)
    # with open(fname_time_intensities_raw, 'rb+') as f:
    #     # >> don't want to save 2x data we need to, so only save interpolated
    #     # fits.append(fname_time_intensities_raw, intensity)
    #     fits.append(fname_time_intensities_raw, i_interp)
    #     fits.append(fname_time_intensities_raw, ticids)
    confirmation = "lc_from_target_list has finished running"
    return confirmation

def get_lc_file_and_data(yourpath, target, sector):
    """ goes in, grabs the data for the target, gets the time index, intensity,and TIC
    if connection error w/ MAST, skips it
    modified [lcg 06262020] - now pulls TICID as well, in case accidentally gets the wrong lc"""
    fitspath = yourpath + 'mastDownload/TESS/' # >> download directory
    targ = "TIC " + str(int(target))
    print(targ)
    try:
        #find and download data products for your target objectname='TIC '+str(int(target)),
        obs_table = Observations.query_criteria(obs_collection='TESS',
                                        dataproduct_type='timeseries',
                                        target_name=str(int(target)),
                                        sequence_number=sector,
                                        objectname=targ)
        data_products_by_obs = Observations.get_product_list(obs_table[0:4])
            
        filter_products = Observations.filter_products(data_products_by_obs,
                                                       description = 'Light curves')
        manifest = Observations.download_products(filter_products, extension='fits')
                
        #get all the paths to lc.fits files
        filepaths = []
        for root, dirs, files in os.walk(fitspath):
            for name in files:
                print(name)
                if name.endswith(("lc.fits")):
                    filepaths.append(root + "/" + name)
        #print(len(filepaths))
        
        if len(filepaths) == 0: #if no lc.fits were downloaded, move on
            print(targ, "no light curve available")
            time1 = 0
            i1 = 0
            ticid = 0
        else: #if there are lc.fits files, open them and get the goods
                #get the goods and then close it #!!!! GET THE TIC FROM THE F I L E 
            f = fits.open(filepaths[0], memmap=False)
            time1 = f[1].data['TIME']
            i1 = f[1].data['PDCSAP_FLUX']
            ticid = f[1].header["TICID"]               
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
        ticid = 0
    
    return time1, i1, ticid

def lc_from_bulk_download(fits_path, target_list, fname_out, fname_targets,
                          fname_notes, path):
    '''Saves interpolated fluxes to fits file fname_out.
    Parameters:
        * fits_path : directory containing all light curve fits files 
                      (ending with '/')
        * target_list : returned by lc_by_camera_ccd()
        * fname_out : name of fits file to save time, flux and ticids into
        * fname_targets : saves ticid of every target saved 
        * fname_notes : saves the ticid of any target it fails on
        * path : directory to save nan mask plots in
    Returns:
        * confirmation : boolean, returns False if failure
    '''
    import fnmatch
    import gc
    
    # >> get list of all light curve fits files
    fnames_all = os.listdir(fits_path)
    fnames = fnmatch.filter(fnames_all, '*fits*')
    
    # >> find all light curves in a group
    fnames_group = []
    target_list = target_list[:,0]
    for target in target_list:
        try:
            fname = list(filter(lambda x: str(int(target)) in x, fnames))[0]
            fnames_group.append(fname)
        except:
            print('Missing ' + str(int(target)))
            with open(fname_notes, 'a') as f:
                f.write(str(int(target)) + '\n')
        
    fnames = fnames_group
    count = 0
    
    ticid_list = []
    intensity = []
    for n in range(len(fnames)):
        
        file = fnames[n]
        print(count)
        
        # >> open file
        with fits.open(fits_path + file) as hdul:
            hdu_data = hdul[1].data
            
            # >> get time array (only for the first light curve)
            if n == 0: 
                time = hdu_data['TIME']
                
            # >> get flux array
            i = hdu_data['PDCSAP_FLUX']
            intensity.append(i)
            
            # >> get ticid
            ticid = hdul[1].header['TICID']
            ticid_list.append(ticid)
            with open(fname_targets, 'a') as f:
                f.write(str(int(ticid)) + '\n')
            
            # >> clear memory
            del hdu_data
            del hdul[1].data
            del hdul[0].data
            gc.collect()
            
        count += 1
    
    # >> interpolate and NaN mask
    print('Interpolating...')
    intensity = np.array(intensity)
    ticid_list = np.array(ticid_list)
    intensity, time = interpolate_all(intensity, time)
    
    # >> save time array, intensity array and ticids to fits file
    print('Saving to fits file...')
    
    hdr = fits.Header()
    hdu = fits.PrimaryHDU(time, header=hdr)
    hdu.writeto(fname_out)
    fits.append(fname_out, intensity)
    fits.append(fname_out, ticid_list)
    confirmation="lc_from_bulk_download has finished running"
    return confirmation

#normalizing each light curve
def normalize(flux, axis=1):
    '''Dividing by median.'''
    medians = np.median(flux, axis = axis, keepdims=True)
    flux = flux / medians - 1.
    return flux

def rms(x, axis=1):
    rms = np.sqrt(np.nanmean(x**2, axis = axis))
    return rms

def standardize(x, ax=1):
    means = np.nanmean(x, axis = ax, keepdims=True) # >> subtract mean
    x = x - means
    stdevs = np.nanstd(x, axis = ax, keepdims=True) # >> divide by standard dev
    
    # >> avoid dividing by 0.0
    stdevs[ np.nonzero(stdevs == 0.) ] = 1e-8
    
    x = x / stdevs
    return x

# def normalize(intensity):
#     """normalizes the intensity from the median value 
#     by dividing out. then sigmaclips using astropy
#     returns a masked array"""
#     sigclip = SigmaClip(sigma=5, maxiters=None, cenfunc='median')
#     intense = []
#     for i in np.arange(len(intensity)):
#         intensity[i] = intensity[i] / np.median(intensity[i])
#         inte = sigclip(intensity[i], masked=True, copy = False)
#         intense.append(inte)
#     intensity = np.ma.asarray(intense)
#     print("Normalization and sigma clipping complete")
#     return intensity

#interpolate and sigma clip
def interpolate_all(flux, time, flux_err=False, interp_tol=20./(24*60),
                    num_sigma=10, DEBUG_INTERP=False, output_dir='./',
                    prefix='', apply_nan_mask=False, DEBUG_MASK=False,
                    ticid=False):
    '''Interpolates each light curves in flux array.'''
    
    flux_interp = []
    for i in flux:
        i_interp = interpolate_lc(i, time, flux_err=flux_err,
                                  interp_tol=interp_tol,
                                  num_sigma=num_sigma,
                                  DEBUG_INTERP=DEBUG_INTERP,
                                  output_dir=output_dir, prefix=prefix)
        flux_interp.append(i_interp)
        
    flux_interp = np.array(flux_interp)
    if apply_nan_mask:
        flux_interp, time = nan_mask(flux_interp, time, DEBUG=DEBUG_MASK,
                               output_dir=output_dir, ticid=ticid)
    
    return flux_interp, time

def interpolate_lc(i, time, flux_err=False, interp_tol=20./(24*60),
                   num_sigma=10, DEBUG_INTERP=False,
                   output_dir='./', prefix=''):
    '''Interpolation for one light curve. Linearly interpolates nan gaps less
    than 20 minutes long. Spline interpolates nan gaps more than 20 minutes
    long (and shorter than orbit gap)'''
    from astropy.stats import SigmaClip
    from scipy import interpolate
    
    # >> plot original light curve
    if DEBUG_INTERP:
        fig, ax = plt.subplots(5, 1, figsize=(8, 3*5))
        ax[0].plot(time, i, '.k', markersize=2)
        ax[0].set_title('original')
    
    # -- sigma clip ----------------------------------------------------------
    sigclip = SigmaClip(sigma=num_sigma, maxiters=None, cenfunc='median')
    clipped_inds = np.nonzero(np.ma.getmask(sigclip(i, masked=True)))
    i[clipped_inds] = np.nan
    if DEBUG_INTERP:
        ax[1].plot(time, i, '.k', markersize=2)
        ax[1].set_title('clipped')
    
    # >> find nan windows
    n = np.shape(i)[0]
    loc_run_start = np.empty(n, dtype=bool)
    loc_run_start[0] = True
    np.not_equal(np.isnan(i)[:-1], np.isnan(i)[1:], out=loc_run_start[1:])
    run_starts = np.nonzero(loc_run_start)[0]

    # >> find nan window lengths
    run_lengths = np.diff(np.append(run_starts, n))
    tdim = time[1]-time[0]
    
    # -- interpolate small nan gaps ------------------------------------------
    interp_gaps = np.nonzero((run_lengths * tdim <= interp_tol) * \
                             np.isnan(i[run_starts]))
    interp_inds = run_starts[interp_gaps]
    interp_lens = run_lengths[interp_gaps]

    i_interp = np.copy(i)
    for a in range(np.shape(interp_inds)[0]):
        start_ind = interp_inds[a]
        end_ind = interp_inds[a] + interp_lens[a]
        i_interp[start_ind:end_ind] = np.interp(time[start_ind:end_ind],
                                                time[np.nonzero(~np.isnan(i))],
                                                i[np.nonzero(~np.isnan(i))])
    i = i_interp
    if DEBUG_INTERP:
        ax[2].plot(time, i, '.k', markersize=2)
        ax[2].set_title('interpolated')
    
    # -- spline interpolate large nan gaps -----------------------------------
    # >> fit spline to non-nan points
    num_inds = np.nonzero( (~np.isnan(i)) * (~np.isnan(time)) )[0]
    ius = interpolate.InterpolatedUnivariateSpline(time[num_inds], i[num_inds])
    
    # >> new time array (take out orbit gap)
    orbit_gap_start = num_inds[ np.argmax(np.diff(time[num_inds])) ]
    orbit_gap_end = num_inds[ orbit_gap_start+1 ]
    orbit_gap_len = orbit_gap_end - orbit_gap_start
    t_spl = np.copy(time)
    t_spl = np.delete(t_spl, range(num_inds[-1], len(t_spl)))
    t_spl = np.delete(t_spl, range(orbit_gap_start, orbit_gap_end))
    t_spl = np.delete(t_spl, range(num_inds[0]))
    
    # >> spline fit for new time array
    i_spl = ius(t_spl)
    
    if DEBUG_INTERP:
        ax[3].plot(t_spl, i_spl, '.')
        ax[3].set_title('spline') 
    
    # >> find nan gaps to spline interpolate over
    interp_gaps = np.nonzero((run_lengths * tdim > interp_tol) * \
                              np.isnan(i[run_starts]) * \
                              (((run_starts > orbit_gap_start) * \
                                (run_starts < orbit_gap_end)) == False))       
    interp_inds = run_starts[interp_gaps]
    interp_lens = run_lengths[interp_gaps]  
    
    # >> spline interpolate nan gaps
    i_interp = np.copy(i)
    for a in range(np.shape(interp_inds)[0]):
        start_ind = interp_inds[a]
        end_ind   = interp_inds[a] + interp_lens[a] - 1

        if not np.isnan(time[start_ind]):
            start_ind_spl = np.argmin(np.abs(t_spl - time[start_ind]))
            end_ind_spl = start_ind_spl + (end_ind-start_ind)
        else:
            end_ind_spl = np.argmin(np.abs(t_spl - time[end_ind]))
            start_ind_spl = end_ind_spl - (end_ind-start_ind)
        i_interp[start_ind:end_ind] = i_spl[start_ind_spl:end_ind_spl]
        
    if DEBUG_INTERP:
        ax[4].plot(time, i_interp, '.k', markersize=2)
        ax[4].set_title('spline interpolate')
        fig.tight_layout()
        fig.savefig(output_dir + prefix + 'interpolate_debug.png',
                    bbox_inches='tight')
        plt.close(fig)
        
    return i_interp
    
def nan_mask(flux, time, flux_err=False, DEBUG=False, debug_ind=1042,
             ticid=False, output_dir='./', prefix='', tol1=0.05, tol2=0.1):
    '''Apply nan mask to flux and time array.
    Returns masked, homogenous flux and time array.
    If there are only a few (less than tol1 % light curves) light curves that
    contribute (more than tol2 % data points are NaNs) to NaN mask, then will
    remove those light curves.'''

    mask = np.nonzero(np.prod(~np.isnan(flux), axis = 0) == False)
    
    # >> plot histogram of number of data points thrown out
    num_masked = []
    num_nan = []
    for lc in flux:
        num_inds = np.nonzero( ~np.isnan(lc) )
        num_masked.append( len( np.intersect1d(num_inds, mask) ) )
        num_nan.append( len(num_inds) )
    plt.figure()
    plt.hist(num_masked, bins=50)
    plt.ylabel('number of light curves')
    plt.xlabel('number of data points masked')
    plt.savefig(output_dir + 'nan_mask.png')
    plt.close()
    
    # >> debugging plots
    if DEBUG:
        fig, ax = plt.subplots()
        ax.plot(time, flux[debug_ind], '.k', markersize=2)
        ax.set_title('removed orbit gap')
        fig.tight_layout()
        fig.savefig(output_dir + prefix + 'nanmask_debug.png',
                    bbox_inches='tight')
        plt.close(fig) 
        
        # >> plot nan-y light curves
        sorted_inds = np.argsort(num_masked)
        for k in range(2): # >> plot top and lowest
            fig, ax = plt.subplots(nrows=10, figsize=(8, 3*10))
            for i in range(10):
                if k == 0:
                    ind = sorted_inds[i]
                else:
                    ind = sorted_inds[-i-1]
                ax[i].plot(time, flux[ind], '.k', markersize=2)
                pf.ticid_label(ax[i], ticid[ind], title=True)
                num_nans = np.count_nonzero(np.isnan(flux[ind]))
                ax[i].text(0.98, 0.98, 'Num NaNs: '+str(num_nans)+\
                           '\nNum masked: '+str(num_masked[ind]),
                           transform=ax[i].transAxes,
                           horizontalalignment='right',
                           verticalalignment='top', fontsize='xx-small')
            if k == 0:
                fig.savefig(output_dir + prefix + 'nanmask_top.png',
                            bbox_inches='tight')
            else:
                fig.savefig(output_dir + prefix + 'nanmask_low.png',
                            bbox_inches='tight')
       
    # >> check if only a few light curves contribute to NaN mask
    num_nan = np.array(num_nan)
    worst_inds = np.nonzero( num_nan > tol2 )
    if len(worst_inds[0]) < tol1 * len(flux): # >> only a few bad light curves
        np.delete(flux, worst_inds, 0)
        
        # >> and calculate new mask
        mask = np.nonzero(np.prod(~np.isnan(flux), axis = 0) == False)    
        
    # >> apply NaN mask
    time = np.delete(time, mask)
    flux = np.delete(flux, mask, 1)
    
    if type(flux_err) != bool:
        flux_err = np.delete(flux_err, mask, 1)
        return flux, time, flux_err
    else:
        return flux, time

#producing the feature vector list -----------------------------

def create_save_featvec(yourpath, times, intensities, sector, camera, ccd):
    """Produces the feature vectors for each light curve and saves them all
    into a single fits file
    Takes: 
        your path (folder you want the file saved into)
        time axis
        all intensity arrays
        sector, camera, ccd values
    returns list of feature vectors
    modified: [lcg 07012020]"""
    folder_name = "Sector" + str(sector) + "Cam" + str(camera) + "CCD" + str(ccd)
    path = yourpath + folder_name
    fname_features = path + "/"+ folder_name + "_features.fits"
    num_data = len(datasets) #how many datasets
    x = time_axis #creates the x axis
    feature_list = np.zeros((num_data, 20)) #MANUALLY UPDATE WHEN CHANGING NUM FEATURES
    print("creating feature vectors about to begin")
    for n in range(num_data):
        feature_list[n] = featvec(x, datasets[n])
        if n % 50 == 0: print(str(n) + " completed")
    
    feature_list = np.asarray(feature_list)
    hdr = fits.Header()
    
    hdu = fits.PrimaryHDU(feature_list, header=hdr)
    hdu.writeto(fname_features)
    
    return feature_list

def featvec(x_axis, sampledata): 
    """calculates the feature vector of the single light curve
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
        
        (from transitleastsquares)
        16 - period
        17 - best duration
        18 - depth
        19 - power
        
        ***if you update the number of features, 
        you have to update the number of features in create_list_featvec
        modified [lcg 06242020]"""
    from transitleastsquares import transitleastsquares
    
    #empty feature vector
    featvec = [] 
    #moments
    featvec.append(np.mean(sampledata)) #mean (don't use moment, always gives 0)
    featvec.append(moment(sampledata, moment = 2)) #variance
    featvec.append(moment(sampledata, moment = 3)) #skew
    featvec.append(moment(sampledata, moment = 4)) #kurtosis
    featvec.append(np.log(np.abs(moment(sampledata, moment = 2)))) #ln variance
    featvec.append(np.log(np.abs(moment(sampledata, moment = 3)))) #ln skew
    featvec.append(np.log(np.abs(moment(sampledata, moment = 4)))) #ln kurtosis
    
    #periods
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
    
    #tls 
    model = transitleastsquares(x_axis, sampledata)
    results = model.power()
    featvec.append(results.period)
    featvec.append(results.duration)
    featvec.append(results.depth)
    featvec.append(results.power)
    
    return(featvec) 

