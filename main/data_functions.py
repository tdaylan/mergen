# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 21:54:56 2020

Data access, data processing, feature vector creation functions.

@author: Lindsey Gordon @lcgordon and Emma Chickles (@??)

Updated: June 26 2020
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

import modellibrary as ml

import batman
import numba
from transitleastsquares import transitleastsquares

def test_data():
    """make sure the module loads in"""
    print("data functions loaded in")
    
def load_in_a_group(sector, camera, ccd, path):
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


#data process an entire group of TICs
def data_access_by_group_fits(yourpath, sectorfile, sector, camera, ccd):
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
    fname_time_intensities_raw = path + "/" + folder_name + "_raw_lightcurves.fits"
    fname_targets = path + "/" + folder_name + "_targets.txt"
    fname_notes = path + "/" + folder_name + "_group_notes.txt"
    
    try:
        os.makedirs(path)
        print ("Successfully created the directory %s" % path) 
        with open(fname_targets, 'a') as file_object:
            file_object.write("This file contains the target TICs for this group. Fits light curves are 1-indexed, so first target is all zeroes \n 00000000")
        with open(fname_notes, 'a') as file_object:
            file_object.write("This file contains group notes, including any TICs that could not be accessed.")
        # get just the list of targets for the specified sector, camera, ccd --------
        target_list = lc_by_camera_ccd(sectorfile, camera, ccd)
        print("there are ", len(target_list), "targets")
    # get the light curve for each target on the list, and save into a text file
        confirmation = lc_from_target_list_fits(yourpath, target_list, fname_time_intensities_raw, fname_targets, fname_notes,
                                                sector)
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


def lc_from_target_list_fits(yourpath, targetList, fname_time_intensities_raw, fname_targets, fname_notes,
                             sector):
    """ runs getting the files and data for all targets on the list
    then appends the time & intensity arrays and the TIC number into text files
    that can later be accessed
    modified [lcg 062620]
    """
    intensity = []
    i_interpolated
    ticids = []
    for n in range(len(targetList)): #for each item on the list
        hdr = fits.Header() #make the header
        
        if n == 0: #for the first target only do you need to get the time index
            target = targetList[n][0] #get that target number
            time1, i1, tic = get_lc_file_and_data(yourpath, target, sector) #grab that data
            
            if type(i1) == np.ndarray: #if the data IS data
                i_interp = ml.interpolate_lc(i1, time1)
                i_interpolated.append(i_interp)
                intensity.append(i1)
                hdr = fits.Header() #make-a the header
                hdu = fits.PrimaryHDU(time1, header=hdr)
                hdu.writeto(fname_time_intensities_raw) #make the fits file
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
                i_interp = ml.interpolate_lc(i1, time1)
                i_interpolated.append(i_interp)
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
    i_interp = np.array(i_interp)
    with open(fname_time_intensities_raw, 'rb+') as f:
        fits.append(fname_time_intensities_raw, intensity)
        fits.append(fname_time_intensities_raw, i_interp)
        fits.append(fname_time_intensities_raw, ticids)
    confirmation = "lc_from_target_list has finished running"
    return confirmation

def get_lc_file_and_data(yourpath, target, sector):
    """ goes in, grabs the data for the target, gets the time index, intensity,and TIC
    if connection error w/ MAST, skips it
    modified [lcg 06262020] - now pulls TICID as well, in case accidentally gets the wrong lc"""
    fitspath = yourpath #+ 'mastDownload/TESS/'
    targ = "TIC " + str(int(target))
    print(targ)
    try:
        #find and download data products for your target objectname='TIC '+str(int(target)),
        obs_table = astroquery.mast.Observations.query_criteria(obs_collection='TESS',
                                                               dataproduct_type='timeseries',
                                                               target_name=str(int(target)),
                                                               sequence_number=sector,
                                                               objectname='TIC '+str(int(target)))
        # obs_table = Observations.query_object(targ, radius=".02 deg")
        data_products_by_obs = Observations.get_product_list(obs_table[0:4])
            
        # #in theory, filter_products should let you sort out the non fits files but i 
        # #simply could not get it to accept it despite following the API guidelines
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
                    #print("appended", name, "to filepaths")
        
        print(len(filepaths))
        
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



#normalizing each light curve
def normalize(intensity):
    """normalizes the intensity from the median value 
    by dividing out. then sigmaclips using astropy
    returns a masked array"""
    sigclip = SigmaClip(sigma=5, maxiters=None, cenfunc='median')
    intense = []
    for i in np.arange(len(intensity)):
        intensity[i] = intensity[i] / np.median(intensity[i])
        inte = sigclip(intensity[i], masked=True, copy = False)
        intense.append(inte)
    intensity = np.ma.asarray(intense)
    print("Normalization and sigma clipping complete")
    return intensity

#interpolate and sigma clip
    
def interpolate_lc(flux, time, flux_err=False, interp_tol=20./(24*60),
                   num_sigma=5, orbit_gap_len = 3, DEBUG=False,
                   spline_interpolate=True):
    '''Interpolates nan gaps less than 20 minutes long.
    output_dir='./',  prefix='''''
    from astropy.stats import SigmaClip
    from scipy import interpolate
    flux_interp = []
    for j in range(len(flux)):
        i = flux[j]
        if DEBUG and j == 1042:
            fig, ax = plt.subplots(6, 1, figsize=(8, 3*6))
            ax[0].plot(time, i, '.k', markersize=2)
            ax[0].set_title('original')
        # >> sigma clip
        sigclip = SigmaClip(sigma=num_sigma, maxiters=None, cenfunc='median')
        clipped_inds = np.nonzero(np.ma.getmask(sigclip(i, masked=True)))
        i[clipped_inds] = np.nan
        if DEBUG and j == 1042:
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
        
        # -- interpolate small nan gaps ---------------------------------------
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
        if DEBUG and j == 1042:
            ax[2].plot(time, i, '.k', markersize=2)
            ax[2].set_title('interpolated')
        
        # -- spline interpolate large nan gaps --------------------------------
        if spline_interpolate:
            orbit_gap_len = np.count_nonzero(np.isnan(time))*tdim
            interp_gaps = np.nonzero((run_lengths * tdim > interp_tol) * \
                                     (run_lengths*tdim < 0.9*orbit_gap_len) * \
                                     np.isnan(i[run_starts]))
            interp_inds = run_starts[interp_gaps]
            interp_lens = run_lengths[interp_gaps]
            
            # >> fit spline to non-nan points
            num_inds = np.nonzero(~np.isnan(i))
            use_splrep = False
            if use_splrep:
                tck = interpolate.splrep(time[num_inds], i[num_inds], k=3)
            else:
                cs= interpolate.CubicSpline(time[num_inds], i[num_inds])
                # i_cs = cs(time[num_inds])
                # num_inds_time = np.nonzero(~np.isnan(time))
                # i_cs = cs(time[num_inds_time])
                t1 = np.linspace(np.min(time[num_inds]), np.max(time[num_inds]),
                                 len(time))
                t1 = np.delete(t1, np.nonzero(np.isnan(time)))
                i_cs = cs(t1)
            if DEBUG and j==1042:
                if use_splrep:
                    i_plot = interpolate.splev(time[num_inds],tck)
                    ax[3].plot(time[num_inds], i_plot, '-')
                else:
                    i_plot = i_cs
                    ax[3].plot(t1, i_cs, '-')
                
                ax[3].set_title('spline') 
            
            i_interp = np.copy(i)
            for a in range(np.shape(interp_inds)[0]):
                start_ind = interp_inds[a]
                end_ind   = interp_inds[a] + interp_lens[a] - 1
                # >> pad [etc 060720]
                # start_ind = max(0, start_ind-10)
                # end_ind = min(len(time)-1, end_ind + 10)
                # t_new = np.linspace(time[start_ind-1]+tdim, time[end_ind-1]+tdim,
                #                     end_ind-start_ind)

                if use_splrep:
                    t_new = np.linspace(time[start_ind], time[end_ind],
                                    end_ind-start_ind)     
                    i_interp[start_ind:end_ind]=interpolate.splev(t_new, tck)
                else:
                    # start_ind_cs = num_inds_time[0].tolist().index(start_ind)
                    # end_ind_cs = num_inds_time[0].tolist().index(end_ind)
                    # start_ind_cs = num_inds[0].tolist().index(start_ind)
                    # end_ind_cs = num_inds[0].tolist().index(end_ind)                    
                    # i_interp[start_ind:end_ind]=i_cs[start_ind_cs:end_ind_cs]
                    # t_new = np.linspace(time[start_ind], time[end_ind],
                    #                     end_ind-start_ind)
                    # i_interp[start_ind:end_ind]=cs(t_new)
                    if not np.isnan(time[start_ind]):
                        start_ind_cs = np.argmin(np.abs(t1 - time[start_ind]))
                        end_ind_cs = start_ind_cs + (end_ind-start_ind)
                    else:
                        end_ind_cs = np.argmin(np.abs(t1 - time[end_ind]))
                        start_ind_cs = end_ind_cs - (end_ind-start_ind)
                    i_interp[start_ind:end_ind] = i_cs[start_ind_cs:end_ind_cs]
            flux_interp.append(i_interp)
            if DEBUG and j==1042:
                ax[4].plot(time, i_interp, '.k', markersize=2)
                ax[4].set_title('spline interpolate')
        else:
            flux_interp.append(i)
        
    # -- remove orbit nan gap -------------------------------------------------
    flux = np.array(flux_interp)
    nan_inds = np.nonzero(np.prod(~np.isnan(flux), 
                                  axis = 0) == False)
    time = np.delete(time, nan_inds)
    flux = np.delete(flux, nan_inds, 1)
    #if DEBUG:
     #   ax[5].plot(time, flux[1042], '.k', markersize=2)
      #  ax[5].set_title('removed orbit gap')
      #  fig.tight_layout()

        # for a in ax.flatten():
        #     format_axes(a, xlabel=True, ylabel=True)
       # fig.savefig(output_dir + prefix + 'interpolate_debug.png',
        #            bbox_inches='tight')
        #plt.close(fig) 
    
    if type(flux_err) != bool:
        flux_err = np.delete(flux_err, nan_inds, 1)
        return flux, time, flux_err
    else:
        return time, flux
    
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
        you have to update the number of features in create_list_featvec!!!!
        modified [lcg 06242020]"""
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

