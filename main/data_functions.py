# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 21:54:56 2020
Data access, data processing, feature vector creation functions.
@author: Lindsey Gordon (@lcgordon) and Emma Chickles (@emmachickles)
Updated: July 8 2020
Data access
* test_data()           : confirms module loaded in 
* lc_by_camera_ccd()    : divides sector TIC list into groups by ccd/camera
* load_data_from_metafiles()    : loads LC from ALL metafiles for sector and
                                  applies NaN mask
* load_group_from_fits()        : loads LC for one group's fits files
* data_access_sector_by_bulk()
* data_access_by_group_fits()
* bulk_download_helper()
* follow_up_on_missed_targets_fits()
* lc_from_target_list()    : Pulls all light curves from a list of TICs
* get_lc_file_and_data()        : Pulls a light curve's fits file by TIC
* tic_list_by_magnitudes        : Gets list of TICs for upper/lower mag. bounds
                        
Data processing
* normalize()       : median normalization
* mean_norm() 	    : mean normalization (for TLS)
* interpolate_all() : sigma clip and interpolate flux array
* interpolate_lc()  : sigma clip and interpolate one light curve
* nan_mask()        : apply NaN mask to flux array

Engineered features
* create_save_featvec()     : creates and saves a fits file containing all features
* featvec()                 : creates a single feature vector for a LC
* feature_gen_from_lc_fits()    : creates features for all of a sector
* get_tess_features : queries Teff, rad, mass, GAIAmag, d 
                      !! query objType from Simbad
* get_tess_feature_txt : queries TESS features (Teff, rad, etc.) for a sector
* build_simbad_database : queries bibcode and object type for TESS objects
* dbscan_param_search : performs grid search for DBSCAN

Depreciated Functions
* load_group_from_txt()
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
# rcParams['lines.color'] = 'k'
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

import pdb
import fnmatch as fm

import numba
# import batman
# from transitleastsquares import transitleastsquares


def test_data():
    """make sure the module loads in"""
    print("Data functions loaded in.")
    
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
    
def load_data_from_metafiles(data_dir, sector, cams=[1,2,3,4],
                             ccds=[1,2,3,4], data_type='SPOC',
                             cadence='2-minute', DEBUG=False,
                             output_dir='./', debug_ind=0,
                             nan_mask_check=True,
                             custom_mask=[]):
    '''Pulls light curves from fits files, and applies nan mask.
    
    Parameters:
        * data_dir : folder containing fits files for each group
        * sector : sector, given as int, or as a list
        * cams : list of cameras
        * ccds : list of CCDs
        * data_type : 'SPOC', 'FFI'
        * cadence : '2-minute', '20-second'
        * DEBUG : makes nan_mask debugging plots. If True, the following are
                  required:
            * output_dir
            * debug_ind
        * nan_mask_check : if True, applies NaN mask
    
    Returns:
        * flux : array of light curve PDCSAP_FLUX,
                 shape=(num light curves, num data points)
        * x : time array, shape=(num data points)
        * ticid : list of TICIDs, shape=(num light curves)
        * target_info : [sector, cam, ccd, data_type, cadence] for each light
                        curve, shape=(num light curves, 5)
    '''
    
    # >> get file names for each group
    fnames = []
    fname_info = []
    for cam in cams:
        for ccd in ccds:
            s = 'Sector{sector}/Sector{sector}Cam{cam}CCD{ccd}/' + \
                'Sector{sector}Cam{cam}CCD{ccd}_lightcurves.fits'
            fnames.append(s.format(sector=sector, cam=cam, ccd=ccd))
            fname_info.append([sector, cam, ccd, data_type, cadence])
                
    # >> pull data from each fits file
    print('Pulling data')
    flux_list = []
    ticid = np.empty((0, 1))
    target_info = [] # >> [sector, cam, ccd, data_type, cadence]
    for i in range(len(fnames)):
        print('Loading ' + fnames[i] + '...')
        with fits.open(data_dir + fnames[i], mmap=False) as hdul:
            if i == 0:
                x = hdul[0].data
            flux = hdul[1].data
            ticid_list = hdul[2].data
    
        flux_list.append(flux)
        ticid = np.append(ticid, ticid_list)
        target_info.extend([fname_info[i]] * len(flux))

    # >> concatenate flux array         
    flux = np.concatenate(flux_list, axis=0)
        
    # >> apply nan mask
    if nan_mask_check:
        print('Applying nan mask')
        flux, x = nan_mask(flux, x, DEBUG=DEBUG, ticid=ticid,
                           debug_ind=debug_ind, target_info=target_info,
                           output_dir=output_dir, custom_mask=custom_mask)

    return flux, x, ticid, np.array(target_info)
    
    
def load_group_from_fits(path, sector, camera, ccd): 
    """ pull the light curves and target list from fits metafiles
    path is the folder in which all the metafiles are saved. ends in a backslash 
    sector camera ccd are integers you want the info from
    modified [lcg 07032020]
    """
    filename_lc = path + "Sector"+str(sector)+"Cam"+str(camera)+"CCD"+str(ccd) + "_lightcurves.fits"
   
    f = fits.open(filename_lc, mmap=False)
    
    time = f[0].data
    intensities = f[1].data
    targets = f[2].data
    f.close()
    
    return time, intensities, targets
    



def data_access_sector_by_bulk(yourpath, sectorfile, sector,
                               bulk_download_dir, custom_mask=[],
                               apply_nan_mask=False):
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
          Also see bulk_download_helper()
    e.g. df.data_access_sector_by_bulk('../../',
                                       '../../all_targets_S020_v1.txt', 20,
                                       '../../tessdata_sector_20/')
    '''
    
    for cam in [1,2,3,4]:
        for ccd in [1,2,3,4]:
            data_access_by_group_fits(yourpath, sectorfile, sector, cam,
                                      ccd, bulk_download=True,
                                      bulk_download_dir=bulk_download_dir,
                                      custom_mask=custom_mask,
                                      apply_nan_mask=apply_nan_mask)
            
def bulk_download_helper(yourpath, shell_script):
    '''If bulk download failed / need to start where you left off. Can also be
    used to go back and check you have all the light curves from a sector.
    Parameters:
        * yourpath : directory to save .fits files in, contains shell_script
        * shell_script : file name for shell script (tesscurl_sector_*_lc.sh)
          from http://archive.stsci.edu/tess/bulk_downloads.html
    e.g. bulk_download_helper('./tessdata_sector_18/',
                              'tesscurl_sector_18_lc.sh')
    '''
    import fnmatch as fm
    with open(yourpath+shell_script, 'r') as f:
        sector_targets = f.readlines()[1:]
        
    downloaded_targets = os.listdir(yourpath)
    
    # >> loop through all the sector_targets
    for i in range(len(sector_targets)):
        
        # >> check if already downloaded
        fname = sector_targets[i].split()[5]
        matches = fm.filter(downloaded_targets, fname)
        
        # >> if not downloaded, download the light curve
        if len(matches) == 0:
            print(str(i) + '/' + str(len(sector_targets)))            
            print(fname)
            command = sector_targets[i].split()[:5] + [yourpath+fname] + \
                [sector_targets[i].split()[6]]
            os.system(' '.join(command))
        else:
            print('Already downloaded '+fname)
            
#data process an entire group of TICs
def data_access_by_group_fits(yourpath, sectorfile, sector, camera, ccd,
                              bulk_download=False, bulk_download_dir='./',
                              custom_mask=[], apply_nan_mask=False):
    """you will need:
        your path into the main folder you're working in - must end with /
        the file for your sector from TESS (full path)
        sector number (as int)
        camera number you want (as int/float)
        ccd number you want (as int/float)
        this ONLY returns the target list and folderpath for the group
        
        Saves a .fits file with primaryHDU=f[0]=time array,
        f[1]=raw intensity array, f[2] = interpolated intensity array (not normalized!)
        , f[3]=TICIDs
        """
    # produce the folder to save everything into and set up file names
    folder_name = "Sector" + str(sector) + "Cam" + str(camera) + "CCD" + str(ccd)
    path = yourpath + folder_name
    fname_time_intensities = path + "/" + folder_name + "_lightcurves.fits"
    fname_targets = path + "/" + folder_name + "_targets.txt"
    fname_notes = path + "/" + folder_name + "_group_notes.txt"
    fname_flagged = path + "/" + folder_name + "_flagged.fits" 
    
    try:
        os.makedirs(path)
        print ("Successfully created the directory %s" % path) 
        with open(fname_targets, 'a') as file_object:
            file_object.write("This file contains the target TICs for this group.")
        with open(fname_notes, 'a') as file_object:
            file_object.write("This file contains group notes, including any TICs that could not be accessed.\n")
        # get just the list of targets for the specified sector, camera, ccd --------
        target_list = lc_by_camera_ccd(sectorfile, camera, ccd)
        print("there are ", len(target_list), "targets")
        
        # >> get the light curve for each target on the list, and save into a
        # >> fits file
        if bulk_download:
            time, intensity, ticids, flagged, ticid_flagged = \
                lc_from_bulk_download(bulk_download_dir, target_list,
                                      fname_time_intensities, fname_targets,
                                      fname_notes, path, fname_flagged,
                                      custom_mask=custom_mask,
                                      apply_nan_mask=apply_nan_mask)
        else: # >> download each light curve
            # !! TODO: add flag option to lc_from_target_list()
            time, intensity, ticids = lc_from_target_list(yourpath, target_list,
                                                    fname_time_intensities,
                                                    fname_targets, fname_notes,
                                                     path=path,
                                                     custom_mask=custom_mask,
                                                     apply_nan_mask=apply_nan_mask)
       
        
    except OSError: #if there is an error creating the folder
        print("There was an OS Error trying to create the folder. Check to see if data is already saved there")
        targets = "empty"
        
    return time, intensity, ticids, path, flagged, ticid_flagged

def follow_up_on_missed_targets_fits(yourpath, sector, camera, ccd):
    """ function to follow up on rejected TIC ids"""
    folder_name = "Sector" + str(sector) + "Cam" + str(camera) + "CCD" + str(ccd)
    path = yourpath + folder_name
    fname_time_intensities = path + "/" + folder_name + "_lightcurves.fits"
    fname_targets = path + "/" + folder_name + "_targets.txt"
    fname_notes = path + "/" + folder_name + "_group_notes.txt"
    fname_notes_followed_up = path + "/" + folder_name + "_targets_still_no_data.txt"
    
    
    retry_targets = np.loadtxt(fname_notes, skiprows=1)
    retry_targets = np.column_stack((retry_targets, retry_targets)) #loak this is a crude way to get around some indexing issues and we're working with it
    
    with open(fname_notes_followed_up, 'a') as file_object:
        file_object.write("Data could not be found for the following TICs after two attempts")
    
    
    for n in range(len(retry_targets)):
        target = retry_targets[n][0] #get that target number
        time1, i1 = get_lc_file_and_data(yourpath, target)
        if type(i1) == np.ndarray: #IF THE DATA IS FORMATTED LKE DATA
            fits.append(fname_time_intensities, i1, header=hdr)
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

    
def lc_from_target_list(yourpath, targetList, fname_time_intensities_raw,
                             fname_targets, fname_notes, path='./',
                             custom_mask=[], apply_nan_mask=False):
    """ runs getting the files and data for all targets on the list
    then appends the time & intensity arrays and the TIC number into text files
    that can later be accessed
    modified [lcg 07092020]
    """
    intensity = []
    ticids = []
    for n in range(len(targetList)): #for each item on the list
        
        if n == 0: #for the first target only do you need to get the time index
            target = targetList[n][0] #get that target number
            time1, i1, tic = get_lc_file_and_data(yourpath, target) #grab that data
            
            if type(i1) == np.ndarray: #if the data IS data
                intensity.append(i1)
                ticids.append(tic)
            else: #if the data is NOT a data
                print("First target failed, no time index was saved")
                with open(fname_notes, 'a') as file_object:
                    file_object.write("\n")
                    file_object.write(str(int(target)))
        else: #only saving the light curve into the fits file because it's all you need
            target = targetList[n][0] #get that target number
            time1, i1, tic = get_lc_file_and_data(yourpath, target)
            if type(i1) == np.ndarray:
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
    intensity_interp, time, ticids, flagged, ticid_flagged = \
        interpolate_all(intensity, time1, ticids, custom_mask=custom_mask,
                        apply_nan_mask=apply_nan_mask)
    
    print('Saving to fits file')
    # i_interp = np.array(i_interp)
    hdr = fits.Header() # >> make the header
    hdu = fits.PrimaryHDU(time, header=hdr)
    hdu.writeto(fname_time_intensities_raw)
    fits.append(fname_time_intensities_raw, intensity_interp)
    fits.append(fname_time_intensities_raw, ticids)
    
    # >> actually i'm going to save the raw intensities just in case
    fits.append(fname_time_intensities_raw, intensity)
    
    # with open(fname_time_intensities_raw, 'rb+') as f:
    #     # >> don't want to save 2x data we need to, so only save interpolated
    #     # fits.append(fname_time_intensities_raw, intensity)
    #     fits.append(fname_time_intensities_raw, i_interp)
    #     fits.append(fname_time_intensities_raw, ticids)
    
    confirmation = "lc_from_target_list has finished running"
    return confirmation

def get_lc_file_and_data(yourpath, target):
    """ goes in, grabs the data for the target, gets the time index, intensity,and TIC
    if connection error w/ MAST, skips it.
    Also masks any flagged data points according to the QUALITY column.
    parameters: 
        * yourpath, where you want the files saved to. must end in /
        * targets, target list of all TICs 
    modified [lcg 07082020] - fixed handling no results, fixed deleting download folder"""
    fitspath = yourpath + 'mastDownload/TESS/' # >> download directory
    targ = "TIC " + str(int(target))
    print(targ)
    try:
        #find and download data products for your target objectname='TIC '+str(int(target)),
        obs_table = Observations.query_criteria(obs_collection='TESS',
                                        dataproduct_type='timeseries',
                                        target_name=str(int(target)),
                                        objectname=targ)
        data_products_by_obs = Observations.get_product_list(obs_table[0:8])
            
        filter_products = Observations.filter_products(data_products_by_obs,
                                                       description = 'Light curves')
        if len(filter_products) != 0:
            manifest = Observations.download_products(filter_products, download_dir= yourpath, extension='fits')
        else: 
            print("Query yielded no matching data produts for ", targ)
            time1 = 0
            i1 = 0
            ticid = 0
            
        #get all the paths to lc.fits files
        filepaths = []
        for root, dirs, files in os.walk(fitspath):
            for name in files:
                #print(name)
                if name.endswith(("lc.fits")):
                    filepaths.append(root + "/" + name)
        #print(len(filepaths))
        #print(filepaths)
        
        if len(filepaths) == 0: #if no lc.fits were downloaded, move on
            print("No lc.fits files available for TIC ", targ)
            time1 = 0
            i1 = 0
            ticid = 0
        else: #if there are lc.fits files, open them and get the goods
                #get the goods and then close it
            f = fits.open(filepaths[0], memmap=False)
            time1 = f[1].data['TIME']
            i1 = f[1].data['PDCSAP_FLUX']
            ticid = f[1].header["TICID"]
            quality = f[1].data['QUALITY']
            f.close()
            
            # >> mask out any nonzero points
            flagged_inds = np.nonzero(quality)
            i1[flagged_inds] = np.nan # >> will be interpolated later
                  
        #then delete all downloads in the folder, no matter what type
        if os.path.isdir(yourpath + "mastDownload") == True:
            shutil.rmtree(yourpath + "mastDownload")
            print("Download folder deleted.")
            
        #corrects for connnection errors
    except (ConnectionError, OSError, TimeoutError, RemoteServiceError):
        print(targ, " could not be accessed due to an error.")
        i1 = 0
        time1 = 0
        ticid = 0
    
    return time1, i1, ticid

def lc_from_bulk_download(fits_path, target_list, fname_out, fname_targets,
                          fname_notes, path, fname_flagged,
                          custom_mask=[], apply_nan_mask=False):
    '''This function opens each _lc.fits file in fits_path, masks flagged data
    points from QUALITY and saves interpolated PDCSAP_FLUX, TIME and TICID to
    fits file fname_out.
    Parameters:
        * fits_path : directory containing all light curve fits files 
                      (ending with '/')
        * target_list : returned by lc_by_camera_ccd()
        * fname_out : name of fits file to save time, flux and ticids into
        * fname_targets : saves ticid of every target saved 
        * fname_notes : saves the ticid of any target it fails on
        * path : directory to save nan mask plots in
        * fname_flagged : name of fits file to save flagged light curves
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
        with fits.open(fits_path + file, mmap=False) as hdul:
            hdu_data = hdul[1].data
            
            # >> get time array (only for the first light curve)
            if n == 0: 
                time = hdu_data['TIME']
                
            # >> get flux array
            i = hdu_data['PDCSAP_FLUX']
            intensity.append(i)
            
            # >> get quality mask
            quality = hdu_data['QUALITY']
            flagged_inds = np.nonzero(quality)
            i[flagged_inds] = np.nan
            
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
    intensity_interp, time, ticid_interp, flagged, ticid_flagged = \
        interpolate_all(intensity, time, ticid_list, custom_mask=custom_mask,
                        apply_nan_mask=apply_nan_mask)
    
    # >> save time array, intensity array and ticids to fits file
    print('Saving to fits file...')
    
    hdr = fits.Header()
    hdu = fits.PrimaryHDU(time, header=hdr)
    hdu.writeto(fname_out)
    fits.append(fname_out, intensity_interp)
    fits.append(fname_out, ticid_interp)
    # >> actually i'm going to save the raw intensities just in case
    fits.append(fname_out, intensity)
    
    # >> save flagged
    if np.shape(flagged)[0] != 0:
        hdr = fits.Header()
        hdu = fits.PrimaryHDU(flagged, header=hdr)
        hdu.writeto(fname_flagged)
        fits.append(fname_flagged, ticid_flagged)
    
    print("lc_from_bulk_download has finished running")
    return time, intensity_interp, ticid_interp, flagged, ticid_flagged

def tic_list_by_magnitudes(path, lowermag, uppermag, n, filelabel):
    """ Creates a fits file of the first n TICs that fall between the given
    magnitude ranges. 
    parameters: 
        * path to where you want things saved
        * lower magnitude limit
        * upper magnitude limit
        * n - number of TICs you want
        * file label (what to call the fits file)
    modified [lcg 07082020]
    """
    catalog_data = Catalogs.query_criteria(catalog="Tic", Tmag=[uppermag, lowermag], objType="STAR")

    T_mags = np.asarray(catalog_data["Tmag"], dtype= float)
    TICIDS = np.asarray(catalog_data["ID"], dtype = int)
    
    tmag_index = np.argsort(T_mags)
    
    sorted_tmags = T_mags[tmag_index]
    sorted_ticids = TICIDS[tmag_index]
    
    hdr = fits.Header() # >> make the header
    hdu = fits.PrimaryHDU(sorted_ticids[0:n], header=hdr)
    hdu.writeto(path + filelabel + ".fits")
    fits.append(path + filelabel + ".fits",sorted_tmags[0:n])
    
    return sorted_ticids, sorted_tmags


#normalizing each light curve
def normalize(flux, axis=1):
    '''Dividing by median.
    !!Current method blows points out of proportion if the median is too close to 0?'''
    medians = np.median(flux, axis = axis, keepdims=True)
    flux = flux / medians - 1.
    return flux

def mean_norm(flux, axis=1): 
    """ normalizes by dividing by mean - necessary for TLS running 
    modified lcg 07192020"""
    means = np.mean(flux, axis = axis, keepdims=True)
    flux = flux / means
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



#interpolate and sigma clip
def interpolate_all(flux, time, ticid, flux_err=False, interp_tol=20./(24*60),
                    num_sigma=10, k=3, DEBUG_INTERP=False, output_dir='./',
                    apply_nan_mask=False, DEBUG_MASK=False, custom_mask=[]):
    '''Interpolates each light curves in flux array.'''
    
    flux_interp = []
    ticid_interp = []
    flagged = []
    ticid_flagged = []
    for i in range(len(flux)):
        i_interp, flag = interpolate_lc(flux[i], time, flux_err=flux_err,
                                        interp_tol=interp_tol,
                                        num_sigma=num_sigma, k=k,
                                        DEBUG_INTERP=DEBUG_INTERP,
                                        output_dir=output_dir,
                                        prefix=str(i)+'-')
        if not flag:
            flux_interp.append(i_interp)
            ticid_interp.append(ticid[i])
        else:
            flagged.append(i_interp)
            ticid_flagged.append(ticid[i])
            print('Spline interpolation failed!')
    
    if apply_nan_mask:
        flux_interp, time = nan_mask(flux_interp, time, DEBUG=DEBUG_MASK,
                                     output_dir=output_dir, ticid=ticid_interp,
                                     custom_mask=custom_mask)
    
    return np.array(flux_interp), time, np.array(ticid_interp), \
            np.array(flagged), np.array(ticid_flagged)

def interpolate_lc(i, time, flux_err=False, interp_tol=20./(24*60),
                   num_sigma=10, k=3, search_range=200, med_tol=2,
                   DEBUG_INTERP=False,
                   output_dir='./', prefix=''):
    '''Interpolation for one light curve. Linearly interpolates nan gaps less
    than 20 minutes long. Spline interpolates nan gaps more than 20 minutes
    long (and shorter than orbit gap)
    Parameters:
        * i : intensity array, shape=(n)
        * time : time array, shape=(n)
        * interp_tol : if nan gap is less than interp_tol days, then will
                       linear interpolate
        * num_sigma : number of sigma to clip
        * k : power of spline
        * search_range : number of data points around interpolate region to 
                         calculate the local standard deviation and median
        * med_tol : checks if median of interpolate region is between
                    med_tol*(local median) and (local median)/med_tol
    
    example code snippet
    import data_functions as df
    from astropy.io import fits
    f = fits.open('tess2019306063752-s0018-0000000005613228-0162-s_lc.fits')
    i_interp, flag = df.interpolate_lc(f[1].data['PDCSAP_FLUX'],
                                      f[1].data['TIME'], DEBUG_INTERP=True)
    '''
    from astropy.stats import SigmaClip
    from scipy import interpolate
    
    # >> plot original light curve
    if DEBUG_INTERP:
        fig, ax = plt.subplots(4, 1, figsize=(8, 3*5))
        ax[0].plot(time, i, '.k')
        ax[0].set_title('original')
        
    # >> get spacing in time array
    dt = np.nanmin( np.diff(time) )
    
    # -- sigma clip -----------------------------------------------------------
    sigclip = SigmaClip(sigma=num_sigma, maxiters=None, cenfunc='median')
    clipped_inds = np.nonzero(np.ma.getmask(sigclip(i, masked=True)))
    i[clipped_inds] = np.nan
    if DEBUG_INTERP:
        time_plot = np.linspace(np.nanmin(time), np.nanmax(time), len(time))
        ax[1].plot(time_plot, i, '.k')
        ax[1].set_title('clipped')
    
    # -- locate nan gaps ------------------------------------------------------
    # >> find all runs
    n = np.shape(i)[0]
    loc_run_start = np.empty(n, dtype=bool)
    loc_run_start[0] = True
    np.not_equal(np.isnan(i)[:-1], np.isnan(i)[1:], out=loc_run_start[1:])
    run_starts = np.nonzero(loc_run_start)[0]

    # >> find nan window lengths
    run_lengths = np.diff(np.append(run_starts, n))
    
    # >> find nan windows
    nan_inds = np.nonzero(np.isnan(i[run_starts]))
    run_starts = run_starts[ nan_inds ]
    run_lengths = run_lengths[ nan_inds ]
    
    # >> create x array
    # !! TODO remove end NaN window from x array
    x = np.arange(len(i))
    
    # >> remove nan windows at the beginning and end
    if run_starts[0] == 0:
        run_starts = np.delete(run_starts, 0)
        run_lengths = np.delete(run_lengths, 0)
    if run_starts[-1] + run_lengths[-1] == len(i):
        x = np.delete(x, range(run_starts[-1], run_starts[-1]+run_lengths[-1]))
        run_starts = np.delete(run_starts, -1)
        run_lengths = np.delete(run_lengths, -1)
        
    # >> remove orbit gap from list
    orbit_gap_ind = np.argmax(run_lengths)
    orbit_gap_start = run_starts[ orbit_gap_ind ]
    orbit_gap_end = orbit_gap_start + run_lengths[ orbit_gap_ind ]    
    run_starts = np.delete(run_starts, orbit_gap_ind)
    run_lengths = np.delete(run_lengths, orbit_gap_ind)
    
    # -- fit a spline ---------------------------------------------------------
    
    # >> get all non-nan points
    num_inds = np.nonzero( (~np.isnan(i)) * (~np.isnan(time)) )[0]
    
    # >> fit spline to non-nan points
    ius = interpolate.InterpolatedUnivariateSpline(num_inds, i[num_inds],
                                                   k=k)
    
    if DEBUG_INTERP:
        x_plot = np.delete(x, range(num_inds[-1], len(x)))
        x_plot = np.delete(x_plot, range(orbit_gap_start, orbit_gap_end))
        x_plot = np.delete(x_plot, range(0, num_inds[0]))
        ax[2].plot(x_plot, ius(x_plot), '.k')
        ax[2].set_title('spline')    
    
    # -- interpolate nan gaps -------------------------------------------------
    i_interp = np.copy(i)
    # rms_lc = np.sqrt(np.mean(i[num_inds]**2)) # >> RMS of entire light curve
    # avg_lc = np.mean(i[num_inds])
    # std_lc = np.std(i[num_inds])
    # >> loop through each orbit gap
    for a in range(len(run_starts)):
        
        flag=False
        if run_lengths[a] * dt > interp_tol: # >> spline interpolate
            start = run_starts[a]
            end = run_starts[a] + run_lengths[a]
            spline_interp = \
                ius(x[start : end])
               
            # >> compare std, median of interpolate region to local std, median
            std_local = np.mean([np.nanstd(i[start-search_range : start]),
                                 np.nanstd(i[end : end+search_range])])
            med_local = np.mean([np.nanmedian(i[start-search_range : start]),
                                 np.nanmedian(i[end : end+search_range])])
            
            if np.std(spline_interp) > std_local or \
                np.median(spline_interp) > med_tol*med_local or\
                    np.median(spline_interp) < med_local/med_tol:
                
            # # >> check if RMS of interpolated region is crazy
            # rms_interp = np.sqrt(np.mean(spline_interp**2))
            # avg_interp = np.mean(spline_interp)
            # # if rms_interp > 1.25*rms_lc: # !! factor
            # if avg_interp > avg_lc+std_lc or avg_interp < avg_lc-std_lc:
                flag=True
            else:
                i_interp[run_starts[a] : run_starts[a] + run_lengths[a]] =\
                    spline_interp
                
        if run_lengths[a] * dt < interp_tol or flag: # >> linear interpolate
            i_interp[run_starts[a] : run_starts[a] + run_lengths[a]] = \
                np.interp(x[run_starts[a] : run_starts[a] + run_lengths[a]],
                          x[num_inds], i[num_inds])
            flag=False
                
    if DEBUG_INTERP:
        ax[3].plot(time_plot, i_interp, '.k')
        ax[3].set_title('interpolated')
        fig.tight_layout()
        fig.savefig(output_dir + prefix + 'interpolate_debug.png',
                    bbox_inches='tight')
        plt.close(fig)        
        
    return i_interp, flag
    
def nan_mask(flux, time, flux_err=False, DEBUG=False, debug_ind=1042,
             ticid=False, target_info=False,
             output_dir='./', prefix='', tol1=0.05, tol2=0.1,
             custom_mask=[]):
    '''Apply nan mask to flux and time array.
    Returns masked, homogenous flux and time array.
    If there are only a few (less than tol1 light curves) light curves that
    contribute (more than tol2 data points are NaNs) to NaN mask, then will
    remove those light curves.
    Parameters:
        * flux : shape=(num light curves, num data points)
        * time : shape=(num data points)
        * flux_err : shape=(num light curves, num data points)
        * ticid : shape=(num light curves)
        * target_info : shape=(num light curves, 5)
        * tol1 : given as fraction of num light curves, determines whether to
          remove the NaN-iest light curves, or remove NaN regions from all
          light curves
        * tol2 : given as fraction of num data points
        * custom_mask : list of indicies to remove from all light curves
    '''
    # >> apply custom NaN mask
    if len(custom_mask) > 0: print('Applying custom NaN mask')
    time = np.delete(time, custom_mask)
    flux = np.delete(flux, custom_mask, 1)

    mask = np.nonzero(np.prod(~np.isnan(flux), axis = 0) == False)
    # >> plot histogram of number of data points thrown out
    num_nan = np.sum( np.isnan(flux), axis=1 )

    def count_masked(x):
        '''Counts number of masked data points for one light curve.'''
        return np.count_nonzero( ~np.isin(mask, np.nonzero(x)) )
    num_masked = np.apply_along_axis(count_masked, axis=1, arr=np.isnan(flux))
    
    plt.figure()
    plt.hist(num_masked, bins=50)
    plt.ylabel('number of light curves')
    plt.xlabel('number of data points masked')
    plt.savefig(output_dir + 'nan_mask.png')
    plt.close()
    
    # >> debugging plots
    if DEBUG:
        fig, ax = plt.subplots()
        ax.plot(time, flux[debug_ind], '.k')
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
                ax[i].plot(time, flux[ind], '.k')
                pf.ticid_label(ax[i], ticid[ind], target_info[ind], title=True)
                num_nans = np.count_nonzero(np.isnan(flux[ind]))
                ax[i].text(0.98, 0.98, 'Num NaNs: '+str(num_nans)+\
                           '\nNum masked: '+str(num_masked[ind]),
                           transform=ax[i].transAxes,
                           horizontalalignment='right',
                           verticalalignment='top', fontsize='xx-small')
            if k == 0:
                fig.tight_layout()
                fig.savefig(output_dir + prefix + 'nanmask_top.png',
                            bbox_inches='tight')
            else:
                fig.tight_layout()
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
    
    # # >> will need to truncate if using multiple sectors
    # new_length = np.min([np.shape(i)[1] for i in flux])
    
    if type(flux_err) != bool:
        flux_err = np.delete(flux_err, mask, 1)
        flux_err = np.delete(flux_err, custom_mask, 1)
        return flux, time, flux_err
    else:
        return flux, time
    
# Target-Wise Metafile Production ----------------------------------
def targetwise_data_access_by(yourpath, target_list, startindex, increment, filelabel):
    """given a list of TICIDs, accesses SPOC light curve for those targets,
    saves the time axis and the intensity into a fits file, and a list of all
    the ticids contained within the file in order. 
    parameters:
        * yourpath = where you want it to be saved
        * target_list = list of TICIDS as integers
        * start index = what index you want to start your search at. can be 
        useful if this stops running for whatever reason and you need to pick up
        again where you left of
        * increment = how many TICIDs maximum per file. 100-500 is pretty good, 
        1000+ is risky simply because of astroquery's crashes
        * filelabel = what you want the output subfolder + files to be called
    returns: path to where everything is saved. 
    requires: targetwise_lc()
    modified [lcg 07112020]
        """
    # produce the folder to save everything into and set up file names
    path = yourpath + filelabel
    fname_notes = path + "/" + filelabel + "_group_notes.txt"
    
    try:
        os.makedirs(path)
        print ("Successfully created the directory %s" % path) 
        with open(fname_notes, 'a') as file_object:
            file_object.write("This file contains group notes, including any TICs that could not be accessed.\n")

        n = startindex

        m = n + increment
        
        while m < len(target_list):
            targets_search = target_list[n:m]
        
            fname_time_intensities = path + "/" + filelabel + "_lightcurves"+str(n) + "-" + str(m)+".fits"
            ticids = lc_from_target_list_diffsectors(yourpath, targets_search,
                                                    fname_time_intensities,
                                                    fname_notes)
            print("finished", m)
        
            n = n + increment
            m = m + increment
        
        
    except OSError: #if there is an error creating the folder
        print("There was an OS Error trying to create the folder. Check to see if data is already saved there")

    return path



def targetwise_lc(yourpath, target_list, fname_time_intensities,fname_notes):
    """ runs getting the files and data for all targets on the list
    then appends the time & intensity arrays and the TIC number into text files
    that can later be accessed
    parameters: 
        * yourpath = folder into which things will get saved
        * target_list = list of ticids, as integers
        * fname_time_intensities = direct path to the file to save into
        * fname_notes = direct path to file to save TICIDS of targets that 
            return no data into
    returns: list of ticids as an array
    requires: get_lc_file_and_data(), interpolate_lc()
    modified [lcg 07112020]
    """

    ticids = []
    for n in range(len(target_list)): #for each item on the list
        
        if n == 0: #for the first target only do you need to get the time index
            target = target_list[0] #get that target number
            time1, i1, tic = get_lc_file_and_data(yourpath, target) #grab that data
            
            if type(i1) == np.ndarray: #if the data IS data
                i_interp, flag = interpolate_lc(i1, time1, flux_err=False, interp_tol=20./(24*60),
                                   num_sigma=10, DEBUG_INTERP=False,
                                   output_dir=yourpath, prefix='')
                TI = [time1, i1]
                TI_array = np.asarray(TI)
                hdr = fits.Header() # >> make the header
                hdu = fits.PrimaryHDU(TI_array, header=hdr)
                hdu.writeto(fname_time_intensities)
                ticids.append(tic)
                
            else: #if the data is NOT a data
                print("First target failed, no time index was saved")
                with open(fname_notes, 'a') as file_object:
                    file_object.write("\n")
                    file_object.write(str(int(target)))
        else: 
            target = target_list[n] #get that target number
            time1, i1, tic = get_lc_file_and_data(yourpath, target) #grab that data
            
            if type(i1) == np.ndarray: #if the data IS data
                i_interp, flag = interpolate_lc(i1, time1, flux_err=False, interp_tol=20./(24*60),
                                   num_sigma=10, DEBUG_INTERP=False,
                                   output_dir=yourpath, prefix='')
                TI = [time1, i1]
                TI_array = np.asarray(TI)
                fits.append(fname_time_intensities, TI_array)
                ticids.append(tic)
                
            else: #if the data is NOT a data
                print("Target failed to return a light curve")
                with open(fname_notes, 'a') as file_object:
                    file_object.write("\n")
                    file_object.write(str(int(target)))
        print(n, " completed")
    fits.append(fname_time_intensities, np.asarray(ticids))
        
    print("lc_from_target_list has finished running")
    return np.asarray(ticids)

#Feature Vector Production -----------------------------

def create_save_featvec(yourpath, times, intensities, filelabel, version=0, save=True):
    """Produces the feature vectors for each light curve and saves them all
    into a single fits file. requires all light curves on the same time axis
    parameters:
        * yourpath = folder you want the file saved into
        * times = a single time axis for all 
        * intensities = array of all light curves (NOT normalized)
        * sector, camera, ccd = integers 
        * version = what version of feature vector to calculate for all. 
            default is 0
        * save = whether or not to save into a fits file
    returns: list of feature vectors + fits file containing all feature vectors
    requires: featvec()
    modified: [lcg 07112020]"""
    

    fname_features = yourpath + "/"+ filelabel + "_features_v"+str(version)+".fits"
    feature_list = []
    if version == 0:
	#median normalize for the v0 features
        intensities = normalize(intensities)
    elif version == 1: 
        from transitleastsquares import transitleastsquares
	#mean normalize the intensity so goes to 1
        intensities = mean_norm(intensities)

    print("Begining Feature Vector Creation Now")
    for n in range(len(intensities)):
        feature_vector = featvec(times, intensities[n], v=version)
        feature_list.append(feature_vector)
        
        if n % 25 == 0: print(str(n) + " completed")
    
    feature_list = np.asarray(feature_list)
    
    if save == True:
        hdr = fits.Header()
        hdr["VERSION"] = version
        hdu = fits.PrimaryHDU(feature_list, header=hdr)
        hdu.writeto(fname_features)
    else: 
        print("Not saving feature vectors to fits")
    
    return feature_list

def featvec(x_axis, sampledata, v=0): 
    """calculates the feature vector of a single light curve
        version 0: features 0-15
        version 1: features 0-19
        0 - Average -  lower alpha
        1 - Variance - upper case beta (B)
        2 - Skewness - upper case gamma
        3 - Kurtosis - upper case delta
        
        4 - ln variance - lowercase beta
        5 - ln skewness - lowercase gamma
        6 - ln kurtosis - lowercase delta
        
        (over 0.1 to 10 days)
        7 - maximum power - upper episolon (E)
        8 - ln maximum power - lower epsilon
        9 - period of maximum power - upper zeta (Z)
        
        10 - slope - upper eta (H)
        11 - ln slope - lower eta
        
        (integration of periodogram over time frame)
        12 - P0 - 0.1-1 days - upper theta
        13 - P1 - 1-3 days - upper iota
        14 - P2 - 3-10 days - upper kappa
        
        (over 0-0.1 days, for moving objects)
        15 - Period of max power - upper lambda
        
        (from transitleastsquares, OPTIONAL based on tls argument)
        16 - best fit period - upper mu (M) - days
        17 - best fit duration - lower mu - days
        18 - best fit depth - upper nu (N) - ppt, measured from bottom
        19 - power of best fit period - lower nu 
        
        for title purposes: 
        features_greek = [r'$\alpha$', 'B', r'$\Gamma$', r'$\Delta$', r'$\beta$', r'$\gamma$',r'$\delta$',
                  "E", r'$\epsilon$', "Z", "H", r'$\eta$', r'$\Theta$', "I", "K", r'$\Lambda$', "M", r'$\mu$'
                  ,"N", r'$\nu$']

	*** version 1 note: you may wish to go into the transitleastsquares's main.py file and
	comment out all 'print' statements in order to save space while running this over lots of light curves
        modified [lcg 07202020]"""
    #empty feature vector
    featvec = []
    if v==0:
        
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
    elif v == 1: 
        model = transitleastsquares(x_axis, sampledata)
        results = model.power(show_progress_bar=False)
        featvec.append(results.period)
        featvec.append(results.duration)
        featvec.append((1 -results.depth))
        featvec.append((results.power.max()))
    
    return featvec 

def feature_gen_from_lc_fits(path, sector, feature_version=0):
    """Given a path to a folder containing ALL the light curve metafiles 
    for a sector, produces the feature vector metafile for each group and then
    one main feature vector metafile containing ALL the features in [0] and the
    TICIDS in [1]. 
    Parameters: 
        * folderpath to where the light curve metafiles are saved
            *must end in a backslash
        * sector number
        * what version of features you want generated (default is 0)
    modified [lcg 07112020]"""
    
    import datetime
    from datetime import datetime
    
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Starting Feature Generation at", dt_string)	

    ticids_all = [10]
    ticids_all = np.asarray(ticids_all)
    sector = sector
    for n in range(1,5):
        camera = int(n)
        for m in range(1,5):
            ccd = int(m)
            file_label = "Sector" + str(sector) + "Cam" + str(camera) + "CCD" + str(ccd)
            folderpath = path + "/" + file_label + "/"

            t, i1, targets = load_group_from_fits(folderpath, sector, camera, ccd)
            ticids_all = np.concatenate((ticids_all, targets))
                    
            i2, t2 = nan_mask(i1, t, flux_err=False, DEBUG=False, debug_ind=1042,
                                ticid=False, output_dir=folderpath, prefix='', tol1=0.05, tol2=0.1)
                        
            i3 = normalize(i2, axis=1)
               
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            print("Starting feature vectors for camera ", camera, "ccd ", ccd, "at ", dt_string)
            
            create_save_featvec(folderpath, t2, i3, file_label, version=0, save=True)
    
    ticids_all = ticids_all[1:]
    feats_all = np.zeros((2,16))

    #make main listing
    for n in range(1,5):
        camera = int(n)
        for m in range(1,5):
            ccd = int(m)
            file_label = "Sector" + str(sector) + "Cam" + str(camera) + "CCD" + str(ccd)
            folderpath = path + "/" + file_label + "/"
            f = fits.open(folderpath + file_label + "_features.fits", mmap=False)
            feats = f[0].data
            feats_all = np.concatenate((feats_all, feats))
            f.close()
            #print(n,m)
    
    feats_all = feats_all[2:]
    
    hdr = fits.Header() # >> make the header
    hdr["Sector"] = sector
    hdr["Version"] = feature_version
    hdr["Date"] = str(datetime.now())
    #hdr["Creator"] = "L. Gordon"
    hdu = fits.PrimaryHDU(feats_all, header=hdr)
    hdu.writeto(folderpath + "Sector"+str(sector) + "_features_v" + str(feature_version) +"_all.fits")
    fits.append(folderpath + "Sector"+str(sector) + "_features_v" + str(feature_version) +"_all.fits", ticids_all)
    
    return feats_all, ticids_all

def get_tess_features(ticid):
    '''Query catalog data https://arxiv.org/pdf/1905.10694.pdf'''
    from astroquery.mast import Catalogs

    target = 'TIC '+str(int(ticid))
    catalog_data = Catalogs.query_object(target, radius=0.02, catalog='TIC')
    Teff = catalog_data[0]["Teff"]

    rad = catalog_data[0]["rad"]
    mass = catalog_data[0]["mass"]
    GAIAmag = catalog_data[0]["GAIAmag"]
    d = catalog_data[0]["d"]
    # Bmag = catalog_data[0]["Bmag"]
    # Vmag = catalog_data[0]["Vmag"]
    objType = catalog_data[0]["objType"]
    Tmag = catalog_data[0]["Tmag"]
    # lum = catalog_data[0]["lum"]

    return target, Teff, rad, mass, GAIAmag, d, objType, Tmag

def get_tess_feature_txt(ticid_list, out='./tess_features_sectorX.txt'):
    '''Queries 'TESS features' (i.e. Teff, rad, mass, GAIAmag, d) for each
    TICID and saves to text file.
    
    Can get ticid_list with:
    with open('all_targets_S019_v1.txt', 'r') as f:
        lines = f.readlines()
    ticid_list = []
    for line in lines[6:]:
        ticid_list.append(int(line.split()[0]))
    '''
    
    # !! 
    # TESS_features = []        
    for i in range(len(ticid_list)):
        print(i)
        try:
            features = get_tess_features(ticid_list[i])
            # TESS_features.append(features)
            with open(out, 'a') as f:
                f.write(' '.join(map(str, features)) + '\n')
        except:
            with open('./failed_get_tess_features.txt', 'a') as f:
                f.write(str(ticid_list[i])+'\n')


    
def build_simbad_database(out='./simbad_database.txt'):
    '''Object type follows format in:
    http://vizier.u-strasbg.fr/cgi-bin/OType?$1'''
    
    # -- querying object type -------------------------------------------------
    customSimbad = Simbad()
    # customSimbad.get_votable_fields()
    customSimbad.add_votable_fields('otype')
    
    # -- querying TICID for each object ---------------------------------------
    # >> first get all the TESS objects in the Simbad database
    res = customSimbad.query_catalog('tic')
    objects = list(res['MAIN_ID'])

    # >> now loop through all of the objects
    for i in range(len(objects)):
        # >> decode bytes object to convert to string
        obj = objects[i].decode('utf-8')
        bibcode = res['COO_BIBCODE'][i].decode('utf-8')
        otype = res['OTYPE'][i].decode('utf-8')
        
        print(obj + ' ' + otype)
        
        # >> now query TICID
        obs_table = Observations.query_criteria(obs_collection='TESS',
                                                dataproduct_type='timeseries',
                                                objectname=obj)
        
        ticids = obs_table['target_name']
        for ticid in ticids:
            with open(out, 'a') as f:
                f.write(ticid + ',' + obj + ',' + otype + ',' + bibcode + '\n')
                
def get_simbad_classifications(ticid_list,
                               simbad_database_txt='./simbad_database.txt'):
    '''Query Simbad classification and bibcode from .txt file (output from
    build_simbad_database).
    Returns a list where simbad_info[i] = [ticid, main_id, obj type, bibcode]
    Object type follows format in:
    http://vizier.u-strasbg.fr/cgi-bin/OType?$1
    '''
    ticid_simbad = []
    main_id_list = []
    otype_list = []
    bibcode_list = []
    with open(simbad_database_txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            ticid, main_id, otype, bibcode = line[:-2].split(',')
            ticid_simbad.append(int(ticid)) 
            main_id_list.append(main_id)
            otype_list.append(otype)
            bibcode_list.append(bibcode)
    intersection, comm1, comm2 = np.intersect1d(ticid_list, ticid_simbad,
                                                return_indices=True)
    simbad_info = []
    for i in comm2:
        simbad_info.append([ticid_simbad[i], main_id_list[i], otype_list[i],
                            bibcode_list[i]])
    return simbad_info

def get_true_classifications(ticid_list,
                             database_dir='./databases/'):
    '''Query classifications and bibcode from *_database.txt file.
    Returns a list where class_info[i] = [ticid, obj type, bibcode]
    Object type follows format in:
    http://vizier.u-strasbg.fr/cgi-bin/OType?$1
    '''
    ticid_classified = []
    class_info = []
    
    # >> find all text files in directory
    fnames = fm.filter(os.listdir(database_dir), '*.txt')
    
    for fname in fnames:
        # >> read text file
        with open(database_dir + fname, 'r') as f:
            lines = f.readlines()
            for line in lines:
                ticid, otype, bibcode = line[:-2].split(',')
                
                # >> only get classifications for ticid_list, avoid repeats
                if float(ticid) in ticid_list and ticid not in ticid_classified:
                    ticid_classified.append(ticid)
                    class_info.append([int(ticid), otype, bibcode])
    return np.array(class_info)
                           
def dbscan_param_search(bottleneck, time, flux, ticid, target_info,
                            eps=list(np.arange(0.1,1.5,0.1)),
                            min_samples=[5],
                            metric=['euclidean', 'manhattan', 'minkowski'],
                            algorithm = ['auto', 'ball_tree', 'kd_tree',
                                         'brute'],
                            leaf_size = [30, 40, 50],
                            p = [1,2,3,4],
                            output_dir='./', DEBUG=False,
                            simbad_database_txt='./simbad_database.txt',
                            database_dir='./databases/', pca=True, tsne=True,
                            confusion_matrix=True):
    '''Performs a grid serach across parameter space for DBSCAN. Calculates
    
    Parameters:
        * bottleneck : array with shape=(num light curves, num features)
            ** is this just the array of features? ^
        * eps, min_samples, metric, algorithm, leaf_size, p : all DBSCAN
          parameters
        * success metric : !!
        * output_dir : output directory, ending with '/'
        * DEBUG : if DEBUG, plots first 5 light curves in each class
        
    TODO : only loop over p if metric = 'minkowski'
    '''
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.metrics import davies_bouldin_score      
    classes = []
    num_classes = []
    counts = []
    num_noisy= []
    parameter_sets=[]
    silhouette_scores=[]
    ch_scores = []
    db_scores = []
    accuracy = []

    with open(output_dir + 'dbscan_param_search.txt', 'a') as f:
        f.write('{} {} {} {} {} {} {} {} {} {} {}\n'.format("eps\t\t", "samp\t\t", "metric\t\t", 
                                                         "alg\t\t", "leaf\t", "p\t",
                                                         "classes\t",
                                                         "silhouette\t\t\t", 'ch\t\t\t', 
                                                         'db\t\t\t', 'acc\t'))

    for i in range(len(eps)):
        for j in range(len(min_samples)):
            for k in range(len(metric)):
                for l in range(len(algorithm)):
                    for m in range(len(leaf_size)):
                        if metric[k] == 'minkowski':
                            p = [1,2,3,4]
                        else:
                            p = [None]
                        for n in range(len(p)):
                            db = DBSCAN(eps=eps[i],
                                        min_samples=min_samples[j],
                                        metric=metric[k],
                                        algorithm=algorithm[l],
                                        leaf_size=leaf_size[m],
                                        p=p[n]).fit(bottleneck)
                            #print(db.labels_)
                            print(np.unique(db.labels_, return_counts=True))
                            classes_1, counts_1 = \
                                np.unique(db.labels_, return_counts=True)
                                
                            param_num = str(len(parameter_sets)-1)
                            title='Parameter Set '+param_num+': '+'{} {} {} {} {} {}'.format(eps[i],
                                                                                        min_samples[j],
                                                                                        metric[k],
                                                                                        algorithm[l],
                                                                                        leaf_size[m],
                                                                                        p[n])
                            
                            prefix='dbscan-p'+param_num                            
                                
                            if confusion_matrix:
                                acc = pf.plot_confusion_matrix(ticid, db.labels_,
                                                               database_dir=database_dir,
                                                               output_dir=output_dir,
                                                               prefix=prefix)
                            else:
                                acc = np.nan
                            accuracy.append(acc)
                                
                            if len(classes_1) > 1:
                                classes.append(classes_1)
                                num_classes.append(len(classes_1))
                                counts.append(counts_1)
                                num_noisy.append(counts_1[0])
                                parameter_sets.append([eps[i], min_samples[j],
                                                       metric[k],
                                                       algorithm[l],
                                                       leaf_size[m],
                                                       p[n]])
                                
                                # >> compute silhouette
                                silhouette = silhouette_score(bottleneck,db.labels_)
                                silhouette_scores.append(silhouette)
                                
                                # >> compute calinski harabasz score
                                ch_score = calinski_harabasz_score(bottleneck,
                                                                db.labels_)
                                ch_scores.append(ch_score)
                                
                                # >> compute davies-bouldin score
                                dav_boul_score = davies_bouldin_score(bottleneck,
                                                             db.labels_)
                                db_scores.append(dav_boul_score)
                                
                            else:
                                silhouette, ch_score, dav_boul_score = \
                                    np.nan, np.nan, np.nan
                                
                            with open(output_dir + 'dbscan_param_search.txt', 'a') as f:
                                f.write('{}\t\t {}\t\t {}\t\t {}\t {}\t \
                                        {}\t {}\t\t\t {}\t\t\t {}\t\t\t {}\t {}\n'.format(eps[i],
                                                                   min_samples[j],
                                                                   metric[k],
                                                                   algorithm[l],
                                                                   leaf_size[m],
                                                                   p[n],
                                                                   len(classes_1),
                                                                   silhouette,
                                                                   ch_score,
                                                                   dav_boul_score,
                                                                   acc))
                                
                            if DEBUG and len(classes_1) > 1:

                                pf.quick_plot_classification(time, flux,
                                                             ticid,
                                                             target_info, bottleneck,
                                                             db.labels_,
                                                             path=output_dir,
                                                             prefix=prefix,
                                                             simbad_database_txt=simbad_database_txt,
                                                             title=title,
                                                             database_dir=database_dir)
                                
                                
                                if pca:
                                    print('Plot PCA...')
                                    pf.plot_pca(bottleneck, db.labels_,
                                                output_dir=output_dir,
                                                prefix=prefix)
                                
                                if tsne:
                                    print('Plot t-SNE...')
                                    pf.plot_tsne(bottleneck, db.labels_,
                                                 output_dir=output_dir,
                                                 prefix=prefix)
                            plt.close('all')
    print("Plot paramscan metrics...")
    pf.plot_paramscan_metrics(output_dir, parameter_sets, 
                              silhouette_scores, db_scores, ch_scores)
    #print(len(parameter_sets), len(num_classes), len(num_noisy), num_noisy)

    pf.plot_paramscan_classes(output_dir, parameter_sets, 
                                  np.asarray(num_classes), np.asarray(num_noisy))

        
    return parameter_sets, num_classes, silhouette_scores, db_scores, ch_scores, accuracy

def load_paramscan_txt(path):
    """ load in the paramscan stuff from the text file
    returns: parameter sets, number of classes, metric scores (in order: silhouettte, db, ch)
    modified [lcg 07292020 - created]"""
    params = np.genfromtxt(path, dtype=(float, int, 'S10', 'S10', int, int, int, np.float32, np.float32, np.float32), names=['eps', 'minsamp', 'metric', 'algorithm', 'leafsize', 'p', 'numclasses', 'silhouette', 'ch', 'db'])
    
    params = np.asarray(params)
    nan_indexes = []
    for n in range(len(params)):
        if np.isnan(params[n][8]):
            nan_indexes.append(int(n))
        
    nan_indexes = np.asarray(nan_indexes)
    
    cleaned_params = np.delete(params, nan_indexes, axis=0)   

    number_classes = np.asarray(cleaned_params['numclasses'])
    metric_scores = np.asarray(cleaned_params[['silhouette', 'db', 'ch']].tolist())
    
    return cleaned_params, number_classes, metric_scores

def hdbscan_param_search(bottleneck, time, flux, ticid, target_info,
                         min_cluster_size=list(range(2,10)),
                         min_samples=list(range(2,10)),
                         metric=['euclidean'],
                         p_space=[1,2,3,4],
                         output_dir='./',
                         database_dir='./databases/'):
    import hdbscan
    # !! wider p range?
    
    param_num=0
    for i in range(len(min_cluster_size)):
        for j in range(len(min_samples)):
            for k in range(len(metric)):
                if metric[k] == 'minkowski':
                    p = p_space
                else:
                    p = [None]
                for l in range(len(p)):
                    param_num += 1
                    
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size[i],
                                                min_samples=min_samples[j],
                                                metric=metric[k])
                    clusterer.fit(bottleneck)
                    classes, counts = \
                        np.unique(clusterer.labels_, return_counts=True)    
                    print(classes, counts)
                    
                    title='Parameter Set '+param_num+': '+'{} {} {}'.format(min_cluster_size[i],
                                                                            min_samples[j],
                                                                            metric[k])
                    
                    prefix='dbscan-p'+param_num                         
                    
                    pf.quick_plot_classification(time, flux,
                                                ticid,
                                                target_info,
                                                clusterer.labels_,
                                                path=output_dir,
                                                prefix=prefix,
                                                title=title,
                                                database_dir=database_dir)
                    acc = pf.plot_confusion_matrix(ticid, clusterer.labels_,
                                                   database_dir=database_dir,
                                                   output_dir=output_dir,
                                                   prefix=prefix)        
                    pf.plot_pca(bottleneck, clusterer.labels_,
                                output_dir=output_dir, prefix=prefix)    
                    pf.plot_tsne(bottleneck, clusterer.labels_,
                                 output_dir=output_dir, prefix=prefix)                    
                        
                        
                        

# DEPRECIATED SECTION -----------------------------------------------------
def load_group_from_txt(sector, camera, ccd, path):
    """loads in a given group's data provided you have it saved in TEXT metafiles already
    path needs to be a string, ending with a forward slash
    camera, ccd, secotr all should be integers
    moved to depreciated 7/8/2020 by lcg
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

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # TESS_features = np.array(TESS_features)
    # hdr = fits.Header()
    # hdu = fits.PrimaryHDU(TESS_features[:,1:-1].astype('float'))
    # hdu.writeto(output_dir + 'tess_features.fits')
    # fits.append(output_dir + 'tess_features.fits', ticid_list)
    # fits.append(output_dir + 'tess_features.fits', TESS_features[:,-1])
            
# def get_abstracts(ticid_list):
#     import time
#     tables = []
#     for i in range(len(ticid_list)):
#         print(str(i) + '/' + str(len(ticid_list)) + '\n')
#         res = Simbad.query_object('TIC ' + str(int(ticid_list[i])))
#         if res == None:
#             pass
#         else:
#             tables.append(res)
#             print(ticid_list[i])
#             print(res)
#         time.sleep(6) # >> to avoid ConnectionError

            # end_ind_spl = np.argmin(np.abs(t_spl - time[end_ind+1]))
            # start_ind_spl = end_ind_spl - (end_ind-start_ind)

    
    # # -- spline interpolate large nan gaps -----------------------------------

    
    # # >> new time array (take out orbit gap)
    # # t_spl = np.copy(time)
    # # t_spl = np.delete(t_spl, range(num_inds[-1], len(t_spl)))
    # # t_spl = np.delete(t_spl, range(orbit_gap_start, orbit_gap_end))
    # # t_spl = np.delete(t_spl, range(num_inds[0]))

    
    # # >> spline fit for new time array
    # i_spl = ius(t_spl)
    
 
        # # >> find starting and ending time for nan gap
        # if not np.isnan(time[run_starts[a]]):
        #     start_ind = np.argmin(np.abs(t_interp - time[run_starts[a]]))
        #     end_ind = start_ind + run_lengths[a]
        # else:
        #     start_time = time[run_starts[a]-1] + dt
        #     start_ind = np.argmin(np.abs(t_interp - start_time))
        #     end_ind = start_ind + run_lengths[a]
            
        # # >> spline interpolate if large nan gap
        # if run_lengths[a] * dt > interp_tol:
        #     spline_interp = fitted_spline[start_ind:end_ind]
            
        #     # >> check if RMS of interpolated section is 5x larger than RMS of 
        #     # >> entire light curve        
        #     rms_lc = np.sqrt(np.mean(i**2))
        #     rms_interp = np.sqrt(np.mean(spline_interp))       
        #     if rms_lc > rms_interp*5.:
        #         i_interp[start_ind:end_ind] = spline_interp
        #         flag=False
        #     else:
        #         # flag=True
        #         flag = False # >> instead of flagging, linearly interpolate
        #         i_interp[start_ind:end_ind] = \
        #             np.interp(t_interp[start_ind:end_ind],
        #                       time[num_inds],
        #                       i[num_inds])  
        # else: # >> linearly interpolate if small nan gap
        #     i_interp[start_ind:end_ind] = \
        #         np.interp(t_interp[start_ind:end_ind],
        #                   time[num_inds],
        #                   i[num_inds])       
                
        #     pdb.set_trace()
            
    
    # # -- interpolate small nan gaps ------------------------------------------
    # interp_gaps = np.nonzero(run_lengths * dt <= interp_tol)
    # # interp_gaps = np.nonzero((run_lengths * tdim <= interp_tol) * \
    # #                          np.isnan(i[run_starts]))    
    # interp_inds = run_starts[interp_gaps]
    # interp_lens = run_lengths[interp_gaps]
    
    # i_interp = np.copy(i)
    # for a in range(np.shape(interp_inds)[0]):
    #     start_ind = interp_inds[a]
    #     end_ind = interp_inds[a] + interp_lens[a]
    #     i_interp[start_ind:end_ind] = np.interp(time[start_ind:end_ind],
    #                                             time[np.nonzero(~np.isnan(i))],
    #                   
    # # >> spline interpolate over remaining nan gaps
    # interp_gaps = np.nonzero( ~np.isin(run_starts, interp_inds) )
    # interp_inds = run_starts[interp_gaps]
    # interp_lens = run_lengths[interp_gaps]
        
    # # >> spline interpolate nan gaps
    # i_interp = np.copy(i)
    # for a in range(np.shape(interp_inds)[0]):
    #     start_ind = interp_inds[a]
    #     # end_ind   = interp_inds[a] + interp_lens[a] - 1
    #     end_ind = interp_inds[a] + interp_lens[a]

    #     if not np.isnan(time[start_ind]):
    #         start_ind_spl = np.argmin(np.abs(t_spl - time[start_ind]))
    #         end_ind_spl = start_ind_spl + (end_ind-start_ind)
    #     else:
    #         start_time = time[start_ind-1] + dt
    #         start_ind_spl = np.argmin(np.abs(t_spl - start_time))
    #         end_ind_spl = start_ind_spl + (end_ind-start_ind)
            
    #     spline_interp = i_spl[start_ind_spl:end_ind_spl]
            
    #     # >> check if RMS of interpolated section is 5x larger than RMS of 
    #     # >> entire light curve
    #     rms_lc = np.sqrt(np.mean(i**2))
    #     rms_interp = np.sqrt(np.mean(spline_interp))
    #     if rms_lc > rms_interp*5.:
    #         i_interp[start_ind:end_ind] = spline_interp
    #         flag=False
    #     else:
    #         # flag=True
    #         flag = False # >> instead of flagging, linearly interpolate
    #         i_interp[start_ind:end_ind] = \
    #             np.interp(time[start_ind:end_ind],
    #                       time[np.nonzero(~np.isnan(i))],
    #                       i[np.nonzero(~np.isnan(i))])     
    #         pdb.set_trace()
        
    # if DEBUG_INTERP:
    #     ax[4].plot(time_plot, i_interp, '.k')
    #     ax[4].set_title('spline interpolate')
    #     fig.tight_layout()
    #     fig.savefig(output_dir + prefix + 'interpolate_debug.png',
    #                 bbox_inches='tight')
    #     plt.close(fig)

    # num_inds = np.nonzero( ~np.isnan(i) )[0]    
    # t_interp = np.arange(np.min(time[num_inds]), np.max(time[num_inds]), dt)
    # # t_interp = np.linspace(np.nanmin(time), np.nanmax(time), len(i))
    # orbit_gap_inds = np.nonzero( (t_interp > time[orbit_gap_start]) * \
    #                               (t_interp < time[orbit_gap_end]) )
    # t_interp = np.delete(t_interp, orbit_gap_inds)
    # fitted_spline = ius(x)
    # ius = interpolate.InterpolatedUnivariateSpline(time[num_inds], i[num_inds],
    #                                                k=k)



        #         ticid_classified.append(int(ticid))
        #         otype_list.append(otype)
        #         bibcode_list.append(bibcode)
                
        # # >> return classifications only for 
        # intersection, comm1, comm2 = np.intersect1d(ticid_list,
        #                                             ticid_classified,
        #                                             return_indices=True)
        # for i in comm2:
        #     simbad_info.append([ticid_simbad[i], otype_list[i],
        #                         bibcode_list[i]])
