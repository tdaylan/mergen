# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 20:57:07 2020

Data Access Functions

@author: conta
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

#import plotting_functions as pf

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
import astropy.coordinates as coord
import astropy.units as u
from astroquery.vizier import Vizier

import pdb
import fnmatch as fm

#import numba

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score   
# import batman
#from transitleastsquares import transitleastsquares

#import model as ml
#import data_functions as df


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

def combine_sectors(sectors, data_dir, custom_masks=None):
    all_flux = []
    all_ticid = []
    all_target_info = []
    all_x = []
    if type(custom_masks) == type(None):
        custom_masks = [[]*len(sectors)]
    for i in range(len(sectors)):
        flux, x, ticid, target_info = \
            load_data_from_metafiles(data_dir, sectors[i], nan_mask_check=True,
                                     custom_mask=custom_masks[i])
            
        mins = np.min(flux, axis = 1, keepdims=True)
        flux = flux - mins
        maxs = np.max(flux, axis=1, keepdims=True)
        flux = flux / maxs            
        all_flux.append(flux)
        all_ticid.append(ticid)
        all_target_info.append(target_info)
        all_x.append(x)
    # >> stitch together !! can only handle 2 sectors
    all_ticid1, comm1, comm2 = np.intersect1d(all_ticid[0], all_ticid[1],
                                             return_indices=True)
        
    flux = np.concatenate([all_flux[0][comm1], all_flux[1][comm2]], axis = -1)

    target_info = []
    for i in range(len(comm1)):
        target_info.append([','.join([all_target_info[0][comm1[i]][0],
                                     all_target_info[1][comm2[i]][0]]),
                           ','.join([all_target_info[0][comm1[i]][1],
                                     all_target_info[1][comm2[i]][1]]),
                           ','.join([all_target_info[0][comm1[i]][2],
                                     all_target_info[1][comm2[i]][2]]),
                           all_target_info[0][comm1[i]][3],
                           all_target_info[0][comm1[i]][4]])
    
    x = np.concatenate(all_x)
    
    ticid = all_ticid[0][comm1]
    
    return flux, x, ticid, np.array(target_info)

def combine_sectors_by_lc(sectors, data_dir, custom_mask=[],
                          output_dir='./', DEBUG=True):
    all_flux = []
    all_ticid = []
    all_target_info = []
    all_x = []
    flux_lengths = []
    for i in range(len(sectors)):
        flux, x, ticid, target_info = \
            load_data_from_metafiles(data_dir, sectors[i], nan_mask_check=False)
        all_flux.append(flux)
        all_ticid.append(ticid)
        all_target_info.append(target_info)
        all_x.append(x)
        flux_lengths.append(np.shape(flux)[1])
        
    # >> truncate
    new_length = np.min(flux_lengths)
    for i in range(len(sectors)):
        all_flux[i] = all_flux[i][:,:new_length]
        
    flux = np.concatenate([all_flux[0], all_flux[1]], axis = 0)
    x = all_x[0][:new_length]
    target_info = np.concatenate([all_target_info[0], all_target_info[1]],
                                 axis=0)
    ticid = np.concatenate([all_ticid[0], all_ticid[1]])    
    flux, x = nan_mask(flux, x, custom_mask=custom_mask, ticid=ticid,
                       target_info=target_info,
                       output_dir=output_dir, DEBUG=True)

    
    return flux, x, ticid, np.array(target_info)

def load_data_from_metafiles(data_dir, sector, cams=[1,2,3,4],
                             ccds=[[1,2,3,4]]*4, data_type='SPOC',
                             cadence='2-minute', DEBUG=False, fast=False,
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
    for i in range(len(cams)):
        cam = cams[i]
        for ccd in ccds[i]:
            if fast:
                s = 'Sector{sector}_20s/Sector{sector}Cam{cam}CCD{ccd}/' + \
                    'Sector{sector}Cam{cam}CCD{ccd}_lightcurves.fits'
            else:
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
        with fits.open(data_dir + fnames[i], memmap=False) as hdul:
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
   
    f = fits.open(filename_lc, memmap=False)
    
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

def load_lygos_csv(file):
    import pandas as pd
    data = pd.read_csv(file, sep = ' ', header = None)
    #print (data)
    t = np.asarray(data[0])
    ints = np.asarray(data[1])
    error = np.asarray(data[2])
    return t, ints, error

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
        with fits.open(fits_path + file, memmap=False) as hdul:
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

def convert_listofRADEC_todegrees(file, outputfile):

    #converting to decimal degrees
    import pandas as pd
    from astropy.coordinates import Angle
    df = pd.read_csv(file)
    print (df)
    
    for n in range(len(df)):
        a = Angle(df['RA'][n], u.degree)
        a = a.degree
        df['RA'][n] = a
        b = Angle(df['DEC'][n], u.degree)
        b = b.degree
        df['DEC'][n] = b
    
    new = df[['RA', 'DEC']].copy()
    print(new)
    new.to_csv(outputfile, index = False)

def fetch_local_bright_TICs(ra_dec_string, targetname):
    """Produces a CSV of all <Tmag 10 targets w/in 2 degrees of the target for reference """
    catalog_data1 = Catalogs.query_criteria(coordinates=ra_dec_string, radius=2,
                                        catalog="TIC", Tmag = [-40, 10])

    correctcols = catalog_data1['ID', 'ra', 'dec', 'Tmag']
    
    print(correctcols)
    
    correctcols.write("/users/conta/urop/Local_TIC_to_" + targetname +".csv")

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

