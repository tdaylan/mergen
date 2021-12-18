# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:00:13 2021
@author: Emma Chickles, Lindsey Gordon
Data access, data processing, feature vector creation functions.
data_utils.py



* data load
* data cleaning
* feature loading

Current functions:
DATA LOADING (SPOC)
* load_data_from_metafiles
* combine_sectors
* combine_sectors_by_lc

DATA DOWNLOADING (SPOC)
* data_access_sector_by_bulk
* bulk_download_helper
* get_lc_file_and_data

DATA LOADING (FFI)
* load_lygos_csv
* load_all_lygos

DATA CLEANING
* normalize
* mean_norm
* rms
* standardize
* interpolate_all
* interpolate_lc
* nan_mask

FEATURE LOADING
* load_ENF_feature_metafile

QUATERNION HANDLING
* convert_to_quat_metafile
* metafile_load_smooth quaternions
* extract_smooth_quaterions

Data access
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
* sector_mask_diag()
* merge_sector_diag()

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
* representation learning() # >> will instead be put in model.py

To Do List:
    - HYPERLEDA LC load in
    - Loading fxns for CAE features
    - Remove redundancy
    - Move imports to init
"""
import numpy as np
from .__init__ import *
import numpy.ma as ma 
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)

from scipy.stats import moment
from scipy.stats import sigmaclip
from scipy.signal import detrend
from scipy import stats
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
rcParams["lines.markersize"] = 2
# rcParams['lines.color'] = 'k'
from scipy.signal import argrelextrema
from scipy import signal

# import plotting_functions as pf
from . import plot_utils as pt

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
from astropy.timeseries import LombScargle

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

import numba

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score   
# import batman
from transitleastsquares import transitleastsquares

# import model as ml
from . import learn_utils as lt

# import ephesus.ephesus.util as ephesus


def create_dir(path):
    print('Setting up '+path)
    try:
        os.makedirs(path)
    except OSError:
        print ("Directory %s already exists" % path)

# -- DOWNLOAD META DATA --------------------------------------------------------

def init_meta_folder(metapath):
    create_dir(metapath+'spoc/')
    create_dir(metapath+'spoc/cat/') # >> external catalogs 
    create_dir(metapath+'spoc/tic/') # >> TESS Input Catalog
    create_dir(metapath+'spoc/targ/') # >> target lists
    create_dir(metapath+'spoc/targ/2m/') # >> 2-minute cadence target lists
    create_dir(metapath+'spoc/targ/20s/') # >> 20-second cadence target lists
    for s in range(1, 45):
        savepath=metapath+'spoc/targ/2m/'
        fname = 'all_targets_S%03d'%s+'_v1.txt'

        if not os.path.exists(savepath+fname):
            url = 'https://tess.mit.edu/wp-content/uploads/'+fname
            os.system('curl -o '+savepath+fname+' '+url)

    for s in range(27, 45):
        savepath=metapath+'spoc/targ/20s/'
        fname = 'all_targets_20s_S%03d'%s+'_v1.txt'
        if not os.path.exists(savepath+fname):
            url = 'https://tess.mit.edu/wp-content/uploads/'+fname
            os.system('curl -o '+savepath+fname+' '+url)
    

# -- DATA LOADING (SPOC) -------------------------------------------------------

def load_data_from_metafiles(lcdir, sector, nan_mask_check=False):
        
    sector_path = lcdir+'sector-%02d'%sector+'/'
    lcfile_list = os.listdir(sector_path)

    time, flux, meta = [], [], []
    for lcfile in lcfile_list:
        data, m = open_fits(fname=sector_path+lcfile)
        if type(data) == type(None):
            data, m = open_fits(fname=sector_path+lcfile)
        time.append(data['TIME'])
        flux.append(data['FLUX'])
        meta.append(m)

    # >> apply nan mask
    if nan_mask_check:
        print('Applying nan mask')
        flux, x = nan_mask(flux, x, DEBUG=DEBUG, ticid=ticid,
                           debug_ind=debug_ind, target_info=target_info,
                           output_dir=output_dir, custom_mask=custom_mask)

    return time, flux, meta


# def load_data_from_metafiles(data_dir, sector, cams=[1,2,3,4],
#                              ccds=[[1,2,3,4]]*4, data_type='SPOC',
#                              cadence='2-minute', DEBUG=False, fast=False,
#                              output_dir='./', debug_ind=0,
#                              nan_mask_check=True,
#                              custom_mask=[]):
#     '''Pulls light curves from fits files, and applies nan mask.
    
#     Parameters:
#         * data_dir : folder containing fits files for each group
#         * sector : sector, given as int, or as a list
#         * cams : list of cameras
#         * ccds : list of CCDs
#         * data_type : 'SPOC', 'FFI'
#         * cadence : '2-minute', '20-second'
#         * DEBUG : makes nan_mask debugging plots. If True, the following are
#                   required:
#             * output_dir
#             * debug_ind
#         * nan_mask_check : if True, applies NaN mask
    
#     Returns:
#         * flux : array of light curve PDCSAP_FLUX,
#                  shape=(num light curves, num data points)
#         * x : time array, shape=(num data points)
#         * ticid : list of TICIDs, shape=(num light curves)
#         * target_info : [sector, cam, ccd, data_type, cadence] for each light
#                         curve, shape=(num light curves, 5)
#     '''
    
#     # >> get file names for each group
#     fnames = []
#     fname_info = []
#     for i in range(len(cams)):
#         cam = cams[i]
#         for ccd in ccds[i]:
#             if fast:
#                 s = 'Sector{sector}_20s/Sector{sector}Cam{cam}CCD{ccd}/' + \
#                     'Sector{sector}Cam{cam}CCD{ccd}_lightcurves.fits'
#             else:
#                 s = 'Sector{sector}/Sector{sector}Cam{cam}CCD{ccd}/' + \
#                     'Sector{sector}Cam{cam}CCD{ccd}_lightcurves.fits'
#             fnames.append(s.format(sector=sector, cam=cam, ccd=ccd))
#             fname_info.append([sector, cam, ccd, data_type, cadence])
                
#     # >> pull data from each fits file
#     print('Pulling data')
#     flux_list = []
#     ticid = np.empty((0, 1))
#     target_info = [] # >> [sector, cam, ccd, data_type, cadence]
#     for i in range(len(fnames)):
#         print('Loading ' + fnames[i] + '...')
#         with fits.open(data_dir + fnames[i], memmap=False) as hdul:
#             if i == 0:
#                 x = hdul[0].data
#             flux = hdul[1].data
#             ticid_list = hdul[2].data
    
#         flux_list.append(flux)
#         ticid = np.append(ticid, ticid_list)
#         target_info.extend([fname_info[i]] * len(flux))

#     # >> concatenate flux array         
#     flux = np.concatenate(flux_list, axis=0)
        
#     # >> apply nan mask
#     if nan_mask_check:
#         print('Applying nan mask')
#         flux, x = nan_mask(flux, x, DEBUG=DEBUG, ticid=ticid,
#                            debug_ind=debug_ind, target_info=target_info,
#                            output_dir=output_dir, custom_mask=custom_mask)

#     return x, flux, ticid.astype('int'), np.array(target_info)

def combine_sectors(sectors, data_dir, custom_masks=None):
    '''Combine sectors by the time axis (i.e. the light curves of stars 
    observed in multiple sectors are stitched together).
    Currently can only handle two sectors at a time.'''
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
    '''Combine sector by the light curve axis (i.e. the light curves of stars
    observed in multiple sectors are treated as two unrelated light curves).
    If the light curves are observed for a longer amount of time in a sector,
    then all of the light curves are truncated.
    TODO: try padding, instead of throwing out data.'''
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

# -- DATA DOWNLOADING (SPOC) ---------------------------------------------------

def data_access_sector_by_bulk(yourpath, sector, custom_mask=[],
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
    
    sectorfile = yourpath+'all_targets_S%03d'%sector+'_v1.txt'
    bulk_download_dir = yourpath

    for cam in [1,2,3,4]:
        for ccd in [1,2,3,4]:
            data_access_by_group_fits(yourpath, sectorfile, sector, cam,
                                      ccd, bulk_download=True,
                                      bulk_download_dir=bulk_download_dir,
                                      custom_mask=custom_mask,
                                      apply_nan_mask=apply_nan_mask)
            
def get_target_lists(data_dir, sectors=[]):
    create_dir(data_dir)

    for sector in sectors:
        out = yourpath + 'targ_lists/all_targets_S%3d'%sector+'_v1.txt'
        load = 'https://tess.mit.edu/wp-content/uploads/all_targets_S%3d'%sector+'_v1.txt'
        os.system('curl -L -o '+ out + ' ' + load)

def bulk_download_lc(yourpath, shell_script='', sector=1):
    '''Downloads all the light curves for a sector. Can also be used to go back
    and check you have all the light curves from a sector.
    Parameters:
        * yourpath : directory to save .fits files in, contains shell_script
        * shell_script : file name for shell script (tesscurl_sector_*_lc.sh)
          from http://archive.stsci.edu/tess/bulk_downloads.html
    e.g. bulk_download_helper('./tessdata_sector_18/',
                              'tesscurl_sector_18_lc.sh')
    TODO:
    * modify to handle 30-min cadence data
    '''
    import fnmatch as fm
    if len(shell_script)==0:
        shell_script='tesscurl_sector_'+str(sector)+'_lc.sh'

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

# -- DATA LOADING (FFI) --------------------------------------------------------

def load_lygos_csv(file):
    import pandas as pd
    data = pd.read_csv(file, sep = ' ', header = None)
    #print (data)
    t = np.asarray(data[0])
    ints = np.asarray(data[1])
    error = np.asarray(data[2])
    return t, ints, error

def load_all_lygos(datapath):
    
    all_t = [] 
    all_i = []
    all_e = []
    all_labels = []
    
    for root, dirs, files in os.walk(datapath):
        for name in files:
            if name.startswith(("rflx")):
                filepath = root + "/" + name 
                print(name)
                label = name.split("_")
                full_label = label[1] + label[2]
                all_labels.append(full_label)
                
                t,i,e = load_lygos_csv(name)
                mean = np.mean(i)
                sigclip = SigmaClip(sigma=4, maxiters=None, cenfunc='median')
                clipped_inds = np.nonzero(np.ma.getmask(sigclip(i)))
                i[clipped_inds] = mean #reset those values to the mean value (or remove??)
    
                all_t.append(t)
                all_i.append(i)
                all_e.append(e)
                
    return all_t, all_i, all_e, all_labels

# -- DATA CLEANING -------------------------------------------------------------

def normalize(flux, axis=1):
    '''Dividing by median.'''
    medians = np.median(flux, axis = axis, keepdims=True)
    flux = flux / medians
    return flux

def mean_norm(flux, axis=1): 
    """normalizes by dividing by mean - necessary for TLS running"""
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
    stdevs[ np.nonzero(stdevs == 0.) ] = 1e-7 #1e-8
    
    x = x / stdevs
    return x

def normalize_minmax(x, ax=1):
    mins = np.min(x, axis=ax, keepdims=True)
    x = x - mins
    maxs = np.max(x, axis=ax, keepdims=True)
    x = x / maxs
    return x

# -- Open and write light curve Fits files -------------------------------------

def open_fits(lcdir='', objid=None, fname=None):
    """Loads preprocessed light curves from Fits file. Must either supply
    objid or fname."""
    if type(fname) == type(None):
        fname = str(int(objid))+'.fits'
    fname = lcdir + fname

    # try:
    #     data  = fits.getdata(fname, 1)
    #     meta = fits.getheader(fname, 0)
    #     return [data, meta]
    # except:
    #     print('Failed to open the following FITS file:')
    #     print(lcdir+fname)
    #     return [None, None]

    try:
        with fits.open(fname) as hdul:
            data = hdul[1].data
            meta = hdul[0].header
        return [data, meta]
    except:
        print('Failed to open the following FITS file:')
        print(lcdir+fname)
        return [None, None]

    gc.collect()


    # lchdu = fits.open(lcdir+fname)
    # data = []
    # for data_name in data_names:
    #     data.append(lchdu[1].data[data_name])

    # meta = lchdu[0].header
    # data.append(meta)

    # lchdu.close()


def write_fits(savepath, meta, data, data_names, table_meta=[],
               verbose=True, verbose_msg='', fname=None, fmt=None,
               n_table_hdu=1, primary_data = None):
    """ 
    * savepath : string, directory to save light curve in
    * meta : primary HDU header data
    * data : second HDU table data, with column names given by data_names
    * data_names : e.g. ['TIME', 'FLUX']
    * table_meta : list of tuples (header_name, value) for the Fits table header
    """

    if type(fname) == type(None):
        objid = meta['TICID']
        fname = str(objid)+'.fits' # >> filename

    hdu_list = []

    if type(meta) == type(None):
        primary_hdr = None
    else:
        primary_hdr = fits.Header(meta)
    primary_hdu = fits.PrimaryHDU(primary_data, header=primary_hdr) # >> metadata
    hdu_list.append(primary_hdu)

    if n_table_hdu == 1:
        if type(fmt) == type(None):
            fmt = ['D'] * len(data)
        table_hdr = fits.Header(table_meta)
        col_list = []
        for i in range(len(data_names)):
            col = fits.Column(name=data_names[i], array=data[i], format=fmt[i])
            col_list.append(col)
        table_hdu = fits.BinTableHDU.from_columns(col_list, header=table_hdr)
        hdu_list.append(table_hdu)
    else:
        for n in range(n_table_hdu):
            data_hdu = data[n]
            if type(fmt) == type(None):
                fmt_hdu = ['D'] * len(data_hdu)
            else:
                fmt_hdu = fmt[n]
            table_hdr = fits.Header(table_meta)
            col_list = []
            for i in range(len(data_names[n])):
                col = fits.Column(name=data_names[n][i], array=data_hdu[i],
                                  format=fmt_hdu[i])
                col_list.append(col)
            table_hdu = fits.BinTableHDU.from_columns(col_list, header=table_hdr)
            hdu_list.append(table_hdu)

    fname = savepath+fname
    hdul = fits.HDUList(hdu_list)
    hdul.writeto(fname, overwrite=True)
    if verbose:
        print(verbose_msg+'\n')
        print('Wrote '+fname)    

# -- Quality flag mask ---------------------------------------------------------

def qual_mask(datapath, verbose=True, v_int=200):
    '''
    Reads and masks flagged data points in all PDCSAP_FLUX light curves of a 
    specified sector.
    * datapath : string, directory with light curve data (includes subdirectory
                 raws/)
    '''

    sectors = os.listdir(datapath+'raws/')
    sectors.sort()

    for sector in sectors:

        raws_sector_path = datapath+'raws/'+sector+'/'
        mask_sector_path = datapath+'mask/'+sector+'/'
        create_dir(mask_sector_path)
        lcfile_list = os.listdir(raws_sector_path)

        for i in range(len(lcfile_list)):
            if i % v_int == 0:
                verbose_msg='Processing light curve '+str(i)+'/'+\
                            str(len(lcfile_list))
                verbose=True
            else:
                verbose=False
            qual_mask_lc(raws_sector_path+lcfile_list[i], mask_sector_path,
                         verbose=verbose, verbose_msg=verbose_msg)

def qual_mask_lc(lcfile, savepath, verbose=True, verbose_msg=''):
    '''
    Reads and masks flagged data points in PDCSAP_FLUX light curves.
    * lcfile : light curve file
    '''

    # >> open light curve file
    lchdu = fits.open(lcfile)

    # >> get light curve
    #    * PDCSAP_FLUX : Systematics corrected photometry using cotrending
    #      basis vectors.
    #    * TIME : stored in BJD-2457000
    flux = lchdu[1].data['PDCSAP_FLUX']
    time = lchdu[1].data['TIME']
    qual = lchdu[1].data['QUALITY']
    meta = lchdu[0].header
    ticid = meta['TICID']
    
    # >> mask out data points with nonzero quality flags
    flagged_inds = np.nonzero(qual)
    flux[flagged_inds] = np.nan

    # >> save masked light curve
    write_fits(savepath, meta, [time, flux], ['TIME', 'FLUX'], verbose=verbose,
               verbose_msg=verbose_msg)
    lchdu.close()


def DAE_preprocessing(lcdir, train_test_ratio=1.0, norm_type='standardization'):
    '''Preprocesses engineered features in preparation for training a deep
    fully-connected autoencoder. Preprocessing steps include:
        1) Reading feature vector for each target
        2) Producing a homogenous input matrix
        3) Partitioning into training and testing sets, if train_test_ratio<1
        4) Normalizing
    Parameters:
        * mg : Mergen object
        * train_test_ratio : partition ratio. If 1, then no partitioning.
        * norm_type : None, or 'standardization'
    '''

    # >> Read data
    lcfile_list = os.listdir(lcdir)
    freq, lspm, meta = [], [], []
    for lcfile in lcfile_list:
        d, m = open_fits(lcdir, fname=lcfile, data_names=['FREQ', 'LSPM'])
        freq.append(d['FREQ'])
        lspm.append(d['LSPM'])
        meta.append(m)

    pdb.set_trace()

    if train_test_ratio < 1:
        print('Partitioning data...')
        x_train, x_test, y_train, y_test, flux_train, flux_test,\
        ticid_train, ticid_test, target_info_train, target_info_test, time =\
            split_data_features(flux, features, time, ticid, target_info,
                                train_test_ratio=train_test_ratio)

    if norm_type == None:
        print('No normalization performed...')
    elif norm_type == 'standardization':
        print('Standardizing feature vectors...')
        x_train = dt.standardize(x_train, ax=0)
        x_test = dt.standardize(x_test, ax=0)

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
             custom_mask=[], use_tol2=True):
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
                pt.ticid_label(ax[i], ticid[ind], target_info[ind], title=True)
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
    if len(worst_inds[0]) < tol1 * len(flux) and use_tol2: # >> only a few bad light curves
        np.delete(flux, worst_inds, 0)
        
        pdb.set_trace()
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

# -- FEATURE LOADING -----------------------------------------------------------

def load_ENF_feature_metafile(folderpath):
    print("Loading engineered features...")

    filepaths = []
    for root, dirs, files in os.walk(folderpath):
        for file in files:
            if file.endswith("features_v0.fits"):
                filepaths.append(root + "/" + file)
                print(root + "/" + file)
            elif file.endswith("features_v1.fits"):
                filepaths.append(root + "/" + file)
                print(root + "/" + file)
        
    f = fits.open(filepaths[0], memmap=False)
    features = np.asarray(f[0].data)
    f.close()
    for n in range(len(filepaths) -1):
        f = fits.open(filepaths[n+1], memmap=False)
        features_new = np.asarray(f[0].data)
        features = np.column_stack((features, features_new))
        f.close()
    return features


# -- QUATERNION HANDLING -------------------------------------------------------

def convert_to_quat_metafile(file, fileoutput):
    f = fits.open(file, memmap=False)
    
    t = f[1].data['TIME']
    Q1 = f[1].data['C1_Q1']
    Q2 = f[1].data['C1_Q2']
    Q3 = f[1].data['C1_Q3']
    f.close()
    
    big_quat_array = np.asarray((t, Q1, Q2, Q3))
    np.savetxt(fileoutput, big_quat_array)

def metafile_load_smooth_quaternions(sector, maintimeaxis, 
                                     quaternion_folder = "/users/conta/urop/quaternions/"):
    
    def quaternion_binning(quaternion_t, q, maintimeaxis):
        sector_start = maintimeaxis[0]
        bins = 900 #30 min times sixty seconds/2 second cadence
                
        def find_nearest_values_index(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx
                
        binning_start = find_nearest_values_index(quaternion_t, sector_start)
        n = binning_start
        m = n + bins
        binned_Q = []
        binned_t = []
                
        while m <= len(quaternion_t):
            bin_t = quaternion_t[n]
            binned_t.append(bin_t)
            bin_q = np.mean(q[n:m])
            binned_Q.append(bin_q)
            n += 900
            m += 900
                
            
        standard_dev = np.std(np.asarray(binned_Q))
        mean_Q = np.mean(binned_Q)
        outlier_indexes = []
                
        for n in range(len(binned_Q)):
            if binned_Q[n] >= mean_Q + 5*standard_dev or binned_Q[n] <= mean_Q - 5*standard_dev:
                outlier_indexes.append(n)
                
                      
        return np.asarray(binned_t), np.asarray(binned_Q), outlier_indexes
        
    from scipy.signal import medfilt
    for root, dirs, files in os.walk(quaternion_folder):
            for name in files:
                if name.endswith(("S"+sector+"-quat.txt")):
                    print(name)
                    filepath = root + "/" + name
                    c = np.genfromtxt(filepath)
                    tQ = c[0]
                    Q1 = c[1]
                    Q2 = c[2]
                    Q3 = c[3]

    q = [Q1, Q2, Q3]

    
    for n in range(3):
        smoothed = medfilt(q[n], kernel_size = 31)
        if n == 0:
            Q1 = smoothed
            tQ_, Q1, Q1_outliers = quaternion_binning(tQ, Q1, maintimeaxis)
        elif n == 1:
            Q2 = smoothed
            tQ_, Q2, Q2_outliers = quaternion_binning(tQ, Q2, maintimeaxis)
        elif n == 2:
            Q3 = smoothed
            tQ_, Q3, Q3_outliers = quaternion_binning(tQ, Q3, maintimeaxis)
    
    outlier_indexes = np.unique(np.concatenate((Q1_outliers, Q2_outliers, Q3_outliers)))
    return tQ_, Q1, Q2, Q3, outlier_indexes  

def extract_smooth_quaterions(path, file, momentum_dump_csv, kernal, maintimeaxis, plot = False):

    from scipy.signal import medfilt
    f = fits.open(file, memmap=False)

    t = f[1].data['TIME']
    Q1 = f[1].data['C1_Q1']
    Q2 = f[1].data['C1_Q2']
    Q3 = f[1].data['C1_Q3']
    
    f.close()
    
    
    q = [Q1, Q2, Q3]
    
    if plot:
        with open(momentum_dump_csv, 'r') as f:
            lines = f.readlines()
            mom_dumps = [ float(line.split()[3][:-1]) for line in lines[6:] ]
            inds = np.nonzero((mom_dumps >= np.min(t)) * \
                              (mom_dumps <= np.max(t)))
            mom_dumps = np.array(mom_dumps)[inds]
    #q is a list of qs
    for n in range(3):
        
        smoothed = medfilt(q[n], kernel_size = kernal)
        
        if plot:
            plt.scatter(t, q[n], label = "original")
            plt.scatter(t, smoothed, label = "smoothed")
            
            for k in mom_dumps:
                plt.axvline(k, color='g', linestyle='--', alpha = 0.1)
            plt.legend(loc = "upper left")
            plt.title("Q" + str(n+1))
            plt.savefig(path + str(n + 1) + "-kernal-" + str(kernal) +"-both.png")
            plt.show()
            #plt.scatter(t, q[n], label = "original")
            plt.scatter(t, smoothed, label = "smoothed")
            for k in mom_dumps:
                plt.axvline(k, color='g', linestyle='--', alpha = 0.1)
            plt.legend(loc="upper left")
            plt.title("Q" + str(n+1) + "Smoothed")
            plt.savefig(path + str(n + 1) + "-kernal-" + str(kernal) +"-median-smoothed-only.png")
            plt.show()
            
        def quaternion_binning(quaternion_t, q, maintimeaxis):
            sector_start = maintimeaxis[0]
            bins = 900 #30 min times sixty seconds/2 second cadence
            
            def find_nearest_values_index(array, value):
                array = np.asarray(array)
                idx = (np.abs(array - value)).argmin()
                return idx
            binning_start = find_nearest_values_index(quaternion_t, sector_start)
            n = binning_start
            m = n + bins
            binned_Q = []
            binned_t = []
            
            while m <= len(t):
                bin_t = quaternion_t[n]
                binned_t.append(bin_t)
                bin_q = np.mean(q[n:m])
                binned_Q.append(bin_q)
                n += 900
                m += 900
            plt.scatter(binned_t, binned_Q)
            plt.show()
        
            standard_dev = np.std(np.asarray(binned_Q))
            mean_Q = np.mean(binned_Q)
            outlier_indexes = []
            
            for n in range(len(binned_Q)):
                if binned_Q[n] >= mean_Q + 5*standard_dev or binned_Q[n] <= mean_Q - 5*standard_dev:
                    outlier_indexes.append(n)
            
            print(outlier_indexes)      
            return outlier_indexes
        
        if n == 0:
            Q1 = smoothed
            Q1_outliers = quaternion_binning(t, Q1, maintimeaxis)
        elif n == 1:
            Q2 = smoothed
            Q2_outliers = quaternion_binning(t, Q2, maintimeaxis)
        elif n == 2:
            Q3 = smoothed
            Q3_outliers = quaternion_binning(t, Q3, maintimeaxis)
    
    outlier_indexes = np.unique(np.concatenate((Q1_outliers, Q2_outliers, Q3_outliers)))
    print(outlier_indexes)
    return t, Q1, Q2, Q3, outlier_indexes  



    
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
    
def combine_sectors_by_time_axis(sectors, data_dir, cutoff=0.5, custom_mask=[],
                                 order=5, tol=0.6, debug=True, norm_type='standardization',
                                 output_dir='./', return_median_flux=False):
    num_sectors = len(sectors)
    all_flux = []
    all_ticid = []
    all_target_info = []
    all_x = []
    all_flux_plot = []
    print('Loading data and applying nanmask')
    for i in range(len(sectors)):
        flux, x, ticid, target_info = \
            load_data_from_metafiles(data_dir, sectors[i], nan_mask_check=True,
                                     custom_mask=custom_mask)
            
        # all_flux_plot.append(normalize(flux))
        if norm_type == 'standardization':
            print('Standardizing fluxes...')
            flux = standardize(flux)
    
        elif norm_type == 'median_normalization':
            print('Normalizing fluxes (dividing by median)...')
            flux = normalize(flux)
            
        elif norm_type == 'minmax_normalization':
            print('Normalizing fluxes (changing minimum and range)...')
            mins = np.min(flux, axis = 1, keepdims=True)
            flux = flux - mins
            maxs = np.max(flux, axis=1, keepdims=True)
            flux = flux / maxs
            
        else:
            print('Light curves are not normalized!')            
            
        all_flux.append(flux)
        all_ticid.append(ticid)
        all_target_info.append(target_info)
        all_x.append(x)
    
    x = np.concatenate(all_x)
    if np.count_nonzero(np.isnan(x)):
        x = np.interp(np.arange(len(x)), np.arange(len(x))[np.nonzero(~np.isnan(x))],
                      x[np.nonzero(~np.isnan(x))])
    flux, flux_plot, target_info, ticid, ticid_rejected = [], [], [], [], []
    
    # !!
    all_ticid, comm1, comm2 = np.intersect1d(all_ticid[0], all_ticid[1],
                                         return_indices=True)
    all_flux[0] = all_flux[0][comm1]
    all_flux[1] = all_flux[1][comm2]
    # all_flux_plot[0] = all_flux_plot[0][comm1]
    # all_flux_plot[1] = all_flux_plot[1][comm2]
    all_target_info[0] = all_target_info[0][comm1]
    all_target_info[1] = all_target_info[1][comm2]
    
    for i in range(len(all_ticid)):
    
        b, a = signal.butter(order, cutoff, btype='high', analog=False)
        y1 = signal.filtfilt(b, a, all_flux[0][i])
        rms1 = np.sqrt(np.mean(y1**2))
        
        y2 = signal.filtfilt(b, a, all_flux[1][i])
        rms2 = np.sqrt(np.mean(y2**2))
        
        if debug and i < 5:
            fig, ax = plt.subplots(2, num_sectors)
            # ax[0,0].plot(all_x[0], all_flux_plot[0][i], '.k', ms=1)
            # ax[0,1].plot(all_x[1], all_flux_plot[1][i], '.k', ms=1) 
            ax[0,0].plot(all_x[0], all_flux[0][i], '.k', ms=1)
            ax[0,1].plot(all_x[1], all_flux[1][i], '.k', ms=1)               
            ax[1,0].plot(all_x[0], y1, '.k', ms=1)
            ax[1,1].plot(all_x[1], y2, '.k', ms=1)
            fig.savefig(output_dir+'highpass_'+str(cutoff)+'_'+str(i)+'.png')   
    
        # >> compare RMS !! assumes combining 2 sectors
        if np.abs(rms2 - rms1) < tol*rms1 and \
            np.abs(rms2 - rms1) < tol*rms2:
            flux.append(np.concatenate([all_flux[0][i], all_flux[1][i]]))
            # flux_plot.append(np.concatenate([all_flux_plot[0][i],
            #                                  all_flux_plot[1][i]]))
            target_info.append([','.join([all_target_info[0][i][0],
                                         all_target_info[1][i][0]]),
                               ','.join([all_target_info[0][i][1],
                                         all_target_info[1][i][1]]),
                               ','.join([all_target_info[0][i][2],
                                         all_target_info[1][i][2]]),
                               all_target_info[0][i][3],
                               all_target_info[0][i][4]])  
            ticid.append(all_ticid[i])
        else:
                       
            print('Excluding TIC '+str(all_ticid[i]))
            if debug:
                fig, ax = plt.subplots(2, 2)
                # ax[0,0].plot(all_x[0], all_flux_plot[0][i], '.k', ms=1)
                # ax[0,1].plot(all_x[1], all_flux_plot[1][i], '.k', ms=1)       
                ax[0,0].plot(all_x[0], all_flux[0][i], '.k', ms=1)
                ax[0,1].plot(all_x[1], all_flux[1][i], '.k', ms=1)                   
                ax[1,0].plot(all_x[0], y1, '.k', ms=1)
                ax[1,1].plot(all_x[1], y2, '.k', ms=1)
                fig.savefig(output_dir+'highpass_TIC_'+str(int(all_ticid[i]))+'.png')    
            ticid_rejected.append(all_ticid[i])
             

    flux = np.array(flux)
    flux_plot = np.array(flux_plot)
    target_info = np.array(target_info)
    ticid = np.array(ticid)
    
    del all_flux
    del all_x
    del all_ticid
    del all_target_info
            
    
    # # >> stitch together !! can only handle 2 sectors
    # all_ticid1, comm1, comm2 = np.intersect1d(all_ticid[0], all_ticid[1],
    #                                          return_indices=True)
        
    # flux = np.concatenate([all_flux[0][comm1], all_flux[1][comm2]], axis = -1)

    # target_info = []
    # for i in range(len(comm1)):
    #     target_info.append([','.join([all_target_info[0][comm1[i]][0],
    #                                  all_target_info[1][comm2[i]][0]]),
    #                        ','.join([all_target_info[0][comm1[i]][1],
    #                                  all_target_info[1][comm2[i]][1]]),
    #                        ','.join([all_target_info[0][comm1[i]][2],
    #                                  all_target_info[1][comm2[i]][2]]),
    #                        all_target_info[0][comm1[i]][3],
    #                        all_target_info[0][comm1[i]][4]])
    # target_info = np.array(target_info)
    
    # ticid = all_ticid[0][comm1]
    
    # return flux, flux_plot, x, ticid, target_info
    return flux, x, ticid, target_info

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

# def load_data_from_metafiles(data_dir, sector, cams=[1,2,3,4],
#                              ccds=[[1,2,3,4]]*4, data_type='SPOC',
#                              cadence='2-minute', DEBUG=False, fast=False,
#                              output_dir='./', debug_ind=0,
#                              nan_mask_check=True,
#                              custom_mask=[]):
#     '''Pulls light curves from fits files, and applies nan mask.
    
#     Parameters:
#         * data_dir : folder containing fits files for each group
#         * sector : sector, given as int, or as a list
#         * cams : list of cameras
#         * ccds : list of CCDs
#         * data_type : 'SPOC', 'FFI'
#         * cadence : '2-minute', '20-second'
#         * DEBUG : makes nan_mask debugging plots. If True, the following are
#                   required:
#             * output_dir
#             * debug_ind
#         * nan_mask_check : if True, applies NaN mask
    
#     Returns:
#         * flux : array of light curve PDCSAP_FLUX,
#                  shape=(num light curves, num data points)
#         * x : time array, shape=(num data points)
#         * ticid : list of TICIDs, shape=(num light curves)
#         * target_info : [sector, cam, ccd, data_type, cadence] for each light
#                         curve, shape=(num light curves, 5)
#     '''
    
#     # >> get file names for each group
#     fnames = []
#     fname_info = []
#     for i in range(len(cams)):
#         cam = cams[i]
#         for ccd in ccds[i]:
#             if fast:
#                 s = 'Sector{sector}_20s/Sector{sector}Cam{cam}CCD{ccd}/' + \
#                     'Sector{sector}Cam{cam}CCD{ccd}_lightcurves.fits'
#             else:
#                 s = 'Sector{sector}/Sector{sector}Cam{cam}CCD{ccd}/' + \
#                     'Sector{sector}Cam{cam}CCD{ccd}_lightcurves.fits'
#             fnames.append(s.format(sector=sector, cam=cam, ccd=ccd))
#             fname_info.append([sector, cam, ccd, data_type, cadence])
                
#     # >> pull data from each fits file
#     print('Pulling data')
#     flux_list = []
#     ticid = np.empty((0, 1))
#     target_info = [] # >> [sector, cam, ccd, data_type, cadence]
#     for i in range(len(fnames)):
#         print('Loading ' + fnames[i] + '...')
#         with fits.open(data_dir + fnames[i], memmap=False) as hdul:
#             if i == 0:
#                 x = hdul[0].data
#             flux = hdul[1].data
#             ticid_list = hdul[2].data
    
#         flux_list.append(flux)
#         ticid = np.append(ticid, ticid_list)
#         target_info.extend([fname_info[i]] * len(flux))

#     # >> concatenate flux array         
#     flux = np.concatenate(flux_list, axis=0)
        
#     # >> apply nan mask
#     if nan_mask_check:
#         print('Applying nan mask')
#         flux, x = nan_mask(flux, x, DEBUG=DEBUG, ticid=ticid,
#                            debug_ind=debug_ind, target_info=target_info,
#                            output_dir=output_dir, custom_mask=custom_mask)

#     return flux, x, ticid, np.array(target_info)
    
    
def load_group_from_fits(path, sector, camera, ccd): 
    """ pull the light curves and target def qlist from fits metafiles
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
    



def data_access_sector_by_bulk(data_dir, sector,
                               bulk_download_dir, custom_mask=[],
                               apply_nan_mask=False, query_tess_feats=False,
                               query_gcvs=True, query_simbad=True, make_fits=True):
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

    print('Accessing Sector '+str(sector))
    sectorpath=data_dir+'Sector'+str(sector)+'/'
    sectorfile = sectorpath+'all_targets_S%03d'%sector+'_v1.txt'
    
    if make_fits:
        for cam in [1,2,3,4]:
            for ccd in [1,2,3,4]:
                data_access_by_group_fits(sectorpath, sectorfile, sector, cam,
                                          ccd, bulk_download=True,
                                          bulk_download_dir=bulk_download_dir,
                                          custom_mask=custom_mask,
                                          apply_nan_mask=apply_nan_mask)
            
    # >> get a list of TICIDs from sectorfile
    ticid_list = np.loadtxt(sectorfile)[:,0] # >> take first column only

    # if query_tess_feats:
    #     # >> download TIC-v8 features
    #     get_tess_feature_txt(ticid_list,
    #                          yourpath+'tess_features_sector'+str(sector)+'.txt')

    database_dir = data_dir+'databases/'

    if query_gcvs:
        # >> query GCVS
        query_vizier(ticid_list=ticid_list, data_dir=data_dir, sector=sector,
                     out=database_dir+'Sector'+str(sector)+'_GCVS.txt', query_mast=False)

    if query_simbad:
        # >> query SIMBAD
        out_f=data_dir+'Sector'+str(sector)+'_simbad.txt'
        ticid_simbad, otypes_simbad, main_id_simbad = \
            query_simbad_classifications(ticid_list, out_f=out_f, data_dir=data_dir,
                                         sector=sector, query_mast=False)
        correct_simbad_to_vizier(in_f=data_dir+'Sector'+str(sector)+'_simbad.txt',
                                 out_f=database_dir+'Sector'+str(sector)+'_simbad_revised.txt', 
                                 simbad_gcvs_conversion=data_dir+'simbad_gcvs_label.txt')
    
            
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
            confirmation = lc_from_target_list(yourpath, target_list,
                                                    fname_time_intensities,
                                                    fname_targets, fname_notes,
                                                     path=path,
                                                     custom_mask=custom_mask,
                                                     apply_nan_mask=apply_nan_mask)
       
        
    except OSError: #if there is an error creating the folder
        print("There was an OS Error trying to create the folder. Check to see if data is already saved there")
        targets = "empty"
        
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

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Data processing :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#normalizing each light curve
def normalize(flux, axis=1):
    '''Dividing by median.
    !!Current method blows points out of proportion if the median is too close to 0?'''
    medians = np.nanmedian(flux, axis = axis, keepdims=True)
    flux = flux / medians
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
                   DEBUG_INTERP=False, orbig_gap_len=0.5,
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
        
    # >> remove orbit gap(s) from list
    orbit_gap_ind = np.argmax(run_lengths)
    orbit_gap_start = run_starts[ orbit_gap_ind ]
    orbit_gap_end = orbit_gap_start + run_lengths[ orbit_gap_ind ]    
    run_starts = np.delete(run_starts, orbit_gap_ind)
    run_lengths = np.delete(run_lengths, orbit_gap_ind)
    
    # !!
    # orbit_gap_inds = np.nonzero(run_lengths > )
    
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
             output_dir='./', prefix='', tol1=0.05, tol2=0.5,
             custom_mask=[], use_tol2=True):
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
                pt.ticid_label(ax[i], ticid[ind], target_info[ind], title=True)
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
    worst_inds = np.nonzero( num_nan > tol2*flux.shape[1] )[0]

    if len(worst_inds)>0 and len(worst_inds)<tol1*len(flux) and use_tol2:
        with open(output_dir+prefix+'removed_light_curves.txt', 'w') as f:
            for i in range(len(worst_inds)):
                f.write('TIC '+ str(ticid[worst_inds[i]])+'\n')

        print('Removing '+str(len(worst_inds))+' light curves')
        flux = np.delete(flux, worst_inds, 0)        

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
    
def sector_mask_diag(sectors=[1,2,3,17,18,19,20], data_dir='./',
                      output_dir='./', custom_masks=None):
    
    num_sectors = len(sectors)
    all_flux = []
    all_ticid = []
    all_target_info = []
    all_x = []
    if type(custom_masks) == type(None):
        custom_masks = [[]]*num_sectors
    for i in range(num_sectors):
        flux, x, ticid, target_info = \
            df.load_data_from_metafiles(data_dir, sectors[i],
                                        nan_mask_check=True,
                                        custom_mask=custom_masks[i])       
        all_flux.append(flux)
        all_ticid.append(ticid)
        all_target_info.append(target_info)
        all_x.append(x)
        
    fig, ax  = plt.subplots(num_sectors)
    for i in range(num_sectors):
        ax[i].plot(all_x[i], all_flux[i][0], '.k', ms=2)
        pt.ticid_label(ax[i], all_ticid[i][0], all_target_info[i][0],
                       title=True)
        ax[i].set_title('Sector '+str(sectors[i])+'\n'+ax[i].get_title(),
                        fontsize='small')
    
    fig.tight_layout()
    fig.savefig(output_dir+'sector_masks.png')

def merge_sector_diag(data_dir, sectors=list(range(1, 29)), output_dir='./',
                      ncols=3):
    
    num_sectors = len(sectors)
    fig, ax = plt.subplots(num_sectors, ncols,
                           figsize=(5*ncols, 1.43*num_sectors))
    for i in range(num_sectors):
        sectorfile=np.loadtxt(data_dir+'Sector'+str(sectors[i])+\
                              '/all_targets_S'+'%03d'%sectors[i]+'_v1.txt')
        for j in range(ncols):
            if ncols == 1:
                a = ax[i]
            else:
                a = ax[i,j]
            ticid = sectorfile[j][0]
            time, flux, ticid = df.get_lc_file_and_data(output_dir, ticid)
            a.plot(time, flux, '.k', ms=2)
            a.set_title('Sector '+str(sectors[i]), fontsize='small')
        
    fig.tight_layout()
    fig.savefig(output_dir + 'sector_lightcurves.png')


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
    


def create_save_featvec_homogenous_time(yourpath, times, intensities, filelabel, version=0, save=True):
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
    modified: [lcg 08212020]"""
    

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

def featvec(x_axis, sampledata, ticid=None, v=0): 
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
        from transitleastsquares import transitleastsquares, period_grid, catalog_info
        model = transitleastsquares(x_axis, sampledata)
        
        if type(ticid) != type(None):
            dt = np.max(x_axis) - np.min(x_axis)            
            ab, mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(TIC_ID=int(ticid))
            # >> find smallest period grid
            rm_set = []
            grid_lengths = [period_grid(1, 1, dt).shape[0]]
            rm_set.append([1,1])
            
            if not np.isnan(radius):                
                grid_lengths.append(period_grid(radius, 1, dt).shape[0])
                rm_set.append([radius, 1])
                
            if not np.isnan(mass):
                grid_lengths.append(period_grid(1, mass, dt).shape[0])
                rm_set.append([1, mass])
                
            
            ind = np.argmin(grid_lengths)
            R_star, M_star = rm_set[ind]
            
        else:
            R_star, M_star = 1,1
        
        results = model.power(show_progress_bar=True, R_star=R_star,
                              M_star=M_star)
        featvec.append(results.period)
        featvec.append(results.duration)
        featvec.append((1 - results.depth))
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
            f = fits.open(folderpath + file_label + "_features.fits", memmap=False)
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

def calculate_variability_statistics(data_dir):
    for sector in range(1,27):
        flux, x, ticid, target_info = \
            load_data_from_metafiles(data_dir, sector, nan_mask_check=False)

        variability = []
        for i in range(len(flux)):
            norm = flux[i] / np.nanmedian(flux[i])
            norm = np.delete(norm, np.nonzero(np.isnan(norm)))
            var = np.percentile(norm, 95) - np.percentile(norm, 5)
            variability.append(var)
        np.savetxt(data_dir+'Sector'+str(sector)+'/Sector'+\
                   str(sector)+'variability_statistics.txt',
                   [ticid, variability])




def get_tess_features(ticid, cols=['Teff', 'rad', 'mass', 'GAIAmag', 'd',\
                                   'objType', 'Tmag']):
    '''Query catalog data https://arxiv.org/pdf/1905.10694.pdf'''
    

    target = 'TIC '+str(int(ticid))
    catalog_data = Catalogs.query_object(target, radius=0.02, catalog='TIC')

    feats = []
    for col in cols:
        feats.append(catalog_data[0][col])

    return target, feats

def get_tess_feature_all(data_dir=''):
    # >> get column names
    columns = np.loadtxt(data_dir+'exo_CTL_08.01xTIC_v8.1_header.csv',
                         dtype='str', delimiter=',')
    columns = np.char.replace(columns, '[', '') # >> clean up
    columns = np.char.split(columns, ']')
    columns = [x[0] for x in columns]
    columns.remove('objID')

    # >> loop through sectors
    # sectors =  [26] # np.arange(8,27)
    for sector in sectors:
        print('Sector '+str(sector))
        output_dir = data_dir + 'Sector'+str(sector)+'/'
        fname = output_dir+'all_targets_S%03d'%sector+'_v1.txt'
        ticid = np.loadtxt(fname)[:,0] # >> take first column

        fname = output_dir+'Sector'+str(sector)+'tic_cat.csv'
        data = pd.read_csv(fname) # !! need to first do get_TIC_catalog_sector
        data_ticid = data['ID'].to_numpy()
        with open(fname, 'r') as f:
            lines=f.readlines()[1:]

        # >> make new file
        fname = output_dir+'Sector'+str(sector)+'tic_cat_all.csv'
        if os.path.exists(fname):
            prog = pd.read_csv(fname, index_col=False)
            prog.to_csv(output_dir+'Sector'+str(sector)+'backup.csv',
                        index=False)
            ticid_prog = prog['ID'].to_numpy()

        with open(fname, 'w') as f:
            f.write(','.join(columns)+'\n')

        for i in range(len(ticid)):
            if i % 100 == 0:
                print(str(i) + '/' + str(len(ticid)))


            if ticid[i] not in ticid_prog:
                if ticid[i] in data_ticid:
                    ind = np.nonzero(data_ticid == ticid[i])[0][0]
                    with open(fname, 'a') as f:
                        f.write(','.join(lines[ind].split(',')[:124])+'\n')
                else:
                    target = 'TIC ' + str(int(ticid[i]))
                    print('Querying '+target)
                    catalog_data = \
                        Catalogs.query_object(target, radius=0.02, catalog='TIC')
                    line = []
                    for col in columns:
                        line.append(catalog_data[0][col])
                    line = np.array(line) # >> clean up
                    line[np.nonzero(line == 'nan')] = ''
                    with open(fname, 'a') as f:
                        f.write(','.join(line) +'\n')
            else:
                print('Already completed '+str(ticid[i]))
                ind = np.nonzero(ticid_prog == ticid[i])[0][0]
                with open(fname, 'a') as f:
                    f.write(','.join(prog.loc[ind].to_numpy().astype('str'))+'\n')
                

    

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

def download_TIC_catalog(output_dir='./'):
    '''Downloads TICv8 catalog in CSV format from:
    https://archive.stsci.edu/tess/tic_ctl.html'''
    url = 'https://archive.stsci.edu/missions/tess/catalogs/tic_v81/'
    # >> make a list of filenames
    fnames = []
    dec = np.arange(2,92,2)
    for i in range(len(dec)-1):
        fname = 'tic_dec%02d'%dec[i]+'_00N__%02d'%dec[i+1]+'_00N.csv.gz'
        fnames.append(fname)
        fname = 'tic_dec%02d'%dec[-i-1]+'_00S__%02d'%dec[-i-2]+'_00S.csv.gz'
        fnames.append(fname)
    fnames.append('tic_dec02_00S__00_00N.csv.gz')
    fnames.append('tic_dec00_00N__02_00N.csv.gz')    

    # >> download csv files
    for fname in fnames:
        if os.path.exists(output_dir+fname[:-3]):
            print('Skipping '+fname+' (already downloaded)')
        else:
            print('Downloading '+fname+' ...')
            os.system('curl -# -o '+output_dir+fname+' '+url+fname)

    # >> extract csv files
    for fname in fnames:
        if os.path.exists(output_dir+fname):
            print('Unzipping '+fname+' ...')
            os.system('gunzip '+output_dir+fname)

    # >> combine all csv files
    os.system('cat tic_dec*.csv > tic_dec_all.csv')

    # >> also download column descriptions
    fname = 'tic_column_description.txt'
    os.system('curl -# -o '+output_dir+fname+' '+url+fname)
    
def get_TIC_catalog_sector(data_dir='data/'):
    '''Will read TICv8 CSV files and return a pandas dataframe as another csv
    (for CTL objects only)'''
    # import glob

    # >> get column descriptions
    # columns = np.loadtxt(data_dir+'tic_column_description.txt',
    #                      dtype='str')[:,0] # >> take first column
    # columns = np.char.replace(columns, '[', '') # >> clean up
    # columns = np.char.replace(columns, ']', '')
    columns = np.loadtxt(data_dir+'exo_CTL_08.01xTIC_v8.1_header.csv',
                         dtype='str', delimiter=',')
    columns = np.char.replace(columns, '[', '') # >> clean up
    columns = np.char.split(columns, ']')
    columns = [x[0] for x in columns]

    # >> get TICIDs for each sector
    ticid = []
    sectors = np.arange(1,27)
    for sector in sectors:
        fname = 'Sector'+str(sector)+'/all_targets_S%03d'%sector+'_v1.txt'
        ticid_sector = np.loadtxt(data_dir+fname)[:,0] # >> take first column
        ticid.append(ticid_sector)

    # # >> initialize sector_df
    # sector_df = {}
    # for i in range(len(sectors)):
    #     sector_df[i] = []

    # for fname in fnames:
        # print('Loading ' + fname + ' ...')
        # for chunk in pd.read_csv(fname, header=None, chunksize=50000,
        #                          low_memory=False):

        #     for i in range(len(sectors)):
        #         # >> find data only for our desired TICIDs
        #         _, _, inds = np.intersect1d(ticid[i], chunk[0],
        #                                     return_indices=True)

        #         if len(inds) > 0:
        #             sector_df[i].append(chunk.iloc[inds])
        #             if len(sector_df[i]) > 1:
        #                 sector_df[i] = [pd.concat(sector_df[i])]
        #             sector_df[i][0].to_csv(data_dir+'Sector'+str(i+1)+'/Sector'+\
        #                                    str(i+1)+'tic_cat.csv')

    sector_df = {}

    # # >> find all TICv8 csv files
    # fnames = glob.glob(data_dir+'tic_dec*')
    fname = data_dir + 'exo_CTL_08.01xTIC_v8.1.csv'

    print('Loading ' + fname + ' ...')
    cat = pd.read_csv(fname, header=None, low_memory=False)

    for i in range(len(sectors)):
        # >> find data only for our desired TICIDs
        _, _, inds = np.intersect1d(ticid[i], cat[0], return_indices=True)

        sector_df[i] = cat.iloc[inds]
        sector_df[i].columns = columns

        sector_df[i].to_csv(data_dir+'Sector'+str(i+1)+'/Sector'+\
                            str(i+1)+'tic_cat.csv', index=False)


    # # >> put column descriptions in each .csv file
    # for i in range(len(sectors)):
    #     sector_df[i][0].columns = columns
    #     sector_df[i][0].to_csv(data_dir+'Sector'+str(i+1)+'/Sector'+\
    #                            str(i+1)+'tic_cat.csv')


def get_TIC_check_success(data_dir):
    for sector in range(1,27):
        print('Sector '+str(sector))
        fname = data_dir+'Sector'+str(sector)+'/all_targets_S%03d'%sector+\
                '_v1.txt'
        ticid = np.loadtxt(fname)[:,0]
        fname = data_dir+'Sector'+str(sector)+'/Sector'+str(sector)+\
                'tic_cat_all.csv'
        df = pd.read_csv(fname, index_col=False)
        inter = np.intersect1d(ticid, df['ID'])
        print('Number of targets in tic_cat_all: '+str(len(inter)))
        print('Right order? '+str(ticid==df['ID']))

        fname = data_dir+'Sector'+str(sector)+'/Sector'+str(sector)+\
                'tic_cat_v2.csv'
        df = pd.read_csv(fname, index_col=False)
        inter = np.intersect1d(ticid, df['ID'])
        print('Number of targets: ' + str(len(ticid)))
        print('Number of targets in tic_cat_v2: '+str(len(inter)))


def get_TIC_catalog_sector_v2(data_dir='data/'):
    '''Will read TICv8 CSV files and return a pandas dataframe as another csv.'''

    # >> get column descriptions
    columns = np.loadtxt(data_dir+'tic_column_description.txt',
                         dtype='str')[:,0] # >> take first column
    columns = np.char.replace(columns, '[', '') # >> clean up
    columns = np.char.replace(columns, ']', '')

    # >> get TICIDs for each sector
    ticid = []
    sectors = np.arange(1,27)
    for sector in sectors:
        fname = 'Sector'+str(sector)+'/all_targets_S%03d'%sector+'_v1.txt'
        ticid_sector = np.loadtxt(data_dir+fname)[:,0] # >> take first column
        ticid.append(ticid_sector)

    # >> initialize sector_df
    sector_df = {}
    for i in range(len(sectors)):
        sector_df[i] = []

    import glob
    fnames = glob.glob(data_dir+'tic_dec*')

    for fname in fnames:
        print('Loading ' + fname + ' ...')
        for chunk in pd.read_csv(fname, header=None, chunksize=50000,
                                 low_memory=False):

            for i in range(len(sectors)):
                # >> find data only for our desired TICIDs
                _, _, inds = np.intersect1d(ticid[i], chunk[0],
                                            return_indices=True)

                if len(inds) > 0:
                    sector_df[i].append(chunk.iloc[inds])
                    if len(sector_df[i]) > 1:
                        sector_df[i] = [pd.concat(sector_df[i])]
                    sector_df[i][0].to_csv(data_dir+'Sector'+str(i+1)+'/Sector'+\
                                           str(i+1)+'tic_cat_v2.csv')

    # >> put column descriptions in each .csv file
    for i in range(len(sectors)):
        sector_df[i][0].columns = columns
        sector_df[i][0].to_csv(data_dir+'Sector'+str(i+1)+'/Sector'+\
                               str(i+1)+'tic_cat_v2.csv')


def build_simbad_database(out='./simbad_database.txt'):
    '''Object type follows format in:
    http://vizier.u-strasbg.fr/cgi-bin/OType?$1
    Can see other Simbad fields with Simbad.list_votable_fields()
    http://simbad.u-strasbg.fr/Pages/guide/sim-fscript.htx
    TODO  change votable field to otypes'''
    
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
        
        #print(obj + ' ' + otype)
        
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

# ::::::::::::::::::::::::;::::::::::::::::;;;;:::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Query catalogs ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def query_simbad(sector='all', data_dir='data/', query_mast=False):
    # import time
    
    customSimbad = Simbad()
    customSimbad.add_votable_fields('otypes')
    # customSimbad.add_votable_fields('biblio')

    if sector=='all':
        sectors = list(range(1,27))
    else:
        sectors=[sector]

    for sector in sectors:
        print('Sector '+str(sector))
        out_f = data_dir+'databases/Sector'+str(sector)+'_simbad.txt'

        ticid_simbad = []
        otypes_simbad = []
        main_id_simbad = []
        bibcode_simbad = []

        with open(out_f, 'a') as f: # >> make file if not already there
            f.write('')    

        with open(out_f, 'r') as f:
            lines = f.readlines()
            ticid_already_classified = []
            for line in lines:
                ticid_already_classified.append(float(line.split(',')[0]))

        if not query_mast:
            tic_cat=pd.read_csv(data_dir+'Sector'+str(sector)+'/Sector'+str(sector)+\
                                     'tic_cat_all.csv', index_col=False)
            ticid_list = tic_cat['ID']

        print(str(len(ticid_list))+' targets')
        print(str(len(ticid_already_classified))+' targets completed')
        ticid_list = np.setdiff1d(ticid_list, ticid_already_classified)
        print(str(len(ticid_list))+' targets to query')

        count = 0
        for tic in ticid_list:

            count += 1
            res = None

            while res is None:
                try:
                    print(str(count)+'/'+str(len(ticid_list))+\
                          ': finding object type for Sector ' +str(sector)+\
                          ' TIC' + str(int(tic)))

                    target = 'TIC ' + str(int(tic))                    
                    if query_mast:
                        # >> get coordinates
                        catalog_data = Catalogs.query_object(target, radius=0.02,
                                                             catalog='TIC')[0]

                    else:
                        ind = np.nonzero(tic_cat['ID'].to_numpy() == tic)[0][0]
                        catalog_data=tic_cat.iloc[ind]
                    # time.sleep(6)


                    # -- get object type from Simbad ---------------------------

                    # >> first just try querying Simbad with the TICID
                    res = customSimbad.query_object(target)
                    # time.sleep(6)

                    # >> if no luck with that, try checking other catalogs
                    catalog_names = ['TYC', 'HIP', 'TWOMASS', 'SDSS', 'ALLWISE',
                                     'GAIA', 'APASS', 'KIC']
                    for name in catalog_names:
                        if type(res) == type(None):
                            if type(catalog_data[name]) != np.ma.core.MaskedConstant:
                                target_new = name + ' ' + str(catalog_data[name])
                                res = customSimbad.query_object(target_new)
                                # time.sleep(6)


                    if type(res) == type(None):
                        print('failed :(')
                        res=0 
                        with open(out_f, 'a') as f:
                            f.write('{},{},{}\n'.format(tic, '', ''))              
                        ticid_simbad.append(tic)
                        otypes_simbad.append('none')
                        main_id_simbad.append('none')                
                    else:
                        otypes = res['OTYPES'][0].decode('utf-8')
                        main_id = res['MAIN_ID'].data[0].decode('utf-8')
                        ticid_simbad.append(tic)
                        otypes_simbad.append(otypes)
                        main_id_simbad.append(main_id)

                        with open(out_f, 'a') as f:
                            f.write('{},{},{}\n'.format(tic, otypes, main_id))

                    # time.sleep(6)
                except:
                    pass
                    print('connection failed! Trying again now')
            
def query_gcvs(data_dir='./', sector='all', tol=0.1, diag_plot=True):
    '''Cross-matches GCVS catalog with TIC catalog.
    * data_dir
    * sector: 'all' or int, currently only handles short-cadence
    * tol: maximum separation of TIC target and GCVS target (in arcsec)
    '''
    data = pd.read_csv(data_dir+'gcvs_database.csv')
    print('Loaded gcvs_database.csv')
    data_coords = coord.SkyCoord(data['RAJ2000'], data['DEJ2000'],
                                 unit=(u.hourangle, u.deg))

    if sector=='all':
        sectors = list(range(1,27))
    else:
        sectors=[sector]

    for sector in sectors:
        prefix = data_dir+'databases/Sector'+str(sector)+'_gcvs'
        out_fname = prefix+'.txt'

        sector_data = pd.read_csv(data_dir+'Sector'+str(sector)+\
                                  '/Sector'+str(sector)+'tic_cat_all.csv',
                                  index_col=False)
        print('Loaded Sector'+str(sector)+'tic_cat_all.csv')

        # >> find GCVS target closest to each TIC target
        if os.path.exists(prefix+'_sep.txt'):
            sep_arcsec = np.loadtxt(prefix+'_sep.txt')
            min_inds = np.loadtxt(prefix+'_sep_inds.txt').astype('int')
            print('Loaded '+prefix+'_sep.txt')
        else:
            min_sep = []
            min_inds = []
            for i in range(len(sector_data)):
                print('TIC '+str(int(sector_data['ID'][i]))+'\t'+str(i)+'/'+\
                      str(len(sector_data)))
                ticid_coord = coord.SkyCoord(sector_data['ra'][i],
                                             sector_data['dec'][i],
                                             unit=(u.deg, u.deg)) 
                sep = ticid_coord.separation(data_coords)
                min_sep.append(np.nanmin(sep))
                ind = np.nanargmin(sep)
                min_inds.append(ind)

            sep_arcsec = np.array([sep.to(u.arcsec).value for sep in min_sep])
            min_inds = np.array(min_inds).astype('int')
            np.savetxt(prefix+'_sep.txt', sep_arcsec)
            np.savetxt(prefix+'_sep_inds.txt', min_inds)

        # >> save the variability type if GCVS target is close enough
        with open(out_fname, 'w') as f:

            for i in range(len(sector_data)):
                if sep_arcsec[i] < tol:
                    ind = min_inds[i]            
                    f.write(str(int(sector_data['ID'][i]))+','+\
                            str(data['VarType'][ind])+','+\
                            str(data['VarName'][ind])+'\n')

                else:
                    f.write(str(int(sector_data['ID'][i]))+',,\n')

        # >> plotting
        if diag_plot:
            # >> make histogram of minimum separations
            fig, ax = plt.subplots()
            bins = 10**np.linspace(np.floor(np.log10(np.nanmin(sep_arcsec))),
                                   np.ceil(np.log10(np.nanmax(sep_arcsec))), 50)
            ax.hist(sep_arcsec, bins=bins, log=True)
            ax.set_xlabel('arcseconds')
            ax.set_ylabel('number of targets in Sector '+str(sector))
            ax.set_xscale('log')
            fig.savefig(prefix+'_sep_arcsec.png')
            
            # >> compare magnitude from TIC and ASAS-SN of cross-matched targets
            tol_tests = [10, 1, 0.1] 
            for tol in tol_tests:
                inds1 = np.nonzero(sep_arcsec < tol)
                inds2 = min_inds[inds1]
                print('Tolerance: '+str(tol)+' arcseconds, number of targets: '+\
                      str(len(inds1[0])))
                plt.figure()
                plt.plot(sector_data['GAIAmag'][inds1[0]], data['magMax'][inds2], '.k')
                plt.xlabel('GAIA magnitude (TIC)')
                plt.ylabel('magMax (GCVS)')
                plt.savefig(prefix+'_tol'+str(tol)+'.png')
                plt.close()
                
# def query_asas_sn(data_dir='./', sector='all', diag_plot=True):
#     '''Cross-matches ASAS-SN catalog with TIC catalog based on matching GAIA IDs
#     * data_dir
#     * sector: 'all' or int, currently only handles short-cadence
#     '''
#     data = pd.read_csv(data_dir+'asas_sn_database.csv')
#     print('Loaded asas_sn_database.csv')
#     data_coords = coord.SkyCoord(data['RAJ2000'], data['DEJ2000'],
#                                  unit=(u.deg, u.deg))

#     if sector=='all':
#         sectors = list(range(1,27))
#     else:
#         sectors=[sector]

#     for sector in sectors:
#         # >> could also have retrieved ra dec from all_targets_S*_v1.txt
#         sector_data = pd.read_csv(data_dir+'Sector'+str(sector)+\
#                                   '/Sector'+str(sector)+'tic_cat_all.csv',
#                                   index_col=False)
#         print('Loaded Sector'+str(sector)+'tic_cat_all.csv')
#         out_fname = data_dir+'databases/Sector'+str(sector)+'_asassn.txt'

#         _, comm1, comm2 = np.intersect1d(sector_data['GAIA'], data['GDR2_ID'],
#                                          return_indices=True)

#         # >> save cross-matched target in text file
#         with open(out_fname, 'w') as f:
#             for i in range(len(sector_data)):            
#                 if i in comm1:
#                     ind = comm2[np.nonzero(comm1 == i)][0]
#                     f.write(str(int(sector_data['ID'][i]))+','+\
#                             str(data['Type'][ind])+','+str(data['ID'][ind])+'\n')
#                 else:
#                     f.write(str(int(sector_data['ID'][i]))+',,\n')
#         print('Saved '+out_fname)

#         if diag_plot:
#             prefix = data_dir+'databases/Sector'+str(sector)+'_'

#             # >> compare magnitude from TIC and ASAS-SN of cross-matched targets
#             plt.figure()
#             plt.plot(sector_data['GAIAmag'][comm1], data['Mean Vmag'][comm2], '.k')
#             plt.xlabel('GAIA magnitude (TIC)')
#             plt.ylabel('Mean Vmag (ASAS-SN)')
#             plt.savefig(prefix+'asassn_mag_cross_match.png')
#             plt.close()

#             # >> get minimum separations between TIC and ASAS-SN targts
#             if os.path.exists(prefix+'asassn_sep.txt'):
#                 sep_arcsec = np.loadtxt(prefix+'asassn_sep.txt')
#                 min_inds = np.loadtxt(prefix+'asassn_sep_inds.txt').astype('int')
#             else:
#                 min_sep = []
#                 min_inds = []
#                 for i in range(len(sector_data)):
#                     print('TIC '+str(int(sector_data['ID'][i]))+'\t'+str(i)+'/'+\
#                           str(len(sector_data)))
#                     ticid_coord = coord.SkyCoord(sector_data['ra'][i],
#                                                  sector_data['dec'][i],
#                                                  unit=(u.deg, u.deg)) 
#                     sep = ticid_coord.separation(data_coords)
#                     min_sep.append(np.min(sep))
#                     min_inds.append(np.argmin(sep))
#                 sep_arcsec = np.array([sep.to(u.arcsec).value for sep in min_sep])
#                 min_inds = np.array(min_inds)
#                 np.savetxt(prefix+'asassn_sep.txt', sep_arcsec)
#                 np.savetxt(prefix+'asassn_sep_inds.txt', min_inds)

#             # >> make histogram of minimum separations
#             fig, ax = plt.subplots()
#             ax.hist(sep_arcsec, bins=10**np.linspace(-2, 4, 30), log=True)
#             ax.set_xlabel('arcseconds')
#             ax.set_ylabel('number of targets in Sector '+str(sector))
#             ax.set_xscale('log')
#             fig.savefig(prefix+'asassn_sep_arcsec.png')

#             fig1, ax1 = plt.subplots()
#             ax1.hist(sep_arcsec[comm1], bins=10**np.linspace(-2, 4, 30), log=True)
#             ax1.set_xlabel('arcseconds')
#             ax1.set_ylabel('number of cross-matched targets in Sector '+str(sector))
#             ax1.set_xscale('log')
#             ax1.set_ylim(ax.get_ylim())
#             fig1.savefig(prefix+'asassn_sep_cross_match.png')

#             # >> compare magnitude from TIC and ASAS-SN of cross-matched targets
#             tol_tests = [10, 1, 0.1]
#             for tol in tol_tests:
#                 inds1 = np.nonzero(sep_arcsec < tol)
#                 inds2 = min_inds[inds1]
#                 plt.figure()
#                 plt.plot(sector_data['GAIAmag'][inds1][0], data['Mean Vmag'][inds2], '.k')
#                 plt.xlabel('GAIA magnitude (TIC)')
#                 plt.ylabel('Mean Vmag (ASAS-SN)')
#                 plt.savefig(prefix+'asassn_mag_tol'+str(tol)+'.png')
#                 plt.close()

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Organizing object types :::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
def get_otype_dict(data_dir='/nfs/blender/data/tdaylan/data/',
                   uncertainty_flags=[':', '?', '*']):
    '''Return a dictionary of object type descriptions.'''
    
    d = {}
        
    with open(data_dir + 'gcvs_labels.txt', 'r') as f:
        lines = f.readlines()
    for line in lines:
        otype, description = line.split(' = ')
        
        # >> remove uncertainty flags
        if otype[-1] in uncertainty_flags:
            otype = otype[:-1]
        
        # >> remove new line character
        description = description.replace('\n', '')
        
        d[otype] = description
        
    return d


def make_parent_dict():    
    d = {'I': ['IA', 'IB'],
         'IN': ['FU', 'INA', 'INB', 'INTIT', 'IN(YY)', 'INAT', 'INS', 'INSA',
                'INSB', 'INST', 'INT', 'INT(YY)', 'IT'],
         'IS': ['ISA', 'ISB'],
         'Fl': ['UV', 'UVN'],
         
         'BCEP': ['BCEPS', 'BCEP(B)'],
         'CEP': ['CEP(B)', 'DCEP', 'DCEPS'],
         'CW': ['CWA', 'CWB'],
         'DSCT': ['DSCTC', 'DSCTC(B)'],
         'L': ['LB', 'LC', 'LPB', 'LP', 'LBV'],
         'RR': ['RR(B)', 'RRAB', 'RRC'],
         'RV': ['RVA', 'RVB'],
         'SR': ['SRA', 'SRB', 'SRC', 'SRD', 'SRS'],
         'ZZ': ['ZZA', 'ZZB', 'ZZO', 'ZZLep'],
         
         'ACV': ['ACVO'],
         
         'N': ['NA', 'NB', 'NC', 'NL', 'NR'],
         'SN': ['SNI', 'SNII'],
         'UG': ['UGSS', 'UGSU', 'UGZ'],
         
         # 'E': ['EA', 'EB', 'EP', 'EW'],
         'D': ['DM', 'DS', 'DW'],
         'K': ['KE', 'KW'],
         
         'X': ['XB', 'XF', 'XI', 'XJ', 'XND', 'XNG', 'XP', 'XPR',
               'XPRM', 'XM', 'XRM', 'XN','XNA','XNGP','XPM','XPNG',
               'XNP'],
         }    

    parents = list(d.keys())

    # >> turn into array
    subclasses = []
    for parent in parents:
        subclasses.extend(d[parent])

    return d, parents, subclasses

def make_variability_tree():
    var_d = {'eruptive':
         ['Fl', 'BE', 'FU', 'GCAS', 'I', 'IA', 'IB', 'IN', 'INA', 'INB', 'INT,IT',
          'IN(YY)', 'IS', 'ISA', 'ISB', 'RCB', 'RS', 'SDOR', 'UV', 'UV', 'UVN',
          'WR', 'INTIT', 'GCAS'],
         'pulsating':
             ['Pu', 'ACYG', 'BCEP', 'BCEPS', 'BLBOO', 'CEP', 'CEP(B)', 'CW', 'CWA',
              'CWB', 'DCEP', 'DCEPS', 'DSCT', 'DSCTC', 'GDOR', 'L', 'LB', 'LC',
              'LPB', 'M', 'PVTEL', 'RPHS', 'RR', 'RR(B)', 'RRAB', 'RRC', 'RV',
              'RVA', 'RVB', 'SR', 'SRA', 'SRB' 'SRC', 'SRD', 'SRS', 'SXPHE',
              'ZZ', 'ZZA', 'ZZB', 'ZZO'],
         'rotating': ['ACV', 'ACVO', 'BY', 'ELL', 'FKCOM', 'PSR',
                      'R', 'SXARI'],
         'cataclysmic':
             ['N', 'NA', 'NB', 'NC', 'NL', 'NR', 'SN', 'SNI', 'SNII', 'UG',
              'UGSS', 'UGSU', 'UGZ', 'ZAND'],
         'eclipsing':
             ['E', 'EA', 'EB', 'EP', 'EW', 'GS', 'PN', 'RS', 'WD', 'WR', 'AR',
              'D', 'DM', 'DS', 'DW', 'K', 'KE', 'KW', 'SD'],
             'xray':
             ['AM', 'X', 'XB', 'XF', 'XI', 'XJ', 'XND', 'XNG', 'XP', 'XPR', 'XPRM',
              'XM'],
             'other': ['VAR']} 
    return var_d

def make_redundant_otype_dict():
    # >> keys are redundant object types, and will be removed if star is also
    # >> classified as any of the associated dictionary values
    var_d = make_variability_tree()

    d = {'**': var_d['eclipsing']+['R'],
         'E':  ['EA', 'EB', 'EP', 'EW', 'GS', 'PN', 'RS', 'WD', 'WR', 'AR', 'D',
                'DM', 'DS', 'DW', 'K', 'KE', 'KW', 'SD'],
         'Er': var_d['eruptive'],
         'ROT': var_d['rotating'],
         'Ro': var_d['rotating'],
         'Pu': var_d['pulsating'],
         'L': var_d['eruptive']+var_d['rotating']+\
         var_d['cataclysmic']+var_d['eclipsing']+var_d['xray']+var_d['other']+\
        ['Pu', 'ACYG', 'BCEP', 'BCEPS', 'BLBOO', 'CEP', 'CEP(B)', 'CW', 'CWA',
              'CWB', 'DCEP', 'DCEPS', 'DSCT', 'DSCTC', 'GDOR', 'LB', 'LC',
              'LPB', 'M', 'PVTEL', 'RPHS', 'RR', 'RR(B)', 'RRAB', 'RRC', 'RV',
              'RVA', 'RVB', 'SR', 'SRA', 'SRB' 'SRC', 'SRD', 'SRS', 'SXPHE',
              'ZZ', 'ZZA', 'ZZB', 'ZZO'],
         'LP': var_d['eruptive']+var_d['pulsating']+var_d['rotating']+\
         var_d['cataclysmic']+var_d['eclipsing']+var_d['xray']+var_d['other'],
         'RR': ['CEP']
         }
    parents = list(d.keys())

    # >> turn into array
    subclasses = []
    for parent in parents:
        subclasses.extend(d[parent])

    return d, parents, subclasses


def merge_otype(otype_list):

    # >> merge classes
    parent_dict, parents, subclasses = make_parent_dict()

    new_otype_list = []
    for otype in otype_list:
        if otype in subclasses:
            # >> find parent
            for parent in parents:
                if otype in parent_dict[parent]:
                    new_otype = parent
            new_otype_list.append(new_otype)
        else:
            new_otype_list.append(otype)
    otype_list = new_otype_list

    # >> remove redundant classes 
    redundant_dict, parents, subclasses = make_redundant_otype_dict()
    new_otype_list = []
    for otype in otype_list:
        if otype in parents:
            if len(np.intersect1d(redundant_dict[otype], otype_list))>0:
                new_otype_list.append('')
            else:
                new_otype_list.append(otype)
        else:
            new_otype_list.append(otype)    

            
    otype_list = np.unique(new_otype_list).astype('str')
    otype = np.delete(otype, np.where(otype == ''))
    return otype_list


def get_parent_otypes(ticid, otypes, remove_classes=['PM','IR','UV','X']):
    '''Finds all the objects with same parent and combines them into the same
    class
    '''

    parent_dict, parents, subclasses =make_parent_dict()
    parents = list(parent_dict.keys())

    # >> turn into array
    subclasses = []
    for parent in parents:
        subclasses.extend(parent_dict[parent])

    new_otypes = []
    for i in range(len(otypes)):
        otype = otypes[i].split('|')

        new_otype=[]
        for o in otype:
            if not o in remove_classes:
                if o in subclasses:
                    for parent in parents: # >> find parent otype
                        if o in parent_dict[parent]:
                            new_o = parent
                    new_otype.append(new_o)
                else:
                    new_otype.append(o)

        # >> remove repeats
        new_otype = np.unique(new_otype)

        # >> remove parent if child in otype list e.g. E|EA or E|EW is redundant
        for parent in parents:
            if parent in new_otype:
                if len(np.intersect1d(new_otype, parent_dict[parent]))>0:
                    new_otype = np.delete(new_otype,
                                          np.nonzero(new_otype==parent))

        new_otypes.append('|'.join(new_otype.astype('str')))
    
    new_otypes = np.array(new_otypes)

    # # >> get rid of empty classes
    # inds = np.nonzero(new_otypes == '')
    # new_otypes = np.delete(new_otypes, inds)
    # ticid = np.delete(ticid, inds)

    inds = np.nonzero(new_otypes == '')
    new_otypes[inds] = 'NONE'
                
    return ticid, new_otypes



def get_parents_only(class_info, parent_dict=None,
                     remove_classes=[], remove_flags=[]):
    '''Finds all the objects with same parent and combines them into the same
    class
    TODO: get rid of this function
    '''
    classes = []
    new_class_info = []

    if type(parent_dict) == type(None):
        parent_dict = make_parent_dict()

    parents = list(parent_dict.keys())

    # >> turn into array
    subclasses = []
    for parent in parents:
        subclasses.extend(parent_dict[parent])

    for i in range(len(class_info)):
        otype_list = class_info[i][1]

        # >> remove any flags
        for flag in remove_flags:
            otype_list = otype_list.replace(flag, '|')
        otype_list = otype_list.split('|')

        new_otype_list=[]
        for otype in otype_list:
            if not otype in remove_classes:
                if otype in subclasses:
                    # >> find parent
                    for parent in parents:
                        if otype in parent_dict[parent]:
                            new_otype = parent

                    new_otype_list.append(new_otype)
                else:
                    new_otype_list.append(otype)

        # >> remove repeats
        new_otype_list = np.unique(new_otype_list)

        # >> don't want e.g. E|EA or E|EW (redundant)
        if 'E' in new_otype_list:
            if len(np.intersect1d(new_otype_list, ['EA', 'EP', 'EW', 'EB']))>0:
                new_otype_list = np.delete(new_otype_list, np.nonzero(new_otype_list=='E'))

        if 'L' in new_otype_list and len(new_otype_list) > 1:
            new_otype_list = np.delete(new_otype_list,
                                       np.nonzero(new_otype_list=='L'))

        if '' in new_otype_list:
            new_otype_list = np.delete(new_otype_list,
                                       np.nonzero(new_otype_list==''))

        # if '|'.join(new_otype_list) == '|AR|EA|EB|RS|SB':
        #     pdb.set_trace()

        new_class_info.append([class_info[i][0], '|'.join(new_otype_list),
                               class_info[i][2]])

    # >> get rid of empty classes
    new_class_info = np.array(new_class_info)
    new_class_info = np.delete(new_class_info,
                               np.nonzero(new_class_info[:,1]==''), 0)
            
    
    return new_class_info

def make_remove_class_list(simbad=False, rmv_flagged=True):
    '''Currently, our pipeline can only do clustering based on photometric data.
    So classes that require spectroscopic data, etc. are removed.'''
    rmv = ['PM', 'IR', 'nan', 'V', 'VAR', 'As', 'SB', 'LM', 'blu', 'EmO', 'S',
           ]
    sequence_descriptors = ['AB', 'HS', 'BS', 'YSO', 'Y', 'sg', 'BD', 's*b']

    if simbad:
        rmv.append('UV')
        rmv.append('X')

    if rmv_flagged:
        flagged = make_flagged_class_list()
    else:
        flagged = []

    return rmv+sequence_descriptors+flagged

def read_otype_txt(otypes, otype_txt, data_dir, simbad=False, add_chars=['+', '/'],
                   uncertainty_flags=[':', '?', '*']):

    rmv_classes = make_remove_class_list(simbad=simbad)

    if simbad:
        with open(data_dir+'simbad_gcvs_label.txt', 'r') as f:
            lines = f.readlines()
        otype_dict = {}
        for line in lines:
            otype, otype_gcvs = line.split(' = ')
            otype_gcvs = otype_gcvs.replace('\n', '')
            otype_dict[otype] = otype_gcvs

    with open(otype_txt, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            tic, otype, main_id = lines[i].split(',')

            # >> make list of labels
            for char in add_chars:
                otype = otype.replace(char, '|')
            otype_list = otype.split('|')

            # >> remove unceratinty flags
            otype_list_new = []
            stop=False
            for o in otype_list:
                if o == 'UGSU':
                    stop=True
                if len(o) > 0 and o != '**':
                    # >> remove uncertainty flags
                    if o[-1] in uncertainty_flags:
                        o = o[:-1]
                    if '(' in o: # >> remove (B) flag
                        o = o[:o.index('(')]

                    # >> convert to GCVS nomenclature
                    if simbad and o in list(otype_dict.keys()): 
                        o = otype_dict[o]

                    # >> remove classes that require external information
                    if o in rmv_classes:
                        o = ''
                    otype_list_new.append(o)

            otypes[float(tic)] = np.unique(otype_list_new)

    return otypes



# def correct_simbad_to_vizier(in_f='./SectorX_simbad.txt',
#                              out_f='./SectorX_simbad_revised.txt',
#                              simbad_gcvs_conversion='./simbad_gcvs_label.txt',
#                              uncertainty_flags=[':', '?', '*']):
#     '''TODO: Clean up args.'''
    
#     with open(simbad_gcvs_conversion, 'r') as f:
#         lines = f.readlines()
#     renamed = {}
#     for line in lines:
#         otype, description = line.split(' = ')
        
#         # >> remove new line character
#         description = description.replace('\n', '')
        
#         renamed[otype] = description    
    
#     with open(in_f, 'r') as f:
#         lines = f.readlines()
        
        
#     for line in lines:
#         tic, otype, main = line.split(',')
#         otype = otype.replace('+', '|')
#         otype_list = otype.split('|')
#         otype_list_new = []
        
#         for o in otype_list:
            
#             if len(o) > 0:
#                 # >> remove uncertainty_flags
#                 if o[-1] in uncertainty_flags:
#                     o = o[:-1]
                    
#                 # >> remove (B)
#                 if '(' in o:
#                     o = o[:o.index('(')]
                    
#                 if o in list(renamed.keys()):
#                     o = renamed[o]
                
#             otype_list_new.append(o)
                
                
#         otype = '|'.join(otype_list_new)
        
        
#         with open(out_f, 'a') as f:
#             f.write(','.join([tic, otype, main]))



def quick_simbad(ticidasstring):
    """ only returns if it has a tyc id"""
    catalogdata = Catalogs.query_object(ticidasstring, radius=0.02, catalog="TIC")[0]
    try: 
        tyc = "TYC " + catalogdata["TYC"]
        customSimbad = Simbad()
        customSimbad.add_votable_fields("otypes")
        res = customSimbad.query_object(tyc)
        objecttype = res['OTYPES'][0].decode('utf-8')
    except: 
        objecttype = "there is no TYC for this object"
    return objecttype



def get_true_classifications(ticid=[], data_dir='./', sector='all'):
    '''Reads Sector*_true_labels.txt, generated from make_true_label_txt()
    * 
    * sector: either 'all' or int'''
    ticid_true = []
    otypes = []
    # database_dir = data_dir+'databases/'
    database_dir = data_dir+'true/'
    
    
    # >> find all text files in directory
    if sector == 'all':
        fnames = fm.filter(os.listdir(database_dir), '*_true_labels.txt')
    else:
        fnames = ['Sector'+str(sector)+'_true_labels.txt']
    
    for fname in fnames:
        data = np.loadtxt(database_dir+fname, delimiter=',', dtype='str')
        ticid_true.extend(data[:,0].astype('float'))
        otypes.extend(data[:,1])

    # >> only return classified targets in ticid list, if given
    if len(ticid) > 0:
        _, inds, _ = np.intersect1d(ticid_true, ticid, return_indices=True)
        ticid_true = np.array(ticid_true)[inds]
        otypes = np.array(otypes)[inds]
    
    return ticid_true, otypes

def load_otype_true_from_datadir(metapath, ticid, sector, savepath):
    '''Reads *-true_labels.txt and returns:
    * otype : Variability types (following nomenclature by GCVS)'''
    
    print('Loading ground truth object types...')

    ticid_true = []
    sector_true = []
    otype_true = []

    # >> read sector files from metapath/spoc/true/
    for fname in sorted(os.listdir(metapath+'spoc/true/')):
        filo = np.loadtxt(metapath+'spoc/true/'+fname, delimiter=',',
                          dtype='str', skiprows=2)
        ticid_sector = filo[:,0].astype('int')
        otype_sector = filo[:,1]
        # filo = pd.read_csv(metapath+'spoc/true/'+fname, delimiter='\s+,',
        #                    skiprows=1)
        # ticid_sector = filo['TICID'].to_numpy().astype('int')
        # otype_sector = filo['TYPE'].to_numpy().astype('str')

        s = int(float(fname[7:9]))
        
        ticid_true.extend(ticid_sector)
        sector_true.extend(np.ones(ticid_sector.shape)*s)
        otype_true.extend(otype_sector)

    # ticid_true = np.array(ticid_true)
    ticid_true = np.array(ticid_true)
    sector_true = np.array(sector_true)

    # >> 
    totype = []
    for i in range(len(ticid)):
        ind = np.nonzero((sector_true == sector[i])*(ticid_true == ticid[i]))
        if len(ind[0]) == 0:
            totype.append('')
        else:
            if otype_true[ind[0][0]] == '':
                totype.append('NONE')
            else:
                totype.append(otype_true[ind[0][0]])
    totype = np.array(totype)

    # otype = np.array(otype)
    # otype_new = []
    # for tic in ticid_true:
    #     ind = np.nonzero(ticid_true == tic)[0][0]
    #     if otype[ind] == 'nan' or otype[ind] == '':
    #         otype_new.append('NONE')
    #     else:
    #         otype_new.append(otype[ind])


    # ticid_true = np.array(ticid_true)
    # otype = np.array(otype)
        
    # orderind = order_array(ticid, ticid_true)
    # otype = otype_new[orderind]

    # return np.array(otype_new)

    np.savetxt(savepath+'totype.txt', np.array([ticid, sector, totype]).T,
               fmt='%s', delimiter=',', header='OBJID,SECTOR,OTYPE')

    return totype

def load_otype_pred_from_txt(ensbpath, sector, ticid):
    '''Reads *-ticid_to_label.txt and returns:
    * otype : Object types'''
    print('Loading predicted object types...')

    fname = ensbpath+'Sector'+str(sector)+'-ticid_to_label.txt'
    fileo = np.loadtxt(fname, delimiter=',', dtype='str', skiprows=1)
    ticid_unsorted = fileo[:,0].astype('float')
    otype = fileo[:,1]

    # >> re-order otype so that ticid_unsorted = ticid
    orgsrti = np.argsort(ticid)      # >> indices that would sort ticid
    orgunsrti = np.argsort(orgsrti)  # >> indices that would return original
                                     # >> ordering of ticid
    
    intsc, _, srtinds = np.intersect1d(ticid, ticid_unsorted,
                                       return_indices=True)
    if len(intsc) != len(ticid):
        sdiff = np.setdiff1d(intsc, ticid)
        print('!! Variability classifications were not found for '+str(sdiff)+\
              ' TICIDs.')





    otype = otype[srtinds][orgunsrti] # >> order otype correctly

    return otype

def order_array(arr1, arr2):
    '''Returns array of indices, which will sort arr2 so that arr1=arr2.
    An example would be two arrays of the same TICIDs, but in different
    order: 
    orderind = order_array(ticid1, ticid2)
    ticid1 == ticid2[orderind]
    '''
    orgsrti = np.argsort(arr1)      # >> indices that would sort arr1
    orgunsrti = np.argsort(orgsrti)  # >> indices that would return original
                                     # >> ordering of arr1
    
    intsc, _, srtinds = np.intersect1d(arr1, arr2, return_indices=True)

    # if len(intsc) != len(ticid):
    #     sdiff = np.setdiff1d(intsc, ticid)
    #     print('!! Variability classifications were not found for '+str(sdiff)+\
    #           ' TICIDs.')

    orderind = srtinds[orgunsrti] # >> will order arr2 to that arr1 = arr2

    return orderind


# def get_true_classifications(ticid_list,
#                              database_dir='./databases/',
#                              single_file=False, sector=None,
#                              useless_classes = ['*', 'IR', 'UV', 'X', 'PM',
#                                                 '?', ':'],
#                              uncertainty_flags = ['*', ':', '?']):
#     '''Query classifications and bibcode from *_database.txt file.
#     Returns a list where class_info[i] = [ticid, obj type, bibcode]
#     Object type follows format in:
#     http://vizier.u-strasbg.fr/cgi-bin/OType?$1
#     '''
#     ticid_classified = []
#     class_info = []
    
#     # >> find all text files in directory
#     if single_file:
#         fnames = ['']
#     else:
#         fnames = fm.filter(os.listdir(database_dir), '*.txt')
    
#     for fname in fnames:
#         # >> read text file
#         with open(database_dir + fname, 'r') as f:
#             lines = f.readlines()
#             for line in lines:
#                 ticid, otype, bibcode = line[:-1].split(',')
                

                
#                 # >> remove any repeats and any empty classes and sort
#                 otype_list = otype.split('|')
#                 # >> remove any candidate indicators
#                 for i in range(len(otype_list)):
#                     if otype_list[i] != '**' and len(otype_list[i])>0:
#                         if otype_list[i][-1] in uncertainty_flags:
#                             otype_list[i] = otype_list[i][:-1]
#                 otype_list = np.unique(otype_list)
#                 # >> remove useless classes
#                 for u_c in useless_classes + ['']:
#                     if u_c in otype_list:
#                         otype_list =np.delete(otype_list,
#                                               np.nonzero(otype_list == u_c))
#                 otype_list.sort()
#                 otype = '|'.join(otype_list)
                
#                 # >> only get classifications for ticid_list, avoid repeats
#                 # >> and only include objects with interesting lables
#                 ticid = float(ticid)
#                 if ticid in ticid_list and len(otype) > 0:
#                     if ticid in ticid_classified:
#                         ind = np.nonzero(np.array(ticid_classified) == ticid)[0][0]
#                         new_class_info = class_info[ind][1] + '|' + otype
#                         new_class_info = new_class_info.split('|')
#                         new_class_info = '|'.join(np.unique(new_class_info))
#                         class_info[ind][1] = new_class_info
#                     else:
#                         ticid_classified.append(ticid)
#                         class_info.append([int(ticid), otype, bibcode])
                    
#     # >> check for any repeats
#     return np.array(class_info)



def get_gcvs_classifications(database_dir='./databases/',
                             remove_flags=True,
                             remove_classes=['D','DM','DS','DW','K','KE','KW','SD',
                                             'GS', 'PN', 'RS', 'WD', 'WR',
                                             'CST','GAL']):
    ''' D, DM, DS, DW, K, KE, KW, SD ... are subsets of eclipsing binaries'''
    ticid = []
    labels = []
    
    fnames = fm.filter(os.listdir(database_dir), '*_GCVS.txt')
    
    for fname in fnames:

        data = np.loadtxt(database_dir+fname, delimiter=',', dtype='str')
        ticid_sector, otype = data[:,0], data[:,1]
        inds = np.nonzero(otype != '')
        ticid_sector, otype = ticid_sector[inds], otype[inds]
        otype = np.char.replace(otype, ' ', '') # >> remove spaces
        otype = np.char.replace(otype, '+', '|')
        otype = np.char.replace(otype, '/', '|')        
        if remove_flags:
            otype = np.char.replace(otype, ':', '')

        for i in range(len(otype)):
            label_list = otype[i].split('|')
            label_list = np.setdiff1d(label_list, remove_classes)
            label_list = np.unique(label_list) # >> remove repeats
            otype[i] = '|'.join(label_list)

        ticid.extend(ticid_sector)
        labels.extend(otype)
                    

    # >> check for any repeats
    return np.array(ticid), np.array(labels)



                           
def dbscan_param_search(bottleneck, time, flux, ticid, target_info,
                        eps=list(np.arange(0.1,1.5,0.1)),
                        min_samples=[5],
                        metric=['euclidean', 'manhattan', 'minkowski'],
                        algorithm = ['auto', 'ball_tree', 'kd_tree',
                                     'brute'],
                        leaf_size = [30, 40, 50],

                        p = [1,2,3,4],
                        output_dir='./', DEBUG=False, single_file=False,
                        simbad_database_txt='./simbad_database.txt',
                        database_dir='./databases/', pca=True, tsne=True,
                        confusion_matrix=True, tsne_clustering=True):
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
   
    classes = []
    num_classes = []
    counts = []
    num_noisy= []
    parameter_sets=[]
    silhouette_scores=[]
    ch_scores = []
    db_scores = []
    accuracy = []
    param_num = 0
    p0=p

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
                            p = p0
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
                                
                            #param_num = str(len(parameter_sets)-1)
                            title='Parameter Set '+str(param_num)+': '+'{} {} {} {} {} {}'.format(eps[i],
                                                                                        min_samples[j],
                                                                                        metric[k],
                                                                                        algorithm[l],
                                                                                        leaf_size[m],
                                                                                        p[n])
                            
                            prefix='dbscan-p'+str(param_num)                            
                                
                            if confusion_matrix:
                                print('Plotting confusion matrix')
                                acc = pt.plot_confusion_matrix(ticid, db.labels_,
                                                               database_dir=database_dir,
                                                               single_file=single_file,
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
                                print('Computing silhouette score')
                                silhouette = silhouette_score(bottleneck,db.labels_)
                                silhouette_scores.append(silhouette)
                                
                                # >> compute calinski harabasz score
                                print('Computing calinski harabasz score')
                                ch_score = calinski_harabasz_score(bottleneck,
                                                                db.labels_)
                                ch_scores.append(ch_score)
                                
                                # >> compute davies-bouldin score
                                print('Computing davies-bouldin score')
                                dav_boul_score = davies_bouldin_score(bottleneck,
                                                             db.labels_)
                                db_scores.append(dav_boul_score)
                                
                            else:
                                silhouette, ch_score, dav_boul_score = \
                                    np.nan, np.nan, np.nan
                                
                            print('Saving results to text file')
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

                                print('Plotting classification results')
                                pt.quick_plot_classification(time, flux,
                                                             ticid,
                                                             target_info, bottleneck,
                                                             db.labels_,
                                                             path=output_dir,
                                                             prefix=prefix,
                                                             simbad_database_txt=simbad_database_txt,
                                                             title=title,
                                                             database_dir=database_dir,
                                                             single_file=single_file)
                                
                                
                                if pca:
                                    print('Plot PCA...')
                                    pt.plot_pca(bottleneck, db.labels_,
                                                output_dir=output_dir,
                                                prefix=prefix)
                                
                                if tsne:
                                    print('Plot t-SNE...')
                                    pt.plot_tsne(bottleneck, db.labels_,
                                                 output_dir=output_dir,
                                                 prefix=prefix)
                                # if tsne_clustering:
                                    
                                    
                            plt.close('all')
                            param_num +=1
    print("Plot paramscan metrics...")
    pt.plot_paramscan_metrics(output_dir+'dbscan-', parameter_sets, 
                              silhouette_scores, db_scores, ch_scores)
    #print(len(parameter_sets), len(num_classes), len(num_noisy), num_noisy)

    pt.plot_paramscan_classes(output_dir+'dbscan-', parameter_sets, 
                                  np.asarray(num_classes), np.asarray(num_noisy))

        
    return parameter_sets, num_classes, silhouette_scores, db_scores, ch_scores, accuracy

def KNN_plotting(path, features, k_values):
    """ This is based on a metric for finding the best possible eps/minsamp
    value from the original DBSCAN paper (Ester et al 1996). Essentially,
    by calculating the average distances to the k-nearest neighbors and plotting
    those values sorted, you can determine by eye (heuristically) the best eps 
    value. It should be eps value = yaxis value of first valley, and minsamp = k.
    
    ** currently uses default values (minkowski p=2) for the n-neighbor search **
    
    inputs: 
        * path to where you want to save the plots
        * features (should have any significant outliers clipped out)
        * k_values: array of integers, ie [2,3,5,10] for the k values
        
    output: 
        * plots the KNN curves into the path
    modified [lcg 08122020 - created]"""
    from sklearn.neighbors import NearestNeighbors
    for n in range(len(k_values)):
        neigh = NearestNeighbors(n_neighbors=k_values[n])
        neigh.fit(features)
    
        k_dist, k_ind = neigh.kneighbors(features, return_distance=True)
        
        avg_kdist = np.mean(k_dist, axis=1)
        avg_kdist_sorted = np.sort(avg_kdist)[::-1]
        
        plt.scatter(np.arange(len(features)), avg_kdist_sorted)
        plt.xlabel("Points")
        plt.ylabel("Average K-Neighbor Distance")
        plt.ylim((0, 50))
        plt.title("K-Neighbor plot for k=" + str(k_values[n]))
        plt.savefig(path + "kneighbors-" +str(k_values[n]) +"-plot-sorted.png")
        plt.close()    

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

def quick_hdbscan_param_search(features, min_samples=[2,3,4,5,6,7,8,15,50],
                               min_cluster_size=[2,3,5,15,50,100],
                               metric=['all'], p0=[1,2,3,4], output_dir='./'):
    
    import hdbscan
    with open(output_dir + 'hdbscan_param_search.txt', 'a') as f:
        f.write('{} {} {} {} {} {} {}\n'.format("min_cluster_size", "min_samples",
                                       "metric", "p", 'num_classes', 
                                       'num_noise', 'other_classes'))    
    if metric[0] == 'all':
        metric = list(hdbscan.dist_metrics.METRIC_MAPPING.keys())
        metric.remove('seuclidean')
        metric.remove('mahalanobis')
        metric.remove('wminkowski')
        metric.remove('haversine')
        metric.remove('cosine')
        metric.remove('arccos')
        metric.remove('pyfunc')        
        
    for i in range(len(min_cluster_size)):
        for j in range(len(metric)):
            if metric[j] == 'minkowski':
                p = p0
            else:
                p = [None]
            for n in range(len(p)):
                for k in range(len(min_samples)):    
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size[i]),
                                                metric=metric[j], min_samples=min_samples[k],
                                                p=p[n], algorithm='best')
                    clusterer.fit(features)
                    classes, counts = np.unique(clusterer.labels_, return_counts=True)
                    
                    with open(output_dir + 'hdbscan_param_search.txt', 'a') as f:
                        f.write('{} {} {} {} {} {} {} {}\n'.format(min_cluster_size[i],
                                                       min_samples[k],
                                                       metric[j], p[n],
                                                       len(np.unique(classes))-1, 
                                                       counts[0], classes, counts))

def hdbscan_param_search(features, time, flux, ticid, target_info,
                            min_cluster_size=list(np.arange(5,30,2)),
                            min_samples = [5,10,15],
                            metric=['euclidean', 'manhattan', 'minkowski'],
                            p0 = [1,2,3,4],
                            output_dir='./', DEBUG=False,
                            database_dir='./databases/',
                            pca=False, tsne=False, confusion_matrix=True,
                            prefix='',
                            data_dir='./data/', save=False,
                            parents=[], labels=[]):
    '''Performs a grid serach across parameter space for HDBSCAN. 
    
    Parameters:
        * features
        * time/flux/ticids/target information
        * min cluster size, metric, p (only for minkowski)
        * output_dir : output directory, ending with '/'
        * DEBUG : if DEBUG, plots first 5 light curves in each class
        * optional to plot pca & tsne coloring for it
        
    '''
    import hdbscan         
    classes = []
    num_classes = []
    counts = []
    num_noisy= []
    parameter_sets=[]
    silhouette_scores=[]
    ch_scores = []
    db_scores = []    
    param_num = 0
    accuracy = []
    
    if metric[0] == 'all':
        metric = list(hdbscan.dist_metrics.METRIC_MAPPING.keys())
        metric.remove('seuclidean')
        metric.remove('mahalanobis')
        metric.remove('wminkowski')
        metric.remove('haversine')
        metric.remove('cosine')
        metric.remove('arccos')
        metric.remove('pyfunc')    

    with open(output_dir + 'hdbscan_param_search.txt', 'a') as f:
        f.write('{} {} {} {}\n'.format("min_cluster_size", "min_samples",
                                       "metric", "p", 'num_classes', 
                                       'silhouette', 'db', 'ch', 'acc'))

    for i in range(len(min_cluster_size)):
        for j in range(len(metric)):
            if metric[j] == 'minkowski':
                p = p0
            else:
                p = [None]
            for n in range(len(p)):
                for k in range(len(min_samples)):
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size[i]),
                                                metric=metric[j], min_samples=min_samples[k],
                                                p=p[n], algorithm='best')
                    clusterer.fit(features)
                    labels = clusterer.labels_
                    
                    if save:
                        hdr=fits.Header()
                        hdu=fits.PrimaryHDU(labels, header=hdr)
                        hdu.writeto(output_dir + 'HDBSCAN_res'+str(param_num)+'.fits')
                    
                    print(np.unique(labels, return_counts=True))
                    classes_1, counts_1 = np.unique(labels, return_counts=True)
                            
                                    
                    
                    title='Parameter Set '+str(param_num)+': '+'{} {} {} {}'.format(min_cluster_size[i],
                                                                                 min_samples[k],
                                                                                 metric[j],p[n])
                                
                    prefix='hdbscan-p'+str(param_num)                            
                                    
                    if len(classes_1) > 1:
                        classes.append(classes_1)
                        num_classes.append(len(classes_1))
                        counts.append(counts_1)
                        num_noisy.append(counts_1[0])
                        parameter_sets.append([min_cluster_size[i],metric[j],p[n]])
                        print('Computing silhouette score')
                        silhouette = silhouette_score(features, labels)
                        silhouette_scores.append(silhouette)
                        
                        # >> compute calinski harabasz score
                        print('Computing calinski harabasz score')
                        ch_score = calinski_harabasz_score(features, labels)
                        ch_scores.append(ch_score)
                        
                        # >> compute davies-bouldin score
                        print('Computing davies-bouldin score')
                        dav_boul_score = davies_bouldin_score(features, labels)
                        db_scores.append(dav_boul_score)                        
                                    
                        if confusion_matrix:
                            print('Computing accuracy')
                            acc = pt.plot_confusion_matrix(ticid, labels,
                                                           database_dir=database_dir,
                                                           output_dir=output_dir,
                                                           prefix=prefix)       
                            
                        else:
                            acc=None
                        accuracy.append(acc)
                                  
                                    
                    with open(output_dir + 'hdbscan_param_search.txt', 'a') as f:
                        f.write(' \t'.join(map(str, [min_cluster_size[i],
                                                     min_samples[k],
                                                     metric[j], p[n],
                                                     len(classes_1),
                                                     silhouette, ch_score,
                                                     dav_boul_score, acc])) + '\n')
                        # s = '{}\t {}\t {}\t {}\t {}\t {}\t {}\t {}\n'
                        # f.write(s.format(min_cluster_size[i], min_samples[k],
                        #                  metric[j], p[n], len(classes_1),
                        #                  silhouette, ch_score,
                        #                  dav_boul_score, acc))
                                    
                    if DEBUG and len(classes_1) > 1:
                        pt.quick_plot_classification(time, flux,ticid,target_info, 
                                                     features, labels,path=output_dir,
                                                     prefix=prefix,
                                                     title=title,
                                                     database_dir=database_dir)
                    
                        pt.plot_cross_identifications(time, flux, ticid,
                                                      target_info, features,
                                                      labels, path=output_dir,
                                                      prefix=prefix,
                                                      database_dir=database_dir,
                                                      data_dir=data_dir)
                        pt.plot_confusion_matrix(ticid, labels,
                                                  database_dir=database_dir,
                                                  output_dir=output_dir,
                                                  prefix=prefix+'merge', merge_classes=True,
                                                  labels=[], parents=parents) 
                    
                        if pca:
                            print('Plot PCA...')
                            pt.plot_pca(features, labels,
                                        output_dir=output_dir,
                                        prefix=prefix)
                                    
                        if tsne:
                            print('Plot t-SNE...')
                            pt.plot_tsne(features,labels,
                                         output_dir=output_dir,
                                         prefix=prefix)                
                    plt.close('all')
                    param_num +=1

        
    return parameter_sets, num_classes, acc         
             

def gmm_param_search(features, ticid, data_dir,
                     num_components=[20, 100, 200, 500],
                     output_dir='./'):
    from sklearn.mixture import GaussianMixture       
    from sklearn.metrics import silhouette_score

    with open(output_dir+'gmm_param_search.txt', 'a') as f:
        f.write('{}\t{}\t{}\t{}\n'.format('num_components', 'recall',
                                          'accuracy', 'silhouette'))

    scores = []
    recall = []
    accuracies = []
    for i in range(len(num_components)):
        clusterer = GaussianMixture(n_components=num_components[i])
        labels = clusterer.fit_predict(features)

        score = silhouette_score(features, labels)
        scores.append(score)

        prefix='gmm_n_'+str(i)+'-'
        cm, assignments, ticid_true, y_true, class_info_new, recalls,\
        false_discovery_rates, counts_true, counts_pred, precisions, accuracy,\
        label_true, label_pred=\
            pt.assign_real_labels(ticid, labels, data_dir+'/databases/',
                                  data_dir, output_dir=output_dir+prefix)
        recall.append(np.mean(recalls))
        accuracies.append(np.mean(accuracy))

        with open(output_dir+'gmm_param_search.txt', 'a') as f:
            f.write('{}\t{}\t{}\t{}\n'.format(num_components[i], np.mean(recalls),
                                              accuracy, score))

    fig, ax = plt.subplots()
    ax.plot(num_components, recall, '.')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Recalls')
    fig.tight_layout()
    fig.savefig(output_dir+'gmm_recall.png')

    fig, ax = plt.subplots()
    ax.plot(num_components, accuracies, '.')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Accuracy')
    fig.tight_layout()
    fig.savefig(output_dir+'gmm_accuracy.png')

    fig, ax = plt.subplots()
    ax.plot(num_components, scores, '.')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Silhouette scores (-1 worst, 1 best)')
    fig.tight_layout()
    fig.savefig(output_dir+'gmm_silhouette.png')

def get_class_objects(ticid_feat, class_info, label):    
    ticid_rare = []
    for i in range(len(class_info)):
        if label in class_info[i][1]:
            ticid_rare.append(int(class_info[i][0]))
    intersection, comm1, comm2 = np.intersect1d(ticid_feat, ticid_rare,
                                                return_indices=True)
    inds_rare = comm1  
    
    return ticid_rare, inds_rare
                 
def lof_param_scan(ticid_feat, features,n_neighbors=list(range(10,40,10)),
                   metric=['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
                            'manhattan', 'braycurtis', 'canberra', 'chebyshev',
                            'correlation', 'dice', 'hamming', 'jaccard',
                            'kulinski', 'minkowski',
                            'rogerstanimoto', 'russellrao', 
                            'sokalmichener', 'sokalsneath', 'sqeuclidean',
                            'yule'],
                   p0=[2,4], algorithm=['auto'],
                   contamination=list(np.arange(0.1, 0.5, 0.1)),
                   rare_classes=['BY', 'rot'],
                   output_dir='./', database_dir='./databases/'):
    
    # >> want to find LOF of rare stuff
    class_info = get_true_classifications(ticid_feat, database_dir=database_dir)
    ticid_rare = {}
    inds_rare = {}
    for label in rare_classes:
        ticid_rare[label] = []
    for i in range(len(class_info)):
        for label in rare_classes:
            if label in class_info[i][1]:
                ticid_rare[label].append(int(class_info[i][0]))
    for label in rare_classes:
        intersection, comm1, comm2 = np.intersect1d(ticid_feat, ticid_rare[label],
                                                    return_indices=True)
        inds_rare[label] = comm1
    
    with open(output_dir + 'lof_param_search.txt', 'a') as f:
        f.write('n_neighbors metric p algorithm contamination rare_LOF\n')
    for i in range(len(n_neighbors)):
        for j in range(len(metric)):
            if metric[j] == 'minkowski':
                p = p0
            else:
                p = [None]
            for k in range(len(p)):
                for l in range(len(algorithm)):
                    for m in range(len(contamination)):
                        clf = LocalOutlierFactor(n_neighbors=int(n_neighbors[i]),
                                                 metric=metric[j],
                                                 p=p[k], algorithm=algorithm[l],
                                                 contamination=contamination[m])
                        fit_predictor = clf.fit_predict(features)
                        negative_factor = clf.negative_outlier_factor_
                        lof = -1 * negative_factor
                        
                        with open(output_dir + 'lof_param_search.txt', 'a') as f:
                            f.write('{} {} {} {} {} '.format(int(n_neighbors[i]),
                                                            metric[j], p[k],
                                                            algorithm[l],
                                                            contamination[m]))                        
                        
                        # >> calculate average lof for the rare things
                        for label in rare_classes:
                            avg_lof = np.mean(lof[comm1])
                            with open(output_dir + 'lof_param_search.txt', 'a') as f:     
                                f.write(str(avg_lof) + ' ')
                        with open(output_dir + 'lof_param_search.txt', 'a') as f:         
                            f.write('\n')
    

def make_confusion_matrix(ticid_pred, ticid_true, y_true_labels, y_pred,
                          debug=False, output_dir='./'):
    from sklearn.metrics import confusion_matrix
    from scipy.optimize import linear_sum_assignment   
    import seaborn as sn
    
    # >> find intersection
    intersection, comm1, comm2 = np.intersect1d(ticid_pred, ticid_true,
                                                return_indices=True)
    ticid_pred = ticid_pred[comm1]
    y_pred = y_pred[comm1]
    ticid_true = ticid_true[comm2]
    y_tru_labels = y_true_labels[comm2]           
        
    columns = np.unique(y_pred).astype('str')

    y_true = []
    for i in range(len(ticid_true)):
        class_num = np.nonzero(y_true_labels == y_true_labels[i])[0][0]
        y_true.append(class_num)
    y_true = np.array(y_true).astype('int')    
    
    cm = confusion_matrix(y_true, y_pred)
    while len(columns) < len(cm):
        columns = np.append(columns, 'X')       
    while len(y_true_labels) < len(cm):
        y_true_labels = np.append(y_true_labels, 'X')     
        
    row_ind, col_ind = linear_sum_assignment(-1*cm)
    cm = cm[:,col_ind]       
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    return cm, accuracy
  
       
def optimize_confusion_matrix(ticid_pred, y_pred, database_dir='./',
                              num_classes=[10,15], num_iter=10):
    from itertools import permutations
    import random
    class_info = get_true_classifications(ticid_pred,
                                          database_dir=database_dir,
                                          single_file=False)  
    ticid_true = class_info[:,0].astype('int')
    classes = []
    for i in range(len(class_info)):
        for label in class_info[i][1].split('|'):
            if label not in classes:
                classes.append(label)
                
    accuracy = []
    for n in num_classes:
        combinations = list(permutations(classes))
        print('Number of combinations: ' + str(len(combinations)))
        for i in range(num_iter):
            labels = random.choice(combinations)
            
            
            ticid_new = []
            y_true = []
            for i in range(len(ticid_true)):
                for j in range(len(labels)):
                    if labels[j] in class_info[i][1] and \
                        ticid_true[i] not in ticid_new:
                        y_true.append(labels[j])
                        ticid_new.append(ticid_true[i])
                        
            y_true = np.array(y_true)
            ticid_true_new = np.array(ticid_new)  
            
            cm, acc = make_confusion_matrix(ticid_pred, ticid_true_new,
                                            labels, y_pred)
            print(labels)
            print('accuracy: ' + str(acc))
            accuracy.append(acc)
    
            
            
                
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: DEPRECIATED SCTION ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def representation_learning(flux, x, ticid, target_info, 
                            output_dir='./',
                            dat_dir = '/Users/studentadmin/Dropbox/TESS_UROP/data/',
                            mom_dump = '/Users/studentadmin/Dropbox/TESS_UROP/Table_of_momentum_dumps.csv',
                            database_dir='/Users/studentadmin/Dropbox/TESS_UROP/data/databases/',
                            p=None,
                            validation_targets=[],
                            norm_type='minmax_normalization',
                            input_rms=True, input_psd=False, load_psd=False,
                            train_test_ratio=0.9, split=False):
    ''' Deprecated 210217
    Parameters you have to change:
        * flux : np.array, with shape (num_samples, num_data_points)
        * x : np.array, with shape (num_data_points)
        * ticid : np.array, with shape (num_samples)
        * target_info : np.array, with shape (num_samples, 5)
        * dat_dir : Dropbox directory with all of our metafiles
        * mom_dump : path to momentum dump csv file
        * data_base_dir : Dropbox directory with all of the database .txt files
        
    Parameters to ignore:
        * p : dictionary of parameters        
        * validation_targets
        * 
    '''
    
    # >> use default parameter set if not given
    if type(p) == type(None):
        p = {'kernel_size': 3,
              'latent_dim': 35,
              'strides': 1,
              'epochs': 10,
              'dropout': 0.,
              'num_filters': 16,
              'num_conv_layers': 12,
              'batch_size': 64,
              'activation': 'elu',
              'optimizer': 'adam',
              'last_activation': 'linear',
              'losses': 'mean_squared_error',
              'lr': 0.0001,
              'initializer': 'random_normal',
              'num_consecutive': 2,
              'pool_size': 2, 
              'pool_strides': 2,
              'kernel_regularizer': None,
              'bias_regularizer': None,
              'activity_regularizer': None,
              'fully_conv': False,
              'encoder_decoder_skip': False,
              'encoder_skip': False,
              'decoder_skip': False,
              'full_feed_forward_highway': False,
              'cvae': False,
              'share_pool_inds': False,
              'batchnorm_before_act': False} 
        
    print('Preprocessing')
    x_train, x_test, y_train, y_test, ticid_train, ticid_test, target_info_train, \
        target_info_test, rms_train, rms_test, x = \
        lt.autoencoder_preprocessing(flux, ticid, x, target_info, p,
                                     validation_targets=validation_targets,
                                     norm_type=norm_type,
                                     input_rms=input_rms, input_psd=input_psd,
                                     load_psd=load_psd,
                                     train_test_ratio=train_test_ratio,
                                     split=split,
                                     output_dir=output_dir)       
        
    print('Training CAE')
    history, model, x_predict = \
        lt.conv_autoencoder(x_train, y_train, x_test, y_test, p,
                            input_rms=True, rms_train=rms_train, rms_test=rms_test,
                            ticid_train=ticid_train, ticid_test=ticid_test,
                            output_dir=output_dir)
        
    print('Diagnostic plots')
    pt.diagnostic_plots(history, model, p, output_dir, x, x_train,
                        x_test, x_predict, mock_data=False, addend=0.,
                        target_info_test=target_info_test,
                        target_info_train=target_info_train,
                        ticid_train=ticid_train,
                        ticid_test=ticid_test, percentage=False,
                        input_features=False,
                        input_rms=input_rms, rms_test=rms_test,
                        input_psd=input_psd,
                        rms_train=rms_train, n_tot=40,
                        plot_epoch = False,
                        plot_in_out = True,
                        plot_in_bottle_out=False,
                        plot_latent_test = True,
                        plot_latent_train = True,
                        plot_kernel=False,
                        plot_intermed_act=True,
                        make_movie = False,
                        plot_lof_test=False,
                        plot_lof_train=False,
                        plot_lof_all=False,
                        plot_reconstruction_error_test=False,
                        plot_reconstruction_error_all=True,
                        load_bottleneck=True)            

    features, flux_feat, ticid_feat, info_feat = \
        lt.bottleneck_preprocessing(None,
                                    np.concatenate([x_train, x_test], axis=0),
                                    np.concatenate([ticid_train, ticid_test]),
                                    np.concatenate([target_info_train,
                                                    target_info_test]),
                                    data_dir=dat_dir,
                                    output_dir=output_dir,
                                    use_learned_features=True,
                                    use_tess_features=False,
                                    use_engineered_features=False,
                                    use_tls_features=False)         
        
    print('Novelty detection')
    pt.plot_lof(x, flux_feat, ticid_feat, features, 20, output_dir,
                n_tot=40, target_info=info_feat, prefix='',
                cross_check_txt=database_dir, debug=False, addend=0.)        
    
    print('DBSCAN parameter search')
    parameter_sets, num_classes, silhouette_scores, db_scores, ch_scores, acc = \
    dbscan_param_search(features, x, flux_feat, ticid_feat,
                            info_feat, DEBUG=False, 
                            output_dir=output_dir, 
                            leaf_size=[30], algorithm=['auto'],
                            min_samples=[5],
                            metric=['minkowski'], p=[3,4],
                            database_dir=database_dir,
                            eps=list(np.arange(1.5, 4., 0.1)),
                            confusion_matrix=False, pca=False, tsne=False,
                            tsne_clustering=False)    
    
    best_ind = np.argmax(silhouette_scores)
    best_param_set = parameter_sets[best_ind]   
        
    parameter_sets, num_classes, silhouette_scores, db_scores, ch_scores, acc = \
    dbscan_param_search(features, x, flux_feat, ticid_feat,
                            info_feat, DEBUG=True, 
                            output_dir=output_dir+'best', single_file=True,
                            leaf_size=[best_param_set[4]],
                            algorithm=[best_param_set[3]],
                            min_samples=[best_param_set[1]],
                            metric=[best_param_set[2]], p=[best_param_set[5]],
                            database_dir=database_dir,
                            eps=[best_param_set[0]])      
