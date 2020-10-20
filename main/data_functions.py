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

import model as ml


def test_data():
    """make sure the module loads in"""
    print("Data functions loaded in.")


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
    '''
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
        ml.autoencoder_preprocessing(flux, ticid, x, target_info, p,
                                     validation_targets=validation_targets,
                                     norm_type=norm_type,
                                     input_rms=input_rms, input_psd=input_psd,
                                     load_psd=load_psd,
                                     train_test_ratio=train_test_ratio,
                                     split=split,
                                     output_dir=output_dir)       
        
    print('Training CAE')
    history, model, x_predict = \
        ml.conv_autoencoder(x_train, y_train, x_test, y_test, p,
                            input_rms=True, rms_train=rms_train, rms_test=rms_test,
                            ticid_train=ticid_train, ticid_test=ticid_test,
                            output_dir=output_dir)
        
    print('Diagnostic plots')
    pf.diagnostic_plots(history, model, p, output_dir, x, x_train,
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
        ml.bottleneck_preprocessing(None,
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
    pf.plot_lof(x, flux_feat, ticid_feat, features, 20, output_dir,
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
                                 order=5, tol=0.4, debug=True, norm_type='standardization',
                                 output_dir='./'):
    from scipy import signal
    all_flux = []
    all_ticid = []
    all_target_info = []
    all_x = []
    print('Loading data and applying nanmask')
    for i in range(len(sectors)):
        flux, x, ticid, target_info = \
            load_data_from_metafiles(data_dir, sectors[i], nan_mask_check=True,
                                     custom_mask=custom_mask)
            
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
    flux, target_info, ticid, ticid_rejected = [], [], [], []
    
    all_ticid, comm1, comm2 = np.intersect1d(all_ticid[0], all_ticid[1],
                                         return_indices=True)
    all_flux[0] = all_flux[0][comm1]
    all_flux[1] = all_flux[1][comm2]
    all_target_info[0] = all_target_info[0][comm1]
    all_target_info[1] = all_target_info[1][comm2]
    
    for i in range(len(all_ticid)):
    
        b, a = signal.butter(order, cutoff, btype='high', analog=False)
        y1 = signal.filtfilt(b, a, all_flux[0][i])
        rms1 = np.sqrt(np.mean(y1**2))
        
        y2 = signal.filtfilt(b, a, all_flux[1][i])
        rms2 = np.sqrt(np.mean(y2**2))
        
        if debug and i < 5:
            fig, ax = plt.subplots(2, 2)
            ax[0,0].plot(all_x[0], all_flux[0][i], '.k', ms=1)
            ax[0,1].plot(all_x[1], all_flux[1][i], '.k', ms=1)            
            ax[1,0].plot(all_x[0], y1, '.k', ms=1)
            ax[1,1].plot(all_x[1], y2, '.k', ms=1)
            fig.savefig(output_dir+'highpass_'+str(cutoff)+'_'+str(i)+'.png')   
    
        # >> compare RMS !! assumes combining 2 sectors
        if np.abs(rms2 - rms1) < tol*rms1 and \
            np.abs(rms2 - rms1) < tol*rms2:
            flux.append(np.concatenate([all_flux[0][i], all_flux[1][i]]))
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
                ax[0,0].plot(all_x[0], all_flux[0][i], '.k', ms=1)
                ax[0,1].plot(all_x[1], all_flux[1][i], '.k', ms=1)            
                ax[1,0].plot(all_x[0], y1, '.k', ms=1)
                ax[1,1].plot(all_x[1], y2, '.k', ms=1)
                fig.savefig(output_dir+'highpass_TIC_'+str(int(all_ticid[i]))+'.png')    
            ticid_rejected.append(all_ticid[i])
             

    flux = np.array(flux)
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
             output_dir='./', prefix='', tol1=0.05, tol2=0.1,
             custom_mask=[], use_tol2=False):
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

def get_tess_features(ticid):
    '''Query catalog data https://arxiv.org/pdf/1905.10694.pdf'''
    

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

def query_associated_catalogs(ticid):
    res=Catalogs.query_object('TIC ' + str(int(ticid)), radius=0.02,
                              catalog='TIC')[0]
    for i in ['HIP', 'TYC', 'UCAC', 'TWOMASS', 'ALLWISE', 'GAIA', 'KIC', 'APASS']:
        print(i + ' ' + str(res[i]) + '\n')

def query_simbad_classifications(ticid_list, output_dir='./', suffix=''):
    '''Call like this:
    query_simbad_classifications([453370125.0, 356473029])
    '''
    import time
    
    customSimbad = Simbad()
    customSimbad.add_votable_fields('otypes')
    # customSimbad.add_votable_fields('biblio')
    
    ticid_simbad = []
    otypes_simbad = []
    main_id_simbad = []
    bibcode_simbad = []
    
    with open(output_dir + 'all_simbad_classifications'+suffix+'.txt', 'a') as f:
        f.write('')    
    
    with open(output_dir + 'all_simbad_classifications'+suffix+'.txt', 'r') as f:
        lines = f.readlines()
        ticid_already_classified = []
        for line in lines:
            ticid_already_classified.append(float(line.split(',')[0]))
    

    for tic in ticid_list:
        
        res=None
        
        while res is None:
            try:
                if tic in ticid_already_classified:
                    print('Skipping TIC')
                    
                else:
                    print('get coords for TIC' + str(int(tic)))
                    
                    # >> get coordinates
                    target = 'TIC ' + str(int(tic))
                    catalog_data = Catalogs.query_object(target, radius=0.02,
                                                         catalog='TIC')[0]
                    # time.sleep(6)
            
                    
                    # -- get object type from Simbad --------------------------------------
                    
                    # >> first just try querying the TICID
                    res = customSimbad.query_object(target)
                    # time.sleep(6)
                    
                    # >> if no luck with that, try checking other IDs
                    if type(res) == type(None):
                        if type(catalog_data['TYC']) != np.ma.core.MaskedConstant:
                            target_new = 'TYC ' + str(catalog_data['TYC'])
                            res = customSimbad.query_object(target_new)
                            # time.sleep(6)
                            
                    if type(res) == type(None):
                        if type(catalog_data['HIP']) != np.ma.core.MaskedConstant:
                            target_new = 'HIP ' + str(catalog_data['HIP'])
                            res = customSimbad.query_object(target_new)
                            # time.sleep(6)
            
                    # # >> UCAC not added to Simbad yet
                    # if type(res) == type(None):
                    #     if type(catalog_data['UCAC']) != np.ma.core.MaskedConstant:
                    #         target_new = 'UCAC ' + str(catalog_data['UCAC'])
                    #         res = customSimbad.query_object(target_new)
                            
                    if type(res) == type(None):
                        if type(catalog_data['TWOMASS']) != np.ma.core.MaskedConstant:
                            target_new = '2MASS ' + str(catalog_data['TWOMASS'])
                            res = customSimbad.query_object(target_new)     
                            # time.sleep(6)
            
                    if type(res) == type(None):
                        if type(catalog_data['SDSS']) != np.ma.core.MaskedConstant:
                            target_new = 'SDSS ' + str(catalog_data['SDSS'])
                            res = customSimbad.query_object(target_new) 
                            # time.sleep(6)
            
                    if type(res) == type(None):
                        if type(catalog_data['ALLWISE']) != np.ma.core.MaskedConstant:
                            target_new = 'ALLWISE ' + str(catalog_data['ALLWISE'])
                            res = customSimbad.query_object(target_new)
                            # time.sleep(6)
                            
                    if type(res) == type(None):
                        if type(catalog_data['GAIA']) != np.ma.core.MaskedConstant:
                            target_new = 'Gaia ' + str(catalog_data['GAIA'])
                            res = customSimbad.query_object(target_new)      
                            # time.sleep(6)
                            
                    if type(res) == type(None):
                        if type(catalog_data['APASS']) != np.ma.core.MaskedConstant:
                            target_new = 'APASS ' + str(catalog_data['APASS'])
                            res = customSimbad.query_object(target_new)        
                            # time.sleep(6)
                            
                    if type(res) == type(None):
                        if type(catalog_data['KIC']) != np.ma.core.MaskedConstant:
                            target_new = 'KIC ' + str(catalog_data['KIC'])
                            res = customSimbad.query_object(target_new)    
                            # time.sleep(6)
                    
                    # # >> if still nothing, query with coordinates
                    # if type(res) == type(None):
                    #     ra = catalog_data['ra']
                    #     dec = catalog_data['dec']            
                    #     coords = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg))
                    #     res = customSimbad.query_region(coords, radius='0d0m2s')         
                    #     time.sleep(6)
                    
                    if type(res) == type(None):
                        print('failed :(')
                        res=0
                        with open(output_dir + 'all_simbad_classifications'+suffix+'.txt', 'a') as f:
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
                        
                        with open(output_dir + 'all_simbad_classifications'+suffix+'.txt', 'a') as f:
                            f.write('{},{},{}\n'.format(tic, otypes, main_id))
                            
                    # time.sleep(6)
            except:
                pass
                print('connection failed! Trying again now')
                    
                    
            
    return ticid_simbad, otypes_simbad, main_id_simbad
        


def query_vizier(ticid_list=None, out='./SectorX_GCVS.txt', catalog='gcvs',
                 dat_dir = '/Users/studentadmin/Dropbox/TESS_UROP/data/',
                 sector=20):
    '''http://www.sai.msu.su/gcvs/gcvs/vartype.htm'''
    
    # Vizier.ROW_LIMIT=-1
    # catalog_list=Vizier.find_catalogs('B/gcvs')
    # catalogs = Vizier.get_catalogs(catalog_list.keys())    
    # catalogs=catalogs[0]
    
    if type(ticid_list) == type(None):
        flux, x, ticid_list, target_info = \
            load_data_from_metafiles(dat_dir, sector, DEBUG=False,
                                     nan_mask_check=False)        
    
    ticid_viz = []
    otypes_viz = []
    main_id_viz = []
    ticid_already_classified = []
    
    # >> make sure output file exists
    with open(out, 'a') as f:
        f.write('')    
    
    with open(out, 'r') as f:
        lines = f.readlines()
        ticid_already_classified = []
        for line in lines:
            ticid_already_classified.append(float(line.split(',')[0]))
            
    
    for tic in ticid_list:
        if tic  in ticid_already_classified:
            print('Skipping '+str(tic))
        else:
            try:
                print('Running '+str(tic))
                target = 'TIC ' + str(int(tic))
                print('Query Catalogs')
                catalog_data = Catalogs.query_object(target, radius=0.02,
                                                     catalog='TIC')[0]
                ra = catalog_data['ra']
                dec = catalog_data['dec']            
                # coords = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg)) 
                # ra = coords.ra.deg
                # dec = coords
                v = Vizier(columns=['VarType', 'VarName'])
                print('Query Vizier')
                res = v.query_region(coord.SkyCoord(ra=ra, dec=dec,
                                                         unit=(u.deg, u.deg),
                                                         frame='icrs'),
                                          radius=0.003*u.deg, catalog=catalog)
                if len(res) > 0:
                    otype = res[0]['VarType'][0]
                    main_id = res[0]['VarName'][0]
                    ticid_viz.append(tic)
                    otypes_viz.append(otype)
                    main_id_viz.append(main_id)
                    # with open(out, 'a') as f:
                    #     f.write('{},{},{}\n'.format(tic, otype, main_id))              
                else:
                    otype = ''
                    main_id = ''
                    
                with open(out, 'a') as f:
                    f.write('{},{},{}\n'.format(tic, otype, main_id))    
            except:
                print('Connection failed! Trying again now')
                
    return ticid_viz, otypes_viz, main_id_viz

def get_otype_dict(data_dir='/Users/studentadmin/Dropbox/TESS_UROP/data/'):
    '''Return a dictionary of descriptions'''
    # d = {'a2': 'Variable Star of alpha2 CVn type',
    #      'ACYG': 'Variables of the Alpha Cygni type',
    #      'IR': 'Infra-Red source',
    #      'UV': 'UV-emission source',
    #      'X': 'X-ray source',
    #      'gB': 'gamma-ray Burst',
    #      'AR': 'Detached systems of the AR Lacertae type',
    #      'EB': 'Eclipsing binary',
    #      'Al': 'Eclipsing binary of Algol type',
    #      'bL': 'Eclipsing binary of beta Lyr type',
    #      'WU': 'Eclipsing binary of W UMa type',
    #      'EP': 'Star showing eclipses by its planet',
    #      'SB': 'Spectroscopic binary',
    #      'EI': 'Ellipsoidal variable Star',
    #      'CV': 'Cataclysmic Variable Star',
    #      'SNR': 'SuperNova Remnant',
    #      'Be': 'Be star',
    #      'Fl': 'Flare star',
    #      'V': 'Variable star',
    #      'HV': 'High-velocity star',
    #      'PM': }
    # d = {'ACYG': 'Variables of the Alpha Cygni type',
    #      'AR': 'Detached systems of the AR Lacertae type',
    #      'D': 'Detached systems',
    #      'DM': 'Detached main-sequence systems',
    #      'DW': 'Detached systems with a subgiant',
    #      'K': 'Contact systems',
    #      'KE': 'Contact systems of early (O-A) spectral type',
    #      'KW': 'Contact systems of the W UMa type',
    #      'SD': 'Semidetached systems',
    #      'GS': 'Systems with one or both giant and supergiant components',
    #      'RS': 'RS Canum Venaticorum-type systems',
    #      'CST': 'Nonvariable stars',
    #      'XPRM': 'X-ray systems consisting of a late-type dwarf (dK-dM) and a pulsar',
    #      'FKCOM': 'FK Comae Berenices-type variables',
    #      'GCAS': 'Eruptive irregular variables of the Gamma Cas type',
    #      'IA': 'Poorly studied irregular variables of early (O-A) spectral type'}
    
    d = {}
    
    with open(data_dir + 'otypes_gcvs.txt', 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        if len(line.split(' '*3)) > 1:
            otype = line.split(' '*3)[0]
            explanation = line.split(' '*3)[1].split('.')[0]
            d[otype] = explanation
    
    with open(data_dir + 'otypes_simbad.txt', 'r') as f:
        lines= f.readlines()
        
    for line in lines:
        if len(line.split('\t')) >= 3:
            otype = line.split('\t')[-2].split()[0]
            if len(otype) > 0:
                if otype[-1] == '*' and otype != '**':
                    otype = otype[:-1]
            explanation = ' '.join(line.split('\t')[-1].split())
            d[otype] = explanation
        
    return d

def get_parents_only(class_info, parents=['EB'],
                     parent_dict = {'EB': ['Al', 'bL', 'WU', 'EP', 'SB', 'SD'],
                                    'ACV': ['ACVO'],
                                    'D': ['DM', 'DS', 'DW'],
                                    'K': ['KE', 'KW'],
                                    'Ir': ['Or', 'RI', 'IA', 'IB', 'INA', 'INB']}):
    '''Finds all the objects with same parent and combines them into the same
    class
    '''
    classes = []
    new_class_info = []
    for i in range(len(class_info)):
        otype_list = class_info[i][1].split('|')
        new_otype_list=[]
        for otype in otype_list:
            for parent in parents:
                if otype in parent_dict[parent]:
                    new_otype = parent
                else:
                    new_otype = otype
                new_otype_list.append(new_otype)
                
        new_otype_list = np.unique(new_otype_list)
        new_class_info.append([class_info[i][0], '|'.join(new_otype_list),
                               class_info[i][2]])
            
    
    return np.array(new_class_info)

def correct_vizier_to_simbad(in_f='./SectorX_GCVS.txt',
                             out_f='./SectorX_GCVS_revised.txt',
                             uncertainty_flags=[':', '?', '*']):
    '''Make sure object types are the same'''
    with open(in_f, 'r') as f:
        lines = f.readlines()
        
    renamed = {'E': 'EB', 'EA': 'Al', 'EB': 'bL', 'EW': 'WU', 'ACV': 'a2',
               'ACVO': 'a2', 'BCEP': 'bC', 'BE':'Be', 'DCEP': 'cC',
               'DSCT': 'dS', 'DSCTC': 'dS', 'ELL': 'El', 'GDOR': 'gD',
               'I': 'Ir', 'IN': 'Or', 'IS': 'RI'}
        
    for line in lines:
        tic, otype, main = line.split(',')
        otype = otype.replace('+', '|')
        otype_list = otype.split('|')
        otype_list_new = []
        
        for o in otype_list:
            
            if len(o) > 0:
                # >> remove uncertainty_flags
                if o[-1] in uncertainty_flags:
                    o = o[:-1]
                    
                # >> remove (B)
                if '(' in o:
                    o = o[:o.index('(')]
                    
                if o in list(renamed.keys()):
                    o = renamed[o]
                # # >> rename object types to Simbad notation
                # if o == 'E':
                #     o = 'EB'
                # elif o == 'EA':
                #     o = 'Al'
                # elif o == 'EB':
                #     o = 'bL'
                # elif o == 'EW':
                #     o = 'WU'
                # elif o == 'ACV' or o == 'ACVO':
                #     o = 'a2'
                # elif o == 'BCEP':
                #     o = 'bC'
                # elif o == 'BE':
                #     o = 'Be'
                # elif o == ''
                
            otype_list_new.append(o)
                
                
        otype = '|'.join(otype_list_new)
        
        
        with open(out_f, 'a') as f:
            f.write(','.join([tic, otype, main]))

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



def get_true_classifications(ticid_list,
                             database_dir='./databases/',
                             single_file=False,
                             useless_classes = ['*', 'IR', 'UV', 'X', 'PM',
                                                '?', ':'],
                             uncertainty_flags = ['*', ':', '?']):
    '''Query classifications and bibcode from *_database.txt file.
    Returns a list where class_info[i] = [ticid, obj type, bibcode]
    Object type follows format in:
    http://vizier.u-strasbg.fr/cgi-bin/OType?$1
    '''
    ticid_classified = []
    class_info = []
    
    # >> find all text files in directory
    if single_file:
        fnames = ['']
    else:
        fnames = fm.filter(os.listdir(database_dir), '*.txt')
    
    for fname in fnames:
        # >> read text file
        with open(database_dir + fname, 'r') as f:
            lines = f.readlines()
            for line in lines:
                ticid, otype, bibcode = line[:-1].split(',')
                

                
                # >> remove any repeats and any empty classes and sort
                otype_list = otype.split('|')
                # >> remove any candidate indicators
                for i in range(len(otype_list)):
                    if otype_list[i] != '**' and len(otype_list[i])>0:
                        if otype_list[i][-1] in uncertainty_flags:
                            otype_list[i] = otype_list[i][:-1]
                otype_list = np.unique(otype_list)
                # >> remove useless classes
                for u_c in useless_classes + ['']:
                    if u_c in otype_list:
                        otype_list =np.delete(otype_list,
                                              np.nonzero(otype_list == u_c))
                otype_list.sort()
                otype = '|'.join(otype_list)
                
                # >> only get classifications for ticid_list, avoid repeats
                # >> and only include objects with interesting lables
                ticid = float(ticid)
                if ticid in ticid_list and len(otype) > 0:
                    if ticid in ticid_classified:
                        ind = np.nonzero(np.array(ticid_classified) == ticid)[0][0]
                        new_class_info = class_info[ind][1] + '|' + otype
                        new_class_info = new_class_info.split('|')
                        new_class_info = '|'.join(np.unique(new_class_info))
                        class_info[ind][1] = new_class_info
                    else:
                        ticid_classified.append(ticid)
                        class_info.append([int(ticid), otype, bibcode])
                    
    # >> check for any repeats
    return np.array(class_info)
                           
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
                                acc = pf.plot_confusion_matrix(ticid, db.labels_,
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
                                pf.quick_plot_classification(time, flux,
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
                                    pf.plot_pca(bottleneck, db.labels_,
                                                output_dir=output_dir,
                                                prefix=prefix)
                                
                                if tsne:
                                    print('Plot t-SNE...')
                                    pf.plot_tsne(bottleneck, db.labels_,
                                                 output_dir=output_dir,
                                                 prefix=prefix)
                                # if tsne_clustering:
                                    
                                    
                            plt.close('all')
                            param_num +=1
    print("Plot paramscan metrics...")
    pf.plot_paramscan_metrics(output_dir+'dbscan-', parameter_sets, 
                              silhouette_scores, db_scores, ch_scores)
    #print(len(parameter_sets), len(num_classes), len(num_noisy), num_noisy)

    pf.plot_paramscan_classes(output_dir+'dbscan-', parameter_sets, 
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
                            simbad_database_txt='./simbad_database.txt',
                            database_dir='./databases/',
                            pca=False, tsne=False, confusion_matrix=True,
                            single_file=False, prefix='',
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
                            acc = pf.plot_confusion_matrix(ticid, labels,
                                                           database_dir=database_dir,
                                                           single_file=single_file,
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
                        pf.quick_plot_classification(time, flux,ticid,target_info, 
                                                     features, labels,path=output_dir,
                                                     prefix=prefix,
                                                     title=title,
                                                     database_dir=database_dir,
                                                     single_file=single_file)
                    
                        pf.plot_cross_identifications(time, flux, ticid,
                                                      target_info, features,
                                                      labels, path=output_dir,
                                                      prefix=prefix,
                                                      database_dir=database_dir,
                                                      data_dir=data_dir)
                        pf.plot_confusion_matrix(ticid, labels,
                                                  database_dir=database_dir,
                                                  single_file=single_file,
                                                  output_dir=output_dir,
                                                  prefix=prefix+'merge', merge_classes=True,
                                                  labels=[], parents=parents) 
                    
                        if pca:
                            print('Plot PCA...')
                            pf.plot_pca(features, labels,
                                        output_dir=output_dir,
                                        prefix=prefix)
                                    
                        if tsne:
                            print('Plot t-SNE...')
                            pf.plot_tsne(features,labels,
                                         output_dir=output_dir,
                                         prefix=prefix)                
                    plt.close('all')
                    param_num +=1

        
    return parameter_sets, num_classes, acc         
             

def gmm_param_search(features, time, flux, ticid, target_info,
                            num_components=[20, 100, 200, 500],
                            output_dir='./', DEBUG=False,
                            simbad_database_txt='./simbad_database.txt',
                            database_dir='./databases/',
                            pca=False, tsne=False, confusion_matrix=True,
                            single_file=False,
                            data_dir='./data/', save=False,
                            parents=[], labels=[]):
    from sklearn.mixture import GaussianMixture       
    classes = []
    num_classes = []
    counts = []
    accuracy = []  

    with open(output_dir + 'hdbscan_param_search.txt', 'a') as f:
        f.write('{} {} {}\n'.format('num_components', 'acc', 'recall'))

    for i in range(len(num_components)):
        clusterer = GaussianMixture(n_components=num_components[i])
        labels = clusterer.fit_predict(features)
                    
        print(np.unique(labels, return_counts=True))
        classes_1, counts_1 = np.unique(labels, return_counts=True)
                
                        
        
        title='Parameter Set '+str(i)+': '+'{}'.format(num_components[i])
                    
        prefix='gmm-p'+str(i)                            
                                    
        acc = pf.plot_confusion_matrix(ticid, labels,
                                       database_dir=database_dir,
                                       single_file=single_file,
                                       output_dir=output_dir,
                                       prefix=prefix)       
        
    return num_classes, acc          

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
    
            
            
                

# DEPRECIATED SECTION -----------------------------------------------------

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# def hdbscan_param_search(bottleneck, time, flux, ticid, target_info,
#                          min_cluster_size=list(range(3,10, 2)),
#                          min_samples=list(range(3,10, 2)),
#                          metric=['euclidean', 'braycurtis'],
#                          p_space=[1,2,3,4],
#                          output_dir='./',
#                          database_dir='./databases/', make_plots=True):
        
#     import hdbscan
#     # !! wider p range?
    
#     if metric[0] == 'all':
#         metric = list(hdbscan.dist_metrics.METRIC_MAPPING.keys())
#         metric.remove('seuclidean')
#         metric.remove('mahalanobis')
#         metric.remove('wminkowski')
#         metric.remove('haversine')
#         metric.remove('cosine')
#         metric.remove('arccos')
#         metric.remove('pyfunc')
    
#     with open(output_dir + 'hdbscan_param_search.txt', 'a') as f:
#         f.write('{}\t {}\t {}\t {}\n'.format("min_cluster_size", 
#                                              'min_samples', 'metric',
#                                              'silhouette'))    
    
#     param_num=0
#     for i in range(len(min_cluster_size)):
#         for j in range(len(min_samples)):
#             for k in range(len(metric)):
#                 if metric[k] == 'minkowski':
#                     p = p_space
#                 else:
#                     p = [None]
#                 for l in range(len(p)):
                    
#                     clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size[i],
#                                                 min_samples=min_samples[j],
#                                                 metric=metric[k], p=p[l])
#                     clusterer.fit(bottleneck)
#                     classes, counts = \
#                         np.unique(clusterer.labels_, return_counts=True)    
#                     print(classes, counts)
                    
                    
#                     if len(classes) > 1:
#                         silhouette = silhouette_score(bottleneck, clusterer.labels_)
#                     else:
#                         silhouette= np.nan
                    
#                     with open(output_dir + 'hdbscan_param_search.txt', 'a') as f:
#                         f.write('{}\t {}\t {}\t {}\n'.format(min_cluster_size[i],
#                                                              min_samples[j],
#                                                              metric[k],
#                                                              silhouette))    
#                     title='Parameter Set '+str(param_num)+': '+'{} {} {}'.format(min_cluster_size[i],
#                                                                             min_samples[j],
#                                                                             metric[k])
                    
#                     prefix='hdbscan-p'+str(param_num)
                    
#                     if make_plots:
#                         acc = pf.plot_confusion_matrix(ticid, clusterer.labels_,
#                                                        database_dir=database_dir,
#                                                        output_dir=output_dir,
#                                                        prefix=prefix)        
#                         pf.plot_pca(bottleneck, clusterer.labels_,
#                                     output_dir=output_dir, prefix=prefix)    
#                         pf.plot_tsne(bottleneck, clusterer.labels_,
#                                      output_dir=output_dir, prefix=prefix)   
#                         pf.quick_plot_classification(time, flux, ticid,
#                                                     target_info, bottleneck,
#                                                     clusterer.labels_,
#                                                     path=output_dir,
#                                                     prefix=prefix,
#                                                     title=title,
#                                                     database_dir=database_dir)                    
                        
#                         plt.figure()
#                         clusterer.condensed_tree_.plot()
#                         plt.savefig(output_dir + prefix + '-tree.png')
                    
#                     param_num += 1
                    
#     return acc    

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
