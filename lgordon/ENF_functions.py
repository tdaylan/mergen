# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 20:52:55 2020

ENF Functions

@author: Lindsey Gordon
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

def load_feature_metafile(folderpath):
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

######### DEFUNCT

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