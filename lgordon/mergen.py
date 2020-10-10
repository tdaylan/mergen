# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 20:27:13 2020

mergen.py
@author: LG and EC
"""

import numpy as np
import numpy.ma as ma 
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)

from pylab import rcParams
rcParams['figure.figsize'] = 10,10

rcParams["lines.markersize"] = 2
from scipy.signal import argrelextrema


import astropy
import astropy.units as u
from astropy.io import fits
import scipy.signal as signal
from astropy.stats import SigmaClip
from astropy.utils import exceptions
from astroquery import exceptions
from astroquery.exceptions import RemoteServiceError
#from astropy.utils.exceptions import AstropyWarning, RemoteServiceError

from datetime import datetime
import os
import shutil
from scipy.stats import moment, sigmaclip

import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import fnmatch

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor

import astroquery
from astroquery.simbad import Simbad
from astroquery.mast import Catalogs
from astroquery.mast import Observations

import data_functions as df
import plotting_functions as pf
import ffi_hyperleda as fh
import model as ml
import data_access as da
import ENF_functions as enf


class mergen(object):
    
    def __init__(self, datapath, datatype, savepath, filelabel,
                 momentum_dump_csv = '/users/conta/urop/Table_of_momentum_dumps.csv',
                 sector = 20, cams = [1,2,3,4], ccds = [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]], 
                 cadence = "2minute", ENF_exists = False, create_ENF = False, version = 0):
        """ Initializes mergen object and  loads in all data
        Data must already be in a metafile format"""
        self.datapath = datapath
        self.datatype = datatype
        self.savepath = savepath
        self.momdumpcsv = momentum_dump_csv
        self.sector = sector
        self.cams = cams
        self.ccds = ccds
        self.cadence = cadence
        self.filelabel = filelabel
        self.isfluxnormalized = False
        
        self.folder_initiate()
        
        if self.datatype == "SPOC":
            #load in SPOC lc from the metafiles
            #this function will interpolate them for you
            print("loading spoc")
            self.spoc_load_lc_from_metafiles()
            
            
        elif self.datatype == "FFI":
            #load in FFI lc from the metafiles
            self.ffi_load_lc_metafiles_all()
            
        if ENF_exists:
            print("Loading in existing feature metafiles")
            self.features = enf.load_feature_metafile(self.datapath)
            
        if create_ENF and version == 0:
            print("Creating ENF v0 features, assuming uniform time axis")
            enf.create_save_featvec_homogenous_time(self.ENFpath, 
                                                    self.time, self.flux, self.filelabel, 
                                                    version=0, save=True)
            
        if create_ENF and version == 1:
            print("Creating ENF v0 features, assuming uniform time axis")
            enf.create_save_featvec_homogenous_time(self.ENFpath, 
                                                    self.time, self.flux, self.filelabel, 
                                                    version=1, save=True)
            
    
    def folder_initiate(self):
        """Makes all the big folders"""
        print("Setting up CAE folder")
        self.CAEpath = self.savepath + "CAE/"
        try:
            os.makedirs(self.CAEpath)
        except OSError:
            print ("Directory %s already exists" % self.CAEpath)
            
        print("Setting up ENF folder")
        self.ENFpath = self.savepath + "ENF/"
        try:
            os.makedirs(self.ENFpath)
        except OSError:
            print ("Directory %s already exists" % self.ENFpath)
        return
    
    
    def ffi_load_lc_metafiles_all(self):
        """ Hopefully loads all metafiles"""
        self.time = None
        self.flux = None
        self.identifiers = None
        
        for root, dirs, files in os.walk(self.datapath):
            for n in range(len(files)):
                if files[n].endswith('_lightcurves.fits'):
                    filepath = os.path.join(root, files[n])
                    f = fits.open(filepath, memmap = False)
                    if self.time is None:
                        self.time = f[0].data
                        self.flux = f[1].data
                    else:
                        self.flux = np.vstack((self.v, f[1].data))
                    f.close()
                        
                elif files[n].endswith("_lightcurve_ids.txt"):
                    filepath = os.path.join(root, files[n])
                    if self.identifiers is None:
                        self.identifiers = np.loadtxt(filepath, dtype='str')
                    else:
                        ids = np.loadtxt(filepath, dtype='str')
                        self.identifiers = np.insert(self.identifiers, len(self.identifiers), ids)
        return
    

    def spoc_load_lc_from_metafiles(self, DEBUG=False, debug_ind=0,
                                    nan_mask_check=True, custom_mask=[]):
    
    # >> get file names for each group
        fnames = []
        print("Loading in SPOC light curves from folder")
        fname_info = []
        for i in range(len(self.cams)):
            cam = self.cams[i]
            for ccd in self.ccds[i]:
                s = '/Sector{sector}Cam{cam}CCD{ccd}/' + \
                    'Sector{sector}Cam{cam}CCD{ccd}_lightcurves.fits'
                fnames.append(s.format(sector=self.sector, cam=cam, ccd=ccd))
                fname_info.append([self.sector, cam, ccd, "SPOC", self.cadence])
                    
        # >> pull data from each fits file
        print('Pulling data')
        flux_list = []
        ticid = np.empty((0, 1))
        target_info = [] # >> [sector, cam, ccd, data_type, cadence]
        for i in range(len(fnames)):
            print('Loading ' + fnames[i] + '...')
            with fits.open(self.datapath + fnames[i], memmap=False) as hdul:
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
            flux, x = df.nan_mask(flux, x, DEBUG=DEBUG, ticid=ticid,
                               debug_ind=debug_ind, target_info=target_info,
                               output_dir=self.savepath, custom_mask=custom_mask)
        self.time = x
        self.flux = flux
        self.identifiers = ticid
        self.target_info = np.array(target_info)
    
        return 

        
    def median_normalize(self):
        '''Dividing by median.
        !!Current method blows points out of proportion if the median is too close to 0?'''
        print("Median Normalizing")
        medians = np.median(self.flux, axis = 1, keepdims=True)
        self.flux = self.flux / medians - 1.
        self.isfluxnormalized = True
        return 
    
    def mean_normalize(self):
        print("Mean normalizing")
        means = np.mean(self.flux, axis=1, keepdims = True)
        self.flux = self.flux / means
        self.isfluxnormalized = True
        return cleanedflux
    
    def pca_linregress(self, plot=False):
        
        print("Beginning PCA Linear Regression Corrections for 3 components")
        from sklearn.linear_model import LinearRegression
        pca = PCA(n_components=3)
        if not self.isfluxnormalized:
            self.median_normalize()
        standardized_flux = df.standardize(self.flux)
        #x should have shape(num samples, num features) (rows, columns)
        # ie each COLUMN is a light curve, so you get principal component of
        p = pca.fit_transform(standardized_flux.T) #take transpose - input is COLUMNS of light curves
        
        components = p.T #now have n_component rows of reduced curves??
        #print(components)
        
        for n in range(len(components)):
            plt.scatter(self.time, components[n])
            plt.title("Principal Component " + str(n))
            plt.savefig(self.savepath + "standardized-pc-" + str(n)+'.png')
            plt.show()
        residuals = np.zeros_like(self.flux)
        for n in range(len(self.flux)):
            reg = LinearRegression().fit(components.T, standardized_flux[n])
            
            y = reg.coef_[0] * components[0] + reg.coef_[1] * components[1] + reg.coef_[2] * components[2]
            residual = standardized_flux[n] - y
            residuals[n] = residual
            if plot:
                score = reg.score(components.T, standardized_flux[n])
                print(reg.coef_, reg.score(components.T, standardized_flux[n]), reg.intercept_)
            
                plt.scatter(self.time, y, label ="lin regress")
                plt.scatter(self.time, standardized_flux[n], label = "original data")
                plt.title("Raw data versus linear regression fit. Fit score: " + str(score))
                plt.legend()
                plt.savefig(self.savepath + "origversuslinregress-" + str(n) + ".png")
                plt.show()
                
                plt.scatter(self.time, residual)
                plt.title("Residual")
                plt.savefig(self.savepath + "residual-" + str(n) + ".png")
                plt.show()
            
        return residuals, components
        
    
    def create_save_featvec(self, version=0, save=True):
        """ documentation """
        fname_features = self.ENFpath + "features_v"+str(version)+".fits"
        feature_list = []
        if version == 0:
            self.median_normalize()
            self.cleanedflux, comp = self.pca_linregress()
        elif version == 1: 
            from transitleastsquares import transitleastsquares
            self.mean_normalize()
            self.cleanedflux, comp = self.pca_linregress()
        print("Begining Feature Vector Creation Now")
        #sigma clip each time you calculate - unsure how better to do this??
        sigclip = SigmaClip(sigma=5, maxiters=None, cenfunc='median')
        for n in range(len(self.cleanedflux)):
            
            times = self.time
            ints = self.cleanedflux[n]
            
            clipped_inds = np.nonzero(np.ma.getmask(sigclip(ints)))
            ints[clipped_inds] = np.nan
            delete_index = np.argwhere(np.isnan(ints))
            times = np.delete(times, delete_index)
            ints = np.delete(ints, delete_index)
            
            try:
                feature_vector = df.featvec(times, ints, v=version)
            except ValueError:
                print("it did the stupid thing where it freaked out about one light curve and idk why")
            
            if version == 1:
                feature_vector = np.nan_to_num(feature_vector, nan=0)
            feature_list.append(feature_vector)
            
            if n % 500 == 0: print(str(n) + " completed")
        
        self.features = np.asarray(feature_list)
        
        if save == True:
            hdr = fits.Header()
            hdr["VERSION"] = version
            hdu = fits.PrimaryHDU(feature_list, header=hdr)
            hdu.writeto(fname_features)
        else: 
            print("Not saving feature vectors to fits")
        return   
    
    
    def plot_lof(self, featuretype = "ENF"):
        """ this is going to redirect you to the actual plot_lof function with 
        everything filled in correctly"""
        
        if self.datatype == "SPOC" and featuretype == "ENF":
        
            if not self.isfluxnormalized:
                self.median_normalize()
            pf.plot_lof(self.time, self.flux, self.identifiers, self.features, 20, self.ENFpath,
                 momentum_dump_csv = self.momdumpcsv,
                 n_neighbors=20, target_info=self.target_info, p=2, metric='minkowski',
                 contamination=0.1, algorithm='auto',
                 prefix='', mock_data=False, addend=1.,
                 n_tot=100, log=False, debug=True, feature_lof=None,
                 bins=50, cross_check_txt=None, single_file=False,
                 fontsize='xx-small', title=True, plot_psd=True, n_pgram=1000)
            
        return
    
    def features_plotting(self, featuretype = "ENF", clustering = "None"):
        
        if not self.isfluxnormalized:
            self.median_normalize()
        if featuretype == "ENF" and self.datatype == "SPOC":
            pf.features_plotting_2D(self.features, self.ENFpath, clustering,
                                 self.time, self.flux, self.identifiers, folder_suffix='',
                                 feature_engineering=True, version=0, eps=0.5, min_samples=5,
                                 metric='minkowski', algorithm='auto', leaf_size=30,
                                 p=2, target_info=self.target_info, kmeans_clusters=4,
                                 momentum_dump_csv= self.momdumpcsv)