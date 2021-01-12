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
from scipy.signal import argrelextrema, medfilt


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
from sklearn.cluster import KMeans, DBSCAN
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
import sn_functions as sn


class mergen(object):
    
    def __init__(self, datapath, datatype, savepath, filelabel,
                 momentum_dump_csv = '/users/conta/urop/Table_of_momentum_dumps.csv',
                 sector = [20], cams = [1,2,3,4], ccds = [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]], 
                 cadence = "2minute", ENF_exists = False, remove_quaternions = False,
                 quaternion_file = '/Users/conta/UROP/S2DataAll/tess2018330083923_sector02-quat.fits'):
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
        self.target_info = None
        
        
        self.folder_initiate()
        
        if self.datatype == "SPOC":
            #load in SPOC lc from the metafiles
            print("loading spoc")
            if len(self.sector) > 1:
                custom_masks = [list(range(500)) + list(range(15800, 17400)), []]
                self.flux, self.time, self.identifiers, self.target_info = da.combine_sectors(self.sector, self.datapath,
                                                             custom_masks=custom_masks)
                self.flux, self.time, self.identifiers, self.target_info = da.combine_sectors_by_lc(self.sector, self.datapath,
                                                                   custom_mask=custom_mask,
                                                                   output_dir=self.savepath)
            elif len(self.sector) == 1:
                self.sector = self.sector[0]
                self.spoc_load_lc_from_metafiles()
            
        elif self.datatype == "FFI":
            #load in FFI lc from the metafiles
            #currently only works for one sector
            print("loading FFIs")
            self.ffi_load_lc_metafiles_all()
            if remove_quaternions:
                outlier_indexes = df.extract_smooth_quaterions(self.savepath, quaternion_file, 
                                                            self.momdumpcsv, 31, 
                                                            self.time)
                print("Current length of time axis: ", len(self.time))
                self.time = np.delete(self.time, outlier_indexes, axis=0)
                self.flux = np.delete(self.flux, outlier_indexes, axis=1)
                print("Updated length of time axis: ", len(self.time))
            
        if ENF_exists:
            print("Loading in existing feature metafiles")
            self.features = enf.load_feature_metafile(self.datapath)
            self.median_normalize()
            self.isolate_outlier_features()
            
      
            
    
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
                        self.flux = np.vstack((self.flux, f[1].data))
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
        self.flux = self.flux / medians
        self.isfluxnormalized = True
        return 
    
    def mean_normalize(self):
        print("Mean normalizing")
        means = np.mean(self.flux, axis=1, keepdims = True)
        self.flux = self.flux / means
        self.isfluxnormalized = True
        return 
    
    def pca_linregress(self, plot=False):
        
        print("Beginning PCA Linear Regression Corrections for 3 components")
        from sklearn.linear_model import LinearRegression
        pca = PCA(n_components=5)
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
            
            y = reg.coef_[0] * components[0] + reg.coef_[1] * components[1] + reg.coef_[2] * components[2] \
                + reg.coef_[3] * components[3] + reg.coef_[4] * components[4]
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
     

    def create_ENF_features(self, version=0, save=True):
        """ documentation """
        fname_features = self.ENFpath + "features_v"+str(version)+".fits"
        feature_list = []
        if version == 0:
            self.median_normalize()
            if self.datatype == "FFI":
                self.flux, comp = self.pca_linregress()
        elif version == 1: 
            from transitleastsquares import transitleastsquares
            self.mean_normalize()
            if self.datatype == "FFI":
                self.flux, comp = self.pca_linregress()
        print("Begining Feature Vector Creation Now")
        #sigma clip each time you calculate - unsure how better to do this??
        sigclip = SigmaClip(sigma=5, maxiters=None, cenfunc='median')
        for n in range(len(self.flux)):
            
            times = self.time
            ints = self.flux[n]
            
            clipped_inds = np.nonzero(np.ma.getmask(sigclip(ints)))
            ints[clipped_inds] = np.nan
            delete_index = np.argwhere(np.isnan(ints))
            times = np.delete(times, delete_index)
            ints = np.delete(ints, delete_index)
            
            try:
                feature_vector = enf.featvec(times, ints, v=version)
            except ValueError:
                print("it did the stupid thing where it freaked out about one light curve and idk why")
            
            if version == 1:
                feature_vector = np.nan_to_num(feature_vector, nan=0)
                with open(self.ENFpath + 'intermediate_v1_saving.txt', 'a') as f:
                    f.write('{} {} \n'.format(self.identifiers[n], feature_vector))
            feature_list.append(feature_vector)
            
            if version == 0:
                if n % 500 == 0: 
                    print(str(n) + " completed")
                if self.cadence == "20_s" and n % 20 == 0:
                    print(str(n) + " completed")
            elif version == 1:
                print(str(n))
        
        self.features = np.asarray(feature_list)
        
        if save == True:
            hdr = fits.Header()
            hdr["VERSION"] = version
            hdu = fits.PrimaryHDU(feature_list, header=hdr)
            hdu.writeto(fname_features)
        else: 
            print("Not saving feature vectors to fits")
        return   
 
    def isolate_outlier_features(self, sigma = 10):
        
        self.outlierfeaturepath = self.ENFpath + "clipped-feature-outliers/"
        try:
            os.makedirs(self.outlierfeaturepath)
        except OSError:
            print("%s already exists" % self.outlierfeaturepath)
        else:
            print("Successfully created the directory %s" % self.outlierfeaturepath)
        
        features_greek = [r'$\alpha$', 'B', r'$\Gamma$', r'$\Delta$', r'$\beta$', r'$\gamma$',r'$\delta$',
                      "E", r'$\epsilon$', "Z", "H", r'$\eta$', r'$\Theta$', "I", "K", r'$\Lambda$', "M", r'$\mu$'
                      ,"N", r'$\nu$']

        #identifying the outlier indexes
        outlier_indexes = []
        for i in range(len(self.features[0])):
            column = self.features[:,i] #get each column
            column_std = np.std(column) #find std
            column_top = np.mean(column) + column_std * sigma #find max limit
            column_bottom = np.mean(column) - (column_std * sigma) #min limit
            for n in range(len(column)):
                #find and note the position of any outliers
                if column[n] < column_bottom or column[n] > column_top or np.isnan(column[n]) ==True: 
                    outlier_indexes.append((int(n), int(i))) #(pos of outlier, which feature)
                    
        outlier_indexes = np.asarray(outlier_indexes)
        self.outlier_indexes = outlier_indexes
        
        #plotting everything 
        target_indexes = outlier_indexes[:,0] #is the index of the target on the lists
        print(target_indexes)
        feature_indexes = outlier_indexes[:,1] #is the index of the feature that it triggered on
        
        #plotting outlier light curves
        for i in range(len(outlier_indexes)):
            target_index = target_indexes[i]
            feature_index = feature_indexes[i]
            plt.figure(figsize=(8,3))
            plt.scatter(self.time, self.flux[target_index], s=0.5)
            target = self.identifiers[target_index]
            
            if np.isnan(self.features[target_index][feature_index]) == True:
                feature_title = features_greek[feature_index] + "=nan"
            else: 
                feature_value = '%s' % float('%.2g' % self.features[target_index][feature_index])
                feature_title = features_greek[feature_index] + "=" + feature_value
            #print(feature_title)
            if self.target_info is not None:
                plt.title("TIC " + str(target) + " " + feature_title, fontsize=8)
            else:
                plt.title(str(target) + " " + feature_title, fontsize=8)
            plt.tight_layout()
            plt.savefig((self.outlierfeaturepath + "featureoutlier-" + str(target) + ".png"))
            plt.show()
            
            
        self.feature_outliers = []
        self.identifiers_outliers = []
        for ind in target_indexes:
            self.feature_outliers.append(self.features[ind])
            self.identifiers_outliers.append(self.identifiers[ind])
  
        self.features = np.delete(self.features, target_indexes, axis=0)
        
        self.flux_outliers = self.flux[target_indexes]
        self.flux = np.delete(self.flux, target_indexes, axis=0)
        
        self.identifiers_outliers = self.identifiers[target_indexes]
        self.identifiers = np.delete(self.identifiers, target_indexes)
        
        if self.target_info is not None:
            self.target_info = np.delete(self.target_info, target_indexes, axis = 0)
        
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
    
    def features_plotting(self, featuretype = "ENF", clustering = "None", eps = 2.5, min_samples = 4, version = 0 ):
        
        if not self.isfluxnormalized:
            self.median_normalize()
        if featuretype == "ENF" and self.datatype == "SPOC":
            pf.features_plotting_2D(self.features, self.ENFpath, clustering,
                                 self.time, self.flux, self.identifiers, folder_suffix='',
                                 feature_engineering=True, version=version, eps=eps, min_samples=min_samples,
                                 metric='minkowski', algorithm='auto', leaf_size=30,
                                 p=2, target_info=self.target_info, kmeans_clusters=4,
                                 momentum_dump_csv= self.momdumpcsv)
        return
         
    def plot_duration_range_lc(self, duration_max = 2):
        """ duration max in hours"""
        folder_path = self.ENFpath + "/duration-" + str(duration_max) + "-hrs/"
        try:
            os.makedirs(folder_path)
        except OSError:
            print ("Creation of the directory %s failed, directory already exists" % folder_path)
            
        rcParams['figure.figsize'] = 16,6
        for n in range(len(self.flux)):
            if self.features[n][1] < (duration_max/24) and self.features[n][1] != 0:
                plt.scatter(self.time, self.flux[n], s = 0.2)
                plt.title("TIC " + str(int(self.identifiers[n])) + "  Duration: " + str(self.features[n][1]))
                plt.savefig(folder_path + str(int(self.identifiers[n])) + ".png")
                plt.show()
                plt.close()
                with open(folder_path + '/duration_info.txt', 'a') as f:
                    f.write('{} {} \n'.format(int(self.identifiers[n]), self.features[n][1]))

    def autoencoder(self, sectors = [27], fast = False,
                    model_init = None, train_test_ratio = 0.9, hyperparameter_optimization = False,
                    lib_dir = None, database_dir = None, single_file = False, 
                    simbad_database_dir = '', run_model = True, iterative = False, diag_plots = True,
                    novelty_detection=True, classification_param_search=False, classification=True, 
                    norm_type = 'standardization', input_rms=True, input_psd=False, n_pgram = 50):
        data_dir = self.datapath
        output_dir = self.CAEpath
        mom_dump = self.momdumpcsv
        sectors = [self.sector]
        cams = self.cams
        ccds = self.ccds
        fast=fast
        
        # weights init
        # model_init = output_dir + 'model'
        model_init = model_init


        # train_test_ratio = 0.1 # >> fraction of training set size to testing set size
        train_test_ratio = train_test_ratio
        
         # >> runs DBSCAN on learned features
        
        # >> normalization options:
        #    * standardization : sets mean to 0. and standard deviation to 1.
        #    * median_normalization : divides by median
        #    * minmax_normalization : sets range of values from 0. to 1.
        #    * none : no normalization
        
        load_psd=False # >> if psd_train.fits, psd_test.fits already exists
        use_tess_features = True
        use_tls_features = False
        input_features=False # >> this option cannot be used yet
        split_at_orbit_gap=False
        DAE = False
        
        # >> move targets out of training set and into testing set (integer)
        # !! TODO: print failure if target not in sector
        # targets = [219107776] # >> EX DRA # !!
        validation_targets = []
        
        if sectors[0] == 1:
            custom_mask = list(range(800)) + list(range(15800, 17400)) + list(range(19576, 20075))
        elif 4 in sectors:
            custom_mask = list(range(7424, 9078))
        else:
            custom_mask = []
        
        custom_masks = [list(range(500)) + list(range(15800, 17400)), []]

        # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
        import talos                    # >> a hyperparameter optimization library
        import pdb
        import tensorflow as tf
        # tf.enable_eager_execution()
        
        import sys
        if lib_dir is not None:
            sys.path.insert(0, lib_dir)     # >> needed if scripts not in current dir

        # >> hyperparameters
        if hyperparameter_optimization:
            p = {'kernel_size': [3,5],
                  'latent_dim': [25],
                  'strides': [2],# 3
                  'epochs': [5],
                  'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
                  'num_filters': [32,64,128],
                  'num_conv_layers': [4,6,8,10],
                  'batch_size': [128],
                  'activation': [tf.keras.activations.softplus,
                                 tf.keras.activations.selu,
                                 tf.keras.activations.relu,
                                 'swish',
                                 tf.keras.activations.exponential,
                                 tf.keras.activations.elu, 'linear'],
                  'optimizer': ['adam', 'adadelta'],
                  'last_activation': ['linear'],
                  'losses': ['mean_squared_error'],
                  'lr': [0.001],
                  'initializer': ['random_normal'],
                  'num_consecutive': [2],
                   'kernel_regularizer': [None],
                  
                   'bias_regularizer': [None],
                  
                   'activity_regularizer': [None],
                 
                  'pool_size': [1]}     
        
        else:
            # >> strides: list, len = num_consecutive
            p = {'kernel_size': 3,
                  'latent_dim': 35,
                  'strides': 1,
                  'epochs': 5,
                  'dropout': 0.2,
                  'num_filters': 16,
                  'num_conv_layers': 4,
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
                  'units': [1024, 512, 64, 16],
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
                  'batchnorm_before_act': True,
                  'concat_ext_feats': False}      
    
        # -- create output directory --------------------------------------------------
            
        if os.path.isdir(output_dir) == False: # >> check if dir already exists
            os.mkdir(output_dir)
            
            
        
        x_train, x_test, y_train, y_test, ticid_train, ticid_test, target_info_train, \
            target_info_test, rms_train, rms_test, x = \
            ml.autoencoder_preprocessing(self.flux, self.time, p, self.identifiers, 
                                         self.target_info, mock_data=False,
                                         sector=sectors[0],
                                         validation_targets=validation_targets,
                                         norm_type=norm_type,
                                         input_rms=input_rms, input_psd=input_psd,
                                         load_psd=load_psd, n_pgram=n_pgram,
                                         train_test_ratio=train_test_ratio,
                                         split=split_at_orbit_gap,
                                         output_dir=output_dir, 
                                         data_dir=data_dir,
                                         use_tess_features=use_tess_features,
                                         use_tls_features=use_tls_features)
            

        if input_psd:
            p['concat_ext_feats'] = True
        
        title='TESS-unsupervised'
        
        # == talos experiment =========================================================
        if hyperparameter_optimization:
            print('Starting hyperparameter optimization...')
            t = talos.Scan(x=x_test,
                            y=x_test,
                            params=p,
                            model=ml.conv_autoencoder,
                            experiment_name=title, 
                            reduction_metric = 'val_loss',
                            minimize_loss=True,
                            reduction_method='correlation',
                            fraction_limit = 0.001)      
            # fraction_limit = 0.001
            analyze_object = talos.Analyze(t)
            data_frame, best_param_ind,p = pf.hyperparam_opt_diagnosis(analyze_object,
                                                               output_dir,
                                                               supervised=False)
        
        # == run model ================================================================
        if run_model:
            print('Training autoencoder...') 
            history, model, x_predict = \
                ml.conv_autoencoder(x_train, x_train, x_test, x_test, p, val=False,
                                    split=split_at_orbit_gap,
                                    ticid_train=ticid_train, ticid_test=ticid_test,
                                    save_model=True, predict=True,
                                    save_bottleneck=True,
                                    output_dir=output_dir,
                                    model_init=model_init) 
            
            if split_at_orbit_gap:
                x_train = np.concatenate(x_train, axis=1)
                x_test = np.concatenate(x_test, axis=1)
                x_predict = np.concatenate(x_predict, axis=1)
            
            
        # == Plots ====================================================================
        if diag_plots:
            print('Creating plots...')
            pf.diagnostic_plots(history, model, p, output_dir, x, x_train,
                                x_test, x_predict, mock_data=False,
                                addend=0.,
                                target_info_test=target_info_test,
                                target_info_train=target_info_train,
                                ticid_train=ticid_train,
                                ticid_test=ticid_test, percentage=False,
                                input_features=input_features,
                                input_rms=input_rms, rms_test=rms_test,
                                input_psd=input_psd,
                                rms_train=rms_train, n_tot=40,
                                plot_epoch = True,
                                plot_in_out = True,
                                plot_in_bottle_out=False,
                                plot_latent_test = True,
                                plot_latent_train = True,
                                plot_kernel=False,
                                plot_intermed_act=False,
                                make_movie = False,
                                plot_lof_test=False,
                                plot_lof_train=False,
                                plot_lof_all=False,
                                plot_reconstruction_error_test=True,
                                plot_reconstruction_error_all=False,
                                load_bottleneck=True)          
            
         
        # if input_psd:
        #     x = x[0]            
        for i in [0,1,2]:
            if i == 0:
                use_learned_features=True
                use_tess_features=False
                use_tls_features=False
                use_engineered_features=False
                use_rms=False
                description='_0_learned'
                DAE=False
            elif i == 1:
                use_learned_features=False
                use_tess_features=True
                use_tls_features=False
                use_engineered_features=False        
                use_rms=False
                description='_1_ext'
                DAE_hyperparam_opt=True
                DAE=True
                p_DAE = {'max_dim': [9, 11, 13, 15, 17, 19], 'step': [1,2,3,4,5,6],
                          'latent_dim': [3,4,5],
                          'activation': ['relu', 'elu'],
                        'last_activation': ['relu', 'elu'],
                          'optimizer': ['adam'],
                          'lr':[0.001, 0.005, 0.01], 'epochs': [20],
                          'losses': ['mean_squared_error'],
                          'batch_size':[128],
                          'initializer': ['glorot_normal', 'glorot_uniform'],
                          'fully_conv': [False]}             
            elif i == 2:
                use_learned_features=True
                use_tess_features=True
                use_tls_features=False
                use_engineered_features=False        
                use_rms=True
                description='_2_learned_RMS_ext'        
                DAE_hyperparam_opt=True
                DAE=True
                p_DAE = {'max_dim': list(np.arange(40, 70, 5)), 'step': [1,2,3,4,5,6],
                          'latent_dim': list(np.arange(12, 50, 5)),
                          'activation': ['relu', 'elu'],
                        'last_activation': ['relu', 'elu'],
                          'optimizer': ['adam'],
                          'lr':[0.001, 0.005, 0.01], 'epochs': [20],
                          'losses': ['mean_squared_error'],
                          'batch_size':[128],
                          'initializer': ['glorot_normal', 'glorot_uniform'],
                          'fully_conv': [False]}                
                
            print('Creating feature space')
            
            if p['concat_ext_feats'] or input_psd:
                features, flux_feat, ticid_feat, info_feat = \
                    ml.bottleneck_preprocessing(sectors[0],
                                                np.concatenate([x_train[0], x_test[0]], axis=0),
                                                np.concatenate([ticid_train, ticid_test]),
                                                np.concatenate([target_info_train,
                                                                target_info_test]),
                                                rms=np.concatenate([rms_train, rms_test]),
                                                data_dir=data_dir, bottleneck_dir=output_dir,
                                                output_dir=output_dir,
                                                use_learned_features=use_learned_features,
                                                use_tess_features=use_tess_features,
                                                use_engineered_features=use_engineered_features,
                                                use_tls_features=use_tls_features,
                                                use_rms=use_rms, norm=True,
                                                cams=cams, ccds=ccds, log=True)    
                    
            
            else:
                features, flux_feat, ticid_feat, info_feat = \
                    ml.bottleneck_preprocessing(sectors[0],
                                                np.concatenate([x_train, x_test], axis=0),
                                                np.concatenate([ticid_train, ticid_test]),
                                                np.concatenate([target_info_train,
                                                                target_info_test]),
                                                rms=np.concatenate([rms_train, rms_test]),
                                                data_dir=data_dir,
                                                bottleneck_dir=output_dir,
                                                output_dir=output_dir,
                                                use_learned_features=True,
                                                use_tess_features=use_tess_features,
                                                use_engineered_features=False,
                                                use_tls_features=use_tls_features,
                                                use_rms=use_rms, norm=True,
                                                cams=cams, ccds=ccds, log=True)  
                    
            print('Plotting feature space')
            pf.latent_space_plot(features, output_dir + 'feature_space.png')    
            
            if DAE:
                if DAE_hyperparam_opt:
        
              
                    t = talos.Scan(x=features,
                                    y=features,
                                    params=p_DAE,
                                    model=ml.deep_autoencoder,
                                    experiment_name='DAE', 
                                    reduction_metric = 'val_loss',
                                    minimize_loss=True,
                                    reduction_method='correlation',
                                    fraction_limit = 0.1)            
                    analyze_object = talos.Analyze(t)
                    data_frame, best_param_ind,p_best = pf.hyperparam_opt_diagnosis(analyze_object,
                                                                        output_dir,
                                                                        supervised=False) 
                    p_DAE=p_best
                    p_DAE['epochs'] = 100
                    
                else:
                        
                    p_DAE = {'max_dim': 50, 'step': 4, 'latent_dim': 42,
                             'activation': 'elu', 'last_activation': 'elu',
                             'optimizer': 'adam',
                             'lr':0.001, 'epochs': 100, 'losses': 'mean_squared_error',
                             'batch_size': 128, 'initializer': 'glorot_uniform',
                             'fully_conv': False}    
                
                    
                    # p_DAE = {'max_dim': 9, 'step': 5, 'latent_dim': 4,
                    #          'activation': 'elu', 'last_activation': 'elu',
                    #          'optimizer': 'adam',
                    #          'lr':0.01, 'epochs': 100, 'losses': 'mean_squared_error',
                    #          'batch_size': 128, 'initializer': 'glorot_normal',
                    #          'fully_conv': False}               
                    
                history_DAE, model_DAE = ml.deep_autoencoder(features, features,
                                                               features, features,
                                                               p_DAE, resize=False,
                                                               batch_norm=True)
                new_features = ml.get_bottleneck(model_DAE, features, p_DAE, DAE=True)
                features=new_features
                
                pf.epoch_plots(history_DAE, p_DAE, output_dir)
                
                print('Plotting feature space')
                pf.latent_space_plot(features, output_dir + 'feature_space' + \
                                     ''+'_DAE.png')        
        
                
            if novelty_detection:
                print('Novelty detection')
                pf.plot_lof(x, flux_feat, ticid_feat, features, 20, output_dir,
                            momentum_dump_csv = self.momdumpcsv,
                            n_tot=200, target_info=info_feat, prefix=str(i),
                            cross_check_txt=database_dir, debug=True, addend=0.,
                            single_file=single_file, log=True, n_pgram=n_pgram,
                            plot_psd=True)
        
            if classification:
                if classification_param_search:
                    df.KNN_plotting(output_dir +'str(i)-', features, [10, 20, 100])
            
                    print('DBSCAN parameter search')
                    parameter_sets, num_classes, silhouette_scores, db_scores, ch_scores, acc = \
                    df.dbscan_param_search(features, x, flux_feat, ticid_feat,
                                            info_feat, DEBUG=False, 
                                            output_dir=output_dir+str(i), 
                                            simbad_database_txt=simbad_database_dir,
                                            leaf_size=[30], algorithm=['auto'],
                                            min_samples=[5],
                                            metric=['minkowski'], p=[3,4],
                                            database_dir=database_dir,
                                            eps=list(np.arange(1.5, 4., 0.1)),
                                            confusion_matrix=False, pca=False, tsne=False,
                                            tsne_clustering=False)      
                    
                    print('Classification with best parameter set')
                    best_ind = np.argmax(silhouette_scores)
                    best_param_set = parameter_sets[best_ind]
                    
                else:
                    best_param_set=[2.0, 3, 'minkowski', 'auto', 30, 4]    
              
            
                
                if classification_param_search:
                    print('HDBSCAN parameter search')
                    acc = df.hdbscan_param_search(features, x, flux_feat, ticid_feat,
                                                  info_feat, output_dir=output_dir,
                                                  p0=[3,4], single_file=single_file,
                                                  database_dir=database_dir, metric=['all'],
                                                  min_samples=[3], min_cluster_size=[3],
                                                  data_dir=data_dir)
                else:
                    # best_param_set = [3, 3, 'manhattan', None]
                    best_param_set = [3, 3, 'canberra', None]
                    print('Run HDBSCAN')
                    _, _, acc = df.hdbscan_param_search(features, x, flux_feat, ticid_feat,
                                                  info_feat, output_dir=output_dir,
                                                  p0=[best_param_set[3]], single_file=single_file,
                                                  database_dir=database_dir,
                                                  metric=[best_param_set[2]],
                                                  min_cluster_size=[best_param_set[0]],
                                                  min_samples=[best_param_set[1]],
                                                  DEBUG=True, pca=True, tsne=True,
                                                  data_dir=data_dir, save=True)  
                
                with open(output_dir + 'param_summary.txt', 'a') as f:
                    f.write('accuracy: ' + str(np.max(acc)))   
                    
                df.gmm_param_search(features, x, flux_feat, ticid_feat, info_feat,
                                 output_dir=output_dir, database_dir=database_dir, 
                                 data_dir=data_dir) 
        
                from sklearn.mixture import GaussianMixture
                gmm = GaussianMixture(n_components=200)
                labels = gmm.fit_predict(features)
                acc = pf.plot_confusion_matrix(ticid_feat, labels,
                                               database_dir=database_dir,
                                               single_file=single_file,
                                               output_dir=output_dir,
                                               prefix='gmm-')          
                pf.quick_plot_classification(x, flux_feat,ticid_feat,info_feat, 
                                             features, labels,path=output_dir,
                                             prefix='gmm-',
                                             database_dir=database_dir,
                                             single_file=single_file)
                pf.plot_cross_identifications(x, flux_feat, ticid_feat,
                                              info_feat, features,
                                              labels, path=output_dir,
                                              database_dir=database_dir,
                                              data_dir=data_dir, prefix='gmm-')
            
                
        # == iterative training =======================================================
                
        ml.iterative_cae(x_train, y_train, x_test, y_test, x, p, ticid_train, 
                          ticid_test, target_info_train, target_info_test, num_split=2,
                          output_dir=output_dir, split=split_at_orbit_gap,
                          input_psd=input_psd)