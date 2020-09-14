# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 17:07:33 2020

@author: Lindsey Gordon
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

import eleanor

import pdb
import fnmatch as fm

import plotting_functions as pf
import data_functions as df



class hyperleda_ffi(object):
    
    def __init__(self, path = "./", momentum_dump_csv = '/users/conta/urop/Table_of_momentum_dumps.csv', tester = True):
        print("Initialized hyperleda ffi processing object")
        self.path = path
        self.tester = tester
        self.momdumpcsv = momentum_dump_csv
        return
        
        
    
    def load_lc_from_files(self, folderpath):
        print("Loading in light curves from", folderpath)
        
        self.folderpath = folderpath

        #make sure they're all txt files
        for filename in os.listdir(self.folderpath):
            file = os.path.join(self.folderpath, filename)
            if filename.startswith("lc_") and not filename.endswith(".txt"):
                os.rename(file, file + ".txt")
        
        #get all file names
        #fitspaths = []
        intensities = []
        identifiers = []
        
        for root, dirs, files in os.walk(self.folderpath):
            
            for n in range(len(files)):
                #print(files[n])
                if files[n].endswith(("cleaned.txt")):
                    #print(os.path.join(root, files[n]))
                    lc_data = np.genfromtxt(os.path.join(root,files[n]), skip_header = 1)
                    u, iden, k = files[n].split("_")
                    identifiers.append(iden)
                    if n == 0:
                        timeaxis = lc_data[:,0]
                        intensities.append(lc_data[:,2])
                    else:
                        intensities.append(lc_data[:,2])
                        
                    if n % 50 == 0:
                        print(n, "completed")
        
        self.intensities = np.asarray(intensities)
        self.timeaxis = timeaxis
        self.identifiers = identifiers
        return
    
    def median_normalize(self):
        '''Dividing by median.
        !!Current method blows points out of proportion if the median is too close to 0?'''
        print("Median Normalizing")
        medians = np.median(self.intensities, axis = 1, keepdims=True)
        self.cleanedflux = self.intensities / medians - 1.
        return
    
    def mean_normalize(self):
        print("Mean normalizing")
        means = np.mean(self.intensities, axis=1, keepdims = True)
        self.cleanedflux = self.intensities / means
        return
    
    
    #i think i'm going to have to just sigma clip each light curve as i go through the feature generation
                    
    def create_save_featvec_homogenous_time(self, filelabel, version=0, save=True):
        """ documentation """

        fname_features = self.folderpath + "/"+ filelabel + "_features_v"+str(version)+".fits"
        feature_list = []
        if version == 0:
            self.median_normalize()
        elif version == 1: 
            from transitleastsquares import transitleastsquares
            #mean normalize the intensity so goes to 1
            self.mean_normalize()
    
        print("Begining Feature Vector Creation Now")
        sigclip = SigmaClip(sigma=5, maxiters=None, cenfunc='median')
        for n in range(len(self.cleanedflux)):
            
            times = self.timeaxis
            ints = self.cleanedflux[n]
            
                
            clipped_inds = np.nonzero(np.ma.getmask(sigclip(ints)))
            ints[clipped_inds] = np.nan
            delete_index = np.argwhere(np.isnan(ints))
            times = np.delete(times, delete_index)
            ints = np.delete(ints, delete_index)
            
            
            feature_vector = df.featvec(times, ints, v=version)
            feature_list.append(feature_vector)
            
            if n % 25 == 0: print(str(n) + " completed")
        
        self.features = np.asarray(feature_list)
        
        if save == True:
            hdr = fits.Header()
            hdr["VERSION"] = version
            hdu = fits.PrimaryHDU(feature_list, header=hdr)
            hdu.writeto(fname_features)
        else: 
            print("Not saving feature vectors to fits")
        
        return   
    
    def load_features(self):
        
        for root, dirs, files in os.walk(self.folderpath):
            
            for n in range(len(files)):
                #print(files[n])
                if files[n].endswith(("_features_v0.fits")):
                    f = fits.open(os.path.join(root,files[n]), memmap=False)
                    self.features = f[0].data
                    f.close()
        
    
    def plot_lof(self, n, n_neighbors=20, n_tot=100):
        """ 
        """
        prefix=''
        p=2
        bins=50
        # -- calculate LOF -------------------------------------------------------
        print('Calculating LOF')
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, p=p)
        fit_predictor = clf.fit_predict(self.features)
        negative_factor = clf.negative_outlier_factor_
        
        lof = -1 * negative_factor
        ranked = np.argsort(lof)
        largest_indices = ranked[::-1][:n_tot] # >> outliers
        smallest_indices = ranked[:n_tot] # >> inliers
        random_inds = list(range(len(lof)))
        import random
        random.Random(4).shuffle(random_inds)
        random_inds = random_inds[:n_tot] # >> random
        ncols=1
    
          
        # >> make histogram of LOF values
        print('skipping LOF histogram')
        #plot_histogram(lof, 20, "Local Outlier Factor (LOF)", time, intensity,
         #              targets, path+'lof-'+prefix+'histogram-insets.png',
          #             insets=True, log=log)
        #plot_histogram(lof, 20, "Local Outlier Factor (LOF)", self.time, self.cleanedflux,
         #              self.identifiers, path+'lof-'+prefix+'histogram.png', insets=False,
          #             log=log)
    

        print('Saving LOF values')
        with open(self.path+'lof-'+prefix+'kneigh' + str(n_neighbors)+'.txt', 'w') as f:
            for i in range(len(self.identifiers)):
                f.write('{} {}\n'.format(self.identifiers[i], lof[i]))
              
            # >> make histogram of LOF values
        print('Make LOF histogram')
        #plot_histogram(lof, 20, "Local Outlier Factor (LOF)", self.time, self.cleanedflux,
         #              self.identifiers, self.path+'lof-'+prefix+'kneigh' + str(n_neighbors)+\
          #                 'histogram-insets.png',
           #            insets=True, log=log)
        #plot_histogram(lof, 20, "Local Outlier Factor (LOF)", self.time, self.cleanedflux,
         #              self.identifiers, self.path+'lof-'+prefix+'kneigh' + str(n_neighbors)+\
          #                 'histogram.png', insets=False,
           #            log=log)
    
            
        # -- momentum dumps ------------------------------------------------------
        # >> get momentum dump times
        print('Loading momentum dump times')
        with open(self.momdumpcsv, 'r') as f:
            lines = f.readlines()
            mom_dumps = [ float(line.split()[3][:-1]) for line in lines[6:] ]
            inds = np.nonzero((mom_dumps >= np.min(self.timeaxis)) * \
                              (mom_dumps <= np.max(self.timeaxis)))
            mom_dumps = np.array(mom_dumps)[inds]
            
       
        # -- plot smallest and largest LOF light curves --------------------------
        print('Plot highest LOF and lowest LOF light curves')
        num_figs = int(n_tot/n) # >> number of figures to generate
        
        for j in range(num_figs):
            
            for i in range(3): # >> loop through smallest, largest, random LOF plots
                fig, ax = plt.subplots(n, ncols, sharex=False,
                                       figsize = (8*ncols, 3*n))
                
                for k in range(n): # >> loop through each row
                
                    axis = ax[k]
                    
                    if i == 0: ind = largest_indices[j*n + k]
                    elif i == 1: ind = smallest_indices[j*n + k]
                    else: ind = random_inds[j*n + k]
                    
                    # >> plot momentum dumps
                    for t in mom_dumps:
                        axis.axvline(t, color='g', linestyle='--')
                        
                    # >> plot light curve
                    axis.plot(self.timeaxis, self.cleanedflux[ind], '.k')
                    axis.text(0.98, 0.02, '%.3g'%lof[ind],
                               transform=axis.transAxes,
                               horizontalalignment='right',
                               verticalalignment='bottom',
                               fontsize='xx-small')                        
                        
                    if k != n - 1:
                        axis.set_xticklabels([])
                        
                # >> label axes

                ax[n-1].set_xlabel('time [BJD - 2457000]')
                    
                # >> save figures
                if i == 0:
                    
                    fig.suptitle(str(n) + ' largest LOF targets', fontsize=16,
                                 y=0.9)
                    fig.tight_layout()
                    fig.savefig(self.path + 'lof-' + prefix + 'kneigh' + \
                                str(n_neighbors) + '-largest_' + str(j*n) + 'to' +\
                                str(j*n + n) + '.png',
                                bbox_inches='tight')
                    plt.close(fig)
                elif i == 1:
                    fig.suptitle(str(n) + ' smallest LOF targets', fontsize=16,
                                 y=0.9)
                    fig.tight_layout()
                    fig.savefig(self.path + 'lof-' + prefix + 'kneigh' + \
                                str(n_neighbors) + '-smallest' + str(j*n) + 'to' +\
                                str(j*n + n) + '.png',
                                bbox_inches='tight')
                    plt.close(fig)
                else:
                    fig.suptitle(str(n) + ' random LOF targets', fontsize=16, y=0.9)
                    
                    # >> save figure
                    fig.tight_layout()
                    fig.savefig(self.path + 'lof-' + prefix + 'kneigh' + str(n_neighbors) \
                                + "-random"+ str(j*n) + 'to' +\
                                str(j*n + n) +".png", bbox_inches='tight')
                    plt.close(fig)
                    
                        
                        
                        
            
            
            
            
