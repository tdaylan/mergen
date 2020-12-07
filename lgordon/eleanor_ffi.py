# -*- coding: utf-8 -*-
"""
Created on Dec 2 2020

@author: Lindsey Gordon @lcgordon

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
from scipy.linalg.misc import LinAlgError

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


    
    
class eleanor_lc(object):
    """
    doctext here
    """

    def __init__(self, path="/Users/conta/urop/", folderlabel = "Sectors1-6_eleanor/", 
                 list = "C:/Users/conta/UROP/SNe - fausnaugh-18.csv" ):

        """ 
        put documentation here
        
        """
        self.path = path
        self.folderlabel = folderlabel
        self.lightcurvefilepath = self.path + self.folderlabel + "eleanor_lc.fits"
        try:
            print(self.path)
            os.mkdir(self.path + self.folderlabel)
        except OSError:
            print('Directory exists already!')
            
        if list is not None: 
            self.list = list
            self.identifiers, self.RAlist, self.DEClist = self.get_radecfromfile()
            self.radecall = np.column_stack((self.RAlist, self.DEClist))
            self.eleanor_lc(plot=True)
            self.gaia_ids, self.times, self.i, self.i_corr = self.open_eleanor_lc_files()
        else:
            self.gaia_ids, self.times, self.i, self.i_corr = self.open_eleanor_lc_files()
            
            
            
        
                
        
    
    def get_radecfromfile(self):
        ''' pulls ra and dec from text file containing all targets
        '''
        ra_all = []
        dec_all = []
        names_all = []
        
        with open(self.list, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                line = line.replace('\n', '')
                name, ra, dec = line.split(',')
                 
                names_all.append(name)
                ra_all.append(ra)
                dec_all.append(dec)
                    
        return np.asarray(names_all), np.asarray(ra_all), np.asarray(dec_all)
    
    def eleanor_lc(self, plot=False):
        """ 
        retrieves + produces eleanor light curves from FFI files
        """
        import eleanor
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        import warnings
        warnings.filterwarnings('ignore')
        from eleanor.utils import SearchError
        
        download_dir_tesscut = os.path.join(os.path.expanduser('~'), '.eleanor', 'tesscut')
        
        download_dir_mastdownload = os.path.join(os.path.expanduser('~'), '.eleanor', 'mastDownload')
        print(download_dir_tesscut, download_dir_mastdownload)
        gaia_ids = []
        
        print(self.radecall[:10])
        for n in range(len(self.radecall)):
        #for n in range(10):
            try:
                
                coords = SkyCoord(ra=self.radecall[n][0], dec=self.radecall[n][1], unit=(u.deg, u.deg))
                    #try:
                files = eleanor.multi_sectors(coords=coords, tic=0, gaia = 0, sectors='all') #by not providing a sector argument, will ONLY retrieve most recent sector
                print(len(files))
                print('Found TIC {0} (Gaia {1}), with TESS magnitude {2}, RA {3}, and Dec {4}'
                      .format(files[0].tic, files[0].gaia, files[0].tess_mag, files[0].coords[0], files[0].coords[1]))
                #data = eleanor.TargetData(files)
                
                for file in files:
                    data = eleanor.TargetData(file)
                    q = data.quality == 0

                    fluxandtime = [data.time[q], data.raw_flux[q], data.corr_flux[q]]
                    lightcurve = np.asarray(fluxandtime)
                    
                    if plot: 
                        #!!! put plotting background here
                        print("Plotting TPF + aperture")
                        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15,4))
                        ax1.imshow(data.tpf[0])
                        ax1.set_title('Target Pixel File')
                        ax2.imshow(data.bkg_tpf[0])
                        ax2.set_title('2D interpolated background');
                        ax3.imshow(data.aperture)
                        ax3.set_title('Aperture')
                        plt.savefig(self.path + self.folderlabel + self.identifiers[n] + ".png")
                   
                    if not os.path.isfile(self.lightcurvefilepath):
                        #setting up fits file + save first one            
                        hdr = fits.Header() # >> make the header
                        hdu = fits.PrimaryHDU(lightcurve, header=hdr)
                        hdu.writeto(self.lightcurvefilepath)
                        print(int(n))
                                                        
                    else: #save the rest
                        fits.append(self.lightcurvefilepath, lightcurve)
                        print(int(n))
                           
                    gaia_ids.append(int(file.gaia))
            except (SearchError, ValueError, LinAlgError):
                print("Search Error or ValueError or LinAlgError occurred")
            
            #try: 
            for root, dirs, files in os.walk(download_dir_tesscut):
                for file in files:
                    try: 
                        os.remove(os.path.join(root, file))
                        #print("Deleted", os.path.join(root, file))
                    except (PermissionError, OSError):
                        #print("Unable to delete", os.path.join(root, file))
                        continue
            for root, dirs, files in os.walk(download_dir_mastdownload):
                for file in files:
                    try:
                        os.remove(os.path.join(root, file))
                        #print("Deleted", os.path.join(root, file))
                    except (PermissionError, OSError):
                        #print("Deleted", os.path.join(root, file))
                        continue
        fits.append(self.lightcurvefilepath, np.asarray(gaia_ids))
        print("All light curves saved into fits file")
        return gaia_ids
    
    def open_eleanor_lc_files(self):
        """ opens the fits file that the eleanor light curves are saved into
        parameters:
            * path to the fits file
        returns:
            * list of gaia_ids
            * time indexes
            * intensities
        modified [lcg 08212020]"""
        f = fits.open(self.lightcurvefilepath, memmap=False)
        gaia_ids = f[-1].data
        target_nums = len(f) - 1
        all_timeindexes = []
        all_intensities = []
        all_i_corrected = []
        for n in range(target_nums):
            all_timeindexes.append(f[n].data[0])
            all_intensities.append(f[n].data[1])
            all_i_corrected.append(f[n].data[2])
                
        f.close()
            
        return gaia_ids, np.asarray(all_timeindexes), np.asarray(all_intensities), np.asarray(all_i_corrected)
    
    def create_save_featvec_different_timeaxes(self):
        """Produces the feature vectors for each light curve and saves them all
        into a single fits file. all light curves have their OWN time axis
        this is set up to work on the eleanor light curves
        Parameters:
            * yourpath = folder you want the file saved into
            * times = all time axes
            * intensities = array of all light curves (NOT normalized)
            * sector, camera, ccd = integers 
            * version = what version of feature vector to calculate for all. 
                default is 0
            * save = whether or not to save into a fits file
        returns: list of feature vectors + fits file containing all feature vectors
        requires: featvec()
        modified: [lcg 08212020]"""
        
        feature_list = []
        self.savetrue = True
        
        if self.version == 0:
            fname_features = self.features0path
            #median normalize for the v0 features
            for n in range(len(self.intensities)):
                self.intensities[n] = df.normalize(self.intensities[n], axis=0)
        elif self.version == 1: 
            fname_features = self.features1path
            import transitleastsquares
            from transitleastsquares import transitleastsquares
            #mean normalize the intensity so goes to 1
            for n in range(len(self.intensities)):
                self.intensities[n] = df.mean_norm(self.intensities[n], axis=0)
    
        print("Begining Feature Vector Creation Now")
        for n in range(len(self.intensities)):
            feature_vector = df.featvec(self.times[n], self.intensities[n], v=self.version)
            feature_list.append(feature_vector)
            
            if n % 25 == 0: print(str(n) + " completed")
        
        feature_list = np.asarray(feature_list)
        
        if self.savetrue:
            hdr = fits.Header()
            hdr["VERSION"] = self.version
            hdu = fits.PrimaryHDU(feature_list, header=hdr)
            hdu.writeto(fname_features)
            fits.append(fname_features, self.gaia_ids)
        else: 
            print("Not saving feature vectors to fits")
        
        return feature_list
    

    

    
    def sigmaclip(self):
        print("Sigma clipping")
        self.sctimes = []
        self.scintensities = []
        self.scintcorr = []
        for i in range(len(self.intensities)):

            sigclip = SigmaClip(sigma=5, maxiters=None, cenfunc='median')
            clipped_inds = np.nonzero(np.ma.getmask(sigclip(self.intensities[i])))
            self.intensities[i][clipped_inds] = np.nan
            delete_index = np.argwhere(np.isnan(self.intensities[i]))
            sctime = np.delete(self.times[i], delete_index)
            self.sctimes.append(sctime)
            scflux = np.delete(self.intensities[i], delete_index)  
            self.scintensities.append(scflux)
            sccorrint = np.delete(self.corrected_intensities[i], delete_index)
            self.scintcorr.append(sccorrint)
            
        self.times = np.asarray(self.sctimes)
        self.intensities = np.asarray(self.scintensities)
        self.corrected_intensities = np.asarray(self.scintcorr)
        
    def normalize(self):
        print("Normalizing")
        for i in range(len(self.times)):
            median = np.median(self.intensities[i])
            self.intensities[i] = self.intensities[i] / median
        

            
        
            
                
