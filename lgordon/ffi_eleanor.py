# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 22:19:31 2020

@author: Lindsey Gordon @lcgordon

All functions for accessing FFI light curves + producing their custom feature vectors

class FFI_lc()

    
Updated: 8/26/2020
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


def test_ffi():
    """make sure the module loads in"""
    print("FFI functions loaded in.")
    
    
class FFI_lc(object):
    """
    init params:
        * path = path to save everything into
        * folderlabel = label for subfolder that everything is going to go into
        * simbadquery - formatted as a criteria query
        * download = True if you want to get data, False if you already have light curve
        and feature files saved in the specified folderlabel folder
        * tls = False - whether or not you want to produce the tls features 
            which currently do not run in spyder (still ugh)
        * momentumdumpcsv = path to the table_of_momentum_dumps.csv for the TESS mission
        * customlist
     
    init: 
        if downloading: 
            * runs build_simbad_extragalactic_database()
            * opens database and gets ra/dec get_radecfromtext()
            * produces and saves LC with eleanor_lc()
            * open eleanor light curves with open_eleanor_lc_files()
            * produces features using create_save_featvec_different_timeaxes()
            if tls = True:
                * produces tls features and saves them.
        else:
            * opens LC with open_eleanor_lc_files()
            * opens features with open_eleanor_features()[0]
                * 0 used to avoid some weird indexing issues i haven't been able to resolve
            
    Other Functions: 
        * features_plotting - plots 2D feature plots, optionally colored by clustering
            * plot_classification - plots first n items in each class
        * plot_lof - calculates and plots the top n LOF scored light curves. 
        * plot_histogram - plots the histogram of given data, optionally plots insets
        * dbscan_param_search - runs and plots dbscan information for the parameter search grid. 
            * column_plot_classification - used by dbscan_param_search, plots 1st 5 LC in
                each class in big 50 curve plots. 
        * features_insets with helpers inset_plotting and get_extrema
    
    modified [lcg 09022020 - plotting]
    """

    def __init__(self, path=None, folderlabel="Vmag19", simbadquery="Vmag <=19",
                 download=True, tls=False, 
                 momentumdumpcsv = "/users/conta/urop/Table_of_momentum_dumps.csv",
                 customlist = False):

        """ 
        put documentation here
        
        """
        if path is None:
            print('Please pass a path to save into')
            return

        if len(folderlabel) == 1:
            print("Only one folder label passed.")
            self.simbadquery = simbadquery
            self.folderlabel = folderlabel[0]
            self.path = path + 'ffi_{}/'.format(self.folderlabel)
            self.catalog = self.path + "simbad_catalog.txt"
            self.lightcurvefilepath = self.path + "{}_lightcurves.fits".format(self.folderlabel)
            self.features0path = self.path + "{}_features_v0.fits".format(self.folderlabel)
            self.features1path = self.path + "{}_features_v1.fits".format(self.folderlabel)
            self.momdumpcsv = momentumdumpcsv
            
            if download: 
                try:
                    print(self.path)
                    os.mkdir(self.path)
                    success = 1
                except OSError:
                    print('Directory exists already!')
                    #this should check to see if there is anything in the directory
                    for root, dirs, files in os.walk(self.path):
                        if len(files) > 0: 
                            print("there are files here")
                            success = 0
                            if os.path.isfile(self.path +"/simbad_catalog.txt") and customlist and len(files) == 1:
                                print("there is one file, the custom input list. accessing.")
                                success = 1
                        else: 
                            print("this folder is empty")
                            success = 1
        
                if success == 1:
                    if not customlist: #if not using a custom input list, make the inputlist
                        print("Producing RA and DEC list")
                        self.build_simbad_extragalactic_database()
                        print("Accessing RA and DEC list") 
                    #go grab everythign from the list
                    self.ralist, self.declist = self.get_radecfromtext()
                    print("Getting and saving eleanor light curves into a fits file")
                    self.radecall = np.column_stack((self.ralist, self.declist))
                    self.gaia_ids = self.eleanor_lc()
                    print("Producing v0 feature vectors")
                    self.gaia_ids, self.times, self.intensities, self.corrected_intensities = self.open_eleanor_lc_files()
                    self.sigmaclip()
                    self.version = 0
                    self.savetrue = True
                    self.features = self.create_save_featvec_different_timeaxes()
                    if tls:
                        print("Producing v1 feature vectors")
                        self.version = 1
                        self.features1 = self.create_save_featvec_different_timeaxes()
                        self.features = np.column_stack((self.features, self.features0))
            else: 
                print("Not downloading anything. Attempting to access LC and Features")
                self.gaia_ids, self.times, self.intensities, self.corrected_intensities = self.open_eleanor_lc_files()
                self.features = self.open_eleanor_features()[0]
                print(self.features[0])
                if len(self.features[0]) == 16 or len(self.features[0]) == 20:
                    self.version = 0
                elif len(self.features[0]) == 4:
                    self.version = 1
                else: 
                    ("something has gone terribly wrong while loading in the features")
        else: 
            ("Loading multiple folders in")
            self.simbadquery = simbadquery
            self.momdumpcsv = momentumdumpcsv
            
            self.labels_all = folderlabel
            
            self.folderlabel = self.labels_all[0]
            self.path = path + 'ffi_{}/'.format(self.folderlabel)
            print(self.path)
            self.catalog = self.path + "simbad_catalog.txt"
            self.lightcurvefilepath = self.path + "{}_lightcurves.fits".format(self.folderlabel)
            self.features0path = self.path + "{}_features_v0.fits".format(self.folderlabel)
            self.features1path = self.path + "{}_features_v1.fits".format(self.folderlabel)
            
            self.gaia_ids, self.times, self.intensities, self.corrected_intensities = self.open_eleanor_lc_files()
            self.features = self.open_eleanor_features()[0]
            print(self.features[0])
            if len(self.features[0]) == 16 or len(self.features[0]) == 20:
                self.version = 0
            elif len(self.features[0]) == 4:
                self.version = 1
            else: 
                ("something has gone terribly wrong while loading in the features")
                
            print("loaded in first folder, ", len(self.intensities), " light curves")
            
            for i in range(len(folderlabel) - 1):
                n = i + 1
                self.folderlabel = self.labels_all[n]
                self.path = path + 'ffi_{}/'.format(self.folderlabel)
                self.lightcurvefilepath = self.path + "{}_lightcurves.fits".format(self.folderlabel)
                self.features0path = self.path + "{}_features_v0.fits".format(self.folderlabel)
                self.features1path = self.path + "{}_features_v1.fits".format(self.folderlabel)
                
                gaias, ts, ints, Icorr = self.open_eleanor_lc_files()
                feats = self.open_eleanor_features()[0]
                
                self.gaia_ids = np.concatenate((self.gaia_ids, gaias))
                self.times = np.concatenate((self.times, ts))
                self.intensities = np.concatenate((self.intensities, ints))
                self.corrected_intensities = np.concatenate((self.corrected_intensities, Icorr))
                self.features = np.concatenate((self.features, feats))
                
                print("loaded in next folder,", len(self.intensities), " light curves")
                
            #now make a concatenation folder? 
            newfolderlabel = 'ffi_output_' + "".join(self.labels_all) + "/"
            self.path = path + newfolderlabel
            try:
                print(self.path)
                os.mkdir(self.path)
                success = 1
            except OSError:
                print('Directory exists already!')
                
                
    def simbad_database_lc_gen(self, simbadquery="Vmag > 18"):
        
        self.simbadquery = simbadquery
        print("Producing RA and DEC list")
        self.build_simbad_extragalactic_database()
        print("Accessing RA and DEC list") #go grab everythign from the list
        self.ralist, self.declist = self.get_radecfromtext()
        print("Getting and saving eleanor light curves into a fits file")
        self.radecall = np.column_stack((self.ralist, self.declist))
        self.gaia_ids = self.eleanor_lc()
        print("Producing v0 feature vectors")
        self.gaia_ids, self.times, self.intensities, self.corrected_intensities = self.open_eleanor_lc_files()
        self.sigmaclip()
        
                
    def build_tess_database(self):
        #do not use
        from astroquery.mast import Catalogs

        catalog_data = Catalogs.query_criteria(catalog="Tic", Tmag=[0,18], d=[20000,100000000])
        print(len(catalog_data))
        
        import numpy as np
        for i in range(len(catalog_data)):
                    # >> decode bytes object to convert to string
                    obj = str(catalog_data["GAIA"][i])
                    ra = str(catalog_data["ra"][i])
                    dec = str(catalog_data["dec"][i])
                   
                    with open("/users/conta/urop/d20kpc_tmag_18_targets.txt", 'a') as f:
                            f.write(obj + ',' + ra + ',' + dec + "," + '\n')
        return

         
    
    def build_simbad_extragalactic_database(self):
        '''Object type follows format in:
        http://vizier.u-strasbg.fr/cgi-bin/OType?$1'''
        
        # -- querying object type -------------------------------------------------
        customSimbad = Simbad()
        customSimbad.TIMEOUT = 1000
        # customSimbad.get_votable_fields()
        customSimbad.add_votable_fields('otype')
        customSimbad.add_votable_fields('distance')
        customSimbad.add_votable_fields('dim_angle')
        customSimbad.add_votable_fields('ra(:;A;ICRS;J2000)', 'dec(:;D;ICRS;2000)')
        table = customSimbad.query_criteria(self.simbadquery)
        objects = list(table['MAIN_ID'])
        ras = list(table['RA___A_ICRS_J2000'])
        decs = list(table['DEC___D_ICRS_2000'])
    
        # >> now loop through all of the objects
        for i in range(len(objects)):
            # >> decode bytes object to convert to string
            obj = objects[i].decode('utf-8')
            ra = ras[i]
            dec = decs[i]
           
            with open(self.catalog, 'a') as f:
                    f.write(obj + ',' + ra + ',' + dec + "," + '\n')
        return
    
    def get_radecfromtext(self):
        ''' pulls ra and dec from text file containing all targets
        '''
        ra_all = []
        dec_all = []
        
        with open(self.catalog, 'r') as f:
            lines = f.readlines()
            for line in lines:
                name, ra, dec, empty = line.split(',')
                    
                ra_all.append(ra)
                dec_all.append(dec)
                    
        return np.asarray(ra_all), np.asarray(dec_all)
    
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
                files = eleanor.multi_sectors(coords=coords, tic=0, sectors='all') #by not providing a sector argument, will ONLY retrieve most recent sector
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
                        plt.savefig("/users/conta/urop/ffi_Vmag7.5/tpf-aperture-" + str(file.gaia) + ".png")
                   
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
            except (SearchError, ValueError):
                print("Search Error or ValueError occurred")
            
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
    
    def open_eleanor_features(self):
        
        """ 
        opens all eleanor features 
        returns a single array of ALL features and list of gaia_ids
        """
        features_filelabel = self.folderlabel + "_features"
        print(features_filelabel)
        filepaths = []
        for root, dirs, files in os.walk(self.path):
            for name in files:
                #print(name)
                if name.startswith((features_filelabel)):
                        filepaths.append(root + "/" + name)
        
        print("found ", len(filepaths), 'feature files')
        f = fits.open(filepaths[0], memmap=False)
        features = np.asarray(f[0].data)
        gaia_ids = np.asarray(f[1].data)
        f.close()
        for n in range(len(filepaths) - 1):
            f = fits.open(filepaths[n+1], memmap=False)
            features_new = np.asarray(f[0].data)
            features = np.column_stack((features, features_new))
            f.close()
        
        return np.asarray(features), np.asarray(gaia_ids)
    
    def clip_feature_outliers(self, sigma=20, plot=True):
        """ 
        plots and then removes any outlier or nan features to avoid messiness in the 
        feature space. 
        parameters: 
            * path to save shit into
            * features (all)
            * time axis (needs a time axis for every flux)
            * flux (all)
            * gaia ids (all)
            * sigma to crop to
            * version of the features - v0 includes both v0 and v1 right now
            * plot=True by default, can skip if you only want to trim and not examine any outliers
            
        returns: 
            features_cropped, gaia_ids_cropped, flux_cropped, time_cropped, outlier_indexes
        modified [lcg 08252020 - adapted for multiple time axes]"""
        
        #set up the directory
        if plot == True: 
            path = self.path + "clipped-feature-outliers/"
            try:
                os.makedirs(path)
            except OSError:
                print ("%s already exists" % path)
            else:
                print ("Successfully created the directory %s" % path)
        
        #labels for the plots
        if self.version==0:
            features_greek = [r'$\alpha$', 'B', r'$\Gamma$', r'$\Delta$', r'$\beta$', r'$\gamma$',r'$\delta$',
                      "E", r'$\epsilon$', "Z", "H", r'$\eta$', r'$\Theta$', "I", "K", r'$\Lambda$', "M", r'$\mu$'
                      ,"N", r'$\nu$']
        elif self.version==1: 
            features_greek = ["M", r'$\mu$',"N", r'$\nu$']
    
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
                    
        #print(np.asarray(outlier_indexes))
            
        outlier_indexes = np.asarray(outlier_indexes)
        target_indexes = outlier_indexes[:,0] #is the index of the target on the lists
        print(target_indexes)
        feature_indexes = outlier_indexes[:,1] #is the index of the feature that it triggered on
        if plot:
            for i in range(len(outlier_indexes)):
                target_index = target_indexes[i]
                feature_index = feature_indexes[i]
                plt.figure(figsize=(8,3))
                plt.scatter(self.times[target_index], self.intensities[target_index], s=0.5)
                target = self.gaia_ids[target_index]
                
                if np.isnan(self.features[target_index][feature_index]) == True:
                    feature_title = features_greek[feature_index] + "=nan"
                else: 
                    feature_value = '%s' % float('%.2g' % self.features[target_index][feature_index])
                    feature_title = features_greek[feature_index] + "=" + feature_value
                print(feature_title)
                
                plt.title("GAIA_ID " + str(int(target)) + " " + feature_title, fontsize=8)
                plt.tight_layout()
                plt.savefig((path + "featureoutlier-TICID" + str(int(target)) + ".png"))
                plt.show()
        else: 
            print("Not plotting outliers!")
                
            
        features_cropped = np.delete(self.features, target_indexes, axis=0)
        gaia_ids_cropped = np.delete(self.gaia_ids, target_indexes)
        flux_cropped = np.delete(self.intensities, target_indexes, axis=0)
        time_cropped = np.delete(self.times, target_indexes, axis=0)
        
        self.features = features_cropped
        self.gaia_ids = gaia_ids_cropped
        self.times = time_cropped
        self.intensities = flux_cropped
        
        self.outlierfeatures = target_indexes
            
        return features_cropped, gaia_ids_cropped, flux_cropped, time_cropped, target_indexes
    
    def plot_lof(self, n=20,n_neighbors=20, n_tot=100):
        """ Plots the 20 most and least interesting light curves based on LOF.
        Parameters:
            * time : ALL time arrays 
            * intensity
            * targets : list of gaia ids
            * n : number of curves to plot in each figure
            * n_neighbors : parameter to run LOF
            * n_tot : total number of light curves to plots (# of figures = n_tot / n)
            * feature vector : assumes x axis is latent dimensions, not time 
        Outputs:
            * Text file with gaia id in column 1, and LOF in column 2 (lof-*.txt)
            * Log histogram of LOF (lof-histogram.png)
            * light curves with highest and lowest LOF
        modified [lcg 08312020 - fixed histogram plotting]
        """
        # -- calculate LOF -------------------------------------------------------
        
        
        self.lofpath = self.path + "LOF/"
        
        try:
            os.makedirs(self.lofpath)
        except OSError:
            print ("Creation of the directory %s failed" % self.lofpath)
            print("New folder created will have -new at the end. Please rename.")
            folder_path = self.lofpath + "-new"
            os.makedirs(self.lofpath)
        else:
            print ("Successfully created the directory %s" % self.lofpath)
            
        print('Calculating LOF')
        clf = LocalOutlierFactor(n_neighbors=n_neighbors)
        fit_predictor = clf.fit_predict(self.features)
        negative_factor = clf.negative_outlier_factor_
        
        lof = -1 * negative_factor
        ranked = np.argsort(lof)
        largest_indices = ranked[::-1][:n_tot] # >> outliers
        smallest_indices = ranked[:n_tot] # >> inliers
        
        # >> save LOF values in txt file
        print('Saving LOF values')
        with open(self.lofpath+'lof.txt', 'w') as f:
            for i in range(len(self.gaia_ids)):
                f.write('{} {}\n'.format(int(self.gaia_ids[i]), lof[i]))
          
        # >> make histogram of LOF values
        print('Make LOF histogram')
        self.plot_histogram(lof, 20, "Local Outlier Factor (LOF)", insets=False, log=False)
        self.plot_histogram(lof, 20, "Local Outlier Factor (LOF)", insets=True, log=False)
        
        # -- plot smallest and largest LOF light curves --------------------------
        print('Plot highest LOF and lowest LOF light curves')
        num_figs = int(n_tot/n) # >> number of figures to generate
        
        for j in range(num_figs):
            
            for i in range(2): # >> loop through smallest and largest LOF plots
                fig, ax = plt.subplots(n, 1, sharex=False, figsize = (8, 3*n))
                
                for k in range(n): # >> loop through each row
                    if i == 0: ind = largest_indices[j*n + k]
                    elif i == 1: ind = smallest_indices[j*n + k]\
                    
                    # >> plot momentum dumps
                    with open(self.momdumpcsv, 'r') as f:
                        lines = f.readlines()
                        mom_dumps = [ float(line.split()[3][:-1]) for line in lines[6:] ]
                        inds = np.nonzero((mom_dumps >= np.min(self.times[ind])) * \
                                          (mom_dumps <= np.max(self.times[ind])))
                        mom_dumps = np.array(mom_dumps)[inds]
                    for t in mom_dumps:
                        ax[k].axvline(t, color='g', linestyle='--')
                        
                    # >> plot light curve
                    ax[k].plot(self.times[ind], self.intensities[ind], '.k')
                    ax[k].text(0.98, 0.02, '%.3g'%lof[ind],
                               transform=ax[k].transAxes,
                               horizontalalignment='right',
                               verticalalignment='bottom',
                               fontsize='xx-small')
                    pf.format_axes(ax[k], ylabel=True)
                    ax[k].set_title("GAIA ID " + str(self.gaia_ids[ind]))
        
                # >> label axes
                ax[n-1].set_xlabel('time [BJD - 2457000]')
                    
                # >> save figures
                if i == 0:
                    
                    fig.suptitle(str(n) + ' largest LOF targets', fontsize=16,
                                 y=0.95)
                    fig.savefig(self.lofpath + 'lof-kneigh' + \
                                str(n_neighbors) + '-largest_' + str(j*n) + 'to' +\
                                str(j*n + n) + '.png',
                                bbox_inches='tight')
                    plt.close(fig)
                elif i == 1:
                    fig.suptitle(str(n) + ' smallest LOF targets', fontsize=16,
                                 y=0.95)
                    fig.savefig(self.lofpath + 'lof-kneigh' + \
                                str(n_neighbors) + '-smallest' + str(j*n) + 'to' +\
                                str(j*n + n) + '.png',
                                bbox_inches='tight')
                    plt.close(fig)
            
            
    def plot_histogram(self, data, bins, x_label, insets=True, log=True):
        """ plot a histogram with one light curve from each bin plotted on top
        data is the histogram data
        bins is bins
        x-label is what you want the xaxis to be labelled as
        insetx is the SAME x-axis to plot
        insety is the full list of light curves
        filename is the exact place you want it saved
        insets is a true/false of if you want them
        modified [lcg 08262020 - FFI version]
        """
        filename = self.path + "histogram.png"
        #this is the very very simple histogram plotting
        fig, ax1 = plt.subplots()
        n_in, bins, patches = ax1.hist(data, bins, log=log)
        
        y_range = np.abs(n_in.max() - n_in.min())
        x_range = np.abs(data.max() - data.min())
        ax1.set_ylabel('Number of light curves')
        ax1.set_xlabel(x_label)
        
        if insets == True:
            filename = self.path + "histogram-insets.png"
            for n in range(len(n_in)): #how many bins?
                if n_in[n] == 0: #if the bin has nothing in it, keep moving
                    continue
                else: 
                    #set up axis and dimension for it
                    axis_name = "axins" + str(n)
                    inset_width = 0.33 * x_range * 0.5
                    inset_x = bins[n] - (0.5*inset_width)
                    inset_y = n_in[n]
                    inset_height = 0.125 * y_range * 0.5
                        #x pos, y pos, width, height
                    axis_name = ax1.inset_axes([inset_x, inset_y, 
                                                inset_width, inset_height], 
                                               transform = ax1.transData) 
                    
                    #identify a light curve from that one
                    for m in range(len(data)):
                        #print(bins[n], bins[n+1])
                        if bins[n] <= data[m] <= bins[n+1]:
                            #print(data[m], m)
                            timeaxis = self.times[m]
                            lc_to_plot = self.intensities[m]
                            lc_ticid = self.gaia_ids[m]
                            break
                        else: 
                            continue
                    
                    axis_name.scatter(timeaxis, lc_to_plot, c='black', s = 0.1, rasterized=True)
                    axis_name.set_title("TIC " + str(int(lc_ticid)), fontsize=6)
        plt.savefig(filename)
        plt.close()

    def features_plotting(self, clustering = 'dbscan', eps=3, min_samples=10,
                             metric='minkowski', algorithm='auto', leaf_size=30,
                             p=2, kmeans_clusters=4):
        """plotting (n 2) features against each other
        parameters: 
            * feature_vectors - array of feature vectors
            * path to where you want everythigns aved - ends in a backslash
            * clustering - what you want to cluster as. options are 'dbscan', 'kmeans', or 
            any other keyword which will do no clustering
            * time axis
            * intensities
            *target ticids
            * folder suffix
            *feature_engineering - default is true
            * version - what version of engineered features, irrelevant integer if feature_engienering is false
            * eps, min_samples, metric, algorithm, leaf_size, p - dbscan parameters, comes with defaults
            *momentum dumps - not sure entirely why it's needed here tbh
            
        returns: only returns labels for dbscan/kmeans clustering. otherwise the only
        output is the files saved into the folder as given thru path
        
        modified [lcg 08252020 - adapted to FFI]
        ** TO DO: make file and graph labels a property of self when you set the version
        """
        #detrmine which of the clustering algoirthms you're using: 
        rcParams['figure.figsize'] = 10,10
        folder_label = "blank"
        if clustering == 'dbscan':
            # !! TODO parameter optimization (eps, min_samples)
            db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
                        algorithm=algorithm, leaf_size=leaf_size,
                        p=p).fit(self.features) #eps is NOT epochs
            classes_dbscan = db.labels_
            numclasses = str(len(set(classes_dbscan)))
            folder_label = "dbscan-colored"

        elif clustering == 'kmeans': 
            Kmean = KMeans(n_clusters=kmeans_clusters, max_iter=700, n_init = 20)
            x = Kmean.fit(self.features)
            classes_kmeans = x.labels_
            folder_label = "kmeans-colored"
        else: 
            print("no clustering chosen")
            folder_label = "2DFeatures"
            
        #makes folder and saves to it    
        folder_path = self.path + folder_label
        try:
            os.makedirs(folder_path)
        except OSError:
            print ("Creation of the directory %s failed" % folder_path)
            print("New folder created will have -new at the end. Please rename.")
            folder_path = folder_path + "-new"
            os.makedirs(folder_path)
        else:
            print ("Successfully created the directory %s" % folder_path) 
     
        if clustering == 'dbscan':
            with open(folder_path + '/dbscan_paramset.txt', 'a') as f:
                f.write('eps {} min samples {} metric {} algorithm {} \
                        leaf_size {} p {} # classes {} \n'.format(eps,min_samples,
                        metric,algorithm, leaf_size, p,numclasses))
            self.plot_classification(labels = classes_dbscan, path = folder_path,n=5)
            pf.plot_pca(self.features, classes_dbscan, output_dir=folder_path+'/')
        elif clustering == 'kmeans':
            print("uhhh nothing right now!! fix me later!")
            self.plot_classification(labels = classes_kmeans, path = folder_path,n=5)
            pf.plot_pca(self.features, classes_kmeans, output_dir=folder_path+'/')
            
        colors = pf.get_colors()
        
        #creates labels
        if self.version==0:
            graph_labels = ["Average", "Variance", "Skewness", "Kurtosis", "Log Variance",
                            "Log Skewness", "Log Kurtosis", "Maximum Power", "Log Maximum Power", 
                                "Period of Maximum Power (0.1 to 10 days)","Slope" , "Log Slope",
                                "P0", "P1", "P2", "Period of Maximum Power (0.001 to 0.1 days)", 
                            "TLS Best fit Period (days)", "TLS Best fit duration (days)", "TLS best fit depth (ppt from transit bottom",
                                "TLS Best fit Power"]
            fname_labels = ["Avg", "Var", "Skew", "Kurt", "LogVar", "LogSkew", "LogKurt",
                                "MaxPower", "LogMaxPower", "Period0_1to10", "Slope", "LogSlope",
                                "P0", "P1", "P2", "Period0to0_1", "TLSPeriod", "TLSDuration", "TLSDepth", "TLSPower"]
        elif self.version == 1: 
            graph_labels = ["TLS Best fit Period (days)", "TLS Best fit duration (days)", "TLS best fit depth (ppt from transit bottom",
                            "TLS Best fit Power"]
            fname_labels = ["TLSPeriod", "TLSDuration", "TLSDepth", "TLSPower"]

        num_features = len(self.features[0])
   
        for n in range(num_features):
            feat1 = self.features[:,n]
            graph_label1 = graph_labels[n]
            fname_label1 = fname_labels[n]
            for m in range(num_features):
                if m == n:
                    continue
                graph_label2 = graph_labels[m]
                fname_label2 = fname_labels[m]                
                feat2 = self.features[:,m]
     
                if clustering == 'dbscan':
                    plt.figure() # >> [etc 060520]
                    plt.clf()
                    for n in range(len(self.features)):
                        plt.scatter(feat1[n], feat2[n], c=colors[classes_dbscan[n]], s=2)
                    plt.xlabel(graph_label1)
                    plt.ylabel(graph_label2)
                    plt.savefig((folder_path+'/' + fname_label1 + "-vs-" + fname_label2 + "-dbscan.png"))
                    plt.show()
                    plt.close()
                     
                elif clustering == 'kmeans':
                    plt.figure() # >> [etc 060520]
                    plt.clf()
                    for n in range(len(self.features)):
                        plt.scatter(feat1[n], feat2[n], c=colors[classes_kmeans[n]], s=2)
                    plt.xlabel(graph_label1)
                    plt.ylabel(graph_label2)
                    plt.savefig(folder_path+'/' + fname_label1 + "-vs-" + fname_label2 + "-kmeans.png")
                    plt.show()
                    plt.close()
                elif clustering == 'none':
                    plt.scatter(feat1, feat2, s = 2, color = 'black')
                    plt.xlabel(graph_label1)
                    plt.ylabel(graph_label2)
                    plt.savefig(folder_path+'/' + fname_label1 + "-vs-" + fname_label2 + ".png")
                    plt.show()
                    plt.close()
                    
        if clustering == 'dbscan':
            np.savetxt(folder_path+"/dbscan-classes.txt", classes_dbscan)
            return classes_dbscan
        if clustering == 'kmeans':
            return classes_kmeans

    def plot_classification(self, labels, path, n=20):
        """ 
        FFI veersion of pf.plot_classification
        plots the first ten items in a class
        """
        
        classes, counts = np.unique(labels, return_counts=True)
        colors=['red', 'blue', 'green', 'purple', 'yellow', 'cyan', 'magenta',
                'skyblue', 'sienna', 'palegreen']*10
        
            
        for i in range(len(classes)): # >> loop through each class
            fig, ax = plt.subplots(n, 1, sharex=False, figsize = (8, 3*n))
            class_inds = np.nonzero(labels == classes[i])[0]
            if classes[i] == -1:
                color = 'black'
            elif classes[i] < len(colors) - 1:
                color = colors[i]
            else:
                color='black'
            
            for k in range(min(n, counts[i])): # >> loop through each row
                ind = class_inds[k]
               
                with open(self.momdumpcsv, 'r') as f:
                    lines = f.readlines()
                    mom_dumps = [ float(line.split()[3][:-1]) for line in lines[6:] ]
                    inds = np.nonzero((mom_dumps >= np.min(self.times[ind])) * \
                                      (mom_dumps <= np.max(self.times[ind])))
                    mom_dumps = np.array(mom_dumps)[inds]
                # >> plot momentum dumps
                for t in mom_dumps:
                    ax[k].plot([t,t], [0, 1], '--g', alpha=0.5,
                               transform=ax[k].transAxes)            
                
                # >> plot light curve
                ax[k].plot(self.times[ind], self.intensities[ind], '.k')
                ax[k].text(0.98, 0.02, str(labels[ind]), transform=ax[k].transAxes,
                           horizontalalignment='right', verticalalignment='bottom',
                           fontsize='xx-small')
                pf.format_axes(ax[k], ylabel=True)
                ax[k].set_title("GAIA ID " + str(self.gaia_ids[ind]))
                ax[k].set_xlabel('time [BJD - 2457000]')
        
            if classes[i] == -1:
                fig.suptitle('Class -1 (outliers)', fontsize=16, y=0.9,
                             color=color)
            else:
                fig.suptitle('Class ' + str(classes[i]), fontsize=16, y=0.9,
                             color=color)
            fig.savefig(path +'/class' + str(classes[i]) + '.png',
                        bbox_inches='tight')
            plt.close(fig)
        return classes, counts
    
    def column_plot_classification(self, output_dir, labels, prefix='prefix', title='title'):
        '''
        plots first five light curves in each class in vertical columns
        '''
        ncols = 10
        nrows = 5
        classes, counts = np.unique(labels, return_counts=True)
        colors = pf.get_colors()
        
        num_figs = int(np.ceil(len(classes) / ncols))
        features_greek = [r'$\alpha$', 'B', r'$\Gamma$', r'$\Delta$', r'$\beta$', r'$\gamma$',r'$\delta$',
                      "E", r'$\epsilon$', "Z", "H", r'$\eta$', r'$\Theta$', "I", "K", r'$\Lambda$', 
                      "M", r'$\mu$',"N", r'$\nu$']
        
        for i in range(num_figs): #
            fig, ax = plt.subplots(nrows, ncols, sharex=False,
                                   figsize=(8*ncols*0.75, 3*nrows))
            fig.suptitle(title)
            
            if i == num_figs - 1 and len(classes) % ncols != 0:
                num_classes = len(classes) % ncols
            else:
                num_classes = ncols
            for j in range(num_classes): # >> loop through columns
                class_num = classes[ncols*i + j]
                
                # >> find all light curves with this  class
                class_inds = np.nonzero(labels == class_num)[0]
                
                if class_num == -1:
                    color = 'black'
                elif class_num < len(colors) - 1:
                    color = colors[class_num]
                else:
                    color='black'
                    
                k=-1
                # >> first plot any Simbad classified light curves
                for k in range(min(nrows, len(class_inds))): 
                    ind = class_inds[k] # >> to index targets
                    ax[k, j].plot(self.times[ind], self.intensities[ind], '.k')
                    ax[k,j].set_title("GAIA ID " + str(self.gaia_ids[ind]), color='black')
                    pf.format_axes(ax[k, j], ylabel=True) 
                    
                features_byclass = self.features[class_inds]
                med_features = np.median(features_byclass, axis=0)
                med_string = str(med_features)
                ax[0, j].set_title('Class '+str(class_num)+ "# Curves:" + str(counts[j]) +
                                   '\n Median Features:' + med_string + 
                                   "\n"+ax[0,j].get_title(),
                                   color=color, fontsize='xx-small')
                ax[-1, j].set_xlabel('Time [BJD - 2457000]')   
                            
                if j == 0:
                    for m in range(nrows):
                        ax[m, 0].set_ylabel('Relative flux')
                        
            fig.tight_layout()
            fig.savefig(output_dir + prefix + '-' + str(i) + '.pdf')
            plt.close(fig)
            
    def dbscan_param_search(self, eps=list(np.arange(0.5,10,0.4)),
                                min_samples=[2,5,10],
                                metric=['euclidean', 'minkowski'],
                                algorithm = ['auto'],
                                leaf_size = [30, 40, 50],
                                p = [1,2,3,4], plotting=False,
                                database_dir='./databases/', pca=True, tsne=True,
                                confusion_matrix=False):
        '''Performs a grid serach across parameter space for DBSCAN. 
        
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
        param_num = 0
        
        
        
        self.dbpath = self.path + "/dbscan-paramscan/"
        
        try:
            os.makedirs(self.dbpath)
        except OSError:
            print ("Creation of the directory %s failed" % self.dbpath)
            print("New folder created will have -new at the end. Please rename.")
            self.dbpath = self.path + "/dbscan-paramscan-new/"
            os.makedirs(self.dbpath)
        else:
            print ("Successfully created the directory %s" % self.dbpath)
        
    
        with open(self.dbpath + 'dbscan_param_search.txt', 'a') as f:
            f.write('{} {} {} {} {} {} {} {} {} {} {}\n'.format("eps", "samp", "metric", 
                                                             "alg", "leaf", "p",
                                                             "#classes", "# noise",
                                                             "silhouette", 'ch', 
                                                             'db', 'acc'))
    
        for i in range(len(eps)):
            for j in range(len(min_samples)):
                for k in range(len(metric)):
                    for l in range(len(algorithm)):
                        for m in range(len(leaf_size)):
                            #if metric[k] == 'minkowski' or 'manhattan':
                             #   p = p
                            #else:
                             #   p = [None]
                            for n in range(len(p)):
                                db = DBSCAN(eps=eps[i],
                                            min_samples=min_samples[j],
                                            metric=metric[k],
                                            algorithm=algorithm[l],
                                            leaf_size=leaf_size[m],
                                            p=p[n]).fit(self.features)
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
                                    acc = pf.plot_confusion_matrix(self.gaia_ids, db.labels_,
                                                                   database_dir=database_dir,
                                                                   output_dir=self.dbpath,
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
                                    silhouette = silhouette_score(self.features, db.labels_)
                                    silhouette_scores.append(silhouette)
                                    
                                    # >> compute calinski harabasz score
                                    ch_score = calinski_harabasz_score(self.features, db.labels_)
                                    ch_scores.append(ch_score)
                                    
                                    # >> compute davies-bouldin score
                                    dav_boul_score = davies_bouldin_score(self.features, db.labels_)
                                    db_scores.append(dav_boul_score)
                                    
                                else:
                                    silhouette, ch_score, dav_boul_score = np.nan, np.nan, np.nan
                                    
                                with open(self.dbpath + 'dbscan_param_search.txt', 'a') as f:
                                    f.write('{} {} {} {} {} {} {} {} {} {} {} {}\n'.format(eps[i],
                                                                       min_samples[j],
                                                                       metric[k],
                                                                       algorithm[l],
                                                                       leaf_size[m],
                                                                       p[n],
                                                                       len(classes_1),
                                                                       counts_1[0],
                                                                       silhouette,
                                                                       ch_score,
                                                                       dav_boul_score,
                                                                       acc))
                                    
                                if plotting and len(classes_1) > 1:
    
                                    self.column_plot_classification(self.dbpath, db.labels_, prefix = prefix,title=title)
                                    
                                    if pca:
                                        print('Plot PCA...')
                                        pf.plot_pca(self.features, db.labels_,
                                                    output_dir=self.dbpath,
                                                    prefix=prefix)
                                    
                                    if tsne:
                                        print('Plot t-SNE...')
                                        pf.plot_tsne(self.features, db.labels_,
                                                     output_dir=self.dbpath,
                                                     prefix=prefix)
                                plt.close('all')
                                param_num +=1
        print("Plot paramscan metrics...")
        pf.plot_paramscan_metrics(self.dbpath, parameter_sets, 
                                  silhouette_scores, db_scores, ch_scores)
    
        pf.plot_paramscan_classes(self.dbpath, parameter_sets, 
                                      np.asarray(num_classes), np.asarray(num_noisy))
    
        return parameter_sets, num_classes, silhouette_scores, db_scores, ch_scores, accuracy        
     
    def hdbscan_param_search(self,min_cluster_size=list(np.arange(5,30,2)),
                             min_samples = [5,10,15],
                             metric=['euclidean', 'manhattan', 'minkowski'],
                             p0 = [1,2,3,4], DEBUG=True,
                             simbad_database_txt='./simbad_database.txt',
                             database_dir='./databases/',
                             pca=True, tsne=True, confusion_matrix=False,
                             single_file=False):
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
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        from sklearn.metrics import davies_bouldin_score
        target_info = np.zeros((len(self.times), 5))  
        output_dir = self.path + "/hdbscan-paramscan/"
        try:
            os.makedirs(output_dir)
        except OSError:
            print ("Creation of the directory %s failed" % output_dir)
            print("New folder created will have -new at the end. Please rename.")
            output_dir = self.path + "/hdbscan-paramscan-new/"
            os.makedirs(output_dir)
        else:
            print ("Successfully created the directory %s" % output_dir)
            
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
                        clusterer.fit(self.features)
                        labels = clusterer.labels_
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
                            silhouette = silhouette_score(self.features, labels)
                            silhouette_scores.append(silhouette)
                            
                            # >> compute calinski harabasz score
                            print('Computing calinski harabasz score')
                            ch_score = calinski_harabasz_score(self.features, labels)
                            ch_scores.append(ch_score)
                            
                            # >> compute davies-bouldin score
                            print('Computing davies-bouldin score')
                            dav_boul_score = davies_bouldin_score(self.features, labels)
                            db_scores.append(dav_boul_score)                        
                                        
                            if confusion_matrix:
                                print('Computing accuracy')
                                acc = pf.plot_confusion_matrix(self.gaia_ids, labels,
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
                                        
                        if DEBUG and len(classes_1) > 1:
                            self.column_plot_classification(output_dir, labels, prefix = prefix,title=title)
                        
                            if pca:
                                print('Plot PCA...')
                                pf.plot_pca(self.features, labels,
                                            output_dir=output_dir,
                                            prefix=prefix)
                                        
                            if tsne:
                                print('Plot t-SNE...')
                                pf.plot_tsne(self.features,labels,
                                             output_dir=output_dir,
                                             prefix=prefix)                
                        plt.close('all')
                        param_num +=1
    
            
        return parameter_sets, num_classes, acc   
               
    def features_insets(self):
        """ Plots 2 features against each other with the extrema points' associated
        light curves plotted as insets along the top and bottom of the plot. 
        
        time is the time axis for the group
        intensity is the full list of intensities
        feature_vectors is the complete list of feature vectors
        targets is the complete list of targets
        folder is the folder into which you wish to save the folder of plots. it 
        should be formatted as a string, ending with a /
        modified [lcg 08262020 - adapted to FFI]
        """   
        folderpath = self.path + "2DFeatures-insets"
        
        try:
            os.makedirs(folderpath)
        except OSError:
            print ("Creation of the directory %s failed" % folderpath)
            print("New folder created will have -new at the end. Please rename.")
            folderpath = folderpath + "-new"
            os.makedirs(folderpath)
        else:
            print ("Successfully created the directory %s" % folderpath) 
            
        folderpath = folderpath + "/" 
        
        if self.version==0:
            graph_labels = ["Average", "Variance", "Skewness", "Kurtosis", "Log Variance",
                                "Log Skewness", "Log Kurtosis", "Maximum Power", "Log Maximum Power", 
                                "Period of Maximum Power (0.1 to 10 days)","Slope" , "Log Slope",
                                "P0", "P1", "P2", "Period of Maximum Power (0.001 to 0.1 days)", 
                                "TLS Best fit Period (days)", "TLS Best fit duration (days)", "TLS best fit depth (ppt from transit bottom",
                                "TLS Best fit Power"]
            fname_labels = ["Avg", "Var", "Skew", "Kurt", "LogVar", "LogSkew", "LogKurt",
                                "MaxPower", "LogMaxPower", "Period0_1to10", "Slope", "LogSlope",
                                "P0", "P1", "P2", "Period0to0_1", "TLSPeriod", "TLSDuration", "TLSDepth", "TLSPower"]
        elif self.version == 1: 
                
            graph_labels = ["TLS Best fit Period (days)", "TLS Best fit duration (days)", "TLS best fit depth (ppt from transit bottom",
                                "TLS Best fit Power"]
            fname_labels = ["TLSPeriod", "TLSDuration", "TLSDepth", "TLSPower"]
            
        for n in range(len(self.features[0])):
            graph_label1 = graph_labels[n]
            fname_label1 = fname_labels[n]
            for m in range(len(self.features[0])):
                if m == n:
                    continue
                graph_label2 = graph_labels[m]
                fname_label2 = fname_labels[m]  
    
                filename = folderpath + fname_label1 + "-vs-" + fname_label2 + ".png"     
                
                inset_indexes = self.get_extrema(n, m)
                
                self.inset_plotting(self.features[:,n], self.features[:,m], graph_label1, 
                               graph_label2, inset_indexes, filename)
                
    
    def inset_plotting(self, datax, datay, label1, label2, inset_indexes, filename):
        """ Plots the extrema of a 2D feature plot as insets on the top and bottom border
        datax and datay are the features being plotted as a scatter plot beneath it
        label1 and label2 are the x and y labels
        insetx is the time axis for the insets
        insety is the complete list of intensities 
        inset_indexes are the identified extrema to be plotted
        filename is the exact path that the plot is to be saved to.
        modified [lcg 08262020 - ffi variant]"""
        
        x_range = datax.max() - datax.min()
        y_range = datay.max() - datay.min()
        y_offset = 0.2 * y_range
        x_offset = 0.01 * x_range
        
        fig, ax1 = plt.subplots()
    
        ax1.scatter(datax, datay, s=2)
        ax1.set_xlim(datax.min() - x_offset, datax.max() + x_offset)
        ax1.set_ylim(datay.min() - y_offset,  datay.max() + y_offset)
        ax1.set_xlabel(label1)
        ax1.set_ylabel(label2)
        
        i_height = y_offset / 2
        i_width = x_range/4.5
        
        x_init = datax.min() 
        y_init = datay.max() + (0.4*y_offset)
        n = 0
        inset_indexes = inset_indexes[0:8]
        while n < (len(inset_indexes)):
            axis_name = "axins" + str(n)
            
        
            axis_name = ax1.inset_axes([x_init, y_init, i_width, i_height], transform = ax1.transData) #x pos, y pos, width, height
            axis_name.scatter(self.times[inset_indexes[n]], self.intensities[inset_indexes[n]], c='black', s = 0.1, rasterized=True)
            
            #this sets where the pointer goes to
            x1, x2 = datax[inset_indexes[n]], datax[inset_indexes[n]] + 0.001*x_range
            y1, y2 =  datay[inset_indexes[n]], datay[inset_indexes[n]] + 0.001*y_range
            axis_name.set_xlim(x1, x2)
            axis_name.set_ylim(y1, y2)
            ax1.indicate_inset_zoom(axis_name)
                  
            #this sets the actual axes limits    
            axis_name.set_xlim(self.times[inset_indexes[n]].min(), self.times[inset_indexes[n]].max())
            axis_name.set_ylim(self.intensities[inset_indexes[n]].min(), self.intensities[inset_indexes[n]].max())
            axis_name.set_title("GAIA ID " + str(int(self.gaia_ids[inset_indexes[n]])), fontsize=6)
            axis_name.set_xticklabels([])
            axis_name.set_yticklabels([])
            
            x_init += 1.1* i_width
            n = n + 1
            
            if n == 4: 
                y_init = datay.min() - (0.8*y_offset)
                x_init = datax.min()
                
        plt.savefig(filename)   
        plt.close()
    
    def get_extrema(self, feat1, feat2):
        """ Identifies the extrema in each direction for the pair of features given. 
        Eliminates any duplicate extrema (ie, the xmax that is also the ymax)
        Returns array of unique indexes of the extrema
        modified [lcg 08262020 - ffi version]"""
        indexes = []
        index_feat1 = np.argsort(self.features[:,feat1])
        index_feat2 = np.argsort(self.features[:,feat2])
        
        indexes.append(index_feat1[0]) #xmin
        indexes.append(index_feat2[-1]) #ymax
        indexes.append(index_feat2[-2]) #second ymax
        indexes.append(index_feat1[-2]) #second xmax
        
        indexes.append(index_feat1[1]) #second xmin
        indexes.append(index_feat2[1]) #second ymin
        indexes.append(index_feat2[0]) #ymin
        indexes.append(index_feat1[-1]) #xmax
        
        indexes.append(index_feat1[-3]) #third xmax
        indexes.append(index_feat2[-3]) #third ymax
        indexes.append(index_feat1[2]) #third xmin
        indexes.append(index_feat2[2]) #third ymin
    
        indexes_unique, ind_order = np.unique(np.asarray(indexes), return_index=True)
        #fixes the ordering of stuff
        indexes_unique = [np.asarray(indexes)[index] for index in sorted(ind_order)]
        
        return indexes_unique
    def KNN_plotting(self, k_values):
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
        modified [lcg 08312020 - ffi version]"""
        self.knnpath = self.path + "knn_plots/"
        
        try:
            os.makedirs(self.knnpath)
        except OSError:
            print ("Creation of the directory %s failed" % self.knnpath)
            print("New folder created will have -new at the end. Please rename.")
            self.knnpath = self.path + "knn_plots-new/"
            os.makedirs(self.knnpath)
        else:
            print ("Successfully created the directory %s" % self.knnpath)
            
        from sklearn.neighbors import NearestNeighbors
        for n in range(len(k_values)):
            neigh = NearestNeighbors(n_neighbors=k_values[n])
            neigh.fit(self.features)
        
            k_dist, k_ind = neigh.kneighbors(self.features, return_distance=True)
            
            avg_kdist = np.mean(k_dist, axis=1)
            avg_kdist_sorted = np.sort(avg_kdist)[::-1]
            
            plt.scatter(np.arange(len(self.features)), avg_kdist_sorted)
            plt.xlabel("Points")
            plt.ylabel("Average K-Neighbor Distance")
            plt.ylim((0, 20))
            plt.title("K-Neighbor plot for k=" + str(k_values[n]))
            plt.savefig(self.knnpath + "kneighbors-" +str(k_values[n]) +"-plot-sorted.png")
            plt.close() 
     
    
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
        
    def cae_truncate(self):
        """ truncates arrays into homogenous cube of data
        modified [lcg 08312020 - created]"""
        lengths = []
        for n in range(len(self.times)):
            lengths.append(len(self.times[n]))
        
        
        crop = np.asarray(lengths).min()
        print(crop)
        
        self.cropped_times = self.times[0][:crop]
        self.cropped_intensities = np.zeros((len(self.intensities), crop))
        
        for n in range(len(self.scintensities)):
            self.cropped_intensities[n] = self.intensities[n][:crop]
            
        
            
                
#%% all the generic versions of the functions - NOT updated.
def eleanor_lc(path, ra_declist, plotting = False):
    """ 
    retrieves + produces eleanor light curves from FFI files
    """
    import eleanor
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    import warnings
    warnings.filterwarnings('ignore')
    from eleanor.utils import SearchError
    
    gaia_ids = []
    filename = path + "eleanor_lightcurves_from_radeclist.fits"
    
    for n in range(len(ra_declist)):
        try:
            coords = SkyCoord(ra=ra_declist[n][0], dec=ra_declist[n][1], unit=(u.deg, u.deg))
            #try:
            files = eleanor.Source(coords=coords, tic=0) #by not providing a sector argument, will ONLY retrieve most recent sector
            print('Found TIC {0} (Gaia {1}), with TESS magnitude {2}, RA {3}, and Dec {4}'
                         .format(files.tic, files.gaia, files.tess_mag, files.coords[0], files.coords[1]))
            data = eleanor.TargetData(files)
            plt.figure(figsize=(16,6))
    
            q = data.quality == 0
            if plotting and n % 20 == 0: 
                    plt.scatter(data.time[q], data.raw_flux[q]/np.nanmedian(data.raw_flux[q])+0.06, c='black', s=0.5)
                    plt.scatter(data.time[q], data.corr_flux[q]/np.nanmedian(data.corr_flux[q]) + 0.03, c='red', s=0.5)
                    plt.ylabel('Normalized Flux')
                    plt.xlabel('Time [BJD - 2457000]')
                    plt.title("(" + str(ra_declist[n][0]) + str(ra_declist[n][1]) + ")")
                    plt.savefig(path + str(n) + 'lightcurveplotted.png')
                    plt.show()
                    plt.close()
                
            fluxandtime = [data.time[q], data.raw_flux[q]]
            lightcurve = np.asarray(fluxandtime)
                #print(lightcurve)
            if n == 0: #setting up fits file + save first one            
                hdr = fits.Header() # >> make the header
                hdu = fits.PrimaryHDU(lightcurve, header=hdr)
                hdu.writeto(filename)
                                            
            elif n != 0: #saving the rest
                fits.append(filename, lightcurve)
                print(int(n))
               
            gaia_ids.append(int(files.gaia))
        except (SearchError, ValueError):
            print("Some kind of error - either no TESS image exists, no GAIA ID exists, or there was a connection issue")
        
        if os.path.isdir("/Users/conta/.eleanor/tesscut") == True:
            shutil.rmtree("/Users/conta/.eleanor/tesscut")
            print("All files deleted")
    fits.append(filename, np.asarray(gaia_ids))
    print("All light curves saved into fits file")
    return gaia_ids

def open_eleanor_features(folderpath):
    """ 
    opens all eleanor features in a given folderpath
    returns a single array of ALL features and list of gaia_ids
    """
    filepaths = []
    for root, dirs, files in os.walk(folderpath):
        for name in files:
            #print(name)
            if name.startswith(("eleanor_features")):
                    filepaths.append(root + "/" + name)
    print(filepaths)
    f = fits.open(filepaths[0], memmap=False)
    features = f[0].data
    gaia_ids = f[1].data
    f.close()
    for n in range(len(filepaths) - 1):
        f = fits.open(filepaths[n+1], memmap=False)
        features_new = f[0].data
        features = np.column_stack((features, features_new))
        f.close()
    
    return features, gaia_ids

def open_eleanor_lc_files(path):
    """ opens the fits file that the eleanor light curves are saved into
    parameters:
        * path to the fits file
    returns:
        * list of gaia_ids
        * time indexes
        * intensities
    modified [lcg 08212020]"""
    f = fits.open(path, memmap=False)
    gaia_ids = f[-1].data
    target_nums = len(f) - 1
    all_timeindexes = []
    all_intensities = []
    for n in range(target_nums):
        all_timeindexes.append(f[n].data[0])
        all_intensities.append(f[n].data[1])
        
    f.close()
    
    return gaia_ids, np.asarray(all_timeindexes), np.asarray(all_intensities)

def create_save_featvec_different_timeaxes(yourpath, times, intensities, gaia_ids, filelabel, version=0, save=True):
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
    

    fname_features = yourpath + "/"+ filelabel + "_features_v"+str(version)+".fits"
    feature_list = []
    
    if version == 0:
	#median normalize for the v0 features
        for n in range(len(intensities)):
            intensities[n] = normalize(intensities[n], axis=0)
    elif version == 1: 
        import transitleastsquares
        from transitleastsquares import transitleastsquares
        #mean normalize the intensity so goes to 1
        for n in range(len(intensities)):
            intensities[n] = mean_norm(intensities[n], axis=0)

    print("Begining Feature Vector Creation Now")
    for n in range(len(intensities)):
        feature_vector = df.featvec(times[n], intensities[n], v=version)
        feature_list.append(feature_vector)
        
        if n % 25 == 0: print(str(n) + " completed")
    
    feature_list = np.asarray(feature_list)
    
    if save == True:
        hdr = fits.Header()
        hdr["VERSION"] = version
        hdu = fits.PrimaryHDU(feature_list, header=hdr)
        hdu.writeto(fname_features)
        fits.append(fname_features, gaia_ids)
    else: 
        print("Not saving feature vectors to fits")
    
    return feature_list

def build_simbad_extragalactic_database(maglim, out='./simbad_v19galaxies.txt'):
    '''Object type follows format in:
    http://vizier.u-strasbg.fr/cgi-bin/OType?$1'''
    
    # -- querying object type -------------------------------------------------
    customSimbad = Simbad()
    customSimbad.TIMEOUT = 1000
    # customSimbad.get_votable_fields()
    customSimbad.add_votable_fields('otype')
    customSimbad.add_votable_fields('ra(:;A;ICRS;J2000)', 'dec(:;D;ICRS;2000)')
    table = customSimbad.query_criteria('Vmag <=' + str(maglim), otype='G')
    objects = list(table['MAIN_ID'])
    ras = list(table['RA___A_ICRS_J2000'])
    decs = list(table['DEC___D_ICRS_2000'])

    # >> now loop through all of the objects
    for i in range(len(objects)):
        # >> decode bytes object to convert to string
        obj = objects[i].decode('utf-8')
        ra = ras[i]
        dec = decs[i]
       
        with open(out, 'a') as f:
                f.write(obj + ',' + ra + ',' + dec + ',' + '\n')
                
def get_radecfromtext(directory):
    ''' pulls ra and dec from text file containing all targets
    '''
    ra_all = []
    dec_all = []
    
    # >> find all text files in directory
    fnames = fm.filter(os.listdir(directory), '*.txt')
    
    for fname in fnames:
        # >> read text file
        with open(directory + fname, 'r') as f:
            lines = f.readlines()
            for line in lines:
                name, ra, dec, empty = line.split(',')
                
                ra_all.append(ra)
                dec_all.append(dec)
                
    return np.asarray(ra_all), np.asarray(dec_all)
def clip_feature_outliers(path, features, time, flux, gaia_ids, sigma, version=0, plot=True):
    """ 
    plots and then removes any outlier or nan features to avoid messiness in the 
    feature space. 
    parameters: 
        * path to save shit into
        * features (all)
        * time axis (needs a time axis for every flux)
        * flux (all)
        * gaia ids (all)
        * sigma to crop to
        * version of the features - v0 includes both v0 and v1 right now
        * plot=True by default, can skip if you only want to trim and not examine any outliers
        
    returns: 
        features_cropped, gaia_ids_cropped, flux_cropped, time_cropped, outlier_indexes
    modified [lcg 08252020 - adapted for multiple time axes]
    """
    path = path + "clipped-feature-outliers/"
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s" % path)
    
    #rcParams['figure.figsize'] = 8,3
    if version==0:
        features_greek = [r'$\alpha$', 'B', r'$\Gamma$', r'$\Delta$', r'$\beta$', r'$\gamma$',r'$\delta$',
                  "E", r'$\epsilon$', "Z", "H", r'$\eta$', r'$\Theta$', "I", "K", r'$\Lambda$', "M", r'$\mu$'
                  ,"N", r'$\nu$']
    elif version==1: 
        features_greek = ["M", r'$\mu$',"N", r'$\nu$']

    outlier_indexes = []
    for i in range(len(features[0])):
        column = features[:,i]
        column_std = np.std(column)
        column_top = np.mean(column) + column_std * sigma
        column_bottom = np.mean(column) - (column_std * sigma)
        for n in range(len(column)):
            #find and note the position of any outliers
            if column[n] < column_bottom or column[n] > column_top or np.isnan(column[n]) ==True: 
                outlier_indexes.append((int(n), int(i)))
                
    print(np.asarray(outlier_indexes))
        
    outlier_indexes = np.asarray(outlier_indexes)

    target_indexes = outlier_indexes[:,0] #is the index of the target on the lists
    print(target_indexes)
    feature_indexes = outlier_indexes[:,1] #is the index of the feature that it triggered on
    if plot:
        for i in range(len(outlier_indexes)):
            target_index = target_indexes[i]
            feature_index = feature_indexes[i]
            plt.figure(figsize=(8,3))
            plt.scatter(time[target_index], flux[target_index], s=0.5)
            target = gaia_ids[target_index]
            #print(features[target_index])
            
            if np.isnan(features[target_index][feature_index]) == True:
                feature_title = features_greek[feature_index] + "=nan"
            else: 
                feature_value = '%s' % float('%.2g' % features[target_index][feature_index])
                feature_title = features_greek[feature_index] + "=" + feature_value
            print(feature_title)
            
            plt.title("GAIA_ID " + str(int(target)) + " " + feature_title, fontsize=8)
            plt.tight_layout()
            plt.savefig((path + "featureoutlier-TICID" + str(int(target)) + ".png"))
            plt.show()
    else: 
        print("Not plotting outliers!")
            
        
    features_cropped = np.delete(features, target_indexes, axis=0)
    gaia_ids_cropped = np.delete(gaia_ids, target_indexes)
    flux_cropped = np.delete(flux, target_indexes, axis=0)
    time_cropped = np.delete(time, target_indexes, axis=0)
        
    return features_cropped, gaia_ids_cropped, flux_cropped, time_cropped, target_indexes

def plot_lof_FFI(time, intensity, targets, features, n, path,
             momentum_dump_csv = '../../Table_of_momentum_dumps.csv',
             n_neighbors=20, target_info=False,
             prefix='', mock_data=False, addend=1., feature_vector=False,
             n_tot=100, log=False):
    """ Plots the 20 most and least interesting light curves based on LOF.
    Parameters:
        * time : array with shape 
        * intensity
        * targets : list of TICIDs
        * n : number of curves to plot in each figure
        * path : output directory
        * n_tot : total number of light curves to plots (number of figures =
                  n_tot / n)
        * feature vector : assumes x axis is latent dimensions, not time  
        * mock_data : if True, will not plot TICID label
        * target_input : [sector, camera, ccd]
    Outputs:
        * Text file with TICID in column 1, and LOF in column 2 (lof-*.txt)
        * Log histogram of LOF (lof-histogram.png)
        * Top 20 light curves with highest and lowest LOF
        * Random 20 light curves
    modified [lcg 07012020 - includes inset histogram plotting]
    """
    # -- calculate LOF -------------------------------------------------------
    print('Calculating LOF')
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)
    fit_predictor = clf.fit_predict(features)
    negative_factor = clf.negative_outlier_factor_
    
    lof = -1 * negative_factor
    ranked = np.argsort(lof)
    largest_indices = ranked[::-1][:n_tot] # >> outliers
    smallest_indices = ranked[:n_tot] # >> inliers
    
    # >> save LOF values in txt file
    print('Saving LOF values')
    with open(path+'lof-'+prefix+'.txt', 'w') as f:
        for i in range(len(targets)):
            f.write('{} {}\n'.format(int(targets[i]), lof[i]))
      
    # >> make histogram of LOF values
    print('Make LOF histogram')
    #plot_histogram(lof, 20, "Local Outlier Factor (LOF)", time, intensity,
     #              targets, path+'lof-'+prefix+'histogram-insets.png',
      #             insets=True, log=log)
    pf.plot_histogram(lof, 20, "Local Outlier Factor (LOF)", time, intensity,
                   targets, path+'lof-'+prefix+'histogram.png', insets=False,
                   log=log)

    # -- plot smallest and largest LOF light curves --------------------------
    print('Plot highest LOF and lowest LOF light curves')
    num_figs = int(n_tot/n) # >> number of figures to generate
    
    for j in range(num_figs):
        
        for i in range(2): # >> loop through smallest and largest LOF plots
            fig, ax = plt.subplots(n, 1, sharex=False, figsize = (8, 3*n))
            
            for k in range(n): # >> loop through each row
                if i == 0: ind = largest_indices[j*n + k]
                elif i == 1: ind = smallest_indices[j*n + k]\
                
                # >> plot momentum dumps
                with open(momentum_dump_csv, 'r') as f:
                    lines = f.readlines()
                    mom_dumps = [ float(line.split()[3][:-1]) for line in lines[6:] ]
                    inds = np.nonzero((mom_dumps >= np.min(time[ind])) * \
                                      (mom_dumps <= np.max(time[ind])))
                    mom_dumps = np.array(mom_dumps)[inds]
                for t in mom_dumps:
                    ax[k].axvline(t, color='g', linestyle='--')
                    
                # >> plot light curve
                ax[k].plot(time[ind], intensity[ind] + addend, '.k')
                ax[k].text(0.98, 0.02, '%.3g'%lof[ind],
                           transform=ax[k].transAxes,
                           horizontalalignment='right',
                           verticalalignment='bottom',
                           fontsize='xx-small')
                pf.format_axes(ax[k], ylabel=True)
                ax[k].set_title("GAIA ID " + str(targets[ind]))
                
    
            # >> label axes
            ax[n-1].set_xlabel('time [BJD - 2457000]')
                
            # >> save figures
            if i == 0:
                
                fig.suptitle(str(n) + ' largest LOF targets', fontsize=16,
                             y=0.9)
                fig.savefig(path + 'lof-' + prefix + 'kneigh' + \
                            str(n_neighbors) + '-largest_' + str(j*n) + 'to' +\
                            str(j*n + n) + '.png',
                            bbox_inches='tight')
                plt.close(fig)
            elif i == 1:
                fig.suptitle(str(n) + ' smallest LOF targets', fontsize=16,
                             y=0.9)
                fig.savefig(path + 'lof-' + prefix + 'kneigh' + \
                            str(n_neighbors) + '-smallest' + str(j*n) + 'to' +\
                            str(j*n + n) + '.png',
                            bbox_inches='tight')
                plt.close(fig)

def features_plotting_FFI(feature_vectors, path, clustering,
                         time, intensity, targets, folder_suffix='',
                         feature_engineering=True, version=0, eps=0.5, min_samples=10,
                         metric='euclidean', algorithm='auto', leaf_size=30,
                         p=2, target_info=False, kmeans_clusters=4,
                         momentum_dump_csv='./Table_of_momentum_dumps.csv'):
    """plotting (n 2) features against each other
    parameters: 
        * feature_vectors - array of feature vectors
        * path to where you want everythigns aved - ends in a backslash
        * clustering - what you want to cluster as. options are 'dbscan', 'kmeans', or 
        any other keyword which will do no clustering
        * time axis
        * intensities
        *target ticids
        * folder suffix
        *feature_engineering - default is true
        * version - what version of engineered features, irrelevant integer if feature_engienering is false
        * eps, min_samples, metric, algorithm, leaf_size, p - dbscan parameters, comes with defaults
        * target_info - default is false
        *momentum dumps - not sure entirely why it's needed here tbh
        
    returns: only returns labels for dbscan/kmeans clustering. otherwise the only
    output is the files saved into the folder as given thru path
    """
    #detrmine which of the clustering algoirthms you're using: 
    rcParams['figure.figsize'] = 10,10
    folder_label = "blank"
    if clustering == 'dbscan':
        # !! TODO parameter optimization (eps, min_samples)
        db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
                    algorithm=algorithm, leaf_size=leaf_size,
                    p=p).fit(feature_vectors) #eps is NOT epochs
        classes_dbscan = db.labels_
        numclasses = str(len(set(classes_dbscan)))
        folder_label = "dbscan-colored"
        

    elif clustering == 'kmeans': 
        Kmean = KMeans(n_clusters=kmeans_clusters, max_iter=700, n_init = 20)
        x = Kmean.fit(feature_vectors)
        classes_kmeans = x.labels_
        folder_label = "kmeans-colored"
    else: 
        print("no clustering chosen")
        folder_label = "2DFeatures"
        
    #makes folder and saves to it    
    folder_path = path + folder_label
    try:
        os.makedirs(folder_path)
    except OSError:
        print ("Creation of the directory %s failed" % folder_path)
        print("New folder created will have -new at the end. Please rename.")
        folder_path = folder_path + "-new"
        os.makedirs(folder_path)
    else:
        print ("Successfully created the directory %s" % folder_path) 
 
    if clustering == 'dbscan':
        with open(folder_path + '/dbscan_paramset.txt', 'a') as f:
            f.write('eps {} min samples {} metric {} algorithm {} \
                    leaf_size {} p {} # classes {} \n'.format(eps,min_samples,
                    metric,algorithm, leaf_size, p,numclasses))
        plot_classification_FFI(time, intensity, targets, db.labels_,
                            folder_path+'/', prefix='dbscan',
                            momentum_dump_csv=momentum_dump_csv,
                            target_info=target_info)
        pf.plot_pca(feature_vectors, db.labels_,
                    output_dir=folder_path+'/')
    elif clustering == 'kmeans':
        plot_classification_FFI(time, intensity, targets, x.labels_,
                            path+folder_label+'/', prefix='kmeans',
                            momentum_dump_csv=momentum_dump_csv,
                            target_info=target_info)
 
    colors = pf.get_colors()
    #creates labels based on if engineered features or not
    if feature_engineering:
        if version==0:
            graph_labels = ["Average", "Variance", "Skewness", "Kurtosis", "Log Variance",
                            "Log Skewness", "Log Kurtosis", "Maximum Power", "Log Maximum Power", 
                            "Period of Maximum Power (0.1 to 10 days)","Slope" , "Log Slope",
                            "P0", "P1", "P2", "Period of Maximum Power (0.001 to 0.1 days)"]
            fname_labels = ["Avg", "Var", "Skew", "Kurt", "LogVar", "LogSkew", "LogKurt",
                            "MaxPower", "LogMaxPower", "Period0_1to10", "Slope", "LogSlope",
                            "P0", "P1", "P2", "Period0to0_1"]
        elif version == 1: 
            
            graph_labels = ["TLS Best fit Period (days)", "TLS Best fit duration (days)", "TLS best fit depth (ppt from transit bottom",
                            "TLS Best fit Power"]
            fname_labels = ["TLSPeriod", "TLSDuration", "TLSDepth", "TLSPower"]
        elif version == 2:
            graph_labels = ["Average", "Variance", "Skewness", "Kurtosis", "Log Variance",
                            "Log Skewness", "Log Kurtosis", "Maximum Power", "Log Maximum Power", 
                            "Period of Maximum Power (0.1 to 10 days)","Slope" , "Log Slope",
                            "P0", "P1", "P2", "Period of Maximum Power (0.001 to 0.1 days)", "TLS Best fit Period (days)", "TLS Best fit duration (days)", "TLS best fit depth (ppt from transit bottom",
                            "TLS Best fit Power"]
            fname_labels = ["Avg", "Var", "Skew", "Kurt", "LogVar", "LogSkew", "LogKurt",
                            "MaxPower", "LogMaxPower", "Period0_1to10", "Slope", "LogSlope",
                            "P0", "P1", "P2", "Period0to0_1", "TLSPeriod", "TLSDuration", "TLSDepth", "TLSPower"]
            
        num_features = len(feature_vectors[0])
    else:
        # >> shape(feature_vectors) = [num_samples, num_features]
        num_features = np.shape(feature_vectors)[1]
        graph_labels = []
        fname_labels = []
        for n in range(num_features):
            graph_labels.append('\u03C6' + str(n))
            fname_labels.append('phi'+str(n))
    for n in range(num_features):
        feat1 = feature_vectors[:,n]
        graph_label1 = graph_labels[n]
        fname_label1 = fname_labels[n]
        for m in range(num_features):
            if m == n:
                continue
            graph_label2 = graph_labels[m]
            fname_label2 = fname_labels[m]                
            feat2 = feature_vectors[:,m]
 
            if clustering == 'dbscan':
                plt.figure() # >> [etc 060520]
                plt.clf()
                for n in range(len(feature_vectors)):
                    plt.scatter(feat1[n], feat2[n], c=colors[classes_dbscan[n]], s=2)
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig((folder_path+'/' + fname_label1 + "-vs-" + fname_label2 + "-dbscan.png"))
                plt.show()
                plt.close()
                 
            elif clustering == 'kmeans':
                plt.figure() # >> [etc 060520]
                plt.clf()
                for n in range(len(feature_vectors)):
                    plt.scatter(feat1[n], feat2[n], c=colors[classes_kmeans[n]], s=2)
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig(folder_path+'/' + fname_label1 + "-vs-" + fname_label2 + "-kmeans.png")
                plt.show()
                plt.close()
            elif clustering == 'none':
                plt.scatter(feat1, feat2, s = 2, color = 'black')
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig(folder_path+'/' + fname_label1 + "-vs-" + fname_label2 + ".png")
                plt.show()
                plt.close()
                
    if clustering == 'dbscan':
        np.savetxt(folder_path+"/dbscan-classes.txt", classes_dbscan)
        return classes_dbscan
    if clustering == 'kmeans':
        return classes_kmeans
    
    
def plot_classification_FFI(time, intensity, targets, labels, path,
                        momentum_dump_csv = './Table_of_momentum_dumps.csv',
                        n=20, target_info=False,
                        prefix='', mock_data=False, addend=1.,
                        feature_vector=False):
    """ 
    """

    classes, counts = np.unique(labels, return_counts=True)
    # !!
    colors=['red', 'blue', 'green', 'purple', 'yellow', 'cyan', 'magenta',
            'skyblue', 'sienna', 'palegreen']*10
    
    # >> get momentum dump times
    
        
    for i in range(len(classes)): # >> loop through each class
        fig, ax = plt.subplots(n, 1, sharex=False, figsize = (8, 3*n))
        class_inds = np.nonzero(labels == classes[i])[0]
        if classes[i] == -1:
            color = 'black'
        elif classes[i] < len(colors) - 1:
            color = colors[i]
        else:
            color='black'
        
        for k in range(min(n, counts[i])): # >> loop through each row
            ind = class_inds[k]
            
            
            with open(momentum_dump_csv, 'r') as f:
                lines = f.readlines()
                mom_dumps = [ float(line.split()[3][:-1]) for line in lines[6:] ]
                inds = np.nonzero((mom_dumps >= np.min(time[ind])) * \
                                  (mom_dumps <= np.max(time[ind])))
                mom_dumps = np.array(mom_dumps)[inds]
            # >> plot momentum dumps
            for t in mom_dumps:
                ax[k].plot([t,t], [0, 1], '--g', alpha=0.5,
                           transform=ax[k].transAxes)            
            
            # >> plot light curve
            ax[k].plot(time[ind], intensity[ind] + addend, '.k')
            ax[k].text(0.98, 0.02, str(labels[ind]), transform=ax[k].transAxes,
                       horizontalalignment='right', verticalalignment='bottom',
                       fontsize='xx-small')
            pf.format_axes(ax[k], ylabel=True)
            ax[k].set_title("GAIA ID " + str(targets[ind]))

        if feature_vector:
            ax[n-1].set_xlabel('\u03C8')
        else:
            ax[n-1].set_xlabel('time [BJD - 2457000]')
    
        if classes[i] == -1:
            fig.suptitle('Class -1 (outliers)', fontsize=16, y=0.9,
                         color=color)
        else:
            fig.suptitle('Class ' + str(classes[i]), fontsize=16, y=0.9,
                         color=color)
        fig.savefig(path + prefix + '-class' + str(classes[i]) + '.png',
                    bbox_inches='tight')
        plt.close(fig)
        
def quick_plot_classification_FFI(time, intensity, targets, target_info, features, labels,
                              path='./', prefix='', addend=1.,
                              simbad_database_txt='./simbad_database.txt',
                              title='', ncols=10, nrows=5,
                              database_dir='./databases/'):
    '''Unfinished. Aim is to give an overview of the classifications, by
    plotting the first 5 light curves of each class. Any light curves
    classified by Simbad will be plotted first in their respective classes.'''
    classes, counts = np.unique(labels, return_counts=True)
    # colors=['red', 'blue', 'green', 'purple', 'yellow', 'cyan', 'magenta',
    #         'skyblue', 'sienna', 'palegreen', 'darksalmon', 'sandybrown',
    #         'lightsalmon', 'lightslategray', 'fuchsia', 'deeppink', 'crimson']*10
    colors = get_colors()
    
    # class_info = df.get_simbad_classifications(targets, simbad_database_txt)
    # ticid_classified = np.array(simbad_info)[:,0].astype('int')    
    class_info = df.get_true_classifications(targets,
                                             database_dir=database_dir)
    ticid_classified = class_info[:,0].astype('int')
    
    num_figs = int(np.ceil(len(classes) / ncols))
    features_greek = [r'$\alpha$', 'B', r'$\Gamma$', r'$\Delta$', r'$\beta$', r'$\gamma$',r'$\delta$',
                  "E", r'$\epsilon$', "Z", "H", r'$\eta$', r'$\Theta$', "I", "K", r'$\Lambda$', "M", r'$\mu$'
                  ,"N", r'$\nu$']
    
    for i in range(num_figs): #
        fig, ax = plt.subplots(nrows, ncols, sharex=True,
                               figsize=(8*ncols*0.75, 3*nrows))
        fig.suptitle(title)
        
        if i == num_figs - 1 and len(classes) % ncols != 0:
            num_classes = len(classes) % ncols
        else:
            num_classes = ncols
        for j in range(num_classes): # >> loop through columns
            class_num = classes[ncols*i + j]
            
            # >> find all light curves with this  class
            class_inds = np.nonzero(labels == class_num)[0]
            
            # >> find light curves with this class and with true classifications
            # inds = np.isin(ticid_simbad, targets[class_inds])
            inds = np.isin(targets[class_inds], ticid_classified)
            classified_inds = class_inds[np.nonzero(inds)]
            not_classified_inds = class_inds[np.nonzero(~inds)]
            
            if class_num == -1:
                color = 'black'
            elif class_num < len(colors) - 1:
                color = colors[class_num]
            else:
                color='black'
                
            k=-1
            # >> first plot any Simbad classified light curves
            for k in range(min(nrows, len(classified_inds))): 
                ind = classified_inds[k] # >> to index targets
                classified_ind = np.nonzero(ticid_classified == targets[ind])[0][0]
                ax[k, j].plot(time, intensity[ind]+addend, '.k')
                # simbad_label(ax[k,j], targets[ind], simbad_info[simbad_ind])
                classification_label(ax[k,j], targets[ind],
                                     class_info[classified_ind])
                ticid_label(ax[k, j], targets[ind], target_info[ind],
                            title=True, color=color)
                format_axes(ax[k, j], ylabel=True)
            
            # >> now plot non-classified light curves
            for l in range(k+1, min(nrows, len(not_classified_inds))):
                ind = not_classified_inds[l]
                ax[l,j].plot(time, intensity[ind]+addend, '.k')
                ticid_label(ax[l,j], targets[ind], target_info[ind],
                            title=True, color=color)
                format_axes(ax[l,j], ylabel=False)
                
            # ax[0, j].set_title('Class ' + str(class_num), color=color)   
                
            #get median features for the class
                #which feature vectors do i need
                #get only those feature vectors
                #take median and convert to string
            relevant_feats  =[]
            for k in range(len(labels)):
                if labels[k] == class_num:
                    relevant_feats.append(int(k))
            #index just this list 
            features_byclass = features[relevant_feats]
            med_features = np.median(features_byclass, axis=0)
            med_string = str(med_features)
            ax[0, j].set_title('Class '+str(class_num)+ "# Curves:" + str(counts[j]) +
                               '\n Median Features:' + med_string + 
                               "\n"+ax[0,j].get_title(),
                               color=color, fontsize='xx-small')
            ax[-1, j].set_xlabel('Time [BJD - 2457000]')   
                        
            if j == 0:
                for m in range(nrows):
                    ax[m, 0].set_ylabel('Relative flux')
                    
        fig.tight_layout()
        fig.savefig(path + prefix + '-' + str(i) + '.pdf')
        plt.close(fig)