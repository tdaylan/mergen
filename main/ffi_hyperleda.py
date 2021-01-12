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


import pdb
import fnmatch as fm

import plotting_functions as pf
import data_functions as df
import model as ml



class hyperleda_ffi(object):
    
    def __init__(self, datapath = "./", savepath = "./", ensemblename = "ensemble",
                 momentum_dump_csv = '/users/conta/urop/Table_of_momentum_dumps.csv',
                 makefeats = True, ENF = True, CAE = True):
        print("Initialized hyperleda ffi processing object")
        self.datapath = datapath
        self.savepath = savepath
        self.ensemblename = ensemblename
        self.ensemblefolder = self.savepath + self.ensemblename + "/"
        self.momdumpcsv = momentum_dump_csv
        self.makefeats = makefeats
        
        print("Making all folders")
        self.folder_initiate()
            
        print("Loading in data")
        try:
            print("Attempting to load in a saved metafile")
            self.fname_lightcurves = self.savepath + self.ensemblename + "/" + self.ensemblename + "_lightcurves.fits"
            self.load_lc_metafile()
            print("Succesfully loaded in light curves")
            print("Opening all features together")
            #self.load_features()
        except:
            print("Metafile does not exist, loading from raw text files")
            self.load_lc_from_files()
            self.save_lc_metafile()
            print("Opening all features together")
        
        try:
            self.load_features()
        except:
            print("Features do not exist yet")
        
        if ENF: 
            if self.makefeats:
                print("Producing v0 features")
                self.create_save_featvec(version=0, save=True)
                
                #print("Producing v1 features")
                #self.create_save_featvec(version = 1, save=True)
            
            print("Opening all features together")
            self.load_features()
            

            self.path = self.enffolder
            self.plot_everything(list(np.arange(0.5,5,0.5)), [3,5,10])
            
        if CAE:
            print("CAE not set up yet")
            self.path = self.CAEpath
            if not ENF:
                self.median_normalize()
                self.cleanedflux, comp = self.pca_linregress()
            target_info = np.zeros((len(self.cleanedflux), 5))
            self.index_for_CAE()
            self.representation_learning(target_info)
        
        if not CAE and not ENF:
            print("Cleaning data")
            self.path = self.ensemblefolder
            self.median_normalize()
            self.cleanedflux, comp =self.pca_linregress()
        return
     
        
    def folder_initiate(self):
        """Makes all the big folders"""
        
        
        print("Setting Up Ensemble Folder")
        try:
            os.makedirs(self.ensemblefolder)
        except OSError:
            print ("Directory %s already exists" % self.ensemblefolder) 
            
        print("Setting up CAE folder")
        self.CAEpath = self.ensemblefolder + "CAE/"
        try:
            os.makedirs(self.CAEpath)
        except OSError:
            print ("Directory %s already exists" % self.CAEpath)
            
        print("Setting up ENF folder")
        self.enffolder = self.ensemblefolder + "ENF/"
        try:
            os.makedirs(self.enffolder)
        except OSError:
            print ("Directory %s already exists" % self.enffolder)
        return
    
    def save_lc_metafile(self):
        self.fname_lightcurves = self.savepath + self.ensemblename + "/" + self.ensemblename + "_lightcurves.fits"
    
        hdr = fits.Header()
        hdu = fits.PrimaryHDU(self.timeaxis, header=hdr)
        hdu.writeto(self.fname_lightcurves)
        fits.append(self.fname_lightcurves, self.intensities)
        
        fname_ids = self.savepath + self.ensemblename + "/" + self.ensemblename + "_lightcurve_ids.txt"



        np.savetxt(fname_ids, np.asarray(self.identifiers), fmt = '%s')
        return 
    
    def load_lc_metafile(self):
        self.fname_lightcurves = self.savepath + self.ensemblename + "/" + self.ensemblename + "_lightcurves.fits"
        
        f = fits.open(self.fname_lightcurves, memmap=False)
        self.timeaxis = f[0].data
        self.intensities = f[1].data
        
        fname_ids = self.savepath + self.ensemblename + "/" + self.ensemblename + "_lightcurve_ids.txt"
        
        self.identifiers = np.loadtxt(fname_ids, dtype='str')
        
        return

    def plot_everything(self, dbscan_eps, min_samples):
        self.path = self.enffolder
        if not self.makefeats: #means that cleaning has not happened yet
            print("Cleaning data")
            self.median_normalize()
            self.cleanedflux, comp =self.pca_linregress()
        print("Removing and plotting outliers in the feature space")
        self.isolate_outlier_features(sigma = 10)
        print("Plotting LOF")
        #self.plot_lof(20)
        print("Plotting marginal distributions, no clustering")
        #self.features_plotting(clustering = 'nocluster')
        #print("plotting marginal distributions, kmeans clustering")
        #self.features_plotting(clustering = 'kmeans',kmeans_clusters=10)
        
        parameter_sets, num_classes, silhouettes, d_b, ch, acc = \
            self.dbscan_param_search(eps=dbscan_eps,
                            min_samples=min_samples,
                            metric=['minkowski'],
                            algorithm = ['auto'],
                            leaf_size = [40],
                            p = [2], plotting=True, appendix = "bigscan",
                            database_dir=None, pca=True, tsne=True,
                            confusion_matrix=False)
        
        best_ind = np.argmax(silhouettes)
        best_param_set = parameter_sets[best_ind] 
        print("Best parameter set is {}".format(best_param_set))
        print("Running DBSCAN with just this parameter set")
        
        parameter_sets, num_classes, silhouettes, d_b, ch, acc = \
            self.dbscan_param_search(eps=[best_param_set[0]],
                            min_samples=[best_param_set[1]],
                            metric=[best_param_set[2]],
                            algorithm = [best_param_set[3]],
                            leaf_size = [best_param_set[4]],
                            p = [best_param_set[5]], plotting=True, appendix="best-params",
                            database_dir=None, pca=True, tsne=True,
                            confusion_matrix=False)   
        
        print("Plotting marginal distributions, best parameter set")
        self.features_plotting(clustering = 'dbscan', eps=best_param_set[0], min_samples=best_param_set[1],
                             metric=best_param_set[2], algorithm=best_param_set[3], leaf_size=best_param_set[4],
                             p=best_param_set[5])
        return
    
    
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
        #self.path = self.enffolder
        self.knnpath = self.path + "knn_plots/"
        
        try:
            os.makedirs(self.knnpath)
        except OSError:
            print ("Creation of the directory %s failed" % self.knnpath)
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
            plt.ylim((0, 30))
            plt.title("K-Neighbor plot for k=" + str(k_values[n]))
            plt.savefig(self.knnpath + "kneighbors-" +str(k_values[n]) +"-plot-sorted.png")
            plt.close() 
    
    def load_lc_from_files(self):

        #make sure they're all txt files
        for root, dirs, files in os.walk(self.datapath):
            for filename in files:
                file = os.path.join(root, filename)
                if filename.startswith("lc_") and not filename.endswith(".txt"):
                    os.rename(file, file + ".txt")
                        
            
        
        #get all file names
        intensities = []
        identifiers = []
        
        #open all files
        for root, dirs, files in os.walk(self.datapath):
            
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
                        
                    if n % 500 == 0:
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
    
    def pca_linregress(self, plot=False):
        print("Beginning PCA Linear Regression Corrections for 3 components")
        from sklearn.linear_model import LinearRegression
        pca = PCA(n_components=3)
    
        standardized_flux = df.standardize(self.cleanedflux)
        
        #x should have shape(num samples, num features) (rows, columns)
        # ie each COLUMN is a light curve, so you get principal component of
        p = pca.fit_transform(standardized_flux.T) #take transpose - input is COLUMNS of light curves
        
        components = p.T #now have n_component rows of reduced curves??
        #print(components)
        
        for n in range(len(components)):
            plt.scatter(self.timeaxis, components[n])
            plt.title("Principal Component " + str(n))
            plt.savefig(self.path + "standardized-pc-" + str(n)+'.png')
            plt.show()
        residuals = np.zeros_like(self.cleanedflux)
        for n in range(len(self.cleanedflux)):
            reg = LinearRegression().fit(components.T, standardized_flux[n])
            
            y = reg.coef_[0] * components[0] + reg.coef_[1] * components[1] + reg.coef_[2] * components[2]
            residual = standardized_flux[n] - y
            residuals[n] = residual
            if plot:
                score = reg.score(components.T, standardized_flux[n])
                print(reg.coef_, reg.score(components.T, standardized_flux[n]), reg.intercept_)
            
                plt.scatter(self.timeaxis, y, label ="lin regress")
                
                plt.scatter(self.timeaxis, standardized_flux[n], label = "original data")
                plt.title("Raw data versus linear regression fit. Fit score: " + str(score))
                plt.legend()
                plt.savefig(self.path + "origversuslinregress-" + str(n) + ".png")
                plt.show()
                
                plt.scatter(self.timeaxis, residual)
                plt.title("Residual")
                plt.savefig(self.path + "residual-" + str(n) + ".png")
                plt.show()
            
        return residuals, components
    
    
    def isolate_outlier_features(self, sigma = 5):
        
        self.outlierfeaturepath = self.path + "clipped-feature-outliers/"
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
            plt.scatter(self.timeaxis, self.cleanedflux[target_index], s=0.5)
            target = self.identifiers[target_index]
            
            if np.isnan(self.features[target_index][feature_index]) == True:
                feature_title = features_greek[feature_index] + "=nan"
            else: 
                feature_value = '%s' % float('%.2g' % self.features[target_index][feature_index])
                feature_title = features_greek[feature_index] + "=" + feature_value
            #print(feature_title)
            
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
        
        self.flux_outliers = self.cleanedflux[target_indexes]
        self.cleanedflux = np.delete(self.cleanedflux, target_indexes, axis=0)
        
        #self.identifiers_outliers = self.identifiers[target_indexes]
        self.identifiers = np.delete(self.identifiers, target_indexes)
        
        return
    
    #i think i'm going to have to just sigma clip each light curve as i go through the feature generation
                    
    def create_save_featvec(self, version=0, save=True):
        """ documentation """

        fname_features = self.enffolder + self.ensemblename + "_features_v"+str(version)+".fits"
        feature_list = []
        if version == 0:
            self.median_normalize()
            self.path = self.enffolder
            self.cleanedflux, comp = self.pca_linregress()
        elif version == 1: 
            from transitleastsquares import transitleastsquares
            self.mean_normalize()
            self.path = self.enffolder
            self.cleanedflux, comp =self.pca_linregress()
    
        print("Begining Feature Vector Creation Now")
        #sigma clip each time you calculate - unsure how better to do this??
        sigclip = SigmaClip(sigma=5, maxiters=None, cenfunc='median')
        for n in range(len(self.cleanedflux)):
            
            times = self.timeaxis
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
      
    
    def load_features(self):
        filepaths = []
        for root, dirs, files in os.walk(self.enffolder):
            for file in files:
                if file.startswith((self.ensemblename)):
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
        self.features = features
        return
        
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
        filename = self.lofpath + "histogram.png"
        #this is the very very simple histogram plotting
        fig, ax1 = plt.subplots()
        n_in, bins, patches = ax1.hist(data, bins, log=log)
        
        y_range = np.abs(n_in.max() - n_in.min())
        x_range = np.abs(data.max() - data.min())
        ax1.set_ylabel('Number of light curves')
        ax1.set_xlabel(x_label)
        
        if insets == True:
            filename = self.lofpath + "histogram-insets.png"
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
                            lc_to_plot = self.cleanedflux[m]
                            ident = self.identifiers[m]
                            break
                        else: 
                            continue
                    
                    axis_name.scatter(self.timeaxis, lc_to_plot, c='black', s = 0.1, rasterized=True)
                    axis_name.set_title(ident, fontsize=6)
        plt.savefig(filename)
        plt.close()
        return
    
    def plot_lof(self, n, n_neighbors=20, n_tot=100):
        """ documentation"""
        prefix=''
        p=2
        bins=50
        
        #set up lof folder
        self.lofpath = self.path + "lof/"
        try:
            os.makedirs(self.lofpath)
        except OSError:
            print ("Directory %s already exists" % self.lofpath)
        
        # -- calculate LOF -------------------------------------------------------
        print('Calculating LOF')
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, p=p)
        fit_predictor = clf.fit_predict(self.features)
        negative_factor = clf.negative_outlier_factor_
        
        lof = -1 * negative_factor
        self.lof = lof
        ranked = np.argsort(lof)
        largest_indices = ranked[::-1][:n_tot] # >> outliers
        smallest_indices = ranked[:n_tot] # >> inliers
        random_inds = list(range(len(lof)))
        import random
        random.Random(4).shuffle(random_inds)
        random_inds = random_inds[:n_tot] # >> random
        ncols=1
    
          
        # >> make histogram of LOF values
        print('LOF histogram')
        self.plot_histogram(lof, 50, "Local Outlier Factor (LOF)", insets=False, log=True)
        self.plot_histogram(lof, 50, "Local Outlier Factor (LOF)", insets=True, log=False)
    

        print('Saving LOF values')
        with open(self.lofpath+'lof-'+prefix+'kneigh' + str(n_neighbors)+'.txt', 'w') as f:
            for i in range(len(self.identifiers)):
                f.write('{} {}\n'.format(self.identifiers[i], lof[i]))
             

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
                    axis.set_title(self.identifiers[ind])
                    #print(self.identifiers[ind])
                    axis.text(0.98, 0.02, '%.3g'%lof[ind],
                               transform=axis.transAxes,
                               horizontalalignment='right',
                               verticalalignment='bottom',
                               fontsize='x-small')                        
                        
                    if k != n - 1:
                        axis.set_xticklabels([])
                        
                # >> label axes

                ax[n-1].set_xlabel('time [BJD - 2457000]')
                    
                # >> save figures
                if i == 0:
                    
                    fig.suptitle(str(n) + ' largest LOF targets', fontsize=16,
                                 y=0.95)
                    fig.tight_layout()
                    fig.savefig(self.lofpath + 'lof-' + prefix + 'kneigh' + \
                                str(n_neighbors) + '-largest_' + str(j*n) + 'to' +\
                                str(j*n + n) + '.png',
                                bbox_inches='tight')
                    plt.close(fig)
                elif i == 1:
                    fig.suptitle(str(n) + ' smallest LOF targets', fontsize=16,
                                 y=0.9)
                    fig.tight_layout()
                    fig.savefig(self.lofpath + 'lof-' + prefix + 'kneigh' + \
                                str(n_neighbors) + '-smallest' + str(j*n) + 'to' +\
                                str(j*n + n) + '.png',
                                bbox_inches='tight')
                    plt.close(fig)
                else:
                    fig.suptitle(str(n) + ' random LOF targets', fontsize=16, y=0.9)
                    
                    # >> save figure
                    fig.tight_layout()
                    fig.savefig(self.lofpath + 'lof-' + prefix + 'kneigh' + str(n_neighbors) \
                                + "-random"+ str(j*n) + 'to' +\
                                str(j*n + n) +".png", bbox_inches='tight')
                    plt.close(fig)
        return
        
    def index_for_CAE(self):
        """ Current CAE setup does not allow for non-number inputs as TICIDs
        Therefore we index all of the fluxes and save the associated names and 
        the indexes into a text file for later reference"""
        self.CAE_index = np.arange(0, len(self.cleanedflux), 1)
        filepath = self.CAEpath + "/indexed_names.txt"
        for n in range(len(self.CAE_index)):
            with open(filepath, 'a') as f:
                f.write('{} {} \n'.format(self.CAE_index[n], self.identifiers[n]))
        return
    
    
    def representation_learning(self, target_info, p=None,
                                validation_targets=[],
                                norm_type='minmax_normalization',
                                input_rms=False, input_psd=False, load_psd=False,
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
            ml.autoencoder_preprocessing(self.cleanedflux, self.CAE_index, self.timeaxis, target_info, p,
                                         validation_targets=validation_targets,
                                         norm_type=norm_type,
                                         input_rms=input_rms, input_psd=input_psd,
                                         load_psd=load_psd,
                                         train_test_ratio=train_test_ratio,
                                         split=split,
                                         output_dir=self.CAEpath)       
            
        print('Training CAE')
        history, model, x_predict = \
            ml.conv_autoencoder(x_train, y_train, x_test, y_test, p,
                                input_rms=input_rms, rms_train=rms_train, rms_test=rms_test,
                                ticid_train=ticid_train, ticid_test=ticid_test,
                                output_dir=self.CAEpath)
            
        print('Diagnostic plots')
        pf.diagnostic_plots(history, model, p, self.CAEpath, x, x_train,
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
                            load_bottleneck=True, FFI = True)            
    
        features, flux_feat, ticid_feat, info_feat = \
            ml.bottleneck_preprocessing(None,
                                        np.concatenate([x_train, x_test], axis=0),
                                        np.concatenate([ticid_train, ticid_test]),
                                        np.concatenate([target_info_train,
                                                        target_info_test]),
                                        data_dir=None,
                                        output_dir=self.CAEpath,
                                        use_learned_features=True,
                                        use_tess_features=False,
                                        use_engineered_features=False,
                                        use_tls_features=False)         
        print("Novelty detection")
        self.features = features
        self.plot_lof(20, n_neighbors=20, n_tot=100)
        
        
        print('DBSCAN parameter search')
        parameter_sets, num_classes, silhouettes, d_b, ch, acc = \
            self.dbscan_param_search(eps=list(np.arange(0.5,5,0.2)),
                            min_samples=[3,5,10],
                            metric=['minkowski'],
                            algorithm = ['auto'],
                            leaf_size = [40],
                            p = [2], plotting=True, appendix = "big-scan-cae",
                            database_dir=None, pca=True, tsne=True,
                            confusion_matrix=False)
        
        best_ind_s = np.argmax(silhouettes)
        best_ind_ch = np.argmax(ch)
        best_ind_db = np.argmin(d_b)
        
        best_param_set = parameter_sets[best_ind_s] 
        print("Best parameter set is {}".format(best_param_set))
        print("Running DBSCAN with just this parameter set")
        
        parameter_sets, num_classes, silhouettes, d_b, ch, acc = \
            self.dbscan_param_search(eps=[best_param_set[0]],
                            min_samples=[best_param_set[1]],
                            metric=[best_param_set[2]],
                            algorithm = [best_param_set[3]],
                            leaf_size = [best_param_set[4]],
                            p = [best_param_set[5]], plotting=True, appendix="best-silhouette-params-cae",
                            database_dir=None, pca=True, tsne=True,
                            confusion_matrix=False)  
            
        if best_ind_s != best_ind_ch:
            best_param_set = parameter_sets[best_ind_ch] 
            print("Best parameter set is {}".format(best_param_set))
            print("Running DBSCAN with just this parameter set")
            
            parameter_sets, num_classes, silhouettes, d_b, ch, acc = \
                self.dbscan_param_search(eps=[best_param_set[0]],
                                min_samples=[best_param_set[1]],
                                metric=[best_param_set[2]],
                                algorithm = [best_param_set[3]],
                                leaf_size = [best_param_set[4]],
                                p = [best_param_set[5]], plotting=True, appendix="best-CH-params-cae",
                                database_dir=None, pca=True, tsne=True,
                                confusion_matrix=False)
        if best_ind_s != best_ind_db and best_ind_ch != best_ind_db:
            best_param_set = parameter_sets[best_ind_db] 
            print("Best parameter set is {}".format(best_param_set))
            print("Running DBSCAN with just this parameter set")
            
            parameter_sets, num_classes, silhouettes, d_b, ch, acc = \
                self.dbscan_param_search(eps=[best_param_set[0]],
                                min_samples=[best_param_set[1]],
                                metric=[best_param_set[2]],
                                algorithm = [best_param_set[3]],
                                leaf_size = [best_param_set[4]],
                                p = [best_param_set[5]], plotting=True, appendix="best-DB-params-cae",
                                database_dir=None, pca=True, tsne=True,
                                confusion_matrix=False)
        
        print("CAE completed")
        return
    
    
    def features_plotting(self, clustering = 'dbscan', eps=3, min_samples=10,
                             metric='minkowski', algorithm='auto', leaf_size=30,
                             p=2, kmeans_clusters=10):
        """plotting (n 2) features against each other

        modified [lcg 09152020 - adapted to hyperleda]
        ** TO DO: make file and graph labels a property of self when you set the version
        """
        #detrmine which of the clustering algoirthms you're using: 
        rcParams['figure.figsize'] = 10,10
        folder_label = "blank"
        if clustering == 'dbscan':
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
        folder_path = self.path + "marginal-distributions-" + clustering + "/"
        try:
            os.makedirs(folder_path)
        except OSError:
            print ("Directory %s already exists" % folder_path)
        else:
            print ("Successfully created the directory %s" % folder_path) 
     
        if clustering == 'dbscan':
            with open(folder_path + 'dbscan_paramset.txt', 'a') as f:
                f.write('eps {} min samples {} metric {} algorithm {} \
                        leaf_size {} p {} # classes {} \n'.format(eps,min_samples,
                        metric,algorithm, leaf_size, p,numclasses))
            self.plot_classification(labels = classes_dbscan, path = folder_path, n=5)
            pf.plot_pca(self.features, classes_dbscan, output_dir=folder_path)
        elif clustering == 'kmeans':
            print("uhhh nothing right now!! fix me later!")
            self.plot_classification(labels = classes_kmeans, path = folder_path, n=5)
            pf.plot_pca(self.features, classes_kmeans, output_dir=folder_path)
            
        colors = pf.get_colors()
        
        #creates labels
        graph_labels = ["Average", "Variance", "Skewness", "Kurtosis", "Log Variance",
                        "Log Skewness", "Log Kurtosis", "Maximum Power", "Log Maximum Power", 
                        "Period of Maximum Power (0.1 to 10 days)","Slope" , "Log Slope",
                        "P0", "P1", "P2", "Period of Maximum Power (0.001 to 0.1 days)", 
                        "TLS Best fit Period (days)", "TLS Best fit duration (days)", "TLS best fit depth (ppt from transit bottom",
                        "TLS Best fit Power"]
        fname_labels = ["Avg", "Var", "Skew", "Kurt", "LogVar", "LogSkew", "LogKurt",
                        "MaxPower", "LogMaxPower", "Period0_1to10", "Slope", "LogSlope",
                        "P0", "P1", "P2", "Period0to0_1", "TLSPeriod", "TLSDuration", "TLSDepth", "TLSPower"]

        num_features = len(self.features[0])
   
        for n in range(num_features):
            feat1 = self.features[:,n]
            graph_label1 = graph_labels[n]
            fname_label1 = fname_labels[n]
            for m in range(num_features):
                if m == n or m < n:
                    continue
                graph_label2 = graph_labels[m]
                fname_label2 = fname_labels[m]                
                feat2 = self.features[:,m]
     
                if clustering == 'dbscan':
                    plt.figure()
                    plt.clf()
                    for n in range(len(self.features)):
                        plt.scatter(feat1[n], feat2[n], c=colors[classes_dbscan[n]], s=2)
                    plt.xlabel(graph_label1)
                    plt.ylabel(graph_label2)
                    plt.savefig((folder_path + fname_label1 + "-vs-" + fname_label2 + "-dbscan.png"))
                    #plt.show()
                    plt.close()
                     
                elif clustering == 'kmeans':
                    plt.figure() #
                    plt.clf()
                    for n in range(len(self.features)):
                        plt.scatter(feat1[n], feat2[n], c=colors[classes_kmeans[n]], s=2)
                    plt.xlabel(graph_label1)
                    plt.ylabel(graph_label2)
                    plt.savefig(folder_path + fname_label1 + "-vs-" + fname_label2 + "-kmeans.png")
                    #plt.show()
                    plt.close()
                else:
                    plt.scatter(feat1, feat2, s = 2, color = 'black')
                    plt.xlabel(graph_label1)
                    plt.ylabel(graph_label2)
                    plt.savefig(folder_path + fname_label1 + "-vs-" + fname_label2 + ".png")
                    #plt.show()
                    plt.close()
                    
        if clustering == 'dbscan':
            np.savetxt(folder_path+"dbscan-classes.txt", classes_dbscan)
            return classes_dbscan
        if clustering == 'kmeans':
            return classes_kmeans
        
    def plot_classification(self, labels, path, n=20):
        
        """ 
        FFI version of pf.plot_classification
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
                    inds = np.nonzero((mom_dumps >= np.min(self.timeaxis)) * \
                                      (mom_dumps <= np.max(self.timeaxis)))
                    mom_dumps = np.array(mom_dumps)[inds]
                # >> plot momentum dumps
                for t in mom_dumps:
                    ax[k].plot([t,t], [0, 1], '--g', alpha=0.5,
                               transform=ax[k].transAxes)            
                
                # >> plot light curve
                ax[k].plot(self.timeaxis, self.cleanedflux[ind], '.k')
                ax[k].text(0.98, 0.02, str(labels[ind]), transform=ax[k].transAxes,
                           horizontalalignment='right', verticalalignment='bottom',
                           fontsize='xx-small')
                pf.format_axes(ax[k], ylabel=True)
                ax[k].set_title(str(self.identifiers[ind]))
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
            
    def dbscan_param_search(self, eps=list(np.arange(0.5,10,0.4)),
                                min_samples=[2,5,10],
                                metric=['euclidean', 'minkowski'],
                                algorithm = ['auto'],
                                leaf_size = [30, 40, 50],
                                p = [1,2,3,4], plotting=False, appendix = "",
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
        
        
        
        self.dbpath = self.path + "dbscan-paramscan{}/".format(appendix)
        
        try:
            os.makedirs(self.dbpath)
        except OSError:
            print ("Directory %s already exists" % self.dbpath)
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
                                
                                #if db.labels_[24037] == -1 or db.labels_[24038] == -1:
                                 #   print("did not classify them")
                                  #  print(db.labels_[24037],db.labels_[24038] )
                                    #continue
                                
                                    
                                    #param_num = str(len(parameter_sets)-1)
                                title='Parameter Set '+str(param_num)+': '+'{} {} {} {} {} {}'.format(eps[i],
                                                                                            min_samples[j],
                                                                                            metric[k],
                                                                                            algorithm[l],
                                                                                            leaf_size[m],
                                                                                            p[n])
                                
                                prefix='dbscan-p'+str(param_num)                            
                                
                                if confusion_matrix:
                                    acc = pf.plot_confusion_matrix(self.identifiers, db.labels_,
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
                    if ind == 24038 or ind == 24037:
                        colorplot = 'red'
                    else: 
                        colorplot = 'black'
                    ax[k, j].plot(self.timeaxis, self.cleanedflux[ind], '.k')
                    ax[k,j].set_title(str(self.identifiers[ind]), color=colorplot)
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
            fig.savefig(output_dir + prefix + '-' + str(i) + '.png')
            plt.close(fig)
    
    def features_insets(self):
        """ Plots 2 features against each other with the extrema points' associated
        light curves plotted as insets along the top and bottom of the plot. 

        modified [lcg 09162020 - adapted to FFI]
        """   
        folderpath = self.path + "2DFeatures-insets/"
        
        try:
            os.makedirs(folderpath)
        except OSError:
            print ("%s already exists" % folderpath)
        else:
            print ("Successfully created the directory %s" % folderpath) 
            

        
        graph_labels = ["Average", "Variance", "Skewness", "Kurtosis", "Log Variance",
                        "Log Skewness", "Log Kurtosis", "Maximum Power", "Log Maximum Power", 
                        "Period of Maximum Power (0.1 to 10 days)","Slope" , "Log Slope",
                        "P0", "P1", "P2", "Period of Maximum Power (0.001 to 0.1 days)", 
                        "TLS Best fit Period (days)", "TLS Best fit duration (days)", "TLS best fit depth (ppt from transit bottom",
                        "TLS Best fit Power"]
        fname_labels = ["Avg", "Var", "Skew", "Kurt", "LogVar", "LogSkew", "LogKurt",
                        "MaxPower", "LogMaxPower", "Period0_1to10", "Slope", "LogSlope",
                        "P0", "P1", "P2", "Period0to0_1", "TLSPeriod", "TLSDuration", "TLSDepth", "TLSPower"]
            
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
            axis_name.scatter(self.timeaxis, self.cleanedflux[inset_indexes[n]], c='black', s = 0.1, rasterized=True)
            
            #this sets where the pointer goes to
            x1, x2 = datax[inset_indexes[n]], datax[inset_indexes[n]] + 0.001*x_range
            y1, y2 =  datay[inset_indexes[n]], datay[inset_indexes[n]] + 0.001*y_range
            axis_name.set_xlim(x1, x2)
            axis_name.set_ylim(y1, y2)
            ax1.indicate_inset_zoom(axis_name)
                  
            #this sets the actual axes limits    
            axis_name.set_xlim(self.timeaxis.min(), self.timeaxis.max())
            axis_name.set_ylim(self.cleanedflux[inset_indexes[n]].min(), self.cleanedflux[inset_indexes[n]].max())
            axis_name.set_title(str(int(self.identifiers[inset_indexes[n]])), fontsize=6)
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
            
            