# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:48:56 2020

@author: Lindsey Gordon

Last updated: April 2020
"""

import numpy as np
import numpy.ma as ma 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)

from pylab import rcParams
rcParams['figure.figsize'] = 16, 6
rcParams["lines.markersize"] = 2
from scipy.signal import argrelextrema

import astropy
from astropy.io import fits
import scipy.signal as signal
from astropy.stats import SigmaClip

from datetime import datetime
import os
import shutil
from scipy.stats import moment, sigmaclip

import sklearn
from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import fnmatch

from sklearn.metrics import confusion_matrix
import feature_functions
from feature_functions import *
from sklearn.neighbors import LocalOutlierFactor

import astroquery
from astroquery.simbad import Simbad
from astroquery.mast import Catalogs
from astroquery.mast import Observations

test(8) #should return 8 * 4

#if doing from scratch, run these: 
#time, intensity, targets = get_data_from_fits()
#intensity = normalize(intensity)
#lc_feat = create_list_featvec(time, intensity)

#if just running on intensity files you already have, run this:
#%%
time, intensity, targets, lc_feat = get_from_files()



#%%
n_choose_2_features_plotting(lc_feat, "5-4", "none")
n_choose_2_features_plotting(lc_feat, "5-4", "kmeans")
n_choose_2_features_plotting(lc_feat, "5-4", "dbscan")

plot_lof(time, intensity, targets, lc_feat, 10, "5-4")

#%%

targets_sector20 = np.loadtxt("/Users/conta/UROP_Spring_2020/all_targets_S020_v1.txt", usecols = 0)

print(targets_sector20)

targets20_5000 = targets_sector20[0:5000]

#targets20_10000

#%%

def lc_from_target_list(targetList):
    fitspath = '/Users/conta/UROP_Spring_2020/mastDownload/TESS/'
    ints = []
    times = []
    targets_TICS = []
    for target in targetList:
        #print(target, type(target), int(target))
        targ = "TIC " + str(int(target))
        print(targ)
        obs_table = Observations.query_object(targ, radius=".02 deg")
        data_products_by_obs = Observations.get_product_list(obs_table[0:2])
        
        filter_products = Observations.filter_products(data_products_by_obs, dataproduct_type = 'timeseries')
        manifest = Observations.download_products(filter_products)
        print(manifest)
        
        filepaths = []
        for root, dirs, files in os.walk(fitspath):
            for name in files:
                if name.endswith(("lc.fits")):
                    filepaths.append(root + "/" + name)
            
        print(filepaths)
        
        for file in filepaths:
                # -- open file -------------------------------------------------------------
            f = fits.open(file, memmap=False)

            time1 = f[1].data['TIME']
            i1 = f[1].data['PDCSAP_FLUX']
            tic1 = f[1].header["OBJECT"]
            times.append(time1)
            ints.append(i1)
            targets_TICS.append(tic1)
            f.close()
            
        if os.path.isdir("mastDownload") == True:
            shutil.rmtree("mastDownload")               #deletes ALL data to conserve space
            print("folder deleted")
        
        #print(ints)
        
    return times, ints, targets_TICS
        
        
times, ints, targets_TICS = lc_from_target_list(targets20_5000)







