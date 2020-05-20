# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:48:56 2020

@author: Lindsey Gordon

Pipeline to produce all files for a given group of data.

Last updated: May 2020
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

import shapely
from shapely import geometry
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

test(8) #should return 8 * 4

mypath = "/Users/conta/UROP_Spring_2020/"
sectorfile = "/Users/conta/UROP_Spring_2020/all_targets_S020_v1.txt"
sector = 20
camera = 1
ccd = 2

times, intensities, failed_to_get, targets = data_process_a_group(mypath, sectorfile, sector, camera, ccd)

features = create_list_featvec(times[0], intensities)

targets_strings = []
for n in range(len(targets)):
    targets_strings.append(("TIC " + str(int(targets_strings[n]))))
    
post_process_plotting(times[0], intensities, features, features, targets_strings, "folder")



#%%
t = np.loadtxt("/Users/conta/UROP_Spring_2020/Sector20Cam1CCD1/sector20_cam1_ccd1_interp_times.txt")
intensities = np.loadtxt("/Users/conta/UROP_Spring_2020/Sector20Cam1CCD1/sector20_cam1_ccd1_processed_intensities.txt")

features = np.loadtxt("/Users/conta/UROP_Spring_2020/Sector20Cam1CCD1/sector_20_cam1_ccd1_features.txt")    

      


                