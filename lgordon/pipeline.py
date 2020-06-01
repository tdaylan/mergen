# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:48:56 2020

@author: Lindsey Gordon

Pipeline to produce all files for a given group of data.

Last updated: May 31 2020
"""
import feature_functions
from feature_functions import *

import numpy as np
import numpy.ma as ma 
import pandas as pd 
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
from astropy.utils import exceptions

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
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

import astroquery
from astroquery.simbad import Simbad
from astroquery.mast import Catalogs
from astroquery.mast import Observations

import shapely
from shapely import geometry
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

test(8) #should return 8 * 4
#%%
mypath = "/Users/conta/UROP_Spring_2020/"
sectorfile = "/Users/conta/UROP_Spring_2020/all_targets_S020_v1.txt"
sector = 20
for n in range(4):
    camera = n
    for i in range(4):
        ccd = i
        times, intensities, failed_to_get, targets, folder_path, features = data_process_a_group(mypath, sectorfile, sector, camera, ccd)
        
        
#%%
post_process_plotting(times, intensities, features, features, targets, "/Users/conta/UROP_Spring_2020/Sector20Cam1CCD2")
