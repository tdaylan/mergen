# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:48:56 2020

@author: Lindsey Gordon

Last updated: April 2020
"""

import numpy as np
import numpy.ma as ma 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

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

test(8) #should return 8 * 4

#if doing from scratch, run these: 
#time, intensity, targets = get_data_from_fits()
#intensity = normalize(intensity)
#lc_feat = create_list_featvec(time, intensity)

#if just running on intensity files you already have, run this:

time, intensity, targets, lc_feat = get_from_files()



#%%
n_choose_2_features_plotting(lc_feat, "5-4", "none")
n_choose_2_features_plotting(lc_feat, "5-4", "kmeans")
n_choose_2_features_plotting(lc_feat, "5-4", "dbscan")

plot_lof(time, intensity, targets, lc_feat, 10, "5-4")

#%%