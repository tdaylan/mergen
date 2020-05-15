# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:00:17 2020

@author: conta
"""


import numpy as np
import numpy.ma as ma 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)

from pylab import rcParams
rcParams['figure.figsize'] = 10,10
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

import shapely
from shapely import geometry
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

test(8) #should return 8 * 4


#%%

time, intensity, targets, lc_feat = get_from_files()

#%%
n_choose_2_insets(time, intensity, lc_feat, targets, "5-15")

#%%

#1st-4th moments (0-3), natural log variance (4), skew (5), kurtosis (6), 
  # power, natural log power, period of max power (0.1 to 10 days) (7-9), 
   # slope, natural log of slope (10-11)
    # integration of periodogram over: period of 0.1-10, period of 0.1-1, period of 1-3,
     #   period of 3-10 days, (12-16)
      #  period of max power for 0.01-0.1 days (for moving objects) (17)

#%%
#use to dig up header info
print_header(0)