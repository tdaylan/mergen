# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:01:26 2020

@author: Lindsey Gordon 

Updated: May 31 2020
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

import shapely
from shapely import geometry
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import classification_functions
from classification_functions import *
import data_functions
from data_functions import *
import plotting_functions
from plotting_functions import *


test_data() #should return 8 * 4
test_plotting()
#%%

t, inty, targ, feats, notes = load_in_a_group(20,1,1,"/Users/conta/UROP_Spring_2020/")
  
classifications = np.loadtxt("/Users/conta/UROP_Spring_2020/Sector20Cam1CCD1/classified_Sector20Cam1CCD1.txt", delimiter = ' ')
classes = classifications[:,1]


#%%
filedb = "/Users/conta/UROP_Spring_2020/plot_output/6-5/dbscan-confusion-matrices-scan.txt"
hand_classes = classifications[:,1] #there are no class 5s for this group!!

dbscan_param_scan(filedb, feats, 0.2, 3, 0.2, 2, 50, 4, hand_classes)
#%%
#color coded plotting of eps/min samples
#plotting for min samples
p = 0
for n in range(len(accuracies)):
    k = n % 14 #tells you what eps value you're on
    if 0 <= k <= 3:
        color = 'red'
    elif 3 < k <=6:
        color = 'pink'
    elif 6 < k <= 9:
        color = 'green'
    elif 9 < k <= 13:
        color = 'blue'
    else:
        color = 'black'
    
    if n == 0:
        plt.scatter(min_samps[0], avg_precision[n], c = color)
    else:
        if n % 14 == 0:
            p = p + 1
        
        plt.scatter(min_samps[p], avg_precision[n], c = color)

plt.xlabel("min samples")
plt.ylabel("avg precision")
plt.title("red: eps 0.2-0.8, pink 1-1.4, green 1.6-2, blue 2.2-2.8")

plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/6-1/dbscan-paramscan-avg-precision-samples-colored.pdf")

#%%
#plotting for eps value ranges

for n in range(len(accuracies)):
    k = n % 14 #what eps value is it
    
    if n <= 70: 
        color = 'red'
    elif 70 <n<= 140:
        color = 'pink'
    elif 140 < n <= 210:
        color = 'green'
    elif 210 < n <= 280:
        color = 'blue'
    elif n > 280:
        color = 'purple'
    
    #color = 'blue'
    plt.scatter(eps_values[k], avg_precision[n], c = color)

plt.xlabel("eps value")
plt.ylabel('avg precision')
plt.title("sample range by color: red: 2-18, pink 22-38, green 42-58, blue 62-78, purple 82-98")

plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/6-1/dbscan-paramscan-avg-precision-eps-colored.pdf")



#%%
#plotting to indicate both actual class and the class predicted by dbscan
#eps 2.2, min samples 18
path = "/Users/conta/UROP_Spring_2020/plot_output/6-5/color-shape-2D-plots-kmeans"
hand_classes = classifications[:,1] #there are no class 5s for this group!!


        
features_2D_colorshape(feats, path, 'kmeans', hand_classes)

        