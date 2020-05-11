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

import shapely
from shapely import geometry
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

test(8) #should return 8 * 4


#%%

time, intensity, targets = get_data_from_fits()

#%%
intensity = normalize(intensity)

lc_feat = create_list_featvec(time, intensity)

n_choose_2_features_plotting(lc_feat, "4-29", "none")
#%%
plot_lof(time, intensity, targets, lc_feat, 10, "4-29")

#%%

#1st-4th moments (0-3), natural log variance (4), skew (5), kurtosis (6), 
  # power, natural log power, period of max power (0.1 to 10 days) (7-9), 
   # slope, natural log of slope (10-11)
    # integration of periodogram over: period of 0.1-10, period of 0.1-1, period of 1-3,
     #   period of 3-10 days, (12-16)
      #  period of max power for 0.01-0.1 days (for moving objects) (17)

#%%
#use to dig up header info
print_header(266)

#%%
all_outliers = []
#period of 0.1-1 (integrated) vs log of max power

#five largest points (outliers) are colored separately
period_01_1_outliers = np.argsort(lc_feat[:,13])[-5:]

plt.scatter(lc_feat[:,13][period_01_1_outliers], lc_feat[:,8][period_01_1_outliers])
plt.show()

for i in range(len(period_01_1_outliers)):
    all_outliers.append(int(period_01_1_outliers[i]))

#for log of max power outliers
logmaxpoweroutlier = np.argmax(lc_feat[:,8])
print(logmaxpoweroutlier)
plt.scatter(lc_feat[:,13],lc_feat[:,8]) 
plt.scatter(lc_feat[:,13][logmaxpoweroutlier], lc_feat[:,8][logmaxpoweroutlier])
plt.show()


all_outliers.append(logmaxpoweroutlier)


#removing average outliers 
average_max = np.argsort(lc_feat[:,0])[-4:]
average_min = np.argsort(lc_feat[:,0])[:4]

print(average_max, average_min)

for i in range(4):
    all_outliers.append(average_max[i])
for i in range(4):
    all_outliers.append(average_min[i])

outliers = np.asarray(all_outliers)
print(outliers, type(outliers))
outliers = np.unique(outliers)
print(outliers)

#plot all the outliers

for i in range(len(outliers)):
    plt.scatter(time, intensity[outliers[i]])
    plt.title(targets[outliers[i]])
    plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-29/feature-outliers/4-29" + targets[outliers[i]] +".png")
    plt.show()


featvec_reduced = np.delete(lc_feat, outliers, 0)   
print(len(featvec_reduced), len(lc_feat))





#%%

#%%
#trying to do parameter scan of dbscan
eps = np.arange(0.1, 5, 0.1)
#eps2 = np.concatenate((eps, eps))

min_samples = np.arange(2,60,1)
#print(eps, min_samples)

numClasses = []
numNoise = []
parameter_list = []
colors = ["red", "blue", "green", "purple"]
for m in range(len(min_samples)):
    
    for n in range(len(eps)):
        eps_n = eps[n]
        samples = min_samples[m]
        parameter_list.append((np.round(eps_n, 2), samples))
        
        db = DBSCAN(eps=eps_n, min_samples=samples).fit(lc_feat) #eps is NOT epochs
        classes_dbscan = db.labels_
        number_of_classes = str(len(set(classes_dbscan)))
        #print("there are " + number_of_classes + " classes")
        numClasses.append(int(number_of_classes))
        #print(eps_n, samples)
        number_noise = 0

        for p in range(len(lc_feat)):
            if classes_dbscan[p] == -1:
                number_noise = number_noise + 1

        numNoise.append(number_noise)
    
#print(numClasses)
#print(numNoise)
#print(parameter_list)
#print(parameter_list[2])
plotting = False
for k in range(len(numClasses)):
    if numClasses[k] > 1 and plotting == True:
        print(parameter_list[k], numClasses[k], numNoise[k])
        db = DBSCAN(eps=eps_n, min_samples=samples).fit(lc_feat) #eps is NOT epochs
        classes_dbscan = db.labels_
        number_of_classes = str(len(set(classes_dbscan)))
        for p in range(len(lc_feat)):
            if classes_dbscan[p]%4 == 0:
                color = "red"
            elif classes_dbscan[p] == -1:
                color = "black"
            elif classes_dbscan[p]%4 == 1:
                color = "blue"
            elif classes_dbscan[p]%4 == 2:
                color = "green"
            elif classes_dbscan[p]%4 == 3:
                color = "purple"
            plt.scatter(logmaxpower[p], logskew[p], c = color, s = 2)
            plt.xlabel("log max power")
            plt.ylabel("log skew")
        plt.show()
    elif numClasses[k] > 1:
        print(parameter_list[k], numClasses[k], numNoise[k])
#%%



#print(T_eff, obj_type, radius, mass, T_mag)


#%%
        
def check_box_location(y_max_tuple, range_x, range_y):
    """ checks if data points lie within the area of the inset plot"""
    inset_x = y_max_tuple[0] + 0.3 * range_x
    inset_y = y_max_tuple[1] - 0.3 * range_y
    inset_width = range_x * 2
    inset_height = range_y /5 
    
    inset_BL = (inset_x, inset_y)
    inset_BR = (inset_x + inset_width, inset_y)
    inset_TL = (inset_x, inset_y + inset_height)
    inset_TR = (inset_x + inset_width, inset_y + inset_height)
    
    conc = np.column_stack((lc_feat[:,13], lc_feat[:,8]))
    polygon = Polygon([inset_BL, inset_BR, inset_TL, inset_TR])
    
    i = 0
    n = len(conc)
    
    while i < n:
        point = Point(conc[i])
        if polygon.contains(point) == True:
            inset_x += 0.01 * range_x
            inset_y += 0.01 * range_y
            i = 0
        elif polygon.contains(point) == False:
            i = i + 1
    return inset_x, inset_y, inset_width, inset_height

fig, ax1 = plt.subplots()
ax1.scatter(lc_feat[:,13], lc_feat[:,8], c = "black")
ax1.set_xlabel("P1")
ax1.set_ylabel("ln max power")

y_max_index = np.argmax(lc_feat[:,8])
targ_y_max = targets[y_max_index]

catalog_data = Catalogs.query_object(targ_y_max, radius=0.02, catalog="TIC")
#https://arxiv.org/pdf/1905.10694.pdf
T_eff = catalog_data[0]["Teff"]
obj_type = catalog_data[0]["objType"]
gaia_mag = catalog_data[0]["GAIAmag"]
radius = catalog_data[0]["rad"]
mass = catalog_data[0]["mass"]
distance = catalog_data[0]["d"]


y_max_tuple = (lc_feat[:,13][y_max_index], lc_feat[:,8][y_max_index])
range_x = lc_feat[:,13].max() - lc_feat[:,13].min()
range_y = lc_feat[:,8].max() - lc_feat[:,8].min()
inset_x, inset_y, inset_width, inset_height = check_box_location(y_max_tuple, range_x, range_y)


#x pos, y pos, width, height
axins1 = ax1.inset_axes([inset_x, inset_y, inset_width, inset_height], transform = ax1.transData)
axins1.scatter(time, intensity[y_max_index], c='black', s = 0.01)

x1, x2, y1, y2 =  lc_feat[y_max_index][13], lc_feat[y_max_index][13] + 0.00001, lc_feat[y_max_index][8], lc_feat[y_max_index][8] + 0.001
axins1.set_xlim(x1, x2)
axins1.set_ylim(y1, y2)
ax1.indicate_inset_zoom(axins1)

axins1.set_xlim(time[0], time[-1])
axins1.set_ylim(intensity[y_max_index].min(), intensity[y_max_index].max())
axins1.set_xlabel("BJD [2457000]")
axins1.set_ylabel("relative flux")
axins1.set_title(targets[y_max_index] + "\nT_eff:" + str(T_eff) + ", ObjType: " + str(obj_type) + ", GAIA mag: " + str(gaia_mag) + "\n Dist: " + str(distance) + ", Radius:" + str(radius) + " Mass:" + str(mass), fontsize=8)

#second inset:

y_min_index = np.argmin(lc_feat[:,8])
targ_y_min = targets[y_min_index]
y_min_tuple = (lc_feat[:,13][y_min_index], lc_feat[:,8][y_min_index])
range_x = lc_feat[:,13].max() - lc_feat[:,13].min()
range_y = lc_feat[:,8].max() - lc_feat[:,8].min()

catalog_data1 = Catalogs.query_object(targ_y_min, radius=0.02, catalog="TIC")

T_eff = catalog_data1[0]["Teff"]
obj_type = catalog_data1[0]["objType"]
gaia_mag = catalog_data1[0]["GAIAmag"]
radius = catalog_data1[0]["rad"]
mass = catalog_data1[0]["mass"]
distance = catalog_data1[0]["d"]

inset_x, inset_y, inset_width, inset_height = check_box_location(y_min_tuple, range_x, range_y)


#x pos, y pos, width, height
axins2 = ax1.inset_axes([inset_x, inset_y, inset_width, inset_height], transform = ax1.transData)
axins2.scatter(time, intensity[y_min_index], c='black', s = 0.01)

x1, x2, y1, y2 =  lc_feat[y_min_index][13], lc_feat[y_min_index][13] + 0.00001, lc_feat[y_min_index][8], lc_feat[y_min_index][8] + 0.001
axins2.set_xlim(x1, x2)
axins2.set_ylim(y1, y2)
ax1.indicate_inset_zoom(axins2)

axins2.set_xlim(time[0], time[-1])
axins2.set_ylim(intensity[y_max_index].min(), intensity[y_max_index].max())
axins2.set_xlabel("BJD [2457000]")
axins2.set_ylabel("relative flux")
axins2.set_title(targets[y_min_index] + "\nT_eff:" + str(T_eff) + ", ObjType: " + str(obj_type) + ", GAIA mag: " + str(gaia_mag) + "\n Dist: " + str(distance) + ", Radius:" + str(radius) + " Mass:" + str(mass), fontsize=8)


plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/5-8/inset-plot-shapely-usage1.png")

plt.show()



