# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:00:17 2020

@author: conta
"""


import numpy as np
import numpy.ma as ma 
import matplotlib.pyplot as plt
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

test(8) #should return 8 * 4


#%%

time, intensity, targets = get_data_from_fits()

#%%
intensity = normalize(intensity)

lc_feat = create_list_featvec(time, intensity)

n_choose_2_features_plotting(lc_feat, "4-29", "none")
#%%
plot_lof(time, intensity, targets, lc_feat, 10, "4-20")


#%%

#1st-4th moments (0-3), natural log variance (4), skew (5), kurtosis (6), 
  # power, natural log power, period of max power (0.1 to 10 days) (7-9), 
   # slope, natural log of slope (10-11)
    # integration of periodogram over: period of 0.1-10, period of 0.1-1, period of 1-3,
     #   period of 3-10 days, (12-16)
      #  period of max power for 0.01-0.1 days (for moving objects) (17)

#%%
#use to dig up header
print_header(266)

#%%
all_outliers = []
#period of 0.1-1 (integrated) vs log of max power
plt.scatter(lc_feat[:,13],lc_feat[:,8]) 

#five largest points (outliers) are colored separately
period_01_1_outliers = np.argsort(lc_feat[:,13])[-5:]

plt.scatter(lc_feat[:,13][period_01_1_outliers], lc_feat[:,8][period_01_1_outliers])
plt.show()

for i in range(len(period_01_1_outliers)):
    print(period_01_1_outliers[i])
    plt.scatter(time, intensity[period_01_1_outliers[i]])
    plt.title(targets[period_01_1_outliers[i]])
    #plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-29/" + targets[period_01_1_outliers[i]] + "-lc.png")
    plt.show()
    all_outliers.append(int(period_01_1_outliers[i]))

#for log of max power outliers
logmaxpoweroutlier = np.argmax(lc_feat[:,8])
print(logmaxpoweroutlier)
plt.scatter(lc_feat[:,13],lc_feat[:,8]) 
plt.scatter(lc_feat[:,13][logmaxpoweroutlier], lc_feat[:,8][logmaxpoweroutlier])
plt.show()

plt.scatter(time, intensity[logmaxpoweroutlier])
plt.title(targets[logmaxpoweroutlier])
#plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-29/" + targets[logmaxpoweroutlier] + "-lc.png")
plt.show()

all_outliers.append(logmaxpoweroutlier)


outliers = np.asarray(all_outliers)
print(outliers, type(outliers))
outliers = np.unique(outliers)
print(outliers)

featvec_reduced = np.delete(lc_feat, outliers, 0)   
print(len(featvec_reduced), len(lc_feat))

plt.scatter(featvec_reduced[:,13],featvec_reduced[:,8]) 



#%%

#so now to poke at other plots:
plt.autoscale(enable=True, axis='both', tight=True)
plt.scatter(featvec_reduced[:,13],featvec_reduced[:,10])
plt.tight_layout()
plt.show()


#%%
#run on all of the features & producing the confusion matrix
#this only works on the first hundred though

predict_on_100 = lc_features[0:100]
db_100 = DBSCAN(eps=100, min_samples=10).fit(predict_on_100)
predicted_100 = db_100.labels_

#producing the confusion matrix
labelled_100 = np.loadtxt("/Users/conta/UROP_Spring_2020/100-labelled/labelled_100.txt", delimiter=',', usecols=1, skiprows=1, unpack=True)
print(predicted_100, labelled_100)

k = confusion_matrix(labelled_100, predicted_100)

print(k)

check_diagonalized(k)

#%%
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

from astroquery.mast import Catalogs

target = "TIC 4132133"

catalog_data = Catalogs.query_object(target, radius=0.02, catalog="TIC")

#https://arxiv.org/pdf/1905.10694.pdf
T_eff = catalog_data[0]["Teff"]
B_mag = catalog_data[0]["Bmag"]
V_mag = catalog_data[0]["Vmag"]
obj_type = catalog_data[0]["objType"]
gaia_mag = catalog_data[0]["GAIAmag"]
radius = catalog_data[0]["rad"]
mass = catalog_data[0]["mass"]
distance = catalog_data[0]["d"]
T_mag = catalog_data[0]["Tmag"]
luminosity = catalog_data[0]["lum"]

#%%
fig, axs = plt.subplots(2, 2, sharex = True, figsize = (8,3), constrained_layout=False)
fig.subplots_adjust(hspace=0)
axs[0,0].scatter(time, intensity[0], s=2, label=targets[0])
axs[0,0].legend(loc="upper left")

axs[0,1].scatter(time, intensity[0], s = 2, color = 'white', label="Teff=" + str(T_eff) + 
   "\n object type=" + str(obj_type) +  
   "\n mass= " + str(mass) +
   "\n gaia magnitude=" + str(gaia_mag))
axs[0,1].legend(loc="upper left")
axs[0,1].get_xaxis().set_visible(False)
axs[0,1].get_yaxis().set_visible(False)
#plt.text(1,1, 'text')
#axs[0].text(0, 0, 'text')

plt.show()
#%%
    clf = LocalOutlierFactor(n_neighbors=2)
    n = 10
    fit_predictor = clf.fit_predict(lc_feat)
    negative_factor = clf.negative_outlier_factor_
    
    lof = -1 * negative_factor
    ranked = np.argsort(lof)
    largest_indices = ranked[::-1][:n]
    smallest_indices = ranked[:n]

    #plot just the largest indices
    #rows, columns
    fig, axs = plt.subplots(n, 2, sharex = True, figsize = (16,n*3), constrained_layout=False)
    fig.subplots_adjust(hspace=0)
    
    targets_search = []
    for k in range(n):
        ind = largest_indices[k]
        axs[k,0].plot(time, intensity[ind], '.k', label=targets[ind])
        targets_search.append(targets[ind])
        axs[k,0].legend(loc="upper left")
        axs[k,0].set_ylabel("relative flux")
        axs[-1,0].set_xlabel("BJD [-2457000]")
    fig.suptitle(str(n) + ' largest LOF targets', fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.96)
    
    T_effs = []
    for i in range(len(targets_search)):
        targ = targets_search[i]
        catalog_data = Catalogs.query_object(targ, radius=0.02, catalog="TIC")[0]
        T_eff = catalog_data["Teff"]
        T_effs.append(T_eff)
    print(T_effs)
    for i in range(n):
        axs[i,1].scatter(time, intensity[0], s = 2, color = 'white', label="Teff=" + str(T_effs[i]))
        axs[i,1].legend(loc="upper left")
        axs[i,1].get_xaxis().set_visible(False)
        axs[i,1].get_yaxis().set_visible(False)

plt.show()
#%%
T_effs = []
for i in range(len(targets_search)):
    targ = targets_search[i]
    catalog_data = Catalogs.query_object(targ, radius=0.02, catalog="TIC")[0]
    T_eff = catalog_data["Teff"]
    T_effs.append(T_eff)
print(T_effs)

#%%
import astropy.coordinates as coord
result_table = Simbad.query_region(coord.SkyCoord(coords, unit="deg"))
print(result_table[0])


