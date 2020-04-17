# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:00:17 2020

@author: conta
"""


import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
rcParams["lines.markersize"] = 2
from scipy.signal import argrelextrema

import astropy
from astropy.io import fits
import scipy.signal as signal

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

test(8) #should return 8 * 4

#%%

time, intensity, targets = get_data_from_fits()
intensity = normalize(intensity)

#%%
lc_feat = create_list_featvec(time, intensity)

#%%
#from https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
#DBSCAN color plots
n_choose_2_features_plotting(lc_feat, "4-17", "none")
#%%
plot_lof(time, intensity, targets, lc_feat, 10, "4-17")

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
#plotting different DBSCAN parameters to try and max out the distro
#avg (0) vs log skew (5)

logmaxpower = lc_feat[:,8]
logskew = lc_feat[:,5]

plt.scatter(logmaxpower, logskew)
#%%
eps = np.arange(0.2, 1.6, 0.2)
#eps2 = np.concatenate((eps, eps))

min_samples = np.arange(2,50,2)
#print(eps, min_samples)

numClasses = []
numNoise = []
parameter_list = []
colors = ["red", "blue", "green", "purple"]
for m in range(4):
    
    for n in range(len(eps)):
        eps_n = eps[n]
        samples = min_samples[m]
        parameter_list.append((int(eps_n), int(samples)))
        
        db = DBSCAN(eps=eps_n, min_samples=samples).fit(lc_feat) #eps is NOT epochs
        classes_dbscan = db.labels_
        number_of_classes = str(len(set(classes_dbscan)))
        #print("there are " + number_of_classes + " classes")
        numClasses.append(int(number_of_classes))
        #print(eps_n, samples)
        number_noise = 0

        for p in range(len(lc_feat)):
            if classes_dbscan[p]%4 == 0:
                color = "red"
            elif classes_dbscan[p] == -1:
                color = "black"
                number_noise = number_noise + 1
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
        numNoise.append(number_noise)
    
print(numClasses)
print(numNoise)
print(parameter_list)
print(parameter_list[2])


