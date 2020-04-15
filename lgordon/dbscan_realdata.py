# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:00:17 2020

@author: conta
"""


import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
rcParams["lines.markersize"] = 5
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

test(8) #should return 8 * 4

#%%

time, intensity, targets = get_data_from_fits()
intensity = normalize(intensity)


lc_feat = create_list_featvec(time, intensity)

#%%
#from https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
#DBSCAN color plots
n_choose_2_features_plotting(lc_feat, "4-13", False, True)

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

eps = np.arange(0.2, 1.5, 0.1)
min_samples = np.arange(2, 50, 2)

number_classes_each = []
number_noise_each = []
#parameter_combo = []

for n in range(len(eps)):
    eps_n = np.round(eps[n], 2)
    eps_string = str(eps_n)

    for m in range(len(min_samples)):
        samples = min_samples[m]
        samples_string = str(np.round(samples, 2))
        #parameters = (eps_n, samples)
        #parameter_combo.append(parameters)
        
        db = DBSCAN(eps=eps_n, min_samples=samples).fit(lc_features) #eps is NOT epochs
        classes_dbscan = db.labels_
        numclasses = str(len(set(classes_dbscan)))
        print("there are " + numclasses + " classes")
        number_classes_each.append(int(numclasses))
        
        avg = lc_features[:,0]
        logskew = lc_features[:,5]
        
        number_noise = 0
        for p in range(len(lc_features)):
            if classes_dbscan[p] == 0:
                color = "red"
            elif classes_dbscan[p] == -1:
                color = "black"
                number_noise = number_noise + 1
            elif classes_dbscan[p] == 1:
                color = "blue"
            elif classes_dbscan[p] == 2:
                color = "green"
            elif classes_dbscan[p] == 3:
                color = "purple"
            #plt.scatter(logskew[p], avg[p], c = color, s = 5)
            #plt.xlabel("log skew")
            #plt.ylabel("average")
            #plt.title("eps = " + eps_string + ", min_samples = " + samples_string + ", algorithm = auto. there are " + numclasses + " classes and " + str(number_noise) + "curves classed as noise") 
        #plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-13/dbscan-parameter-scan/4-13-avg-vs-log-skew-eps-" + eps_string +"-minsamples-" + samples_string +".png")
        #plt.show()
        number_noise_each.append(int(number_noise))

print(number_classes_each, number_noise_each)
#also want to know how many have been classed as noise - want to minimize that!

#%%
#need to argsort and index it
#plt.scatter(number_classes_each, number_noise_each)
index_numclass = np.argsort(np.asarray(number_classes_each))
#print(index_numclass)

num_classes_sorted = np.take_along_axis(np.asarray(number_classes_each), index_numclass, axis=0)
#print(num_classes_sorted)
num_noise_sorted = np.take_along_axis(np.asarray(number_noise_each), index_numclass, axis=0)
plt.scatter(num_classes_sorted, num_noise_sorted)
plt.xlim(0, 20)
plt.ylim(0,200)
plt.xlabel("number of classes")
plt.ylabel("number of noisy points")

#want to be able to back grab the points that give good results

#%%

#there are 13 eps values being looked at
#there are 24 min sample values being looked at. 
#total combos: 312

x = np.linspace(0,312,312)
plt.plot(x, number_classes_each)
plt.xlabel("combination number")
plt.ylabel("number of classes")
plt.show()
plt.plot(x, number_noise_each)

#%%


#%%
print(time)
#%%


