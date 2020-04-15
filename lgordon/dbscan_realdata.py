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

#importing data (uses emma's code)
fitspath = '/Users/conta/UROP_Spring_2020/tessdata_lc_sector20_1000/'
fnames_all = os.listdir(fitspath)
fnames = fnmatch.filter(fnames_all, '*fits*')

interp_tol = 20. / (24*60) # >> interpolate small gaps (less than 20 minutes)

intensity = []
targets = []
for file in fnames:
    # -- open file -------------------------------------------------------------
    f = fits.open(fitspath + file)

    # >> get data
    time = f[1].data['TIME']
    i = f[1].data['PDCSAP_FLUX']
    tic = f[1].header["OBJECT"]
    targets.append(tic)
    # -- find small nan gaps ---------------------------------------------------
    # >> adapted from https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    # >> find run starts
    n = np.shape(i)[0]
    loc_run_start = np.empty(n, dtype=bool)
    loc_run_start[0] = True
    np.not_equal(np.isnan(i)[:-1], np.isnan(i)[1:], out=loc_run_start[1:])
    run_starts = np.nonzero(loc_run_start)[0]

    # >> find run lengths
    run_lengths = np.diff(np.append(run_starts, n))

    tdim = time[1] - time[0]
    interp_inds = run_starts[np.nonzero((run_lengths * tdim <= interp_tol) * \
                                        np.isnan(i[run_starts]))]
    interp_lens = run_lengths[np.nonzero((run_lengths * tdim <= interp_tol) * \
                                         np.isnan(i[run_starts]))]

    # -- interpolation ---------------------------------------------------------
    # >> interpolate small gaps
    i_interp = np.copy(i)
    for a in range(np.shape(interp_inds)[0]):
        start_ind = interp_inds[a]
        end_ind = interp_inds[a] + interp_lens[a]
        i_interp[start_ind:end_ind] = np.interp(time[start_ind:end_ind],
                                                time[np.nonzero(~np.isnan(i))],
                                                i[np.nonzero(~np.isnan(i))])
    intensity.append(i_interp)

# -- remove orbit nan gap ------------------------------------------------------
intensity = np.array(intensity)
# nan_inds = np.nonzero(np.prod(np.isnan(intensity)==False), axis = 0))
nan_inds = np.nonzero(np.prod(np.isnan(intensity)==False, axis = 0) == False)
time = np.delete(time, nan_inds)
intensity = np.delete(intensity, nan_inds, 1) #each row of intensity is one interpolated light curve.

intensity = normalize(intensity)

lc_features = create_list_featvec(time, intensity, 12)

#%%
#from https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
#DBSCAN color plots
db = DBSCAN(eps=0.5, min_samples=10).fit(lc_features) #eps is NOT epochs
classes_dbscan = db.labels_

print(classes_dbscan)
numclasses = str(len(set(classes_dbscan)))
print("there are " + numclasses + "classes")

#coloring kmeans
for n in range(12):
    feat1 = lc_features[:,n]
    if n == 0:
        graph_label1 = "Average"
        fname_label1 = "Avg"
    elif n == 1: 
        graph_label1 = "Variance"
        fname_label1 = "Var"
    elif n == 2:
        graph_label1 = "Skewness"
        fname_label1 = "Skew"
    elif n == 3:
        graph_label1 = "Kurtosis"
        fname_label1 = "Kurt"
    elif n == 4:
        graph_label1 = "Log Variance"
        fname_label1 = "LogVar"
    elif n == 5:
        graph_label1 = "Log Skewness"
        fname_label1 = "LogSkew"
    elif n == 6: 
        graph_label1 = "Log Kurtosis"
        fname_label1 = "LogKurt"
    elif n == 7: 
        graph_label1 = "Maximum Power"
        fname_label1 = "MaxPower"
    elif n == 8: 
        graph_label1 = "Log Maximum Power"
        fname_label1 = "LogMaxPower"
    elif n == 9: 
        graph_label1 = "Period [BJD - 2457000]"
        fname_label1 = "Period"
    elif n == 10: 
        graph_label1 = "Slope"
        fname_label1 = "Slope"
    elif n == 11: 
        graph_label1 = "Log Slope"
        fname_label1 = "LogSlope"
            
    for m in range(12):
        if m == n:
            break
        if m == 0:
            graph_label2 = "Average"
            fname_label2 = "Avg"
        elif m == 1: 
            graph_label2 = "Variance"
            fname_label2 = "Var"
        elif m == 2:
            graph_label2 = "Skewness"
            fname_label2 = "Skew"
        elif m == 3:
            graph_label2 = "Kurtosis"
            fname_label2 = "Kurt"
        elif m == 4:
            graph_label2 = "Log Variance"
            fname_label2 = "LogVar"
        elif m == 5:
            graph_label2 = "Log Skewness"
            fname_label2 = "LogSkew"
        elif m == 6: 
            graph_label2 = "Log Kurtosis"
            fname_label2 = "LogKurt"
        elif m == 7: 
            graph_label2 = "Maximum Power"
            fname_label2 = "MaxPower"
        elif m == 8: 
            graph_label2 = "Log Maximum Power"
            fname_label2 = "LogMaxPower"
        elif m == 9: 
            graph_label2 = "Period [BJD - 2457000]"
            fname_label2 = "Period"
        elif m == 10:
            graph_label2 = "Slope"
            fname_label2 = "Slope"
        elif m == 11: 
            graph_label2 = "Log Slope"
            fname_label2 = "LogSlope"
        feat2 = lc_features[:,m]
        plt.autoscale(enable=True, axis='both', tight=True)
        for p in range(len(lc_features)):
            if classes_dbscan[p] == 0:
                color = "red"
            elif classes_dbscan[p] == -1:
                color = "black"
            elif classes_dbscan[p] == 1:
                color = "blue"
            elif classes_dbscan[p] == 2:
                color = "green"
            elif classes_dbscan[p] == 3:
                color = "purple"
            plt.scatter(feat1[p], feat2[p], c = color, s = 5)
        plt.xlabel(graph_label1)
        plt.ylabel(graph_label2)
        plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-13/dbscan-colored/4-13-" + fname_label1 + "-vs-" + fname_label2 + "-dbscan-colored.png")
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
n_choose_2_features_plotting(lc_feat, "4-13", False, True)