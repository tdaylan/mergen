# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 08:39:09 2020

@author: conta
"""

#3-24 running kmeans on real data
#you have to load feature_functions python module to get it to run!

import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
rcParams["lines.markersize"] = 5

import astropy
from astropy.io import fits
import scipy.signal as signal

from datetime import datetime
import os
from scipy.stats import moment

import sklearn
from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import fnmatch

from sklearn.metrics import confusion_matrix
import feature_functions
from feature_functions import *

test(8) #should return 8 * 14

#%%
# emma's interpolation code

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
#%%
lc_feat = create_list_featvec(time, intensity[0:100], 12)

Kmean = KMeans(n_clusters=4, max_iter=700, n_init = 20)
#%%
#n choose 2 plotting of features + features w/ cluster centers
for n in range(7):
    feat1 = lc_feat[:,n]
    if n == 0:
        label1 = "average"
    elif n == 1: 
        label1 = "ln variance"
    elif n == 2:
        label1 = "ln skew"
    elif n == 3:
        label1 = "ln kurtosis"
    elif n == 4:
        label1 = "ln power"
    elif n == 5:
        label1 = "frequency"
    elif n==6:
        label1 = "ln slope"
    for m in range(7):
        if m == n:
            break
        if m == 0:
            label2 = "average"
        elif m == 1: 
            label2 = "ln variance"
        elif m == 2:
            label2 = "ln skew"
        elif m == 3:
            label2 = "ln kurtosis"
        elif m == 4:
            label = "ln power"
        elif m == 5:
            label2 = "frequency"
        elif m ==6:
            label2 = "ln slope"
        feat2 = lc_feat[:,m]
        plt.scatter(feat1, feat2, c="blue", s = 5)
        plt.xlabel(label1)
        plt.ylabel(label2)
        #plt.title(label1 + " vs " + label2)
       # plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-1/4-1_" + label1 + "_vs_" + label2 + "_realdata.png")
        plt.show()
        
        combined_features = np.column_stack((feat1, feat2))

        Kmean.fit(combined_features)

        centers = Kmean.cluster_centers_
        center1 = centers[0,:]
        center2 = centers[1,:]
        center3 = centers[2,:]
        center4 = centers[3,:]
        #print(center1, center2)
        
        plt.scatter(feat1, feat2, c="blue", s = 5)
        plt.scatter(center1[0], center1[1], s=200, c='g')
        plt.scatter(center2[0], center2[1], s=200, c='r')
        plt.scatter(center3[0], center3[1], s=200, c='y')
        plt.scatter(center4[0], center4[1], s=200, c='purple')
        #plt.title(label1 + " vs " + label2)
        plt.xlabel(label1)
        plt.ylabel(label2)
        #plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-1/4-1_" + label1 + "_vs_" + label2 + "_clustered_realdata.png")
        plt.show()

#%%

#%%
Kmean.fit(lc_feat)
cluster_centers = Kmean.cluster_centers_
#this sorts the coordinates for each cluster
center0 = cluster_centers[0,:]
center1 = cluster_centers[1,:]
center2 = cluster_centers[2,:]
center3 = cluster_centers[3,:]
#this sorts the coordinates for each feature
feature_center0 = cluster_centers[:,0]
feature_center1 = cluster_centers[:,1]
feature_center2 = cluster_centers[:,2]
feature_center3 = cluster_centers[:,3]
feature_center4 = cluster_centers[:,4]
feature_center5 = cluster_centers[:,5]
feature_center6 = cluster_centers[:,6]
#%%
predict_on_100 = lc_feat[0:100]
#print(predict_on_100)
predicted_100 = Kmean.predict(predict_on_100)
print(predicted_100)

#producing the confusion matrix
labelled_100 = np.loadtxt("/Users/conta/UROP_Spring_2020/100-labelled/labelled_100.txt", delimiter=',', usecols=1, skiprows=1, unpack=True)
print(labelled_100)


z = confusion_matrix(labelled_100, predicted_100)
print(z)

check_diagonalized(z)

#horizontal axis is predicted. vertical axis is actual. 
#row 0 is for all the actual zeros: 55 predicted as 0, 0 predicted as 1/2/3.
#row 1 is for all actual 1s. 21 predicted as 0, 0 predicted as 1/2/3
#row 2 is for all actual 2s. 22 predicted as 0, 0 as 1/2/3
#row 3 is for all actual 3s. 2 predicted as 0, 0 as 1/2/3

#%%

x = Kmean.fit(lc_feat)
classes = x.labels_
#plotting the indexes vs what feature it returns as
k = np.linspace(0, len(classes), len(classes))
plt.scatter(k,classes)
plt.title("Graphing all of the labels for Kmeans on real data")
plt.xlabel("index number of the data vector")
plt.ylabel("classification given by kmeans")
#plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/3-27/3-27-class_results.png")
#%%
#plotting the one it identifies as outliers
identified = []
i = 0
while i < len(classes):
    if classes[i] > 0:
        identified.append(i)
    i = i + 1
    
print(identified)
for j in identified:
    plt.plot(time, intensity[j], '.')
    class_string = str(classes[j])
    j_string = str(j)
    plt.title(targets[j] + "kmeans classed as: " + class_string)
    #plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/3-27/3-27-" + j_string + "-plotted.png")
    plt.show()
    
#%%
freq = lc_feat[:,9]
ln_variance = lc_feat[:,4]

plt.scatter(ln_variance, freq)
plt.xlabel("ln variance")
plt.ylabel("freq")
plt.title("ln variance vs freq. for 1st 100")
plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-6/4-6-variance-frequency-first100.png")