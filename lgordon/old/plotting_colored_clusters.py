# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:08:47 2020

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

test(8) #should return 8 * 4
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

lc_feat = create_list_featvec(time, intensity, 13)

#%%
#1st-4th moments, natural log variance, skew, kurtosis, power, natural log power, frequency, slope, natural log of slope
Kmean = KMeans(n_clusters=4, max_iter=700, n_init = 20)

x = Kmean.fit(lc_feat)
classes_kmeans = x.labels_

print(classes_kmeans)

#coloring kmeans
for n in range(13):
    feat1 = lc_feat[:,n]
    if n == 0:
        label1 = "average"
    elif n == 1: 
        label1 = "variance"
    elif n == 2:
        label1 = "skewness"
    elif n == 3:
        label1 = "kurtosis"
    elif n == 4:
        label1 = "log_variance"
    elif n == 5:
        label1 = "log_skewness"
    elif n == 6: 
        label1 = "log_kurtosis"
    elif n == 7: 
        label1  = "Maximum_power"
    elif n == 8: 
        label1 = "log_maximum_power"
    elif n == 9: 
        label1 = "period BJD"
    elif n == 10: 
        label1 = "slope"
    elif n == 11: 
        label1 = "log_slope"
    elif n == 12:
        la
        
    for m in range(12):
        if m == n:
            break
        if m == 0:
            label2 = "average"
        elif m == 1: 
            label2 = "variance"
        elif m == 2:
            label2 = "skewness"
        elif m == 3:
            label2 = "kurtosis"
        elif m == 4:
            label2 = "log_variance"
        elif m == 5:
            label2 = "log_skewness"
        elif m == 6: 
            label2 = "log_kurtosis"
        elif m == 7: 
            label2  = "maximum_power"
        elif m == 8: 
            label2 = "log_maximum_power"
        elif m == 9: 
            label2 = "period BJD"
        elif m == 10: 
            label2 = "slope"
        elif m == 11: 
            label2 = "log_slope"
        feat2 = lc_feat[:,m]
        plt.autoscale(enable=True, axis='both', tight=True)
        
        for p in range(len(lc_feat)):
            if classes_kmeans[p] == 0:
                color = "red"
            elif classes_kmeans[p] == 1:
                color = "blue"
            elif classes_kmeans[p] == 2:
                color = "green"
            elif classes_kmeans[p] == 3:
                color = "purple"
            plt.scatter(feat1[p], feat2[p], c = color)
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-10/4-10-" + label1 + "-vs-" + label2 + "-kmeans-colored.png")
        plt.show()

#%% plotting 20 of each kmeans cluster
cluster_0 = []
cluster_1 = []
cluster_2 = []
cluster_3 = []

for n in range(len(intensity)):
    if classes_kmeans[n] == 0:
        cluster_0.append(n)
    elif classes_kmeans[n] == 1:
        cluster_1.append(n)
    elif classes_kmeans[n] ==2:
        cluster_2.append(n)
    elif classes_kmeans[n] == 3:
        cluster_3.append(n)

# subplotting
#cluster 0
fig0 = plt.figure(figsize=(20,60))
for k in range(20):
    l = cluster_0[k] #get the index of the intensity from the cluster list
    l_str = str(l)
    plot_index = k + 1
    fig0.title("Cluster 0 Examples")
    ax1 = fig0.add_subplot(10,2, plot_index)
    ax1.scatter(time, intensity[l], c="red")
    ax1.set_xlabel("Time [BJD -2457000]")
    ax1.set_ylabel("Relative Flux")
    ax1.set_title(targets[l])
    plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-10/kmeans_examples/4-10-kmeans-cluster-0-examples.png")

#cluster 1   
fig1 = plt.figure(figsize=(20,60))
for k in range(20):
    l = cluster_1[k] #get the index of the intensity from the cluster list
    l_str = str(l)
    plot_index = k + 1
    ax1 = fig1.add_subplot(10,2, plot_index)
    ax1.scatter(time, intensity[l], c="blue")
    ax1.set_xlabel("Time [BJD -2457000]")
    ax1.set_ylabel("Relative Flux")
    ax1.set_title(targets[l])
    plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-10/kmeans_examples/4-10-kmeans-cluster-1-examples.png")
#cluster 2
fig2 = plt.figure(figsize=(20,60))
for k in range(20):
    l = cluster_2[k] #get the index of the intensity from the cluster list
    l_str = str(l)
    plot_index = k + 1
    ax1 = fig2.add_subplot(10,2, plot_index)
    ax1.scatter(time, intensity[l], c="green")
    ax1.set_xlabel("Time [BJD -2457000]")
    ax1.set_ylabel("Relative Flux")
    ax1.set_title(targets[l]) 
    plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-10/kmeans_examples/4-10-kmeans-cluster-2-examples.png")

#cluster 3
fig3 = plt.figure(figsize=(20,60))
for k in range(20):
    l = cluster_3[k] #get the index of the intensity from the cluster list
    l_str = str(l)
    plot_index = k + 1
    ax1 = fig3.add_subplot(10,2, plot_index)
    ax1.scatter(time, intensity[l], c="purple")
    ax1.set_xlabel("Time [BJD -2457000]")
    ax1.set_ylabel("Relative Flux")
    ax1.set_title(targets[l])
    plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-10/kmeans_examples/4-10-kmeans-cluster-3-examples.png")
    

#%%
#DBSCAN color plots
n_choose_2_features_plotting(lc_feat, "4-13", False, True)

#%% plotting 20 dbscan examples from each class
cluster_0_db = []
cluster_1_db = []
cluster_2_db = []
cluster_noise_db = []

for n in range(len(intensity)):
    if classes_dbscan[n] == 0:
        cluster_0_db.append(n)
    elif classes_dbscan[n] == 1:
        cluster_1_db.append(n)
    elif classes_dbscan[n] ==2:
        cluster_2_db.append(n)
    elif classes_dbscan[n] == -1:
        cluster_noise_db.append(n)

#cluster 0
if len(cluster_0_db) < 20: 
    p = len(cluster_0_db)
else:
    p = 20
    
height = p*5/2
fig_0 = plt.figure(figsize=(24,height)) #this must be a multiple of 8x3
fig_0.suptitle("DBSCAN Cluster 0", fontsize=30)
fig_0.tight_layout()
fig_0.subplots_adjust(top=0.93)
for k in range(p):
    l = cluster_0_db[k] #get the index of the intensity from the cluster list
    l_str = str(l)
    plot_index = k + 1
    half_p = p/2
    ax1 = fig_0.add_subplot(half_p,2, plot_index)
    ax1.scatter(time, intensity[l], c="red")
    ax1.set_xlabel("Time [BJD -2457000]")
    ax1.set_ylabel("Relative Flux")
    ax1.set_title(targets[l])    
    plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-13/dbscan-examples/4-13-dbscan-cluster-0.png")

#cluster 1 
if len(cluster_1_db) < 20: 
    p = len(cluster_1_db)
else:
    p = 20

height = p*5/2
fig_1 = plt.figure(figsize=(24,height)) #this must be a multiple of 8x3
fig_1.suptitle("DBSCAN Cluster 1", fontsize=30)
fig_1.tight_layout()
fig_1.subplots_adjust(top=0.93)
for k in range(p):
    l = cluster_1_db[k] #get the index of the intensity from the cluster list
    l_str = str(l)
    plot_index = k + 1
    half_p = p/2
    ax1 = fig_1.add_subplot(half_p,2, plot_index)
    ax1.scatter(time, intensity[l], c="blue")
    ax1.set_xlabel("Time [BJD -2457000]")
    ax1.set_ylabel("Relative Flux")
    ax1.set_title(targets[l])    
    plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-13/dbscan-examples/4-13-dbscan-cluster-1.png")

#cluster 2
if len(cluster_2_db) < 20: 
    p = len(cluster_2_db)
else:
    p = 20
    
height = p*5/2
fig_2 = plt.figure(figsize=(24,height)) #this must be a multiple of 8x3
fig_2.suptitle("DBSCAN Cluster 2", fontsize=30)
fig_2.tight_layout()
fig_2.subplots_adjust(top=0.93)
for k in range(p):
    l = cluster_2_db[k] #get the index of the intensity from the cluster list
    l_str = str(l)
    plot_index = k + 1
    
    half_p = p/2
    ax1 = fig_2.add_subplot(half_p,2, plot_index)
    ax1.scatter(time, intensity[l], c="green")
    ax1.set_xlabel("Time [BJD -2457000]")
    ax1.set_ylabel("Relative Flux")
    ax1.set_title(targets[l])    
    plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-13/dbscan-examples/4-13-dbscan-cluster-2.png")

#cluster noise
if len(cluster_noise_db) < 20: 
    p = len(cluster_noise_db)
else:
    p = 20
    
height = p*5/2
fig_noise = plt.figure(figsize=(24,height)) #this must be a multiple of 8x3
fig_noise.suptitle("DBSCAN ID'd As Noise", fontsize=30)
fig_noise.tight_layout()
fig_noise.subplots_adjust(top=0.93)
for k in range(p):
    l = cluster_noise_db[k] #get the index of the intensity from the cluster list
    l_str = str(l)
    plot_index = k + 1
    half_p = p/2
    ax1 = fig_noise.add_subplot(half_p,2, plot_index)
    ax1.scatter(time, intensity[l], c="black")
    ax1.set_xlabel("Time [BJD -2457000]")
    ax1.set_ylabel("Relative Flux")
    ax1.set_title(targets[l])    
    plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-13/dbscan-examples/4-13-dbscan-cluster-noise.png")
    

    
    
    