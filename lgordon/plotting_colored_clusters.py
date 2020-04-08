# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:08:47 2020

@author: conta
"""


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

#%%
lc_feat = create_list_featvec(time, intensity, 12)

#%%
#1st-4th moments, natural log variance, skew, kurtosis, power, natural log power, frequency, slope, natural log of slope
Kmean = KMeans(n_clusters=4, max_iter=700, n_init = 20)

x = Kmean.fit(lc_feat)
classes_kmeans = x.labels_

print(classes_kmeans)

#coloring kmeans
for n in range(12):
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
        label1 = "frequency"
    elif n == 10: 
        label1 = "slope"
    elif n == 11: 
        label1 = "log_slope"
        
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
            label2 = "frequency"
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
                color = "yellow"
            plt.scatter(feat1[p], feat2[p], c = color)
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-8/4-8-" + label1 + "-vs-" + label2 + "-kmeans-colored.png")
        plt.show()
    

#%%
#DBSCAN color plots
db = DBSCAN(eps=0.5, min_samples=10).fit(lc_feat) #eps is NOT epochs
classes_dbscan = db.labels_

print(classes_dbscan)
numclasses = str(len(set(classes_dbscan)))
print("there are " + numclasses + "classes")

#coloring kmeans
for n in range(12):
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
        label1 = "frequency"
    elif n == 10: 
        label1 = "slope"
    elif n == 11: 
        label1 = "log_slope"
        
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
            label2 = "frequency"
        elif m == 10: 
            label2 = "slope"
        elif m == 11: 
            label2 = "log_slope"
        feat2 = lc_feat[:,m]
        plt.autoscale(enable=True, axis='both', tight=True)
        for p in range(len(lc_feat)):
            if classes_dbscan[p] == 0:
                color = "red"
            elif classes_dbscan[p] == -1:
                color = "black"
            elif classes_dbscan[p] == 1:
                color = "blue"
            elif classes_dbscan[p] == 2:
                color = "green"
            elif classes_dbscan[p] == 3:
                color = "yellow"
            plt.scatter(feat1[p], feat2[p], c = color, s = 5)
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-3/4-3-" + label1 + "-vs-" + label2 + "-dbscan-colored.png")
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

for k in range(20):
    l = cluster_0[k]
    l_str = str(l)
    plt.scatter(time, intensity[l], c='blue')
    plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-8/4-8-kmeans-cluster-0-" + l_str + ".png")
    plt.show()

for k in range(20):
    l = cluster_1[k]
    l_str = str(l)
    plt.scatter(time, intensity[l], c='red')
    plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-8/4-8-kmeans-cluster-1-" + l_str + ".png")
    plt.show() 

for k in range(20):
    l = cluster_2[k]
    l_str = str(l)
    plt.scatter(time, intensity[l], c='green')
    plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-8/4-8-kmeans-cluster-2-" + l_str + ".png")
    plt.show()
    
for k in range(20):
    l = cluster_3[k]
    l_str = str(l)
    plt.scatter(time, intensity[l], c='orange')
    plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-8/4-8-kmeans-cluster-3-" + l_str + ".png")
    plt.show()
    
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

if len(cluster_0_db) < 20: 
    p = len(cluster_0_db)
else:
    p = 20
    
for k in range(p):
    l = cluster_0_db[k]
    l_str = str(l)
    plt.scatter(time, intensity[l], c='blue')
    plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-8/4-8-dbscan-cluster-0-" + l_str + ".png")
    plt.show()
    
    
if len(cluster_1_db) < 20: 
    p = len(cluster_1_db)
else:
    p = 20
    
for k in range(p):
    l = cluster_1_db[k]
    l_str = str(l)
    plt.scatter(time, intensity[l], c='red')
    plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-8/4-8-dbscan-cluster-1-" + l_str + ".png")
    plt.show() 

if len(cluster_2_db) < 20: 
    p = len(cluster_2_db)
else:
    p = 20

for k in range(p):
    l = cluster_2_db[k]
    l_str = str(l)
    plt.scatter(time, intensity[l], c='green')
    plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-8/4-8-dbscan-cluster-2-" + l_str + ".png")
    plt.show()

if len(cluster_noise_db) < 20: 
    p = len(cluster_noise_db)
else:
    p = 20
    
for k in range(p):
    l = cluster_noise_db[k]
    l_str = str(l)
    plt.scatter(time, intensity[l], c='orange')
    plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-8/4-8-dbscan-cluster-noise-" + l_str + ".png")
    plt.show()
    
#%%
plt.scatter(time, intensity[0]) 
plt.show()
num_points = len(time)

quarter_points = int(num_points/4)



plt.scatter(time[0:quarter_points], intensity[0][0:quarter_points])
plt.scatter(time[quarter_points: 2 * quarter_points], intensity[0][quarter_points: 2*quarter_points])
plt.scatter(time[2*quarter_points: 3 * quarter_points], intensity[0][2*quarter_points: 3*quarter_points])
plt.scatter(time[3*quarter_points: 4 * quarter_points], intensity[0][3*quarter_points: 4*quarter_points])
    
#%%
#create an array containing the moment for each quarter of the data. calculate the std. 
#if all are within 1 std, then return FALSE (0). if not, return TRUE (1), indicating that something is wrong with one section's moment
    
    
    
    
    
    