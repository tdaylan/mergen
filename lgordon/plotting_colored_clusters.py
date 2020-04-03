# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:08:47 2020

@author: conta
"""


import numpy as np
import matplotlib.pyplot as plt

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

test(8) #should return 8 * 4

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
        label1 = "skew"
    elif n == 3:
        label1 = "kurtosis"
    elif n == 4:
        label1 = "log variance"
    elif n == 5:
        label1 = "log skew"
    elif n == 6: 
        label1 = "log kurtosis"
    elif n == 7: 
        label1  = "power"
    elif n == 8: 
        label1 = "log power"
    elif n == 9: 
        label1 = "period of maximum power"
    elif n == 10: 
        label1 = "slope"
    elif n == 11: 
        label1 = "log slope"
        
    for m in range(12):
        if m == n:
            break
        if m == 0:
            label2 = "average"
        elif m == 1: 
            label2 = "variance"
        elif m == 2:
            label2 = "skew"
        elif m == 3:
            label2 = "kurtosis"
        elif m == 4:
            label2 = "log variance"
        elif m == 5:
            label2 = "log skew"
        elif m == 6: 
            label2 = "log kurtosis"
        elif m == 7: 
            label2  = "power"
        elif m == 8: 
            label2 = "log power"
        elif m == 9: 
            label2 = "period of maximum power"
        elif m == 10: 
            label2 = "slope"
        elif m == 11: 
            label2 = "log slope"
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
        plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-3/4-3-" + label1 + "-vs-" + label2 + "-kmeans-colored.png")
        plt.show()
    
    
#%%
#DBSCAN color plots
db = DBSCAN(eps=0.5, min_samples=10).fit(lc_feat) #eps is NOT epochs
classes_dbscan = db.labels_

print(classes_dbscan)

#coloring kmeans
for n in range(12):
    feat1 = lc_feat[:,n]
    if n == 0:
        label1 = "average"
    elif n == 1: 
        label1 = "variance"
    elif n == 2:
        label1 = "skew"
    elif n == 3:
        label1 = "kurtosis"
    elif n == 4:
        label1 = "log variance"
    elif n == 5:
        label1 = "log skew"
    elif n == 6: 
        label1 = "log kurtosis"
    elif n == 7: 
        label1  = "power"
    elif n == 8: 
        label1 = "log power"
    elif n == 9: 
        label1 = "period of maximum power"
    elif n == 10: 
        label1 = "slope"
    elif n == 11: 
        label1 = "log slope"
        
    for m in range(12):
        if m == n:
            break
        if m == 0:
            label2 = "average"
        elif m == 1: 
            label2 = "variance"
        elif m == 2:
            label2 = "skew"
        elif m == 3:
            label2 = "kurtosis"
        elif m == 4:
            label2 = "log variance"
        elif m == 5:
            label2 = "log skew"
        elif m == 6: 
            label2 = "log kurtosis"
        elif m == 7: 
            label2  = "power"
        elif m == 8: 
            label2 = "log power"
        elif m == 9: 
            label2 = "period of maximum power"
        elif m == 10: 
            label2 = "slope"
        elif m == 11: 
            label2 = "log slope"
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
            plt.scatter(feat1[p], feat2[p], c = color)
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/4-3/4-3-" + label1 + "-vs-" + label2 + "-dbscan-colored.png")
        plt.show()

    