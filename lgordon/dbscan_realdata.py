# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 20:00:17 2020

@author: conta
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10

import astropy
from astropy.io import fits
import scipy.signal as signal
from scipy.stats import moment

import sklearn
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import os
import fnmatch
from sklearn.preprocessing import Normalizer
from feature_functions import *

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

#%%
lc_features = create_list_featvec(intensity, 6)

#%%
#from https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
for n in range(0,6):
    plt.autoscale(enable=True, axis='both', tight=True)
    feat1 = lc_features[:,n]
    if n == 0:
        label1 = "average"
    elif n == 1: 
        label1 = "variance"
    elif n == 2:
        label1 = "skew"
    elif n == 3:
        label1 = "kurtosis"
    elif n == 4:
        label1 = "power"
    elif n == 5:
        label1 = "frequency"
        
    for m in range(6):
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
            label = "power"
        elif m == 5:
            label2 = "frequency"
        feat2 = lc_features[:,m]
        plt.autoscale(enable=True, axis='both', tight=True)
        plt.scatter(feat1, feat2, c="blue")
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.title(label1 + " vs " + label2)
        #plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/3-25/3-25_" + label1 + "_vs_" + label2 + "_realdata.png")
        plt.show()
        
        combined_features = np.column_stack((feat1, feat2))

        db = DBSCAN(eps=100, min_samples=10).fit(combined_features)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print('Estimated number of clusters: %d' % n_clusters_)

        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
        
            class_member_mask = (labels == k)
        
            xy = combined_features[class_member_mask & core_samples_mask]
            yz = combined_features[class_member_mask & ~core_samples_mask]
            xy = np.concatenate((xy, yz))
            #print(xy)
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)
            
        plt.title('DBSCAN clustering_' + label1 + " vs " + label2)
        plt.xlabel(label1)
        plt.ylabel(label2)
        #plt.legend(["cluster1", "cluster2", "cluster3", "noise"])
        #plt.savefig("3-19-dbscanclustering-fakedata.png")
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

