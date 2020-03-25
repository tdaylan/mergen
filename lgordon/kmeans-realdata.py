# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 08:39:09 2020

@author: conta
"""

#3-24 running kmeans on real data

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
from sklearn import metrics
import fnmatch

from sklearn.metrics import confusion_matrix

def moments(dataset): 
    """calculates the 1st through 4th moment of the given data"""
    moments = []
    #moments.append(moment(dataset, moment = 0)) #total prob, should always be 1
    moments.append(moment(dataset, moment = 1)) # expectation value
    moments.append(moment(dataset, moment = 2)) #variance
    moments.append(moment(dataset, moment = 3)) #skew
    moments.append(moment(dataset, moment = 4)) #kurtosis
    return(moments)

def featvec(x_axis, sampledata): 
    """calculates the feature vector of the given data. currently returns: 1st-4th moments, power, frequency"""
    featvec = moments(sampledata)
    
    f = np.linspace(0.01, 20, 100)
    pg = signal.lombscargle(x_axis, sampledata, f, normalize = True)
    
    power = pg[pg.argmax()]
    featvec.append(power)
    
    frequency = f[pg.argmax()]
    featvec.append(frequency)
    return(featvec) #1st, 2nd, 3rd, 4th moments, power, frequency
    

#normalizing each light curve
def normalize(intensity):
    """normalizes the intensity from the median value"""
    norm_intensity = []
    for i in np.arange(len(intensity)):
        median = np.median(intensity[i])
        normalized = intensity[i]/median
        norm_intensity.append(normalized)
    return norm_intensity
#%%
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
# read tess data (sector 20)
# time in TESS-truncated JD (BJD - 2457000)
# emma 03/16/2020
#
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

fitspath = '/Users/conta/UROP_Spring_2020/tessdata_lc_sector20_1000/'
fnames_all = os.listdir(fitspath)
fnames = fnmatch.filter(fnames_all, '*fits*')

interp_tol = 20. / (24*60) # >> interpolate small gaps (less than 20 minutes)

intensity = []

for file in fnames:
    # -- open file -------------------------------------------------------------
    f = fits.open(fitspath + file)

    # >> get data
    time = f[1].data['TIME']
    i = f[1].data['PDCSAP_FLUX']

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

#producing features
numcurves = len(intensity) #how many lightcurves are there
numpoints = len(intensity[0])
#print(numcurves, numpoints)
lc_feat = np.zeros((numcurves,6)) #array of zeroes, 10 curves x 6 features
x = np.linspace(0, numpoints, num=numpoints)
#print(x)

for n in np.arange(numcurves):
    lc_feat[n] = featvec(x, intensity[n])

#%%
Kmean = KMeans(n_clusters=4, max_iter=700, n_init = 20)
#%%
#n choose 2 plotting of features + features w/ cluster centers
for n in range(0,6):
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
        feat2 = lc_feat[:,m]
        plt.scatter(feat1, feat2, c="blue")
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.title(label1 + " vs " + label2)
        #plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/3-25/3-25_" + label1 + "_vs_" + label2 + "_realdata.png")
        plt.show()
        
        combined_features = np.column_stack((feat1, feat2))

        Kmean.fit(combined_features)

        centers = Kmean.cluster_centers_
        center1 = centers[0,:]
        center2 = centers[1,:]
        center3 = centers[2,:]
        center4 = centers[3,:]
        #print(center1, center2)
        
        plt.scatter(feat1, feat2, c="blue")
        plt.scatter(center1[0], center1[1], s=200, c='g')
        plt.scatter(center2[0], center2[1], s=200, c='g')
        plt.scatter(center3[0], center3[1], s=200, c='g')
        plt.scatter(center4[0], center4[1], s=200, c='g')
        plt.title(label1 + " vs " + label2)
        plt.xlabel(label1)
        plt.ylabel(label2)
        #plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/3-25/3-25_" + label1 + "_vs_" + label2 + "_clustered_realdata.png")
        plt.show()

#%%
Kmean.fit(lc_feat)



predict_on_100 = lc_feat[0:100]
#print(predict_on_100)
predicted_100 = Kmean.predict(predict_on_100)
print(predicted_100)

#producing the confusion matrix
labelled_100 = np.loadtxt("/Users/conta/UROP_Spring_2020/100-labelled/plots/labelled_100.txt", delimiter=',', usecols=1, skiprows=1, unpack=True)
print(labelled_100)


confusion_matrix(labelled_100, predicted_100)

#horizontal axis is predicted. vertical axis is actual. 
#row 0 is for all the actual zeros: 55 predicted as 0, 0 predicted as 1/2/3.
#row 1 is for all actual 1s. 21 predicted as 0, 0 predicted as 1/2/3
#row 2 is for all actual 2s. 22 predicted as 0, 0 as 1/2/3
#row 3 is for all actual 3s. 2 predicted as 0, 0 as 1/2/3

#%%

x = Kmean.fit(lc_feat)
classes = x.labels_
k = np.linspace(0, len(classes), len(classes))
plt.scatter(k,classes)
plt.title("Graphing all of the labels for Kmeans on real data")
plt.xlabel("feature vector")
plt.ylabel("classification")
plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/3-25/3-25-class_results.png")



