# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 08:39:09 2020

@author: conta
"""

#3-22 running kmeans on real data (hopefully)

import pandas as pd
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

def moments(dataset): 
    moments = []
    #moments.append(moment(dataset, moment = 0)) #total prob, should always be 1
    moments.append(moment(dataset, moment = 1)) # expectation value
    moments.append(moment(dataset, moment = 2)) #variance
    moments.append(moment(dataset, moment = 3)) #skew
    moments.append(moment(dataset, moment = 4)) #kurtosis
    return(moments)

def featvec(x_axis, sampledata): 
    featvec = moments(sampledata)
    
    f = np.linspace(0.01, 20, 100)
    pg = signal.lombscargle(x_axis, sampledata, f, normalize = True)
    
    power = pg[pg.argmax()]
    featvec.append(power)
    
    frequency = f[pg.argmax()]
    featvec.append(frequency)
    return(featvec) #1st, 2nd, 3rd, 4th moments, power, frequency


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
# read tess data (sector 20)
# time in TESS-truncated JD (BJD - 2457000)
# emma 03/16/2020
#
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

fitspath = './UROP_Spring_2020/tessdata_lc_sector20_1000/'
output_dir = './lightcurves032220/'

fnames_all = os.listdir(fitspath)
fnames = fnmatch.filter(fnames_all, '*fits*')

# >> remove buggy fits file (interrupted download)
#fnames.pop(fnames.index('tess2019357164649-s0020-0000000156168236-0165-s_lc.fits'))

interp_tol = 20. / (24*60) # >> interpolate small gaps (less than 20 minutes)

intensity = []

for file in fnames:
    # -- open file -------------------------------------------------------------
    f = fits.open(fitspath + file)
    # print(f.info())
    # print(f[1].header)

    # >> get data
    time = f[1].data['TIME']
    i = f[1].data['PDCSAP_FLUX']

    # >> plot
    #fig, ax = plt.subplots(1,1)
    #ax.plot(time, i, '.')
    #plt.savefig(output_dir + file[14:-5] + '.png')
    #plt.close(fig)

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
intensity = np.delete(intensity, nan_inds, 1)
#each row of intensity is one interpolated light curve.
#%%
#producing features
numcurves = len(intensity) #how many lightcurves are there
numpoints = len(intensity[0])
#print(numcurves, numpoints)
lc_feat = np.zeros((numcurves,6)) #array of zeroes, 10 curves x 6 features
x = np.linspace(0, numpoints, num=numpoints)
#print(x)

for n in np.arange(numcurves):
    lc_feat[n] = featvec(x, intensity[n])


Kmean = KMeans(n_clusters=2, max_iter=700, n_init = 20)
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
        plt.savefig("UROP_Spring_2020/3-23_" + label1 + "_vs_" + label2 + "_realdata.png")
        plt.show()
        
        combined_features = np.column_stack((feat1, feat2))

        Kmean.fit(combined_features)

        centers = Kmean.cluster_centers_
        center1 = centers[0,:]
        center2 = centers[1,:]
        #print(center1, center2)
        
        plt.scatter(feat1, feat2, c="blue")
        plt.scatter(center1[0], center1[1], s=200, c='g')
        plt.scatter(center2[0], center2[1], s=200, c='g')
        plt.title(label1 + " vs " + label2)
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.savefig("UROP_Spring_2020/3-23_" + label1 + "_vs_" + label2 + "_clustered_realdata.png")
        plt.show()



