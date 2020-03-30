# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:23:11 2020

@author: conta
"""
#3-19-20 running feature generation on real data that has been interpolated
#interpolation taken from emma's 3-16 code


from astropy.io import fits
import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

import pandas as pd
import scipy.signal as signal

from datetime import datetime
import os
from scipy.stats import moment

# import and interpolate data

fitspath = './tessdata_lc_sector20_1000/'
output_dir = './lightcurves031920/'

fnames_all = os.listdir(fitspath)
fnames = fnmatch.filter(fnames_all, '*fits*')
numfiles = len(fnames)

interp_tol = 20. / (24*60) # >> interpolate small gaps (less than 20 minutes)

intensity = []

for file in fnames:
    # -- open file -------------------------------------------------------------
    f = fits.open(fitspath + file)

    # >> get data
    time = f[1].data['TIME']
    i = f[1].data['PDCSAP_FLUX']

    # -- find small nan gaps ---------------------------------------------------
    # >> adapted from https://gist.github.com/alimanfoo/
    #    c5977e87111abe8127453b21204c1065
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
nan_inds = np.nonzero(np.prod(np.isnan(np.array(intensity)), axis = 0))
time = np.delete(time, nan_inds)
intensity = np.delete(intensity, nan_inds, 1)
#so the above gives the time for all of them, and the intensity[i]

#defining functions to produce features
def moments(dataset): 
    moments = []
    #moments.append(moment(dataset, moment = 0)) #total prob, should always be 1
    moments.append(moment(dataset, moment = 1)) # expectation value
    moments.append(moment(dataset, moment = 2)) #variance
    moments.append(moment(dataset, moment = 3)) #skew
    moments.append(moment(dataset, moment = 4)) #kurtosis
    return(moments)

def featvec(sampledata): 
    featvec = moments(sampledata)
    
    input_dim = len(sampledata)
    max_amp = np.argmax(sampledata)
    f = np.linspace(0.01, 20, 100)
    pg = signal.lombscargle(np.linspace(0, max_amp, input_dim), sampledata, f, normalize = True)
    
    featvec.append(f[pg.argmax()])
    
    
    featvec.append(max_amp)
    
    return(featvec) #1st, 2nd, 3rd, 4th moments, period, max_amp (6 features)

test = featvec(intensity[0])
#print(test)

listsamp = np.zeros((numfiles, 6))

for n in np.arange(numfiles):
    listsamp[n] = featvec(intensity[n])

print(listsamp)

#so i'm not too sure why it gives nan values for the moments?? this is weird. 
#my best guess is that there is something with the data's average coming out to dividing by zero or something like that


#after a quick test, determined that KMeans will not run on NaNs! So that's cool and exciting. :/














