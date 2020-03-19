# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:33:21 2020

@author: conta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Reshape, Activation
#from keras.layers import Embedding
#from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D


import astropy
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.utils.data import get_pkg_data_filename
import scipy.signal as signal

from datetime import datetime
import os
from scipy.stats import moment

#import tdpy
#print(tdpy.__path__)
#from tdpy import mcmc

#producing test data
batch_size  = 2000   # >> 1000 lightcurves for each class
test_size   = 50      # >> 25 for each class
num_classes = 1
epochs      = 5
input_dim   = 500     # >> number of data points in light curve
half_batch = int(batch_size/2) #for doing the two halves of the fake data
noise = [0.2] 

# gaussians
height = 20.
center = 15.
stdev  = 10.
xmax   = 30.

def gaussian(datapoints, a, b, c):
    x = np.linspace(0, xmax, datapoints)
    return  a * np.exp(-(x-b)**2 / 2*c**2) + np.random.normal(size=(datapoints))
x_test_flat = 1 + np.random.normal(size = (test_size, input_dim)) #produces a 50x500 item array, which gives the 500 flat data sets their 500 x values
y_test_flat = np.zeros((test_size, num_classes)) #produces a 50x2 array - 50 flat data sets, each with two possible classes. 
if num_classes == 2:
    y_test_flat[:,0] = 1
else:
    y_test_flat[:,0] = 0


# 50 gaussians (class 1)
x_test_bump = np.zeros((test_size, input_dim)) #another 500x500 item array, all zeroes
for i in range(test_size):
    x_test_bump[i] = gaussian(input_dim, a = height, b = center, c = stdev)
    #set the ith value of the x_train to be the gaussian output for that list item
y_test_bump = np.zeros((test_size, num_classes)) #creates y data, 500x2, all zeroes
if num_classes == 2:
    y_test_bump[:,1] = 1
else: 
    y_test_bump[:,0] = 1

#combine each type into one
x_test = np.concatenate((x_test_flat, x_test_bump), axis=0)
y_test = np.concatenate((y_test_flat, y_test_bump), axis=0)

#plotting testing data
"""
plt.ion()
plt.figure(0)
plt.title('Input light curves')
plt.plot(np.linspace(0, xmax, input_dim), x_test[0], '-',
         label = 'flat')
n = int(test_size/2)
plt.plot(np.linspace(0, xmax, input_dim), x_test[int(test_size)], '-',
         label = 'bump')
plt.legend()
""" 

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
    
    f = np.linspace(0.01, 20, 100000)
    pg =signal.lombscargle(np.linspace(0, xmax, input_dim), sampledata, f, normalize = True)
    
    featvec.append(f[pg.argmax()])
    return(featvec) #1st, 2nd, 3rd, 4th moments, period

test = featvec(x_test[0])
test2 = featvec(x_test[int(test_size)])

numdata = len(x_test)

listsamp = np.zeros((numdata, 5))

for n in np.arange(numdata):
    listsamp[n] = featvec(x_test[n])

#%%
from datetime import datetime
now = datetime.now()
timestamp = datetime.timestamp(now)
timestamp = str(timestamp)
#histograms of these features
firstmoment = listsamp[:,0]
secondmoment =listsamp[:,1]
thirdmoment = listsamp[:,2]
fourthmoment = listsamp[:,3]
periodogram_max = listsamp[:,4]
num_bins = 25

plt.hist(secondmoment, bins = num_bins)
plt.title("second moment variance histogram")
plt.xlabel("values")
plt.ylabel("instances")
plt.savefig(timestamp + "_secondmomenthistogram.png")
plt.show()

plt.hist(thirdmoment, bins = num_bins)
plt.title("third moment skew histogram")
plt.savefig(timestamp + "_thirdmomenthistogram.png")
plt.show()

plt.hist(fourthmoment, bins = num_bins)
plt.title("fourth moment kurtosis histogram")
plt.savefig(timestamp + "_fourthmomenthistogram.png")
plt.show()

plt.hist(periodogram_max, bins = num_bins)
plt.title("periodogram max value histogram")
plt.savefig(timestamp + "_periodhistogram.png")
plt.show()

#%%
#clustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
#%%
"""
from tutorial at https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
K= -2 * np.random.rand(100,2)
K1 = 1 + 2 * np.random.rand(50,2)
K[50:100, :] = K1
plt.scatter(K[:,0], K[:,1], s = 50, c = 'b')
plt.show()

Kmean = KMeans(n_clusters=2)
Kmean.fit(K)

plt.scatter(K[:,0], K[:,1], s =50, c='b')
plt.scatter(-0.94665068, -0.97138368, s=200, c='g', marker='s')
plt.scatter(2.01559419, 2.02597093, s=200, c='r', marker='s')
plt.show()

sample_test=np.array([-3.0,-3.0])
second_test=sample_test.reshape(1, -1)
Kmean.predict(second_test)
"""
variance = listsamp[:,1] #variance
period = listsamp[:,4] #period
plt.scatter(variance, period)
plt.show()

var_period = np.column_stack((variance, period))

Kmean = KMeans(n_clusters=2)
Kmean.fit(var_period)

centers = Kmean.cluster_centers_
center1 = centers[0,:]
center2 = centers[1,:]
print(center1, center2)

plt.scatter(variance, period)
plt.scatter(center1[0], center1[1], s=200, c='g')
plt.scatter(center2[0], center2[1], s=200, c='g')

sample_test=np.array([1,0])
second_test = sample_test.reshape(1,-1)
Kmean.predict(second_test)
#this should appear in the first of the two clusters and it does! yay clustering. 




