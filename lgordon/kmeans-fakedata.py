# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 21:36:36 2020

@author: conta
"""

#running kmeans on fake data 3-22-20


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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
#%%
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
    """Produces a gaussian function"""
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


def moments(dataset):
    """calculates the 1st through 4th moment of the given data"""
    moments = []
    #moments.append(moment(dataset, moment = 0)) #total prob, should always be 1
    moments.append(moment(dataset, moment = 1)) # expectation value
    moments.append(moment(dataset, moment = 2)) #variance
    moments.append(moment(dataset, moment = 3)) #skew
    moments.append(moment(dataset, moment = 4)) #kurtosis
    return(moments)

def featvec(sampledata): 
    """calculates the feature vector of the given data. currently returns: 1st-4th moments, power, frequency"""
    featvec = moments(sampledata)
    
    f = np.linspace(0.01, 20, 100)
    pg = signal.lombscargle(np.linspace(0, xmax, input_dim), sampledata, f, normalize = True)
    
    power = pg[pg.argmax()]
    featvec.append(power)
    
    frequency = f[pg.argmax()]
    featvec.append(frequency)
    return(featvec) #1st, 2nd, 3rd, 4th moments, power, frequency

numdata = len(x_test)
print(numdata)
listsamp = np.zeros((numdata, 6))

for n in np.arange(numdata):
    listsamp[n] = featvec(x_test[n])

Kmean = KMeans(n_clusters=2, max_iter=700, n_init = 20)

#n choose 2 plotting of features + features w/ cluster centers
for n in range(0,6):
    feat1 = listsamp[:,n]
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
        feat2 = listsamp[:,m]
        plt.scatter(feat1[0:50], feat2[0:50], c="red")
        plt.scatter(feat1[50:100], feat2[50:100], c="blue")
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.title(label1 + " vs " + label2)
        #plt.savefig("3-22_" + label1 + "_vs_" + label2 + ".png")
        plt.show()
        
        combined_features = np.column_stack((feat1, feat2))

        Kmean = KMeans(n_clusters=2, max_iter=500)
        Kmean.fit(combined_features)

        centers = Kmean.cluster_centers_
        center1 = centers[0,:]
        center2 = centers[1,:]
        #print(center1, center2)
        
        plt.scatter(feat1[0:50], feat2[0:50], c="red")
        plt.scatter(feat1[50:100], feat2[50:100], c="blue")
        plt.scatter(center1[0], center1[1], s=200, c='g')
        plt.scatter(center2[0], center2[1], s=200, c='g')
        plt.title(label1 + " vs " + label2)
        #plt.savefig("3-22_" + label1 + "_vs_" + label2 + "clustered.png")
        plt.show()

#run on all of the features
Kmean.fit(listsamp)

#producing the confusion matrix

newlist = np.zeros((20, 6))
newlist[0:10] = listsamp[20:30] #these are flat
newlist[10:20] = listsamp[70:80] #these are bump

#print(newlist) #should be the first 10 are flat, second ten are bump

fakedata_pred = Kmean.predict(newlist)
#print(fakedata_pred) #predicted classes for the 20 testers

fakedata_true = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1] 

confusion_matrix(fakedata_true, fakedata_pred)

true_neg, false_pos, false_neg, true_pos = confusion_matrix(fakedata_true, fakedata_pred).ravel() #ONLY works for binary classification
print(true_neg, false_pos, false_neg, true_pos)











