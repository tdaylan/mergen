# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:25:00 2020

@author: conta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import astropy
from astropy.io import fits
import scipy.signal as signal
from scipy.stats import moment

import sklearn
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer

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
    
def check_diagonalized(c_matrix):
    """Metric for optimization of diagonal of confusion matrix"""
    num_labels = len(c_matrix)
    total = np.sum(c_matrix, axis=None)
    diagonal = 0
    n = 0
    while n < num_labels:
        diagonal = diagonal + c_matrix[n][n]
        n = n+1
    fraction_diagonal = diagonal/total
    return fraction_diagonal

def gaussian(datapoints, a, b, c):
    """Produces a gaussian function"""
    x = np.linspace(0, xmax, datapoints)
    return  a * np.exp(-(x-b)**2 / 2*c**2) + np.random.normal(size=(datapoints))

def create_list_featvec(datasets, num_features):
    """creates the list of one feature vector for every dataset put in. 
    datasets is the array of all of the datasets being turned into feature vectors (ie, 1200 light curves)
    num_features is the number of features that are produced by the current iteration of featvec"""
    num_data = len(datasets)
    feature_list = np.zeros((num_data, num_features))
    for n in np.arange(num_data):
        feature_list[n] = featvec(datasets[n])
    return feature_list

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

#50 flat curves (class 0)
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

#creating feature vectors:

listsamp = create_list_featvec(x_test, 6)

#%%
#from https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
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
        plt.scatter(feat1[0:50], feat2[0:50], c="blue")
        plt.scatter(feat1[50:100], feat2[50:100], c="green")
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.title(label1 + " vs " + label2)
        #plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/3-25/3-25_" + label1 + "_vs_" + label2 + "_realdata.png")
        plt.show()
        
        combined_features = np.column_stack((feat1, feat2))

        db = DBSCAN(eps=0.3, min_samples=10).fit(combined_features)
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
        

#run on all of the features & producing the confusion matrix

db = DBSCAN(eps=50, min_samples=10).fit(listsamp)
fakedata_pred = db.labels_

fakedata_true = np.concatenate((np.zeros((1,50)), np.ones((1,50))), axis=None)

k = confusion_matrix(fakedata_true, fakedata_pred)
print(k)

check_diagonalized(k)
        

