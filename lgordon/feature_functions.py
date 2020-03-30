# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 00:18:52 2020

@author: conta
"""

#all the functions I'm consistently using across files. updated 3/30/2020
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os
from scipy.stats import moment

def test(num):
    print(num * 14)

def moments(dataset): 
    """calculates the 1st through 4th moment of the given data"""
    moments = []
    #moments.append(moment(dataset, moment = 0)) #total prob, should always be 1
    moments.append(np.mean(dataset)) #mean
    #moments.append(moment(dataset, moment = 1)) # expectation value -> always is zero??
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
    """normalizes the intensity from the median value using sklearn's preprocssing Normalizer"""
    for n in np.arange(len(intensity)): 
        int_1 = intensity[n]
        min_1 = int_1[int_1.argmin()]
        max_1 = int_1[int_1.argmax()]
        intensity[n] = (int_1 - min_1) / (max_1 - min_1) 
    return intensity

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
    num_data = len(datasets) #how many datasets
    num_points = len(datasets[0]) #how many points per dataset
    x = np.linspace(0,num_points, num=num_points) #creates the x axis
    feature_list = np.zeros((num_data, num_features))
    for n in np.arange(num_data):
        feature_list[n] = featvec(x, datasets[n])
    return feature_list

def plot_lc(time, intensity):
    """takes input time and intensity and returns lightcurve plot with 8x3 scaling"""
    plt.figure(figsize=(8,3))
    plt.autoscale(enable=True, axis='both', tight=False)
    plt.plot(time, intensity, '.')
    plt.show()

#def make_n_choose_2(featurevectors, savefig)