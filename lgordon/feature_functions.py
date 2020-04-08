# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 00:18:52 2020

@author: conta
"""

#all the functions I'm consistently using across files. updated 3/30/2020
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.stats import moment
from scipy import stats
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
rcParams["lines.markersize"] = 5

def test(num):
    print(num * 4)
    
def create_list_featvec(time_axis, datasets, num_features):
    """input: all of the datasets being turned into feature vectors (ie, intensity)
        num_features is the number of features currently being worked on. 
    
    returns a list of featurevectors, one for each input . """
    num_data = len(datasets) #how many datasets
    x = time_axis #creates the x axis
    feature_list = np.zeros((num_data, num_features))
    for n in np.arange(num_data):
        feature_list[n] = featvec(x, datasets[n])
    return feature_list

def featvec(x_axis, sampledata): 
    """calculates the feature vector of the single set of data (ie, intensity[0])
    currently returns 12: 
        1st-4th moments, 
        natural log variance, skew, kurtosis, 
        power, natural log power, frequency, 
        slope, natural log of slope"""
    featvec = moments(sampledata)
    
    f = np.linspace(-0.001, 5, 3000)
    pg = signal.lombscargle(x_axis, sampledata, f, normalize = True)
    
    power = pg[pg.argmax()]
    featvec.append(power)
    featvec.append(np.log(np.abs(power)))
    
    frequency = f[pg.argmax()]
    featvec.append(frequency)
    
    slope = stats.linregress(x_axis, sampledata)[0]
    featvec.append(slope)
    featvec.append(np.log(np.abs(slope)))
    print("done")
    return(featvec) #1st, 2nd, 3rd, 4th moments, power, frequency, slope

def moments(dataset): 
    """calculates the 1st through 4th moment of a single row of data (ie, intensity[0])"""
    moments = []
    moments.append(np.mean(dataset)) #mean (don't use moment, always gives 0)
    moments.append(moment(dataset, moment = 2)) #variance
    moments.append(moment(dataset, moment = 3)) #skew
    moments.append(moment(dataset, moment = 4)) #kurtosis
    moments.append(np.log(np.abs(moment(dataset, moment = 2)))) #ln variance
    moments.append(np.log(np.abs(moment(dataset, moment = 3)))) #ln skew
    moments.append(np.log(np.abs(moment(dataset, moment = 4)))) #ln kurtosis
    return(moments)


    

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



def plot_lc(time, intensity):
    """takes input time and intensity and returns lightcurve plot with 8x3 scaling"""
    plt.figure(figsize=(8,3))
    plt.autoscale(enable=True, axis='both', tight=False)
    plt.plot(time, intensity, '.')
    plt.show()

#def make_n_choose_2(featurevectors, savefig)