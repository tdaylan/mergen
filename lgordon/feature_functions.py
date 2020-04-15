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
from scipy.signal import argrelextrema

import sklearn
from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import fnmatch

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
        power, natural log power, period of max power, 
        slope, natural log of slope
        integration of periodogram over entire f range (0.1 to 10)"""
    featvec = moments(sampledata) #produces moments and log moments
    
    
    f = np.linspace(0.09, 10, 3000)
    pg = signal.lombscargle(x_axis, sampledata, f, normalize = True)
    rel_maxes = argrelextrema(pg, np.greater)
    
    powers = []
    indexes = []
    for n in range(len(rel_maxes[0])):
        #print(rel_maxes[0][n]) #accessing each index
        index = rel_maxes[0][n]
        indexes.append(index)
        power_level_at_rel_max = pg[index]
        #print(power_level_at_rel_max)
        powers.append(power_level_at_rel_max)
    
    max_power = np.max(powers)
    #print("the max power is" + str(max_power))
    index_of_max_power = np.argmax(powers)
    #print("the index of that power is" + str(index_of_max_power))
    index_of_f_max = rel_maxes[0][index_of_max_power]
    f_max_power = f[index_of_f_max]
    #print("the frequency at that index is" + str(f_max_power))
    period_max_power = 2*np.pi / f_max_power
    featvec.append(max_power)
    featvec.append(np.log(np.abs(max_power)))
    featvec.append(period_max_power)

    #power = pg[pg.argmax()]
    #frequency = f[pg.argmax()]
    
    slope = stats.linregress(x_axis, sampledata)[0]
    featvec.append(slope)
    featvec.append(np.log(np.abs(slope)))
    
    integrating = np.trapz(pg, f)
    featvec.append(integrating)
    print("done")
    return(featvec) 

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



def plot_lc(time, intensity, index):
    """takes input time and intensity and returns lightcurve plot with 8x3 scaling"""
    plt.figure(figsize=(8,3))
    plt.autoscale(enable=True, axis='both', tight=False)
    plt.plot(time, intensity[index], '.', s=5)
    plt.title(targets[index])
    plt.show()


def n_choose_2_features_plotting(feature_vectors, date, kmeans, dbscan):
    """plotting (n 2) features against each other, currently with no clustering
    feature_vectors must be a list of feature vectors
    date must be a string in the format of the folder you are saving into ie "4-13"
    this function does NOT plot kmeans/dbscan colors
    """
    if dbscan == True:
        db = DBSCAN(eps=0.5, min_samples=10).fit(feature_vectors) #eps is NOT epochs
        classes_dbscan = db.labels_
        numclasses = str(len(set(classes_dbscan)))
        clustering = 'dbscan'
    elif kmeans == True: 
        Kmean = KMeans(n_clusters=4, max_iter=700, n_init = 20)
        x = Kmean.fit(lc_feat)
        classes_kmeans = x.labels_
        clustering = 'kmeans'
    else: 
        print("no clustering")
        clustering = 'none'
    for n in range(12):
        feat1 = feature_vectors[:,n]
        if n == 0:
            graph_label1 = "Average"
            fname_label1 = "Avg"
        elif n == 1: 
            graph_label1 = "Variance"
            fname_label1 = "Var"
        elif n == 2:
            graph_label1 = "Skewness"
            fname_label1 = "Skew"
        elif n == 3:
            graph_label1 = "Kurtosis"
            fname_label1 = "Kurt"
        elif n == 4:
            graph_label1 = "Log Variance"
            fname_label1 = "LogVar"
        elif n == 5:
            graph_label1 = "Log Skewness"
            fname_label1 = "LogSkew"
        elif n == 6: 
            graph_label1 = "Log Kurtosis"
            fname_label1 = "LogKurt"
        elif n == 7: 
            graph_label1 = "Maximum Power"
            fname_label1 = "MaxPower"
        elif n == 8: 
            graph_label1 = "Log Maximum Power"
            fname_label1 = "LogMaxPower"
        elif n == 9: 
            graph_label1 = "Period [BJD - 2457000]"
            fname_label1 = "Period"
        elif n == 10: 
            graph_label1 = "Slope"
            fname_label1 = "Slope"
        elif n == 11: 
            graph_label1 = "Log Slope"
            fname_label1 = "LogSlope"
        elif n==12:
            graph_label1 = "Power integrated over f=[0.1,10]"
            fname_label1 = "IntPower"
            
        for m in range(12):
            if m == n:
                break
            if m == 0:
                graph_label2 = "Average"
                fname_label2 = "Avg"
            elif m == 1: 
                graph_label2 = "Variance"
                fname_label2 = "Var"
            elif m == 2:
                graph_label2 = "Skewness"
                fname_label2 = "Skew"
            elif m == 3:
                graph_label2 = "Kurtosis"
                fname_label2 = "Kurt"
            elif m == 4:
                graph_label2 = "Log Variance"
                fname_label2 = "LogVar"
            elif m == 5:
                graph_label2 = "Log Skewness"
                fname_label2 = "LogSkew"
            elif m == 6: 
                graph_label2 = "Log Kurtosis"
                fname_label2 = "LogKurt"
            elif m == 7: 
                graph_label2 = "Maximum Power"
                fname_label2 = "MaxPower"
            elif m == 8: 
                graph_label2 = "Log Maximum Power"
                fname_label2 = "LogMaxPower"
            elif m == 9: 
                graph_label2 = "Period [BJD - 2457000]"
                fname_label2 = "Period"
            elif m == 10:
                graph_label2 = "Slope"
                fname_label2 = "Slope"
            elif m == 11: 
                graph_label2 = "Log Slope"
                fname_label2 = "LogSlope"
            elif m==12:
                graph_label2 = "Power integrated over f=[0.1,10]"
                fname_label2 = "IntPower"
            feat2 = feature_vectors[:,m]
            
            if clustering == 'dbscan':
                for p in range(len(feature_vectors)):
                    if classes_dbscan[p] == 0:
                        color = "red"
                    elif classes_dbscan[p] == -1:
                        color = "black"
                    elif classes_dbscan[p] == 1:
                        color = "blue"
                    elif classes_dbscan[p] == 2:
                        color = "green"
                    elif classes_dbscan[p] == 3:
                        color = "purple"
                    plt.scatter(feat1[p], feat2[p], c = color, s = 5)
                plt.autoscale(enable=True, axis='both', tight=True)
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig(("/Users/conta/UROP_Spring_2020/plot_output/" + date + "/" + date + "-" + fname_label1 + "-vs-" + fname_label2 + "-dbscan.png"))
                plt.show()
            elif clustering == "kmeans":
                print("have not set this up yet")
            elif cluster == "none":
                plt.scatter(feat1, feat2, s = 5, color = 'blue')
                plt.autoscale(enable=True, axis='both', tight=True)
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig(("/Users/conta/UROP_Spring_2020/plot_output/" + date + "/" + date + "-" + fname_label1 + "-vs-" + fname_label2 + ".png"))
                plt.show()
                
                
