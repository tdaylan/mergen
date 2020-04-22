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
rcParams["lines.markersize"] = 2
from scipy.signal import argrelextrema

import sklearn
from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import fnmatch

from datetime import datetime
import os
from scipy.stats import moment, sigmaclip

import astropy
from astropy.io import fits
import scipy.signal as signal

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor

import astroquery
from astroquery.simbad import Simbad

def test(num):
    print(num * 4)
    
def create_list_featvec(time_axis, datasets):
    """input: all of the datasets being turned into feature vectors (ie, intensity)
        num_features is the number of features currently being worked on. 
    
    returns a list of featurevectors, one for each input . """
    num_data = len(datasets) #how many datasets
    x = time_axis #creates the x axis
    feature_list = np.zeros((num_data, 17)) #MANUALLY UPDATE WHEN CHANGING NUM FEATURES
    for n in np.arange(num_data):
        feature_list[n] = featvec(x, datasets[n])
    return feature_list

def featvec(x_axis, sampledata): 
    """calculates the feature vector of the single set of data (ie, intensity[0])
    currently returns 17: 
        1st-4th moments, 
        natural log variance, skew, kurtosis, 
        power, natural log power, period of max power, 
        slope, natural log of slope
        integration of periodogram over: period of 0.1-10, period of 0.1-1, period of 1-3,
        period of 3-10 days,
        period of max power for 0.01-0.1 days (for moving objects)
        ***if you update the number of features, you have to update the number of features in 
        create_list_featvec!!!!"""
    featvec = moments(sampledata) #produces moments and log moments
    
    
    f = np.linspace(0.6, 62.8, 5000)  #period range converted to frequencies
    periods = np.linspace(0.1, 10, 5000)#0.1 to 10 day period
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
    
    slope = stats.linregress(x_axis, sampledata)[0]
    featvec.append(slope)
    featvec.append(np.log(np.abs(slope)))
    
    
    integrating = np.trapz(pg, periods) #integrates the whole 0.1-10 day range
    integrating1 = np.trapz(pg[457:5000], periods[457:5000]) #0.1 days to 1 days
    integrating2 = np.trapz(pg[121:457], periods[121:457])#1-3 days
    integrating3 = np.trapz(pg[0:121], periods[0:121]) #3-10 days
    
    featvec.append(integrating)
    featvec.append(integrating1)
    featvec.append(integrating2)
    featvec.append(integrating3)
    
    f2 = np.linspace(62.8, 6283.2, 20)  #period range converted to frequencies
    p2 = np.linspace(0.001, 0.1, 20)#0.001 to 1 day periods
    pg2 = signal.lombscargle(x_axis, sampledata, f2, normalize = True)
    rel_maxes2 = argrelextrema(pg2, np.greater)
    powers2 = []
    indexes2 = []
    for n in range(len(rel_maxes2[0])):
        index2 = rel_maxes2[0][n]
        indexes2.append(index2)
        power_level_at_rel_max2 = pg2[index2]
        powers2.append(power_level_at_rel_max2)
    max_power2 = np.max(powers2)
    index_of_max_power2 = np.argmax(powers2)
    index_of_f_max2 = rel_maxes2[0][index_of_max_power2]
    f_max_power2 = f2[index_of_f_max2]
    period_max_power2 = 2*np.pi / f_max_power2
    featvec.append(period_max_power2)
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
    for i in np.arange(len(intensity)):
        intensity[i] = intensity[i] / np.median(intensity[i])
        #int_1 = intensity[n]
        #min_1 = int_1[int_1.argmin()]
        #max_1 = int_1[int_1.argmax()]
        #intensity[n] = (int_1 - min_1) / (max_1 - min_1) 
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


def n_choose_2_features_plotting(feature_vectors, date, clustering):
    """plotting (n 2) features against each other, currently with no clustering
    feature_vectors must be a list of feature vectors
    date must be a string in the format of the folder you are saving into ie "4-13"
    this function does NOT plot kmeans/dbscan colors
    """
    cluster = "empty"
    if clustering == 'dbscan':
        db = DBSCAN(eps=0.5, min_samples=10).fit(feature_vectors) #eps is NOT epochs
        classes_dbscan = db.labels_
        numclasses = str(len(set(classes_dbscan)))
        cluster = 'dbscan'
    elif clustering == 'kmeans': 
        Kmean = KMeans(n_clusters=4, max_iter=700, n_init = 20)
        x = Kmean.fit(lc_feat)
        classes_kmeans = x.labels_
        cluster = 'kmeans'
    else: 
        print("no clustering chosen")
        cluster = 'none'
        
    for n in range(16):
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
            graph_label1 = "Power integrated over T = 0.1 to 10 days"
            fname_label1 = "IntPower01-10"
        elif n == 13:
            graph_label1 = "Power integrated over T = 0.1 to 1 days"
            fname_label1 = "IntPower01-1"
        elif n == 14:
            graph_label1 = "Power integrated over T = 1 to 3 days"
            fname_label1 = "IntPower1-3"
        elif n == 15:
            graph_label1 = "Power integrated over T = 3 to 10 days"
            fname_label1 = "IntPower3-10"
            
        for m in range(16):
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
                graph_label2 = "Power integrated over T = 0.1 to 10 days"
                fname_label2 = "IntPower01-10"
            elif m == 13:
                graph_label2 = "Power integrated over T = 0.1 to 1 days"
                fname_label2 = "IntPower01-1"
            elif m == 14:
                graph_label2 = "Power integrated over T = 1 to 3 days"
                fname_label2 = "IntPower1-3"
            elif m == 15:
                graph_label2 = "Power integrated over T = 3 to 10 days"
                fname_label2 = "IntPower3-10"
                
            feat2 = feature_vectors[:,m]
            
            if cluster == 'dbscan':
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
                #plt.autoscale(enable=True, axis='both', tight=True)
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig(("/Users/conta/UROP_Spring_2020/plot_output/" + date + "/dbscan-colored/" + date + "-" + fname_label1 + "-vs-" + fname_label2 + "-dbscan.png"))
                plt.show()
            elif cluster == 'kmeans':
                for p in range(len(lc_feat)):
                    if classes_kmeans[p] == 0:
                        color = "red"
                    elif classes_kmeans[p] == 1:
                        color = "blue"
                    elif classes_kmeans[p] == 2:
                        color = "green"
                    elif classes_kmeans[p] == 3:
                        color = "purple"
                    plt.scatter(feat1[p], feat2[p], c = color)
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/" + date + "/kmeans-colored/" + date + "-" + fname_label1 + "-vs-" + fname_label2 + "-kmeans.png")
                plt.show()
            elif cluster == 'none':
                plt.scatter(feat1, feat2, s = 2, color = 'blue')
                #plt.autoscale(enable=True, axis='both', tight=True)
                plt.xlabel(graph_label1)
                plt.ylabel(graph_label2)
                plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/" + date + "/nchoose2/" + date + "-" + fname_label1 + "-vs-" + fname_label2 + ".png")
                plt.show()
                
def get_data_from_fits():
    """ imports data from fits files. 
        based on emma's code"""    

    fitspath = '/Users/conta/UROP_Spring_2020/tessdata_lc_sector20_1000/'
    fnames_all = os.listdir(fitspath)
    fnames = fnmatch.filter(fnames_all, '*fits*')
    
    interp_tol = 20. / (24*60) # >> interpolate small gaps (less than 20 minutes)
    
    intensity = []
    targets = []
    #coordinates = []
    for file in fnames:
        # -- open file -------------------------------------------------------------
        f = fits.open(fitspath + file)
    
        # >> get data
        time = f[1].data['TIME']
        i = f[1].data['PDCSAP_FLUX']
        tic = f[1].header["OBJECT"]
        targets.append(tic)

        #ra = f[1].header["RA_OBJ"]
        #dec = f[1].header["DEC_OBJ"]
        #coords = str(ra) + " " + str(dec)
        #coordinates.append(coords)
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
    return time, intensity, targets


def plot_lof(time, intensity, targets, features, n, date):
    """plots the 20 most and least interesting light curves based on LOF
    takes input: time, intensity, targets, featurelist, n number of curves you want, date as a string """
    from sklearn.neighbors import LocalOutlierFactor

    clf = LocalOutlierFactor(n_neighbors=2)
    
    fit_predictor = clf.fit_predict(features)
    negative_factor = clf.negative_outlier_factor_
    
    lof = -1 * negative_factor
    ranked = np.argsort(lof)
    largest_indices = ranked[::-1][:n]
    smallest_indices = ranked[:n]

    #plot just the largest indices
    #rows, columns
    fig, axs = plt.subplots(n, 1, sharex = True, figsize = (8,n*3), constrained_layout=False)
    fig.subplots_adjust(hspace=0)
    
    for k in range(n):
        ind = largest_indices[k]
        axs[k].plot(time, intensity[ind], '.k', label=targets[ind])
        axs[k].legend(loc="upper left")
        axs[k].set_ylabel("relative flux")
        axs[-1].set_xlabel("BJD [-2457000]")
    fig.suptitle(str(n) + ' largest LOF targets', fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.96)
    fig.savefig("/Users/conta/UROP_Spring_2020/plot_output/" + date + "/" + date + "-"+ str(n) + "-largest-lof.png")

    #plot the smallest indices
    fig1, axs1 = plt.subplots(n, 1, sharex = True, figsize = (8,n*3), constrained_layout=False)
    fig1.subplots_adjust(hspace=0)
    
    for m in range(n):
        ind = smallest_indices[m]
        axs1[m].plot(time, intensity[ind], '.k', label=targets[ind])
        axs1[m].legend(loc="upper left")
        axs1[m].set_ylabel("relative flux")
        axs1[-1].set_xlabel("BJD [-2457000]")
    fig1.suptitle(str(n) + ' smallest LOF targets', fontsize=16)
    fig1.tight_layout()
    fig1.subplots_adjust(top=0.96)
    fig1.savefig("/Users/conta/UROP_Spring_2020/plot_output/" + date + "/" + date + "-"+ str(n) + "-smallest-lof.png")
                
