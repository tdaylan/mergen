# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 00:18:52 2020

@author: Lindsey Gordon 

Functions used across files. Last updated June 30th 2020
"""

#Imports ---------------------------------------
import numpy as np
import numpy.ma as ma 
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)

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
import shutil
from scipy.stats import moment, sigmaclip

import astropy
from astropy.io import fits
import scipy.signal as signal
from astropy.stats import SigmaClip
from astropy.utils import exceptions

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

import astroquery
from astroquery.simbad import Simbad
from astroquery.mast import Catalogs
from astroquery.mast import Observations
from astroquery import exceptions
from astroquery.exceptions import RemoteServiceError


#Testing that this file imported correctly ------


def classification_test():
    print("classification functions loaded in")
    


#confusion matrix functions ------------------------------------------------
def dbscan_param_scan(path, features, epsmin, epsmax, epsstep, sampmin, sampmax, sampstep, hand_classes):
    """run parameter scan for dbscan over given range of eps and samples,
    knowing the hand classified values"""
    filedb = path + "/dbscan-confusion-matrices-scan.txt"
    #feature optimizing for dbscan
    #0 flat 1 sine 2 multiple transits 3 flares 4 single transits 5 not sure
    text1 = "\n Eps values between " + str(epsmin) + " and " + str(epsmax) + ". Min samples between " + str(sampmin) + " and " + str(sampmax)
    with open(filedb, 'a') as file_object:
            file_object.write("This file contains the confusion matrices for the given features undergoing DBSCAN optimization")
            file_object.write(text1)
    
    eps_values = np.arange(epsmin, epsmax, epsstep)
    min_samps = np.arange(sampmin,sampmax,sampstep)
    avg_precision = []
    avg_recall = []
    accuracies = []
    for i in range(len(min_samps)):
        for n in range(len(eps_values)):
            #dbscan predicting on features
            #feature vectors -> feats
            db_run = DBSCAN(eps=eps_values[n], min_samples=min_samps[i]).fit(feats) #run dbscan on all features
            predicted_classes = db_run.labels_
                    
            #produce a confusion matrix
            db_matrix = confusion_matrix(hand_classes, predicted_classes)
            #print(db_matrix)
            noise_true = IsItIdentifyingNoise(predicted_classes)
            #check main diagonal
            db_accuracy = matrix_accuracy(db_matrix)     
            accuracies.append(db_accuracy)
            
            db_precision = matrix_precision(db_matrix)
            avg_precision.append(np.average(db_precision))
            #print(db_precision)
            
            db_recall = matrix_recall(db_matrix)
            avg_recall.append(np.average(db_recall))
            
            with open(filedb, 'a') as file_object:
                #file_object.write("\n")
                file_object.write("\n eps value:" + str(eps_values[n]) + " min samples: " + str(min_samps[i]))
                if noise_true == 'True':
                    file_object.write("\n The 0th row and column represent a noise class (-1)")
                #file_object.write("\n")
                file_object.write("\n" + str(db_matrix) + "\n Accuracy:" + str(db_accuracy) + "\n Precisions:" + str(db_precision) + "\n Recalls:" + str(db_recall) + "\n")
        #then do color-coded plotting for the different ranges: 
        
    #plotting eps value ranges: 
    num_eps = len(eps_values)
    num_samps = len(min_samps)
    num_combos = num_eps*num_samps
    color_division = num_combos / 5
    y_axes = [accuracies, avg_precision, avg_recall]
    y_labels = ["accuracies", "average-precision", "average-recall"]
    for m in range(3):
        y_axis = y_axes[m]
        y_label = y_labels[m]
        for n in range(num_combos):
            k = n % num_eps #what eps value is it
            
            if n <= color_division: 
                color = 'red'
            elif color_division <n<= 2*color_division:
                color = 'pink'
            elif 2*color_division < n <= 3*color_division:
                color = 'green'
            elif 3*color_division < n <= 4*color_division:
                color = 'blue'
            elif n > 4*color_division:
                color = 'purple'
            
            plt.scatter(eps_values[k], y_axis[n], c = color)
        
        plt.xlabel("eps value")
        plt.ylabel(y_label)
        plt.title("sample range by color: red, pink, green, blue, purple, are increasing by # of samples")
        
        plt.savefig(path + "/dbscan-paramscan-" + y_label +"-eps-colored.pdf")
        plt.show()

    return accuracies, avg_precision, avg_recall

def matrix_accuracy(c_matrix):
    """calculate the accuracy of the matrix"""
    num_labels = len(c_matrix)
    total = np.sum(c_matrix, axis=None)
    diagonal = 0
    n = 0
    while n < num_labels:
        diagonal = diagonal + c_matrix[n][n]
        n = n+1
    accuracy = diagonal/total
    return accuracy

def IsItIdentifyingNoise(predicted_classes):
    """check if it identified any as a noise category """
    noise = "False"
    for n in range(len(predicted_classes)):
        if predicted_classes[n] == -1:
            noise = "True"
    return noise

def matrix_precision(matrix):
    """calculates the precision of each class"""
    precisions = []
    for n in range(len(matrix)):
        column = matrix[:,n]
        column_total = np.sum(column)
        
        correct = matrix[n][n]
        if column_total == 0:
            prec = 0
        else:
            prec = correct/column_total
        
        precisions.append(prec)
    
    return np.asarray(precisions)

def matrix_recall(matrix):
    """calculates the recall of each class"""
    recalls = []
    for n in range(len(matrix)):
        row = matrix[n]
        row_total = np.sum(row)
        
        correct = matrix[n][n]
        if row_total == 0:
            rec = 0
        else:
            rec = correct/row_total
        
        recalls.append(rec)
    
    return np.asarray(recalls)
