
"""
Created on Mon May 18 17:01:26 2020

@author: Lindsey Gordon 

Updated: May 31 2020
"""

import numpy as np
import numpy.ma as ma 
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)

from pylab import rcParams
rcParams['figure.figsize'] = 10,10
rcParams["lines.markersize"] = 2
from scipy.signal import argrelextrema


import astropy
from astropy.io import fits
import scipy.signal as signal
from astropy.stats import SigmaClip
from astropy.utils import exceptions
from astroquery import exceptions
from astroquery.exceptions import RemoteServiceError
#from astropy.utils.exceptions import AstropyWarning, RemoteServiceError

from datetime import datetime
import os
import shutil
from scipy.stats import moment, sigmaclip

import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import fnmatch

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor

import astroquery
from astroquery.simbad import Simbad
from astroquery.mast import Catalogs
from astroquery.mast import Observations

import shapely
from shapely import geometry
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
#%%
import classification_functions
from classification_functions import *
import data_functions
from data_functions import *
import plotting_functions
from plotting_functions import *


test_data() #should return 8 * 4
test_plotting()
#%%
#convert into fits files
#add TLS

#how to put a time limit on this bitch?
# there was something else to do that was relevant - redoing interpolation routines
#let's put the new features in
t, inty, targ, feats, notes = load_in_a_group(20,1,1,"/Users/conta/UROP_Spring_2020/")

#%%
import batman
import numba
#%%
from transitleastsquares import transitleastsquares
#%%
model = transitleastsquares(t, inty)
results = model.power()
print(results)
#%%
t=np.loadtxt("/Users/conta/UROP_Spring_2020/Sector20Cam1CCD1/Sector20Cam1CCD1_times_processed.txt")
inty = np.loadtxt("/Users/conta/UROP_Spring_2020/Sector20Cam1CCD1/Sector20Cam1CCD1_intensities_processed.txt")[0]

#%%%

def interp_norm_sigmaclip(yourpath, sector, camera, ccd):
    """interpolates, normalizes, and sigma clips all light curves
    then produces feature vectors for them"""
    folder_name = "Sector" + str(sector) + "Cam" + str(camera) + "CCD" + str(ccd)
    path = yourpath + folder_name
    fname_time_intensities_raw = path + "/" + folder_name + "_raw_lightcurves.fits"
    fname_lc_processed = path + "/" + folder_name + "_processed_lightcurves.fits"
    
    f = fits.open(fname_time_intensities_raw)
    t = f[0].data
    
    i = np.zeroes((len(f), len(t))) #f rows, t columns
    
    for n in range(len(f)): 
        i[n] = f[n].data
    
    interp_t, interp_i = interpolate_lc(i, t, flux_err=False, interp_tol=20./(24*60),num_sigma=5, orbit_gap_len = 3, DEBUG=False,spline_interpolate=True)
    i_norm = normalize(interp_i)
    
    hdr = fits.Header()
    hdu = fits.PrimaryHDU(interp_t, header=hdr)
    hdu.writeto(fname_lc_processed)
    
    for n in range(len(interp_i)):
        fits.append(interp_i[n], header=hdr)
    
    
    
    return interp_t, interp_i

def produce_save_feature_vectors(yourpath, times, intensities, targets, sector, camera, ccd):
    folder_name = "Sector" + str(sector) + "Cam" + str(camera) + "CCD" + str(ccd)
    path = yourpath + folder_name
    fname_features = path + "/"+ folder_name + "_features.fits"
    
    hdr = fits.Header()
    hdu = fits.PrimaryHDU(folder_name,header=hdr)
    hdu.writeto(fname_features)
    
    for n in range(len(intensities)):
        feat = featvec(times, intensities[n])
        fits.append(fnames_features, feat, header=hdr)
    


#%%

def featvec(x_axis, sampledata): 
    """calculates the feature vector of the single set of data (ie, intensity[0])
    currently returns 16: 
        0 - Average
        1 - Variance
        2 - Skewness
        3 - Kurtosis
        
        4 - ln variance
        5 - ln skewness
        6 - ln kurtosis
        
        (over 0.1 to 10 days)
        7 - maximum power
        8 - ln maximum power
        9 - period of maximum power
        
        10 - slope
        11 - ln slope
        
        (integration of periodogram over time frame)
        12 - P0 - 0.1-1 days
        13 - P1 - 1-3 days
        14 - P2 - 3-10 days
        
        (over 0-0.1 days, for moving objects)
        15 - Period of max power
        
        
        ***if you update the number of features, 
        you have to update the number of features in create_list_featvec!!!!"""
    #empty feature vector
    featvec = [] 
    #moments
    featvec.append(np.mean(sampledata)) #mean (don't use moment, always gives 0)
    featvec.append(moment(sampledata, moment = 2)) #variance
    featvec.append(moment(sampledata, moment = 3)) #skew
    featvec.append(moment(sampledata, moment = 4)) #kurtosis
    featvec.append(np.log(np.abs(moment(sampledata, moment = 2)))) #ln variance
    featvec.append(np.log(np.abs(moment(sampledata, moment = 3)))) #ln skew
    featvec.append(np.log(np.abs(moment(sampledata, moment = 4)))) #ln kurtosis
    
    #periods
    f = np.linspace(0.6, 62.8, 5000)  #period range converted to frequencies
    periods = np.linspace(0.1, 10, 5000)#0.1 to 10 day period
    pg = signal.lombscargle(x_axis, sampledata, f, normalize = True)
    rel_maxes = argrelextrema(pg, np.greater)
    
    powers = []
    indexes = []
    for n in range(len(rel_maxes[0])):
        index = rel_maxes[0][n]
        indexes.append(index)
        power_level_at_rel_max = pg[index]
        powers.append(power_level_at_rel_max)
    
    max_power = np.max(powers)
    index_of_max_power = np.argmax(powers)
    index_of_f_max = rel_maxes[0][index_of_max_power]
    f_max_power = f[index_of_f_max]
    period_max_power = 2*np.pi / f_max_power
    
    featvec.append(max_power)
    featvec.append(np.log(np.abs(max_power)))
    featvec.append(period_max_power)
    
    slope = stats.linregress(x_axis, sampledata)[0]
    featvec.append(slope)
    featvec.append(np.log(np.abs(slope)))
    
    #integrates the whole 0.1-10 day range
    integrating1 = np.trapz(pg[457:5000], periods[457:5000]) #0.1 days to 1 days
    integrating2 = np.trapz(pg[121:457], periods[121:457])#1-3 days
    integrating3 = np.trapz(pg[0:121], periods[0:121]) #3-10 days
    
    featvec.append(integrating1)
    featvec.append(integrating2)
    featvec.append(integrating3)
    
    #for 0.001 to 1 day periods
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
    #print("done")
    return(featvec) 
    



#%%
data_access_by_group_fits("/Users/conta/UROP_Spring_2020/", "/Users/conta/UROP_Spring_2020/all_targets_S020_v1.txt", 20, 2, 1)
#%%

def get_lc_file_and_data(yourpath, target):
    """ goes in, grabs the data for the target, gets the time index, intensity,
    etc. for the image. if connection error w/ MAST, skips it"""
    fitspath = yourpath + 'mastDownload/TESS/'
    targ = "TIC " + str(int(target))
    print(targ)
    try:
        #find and download data products for your target
        obs_table = Observations.query_object(targ, radius=".02 deg")
        data_products_by_obs = Observations.get_product_list(obs_table[0:5])
            
        #in theory, filter_products should let you sort out the non fits files but i 
        #simply could not get it to accept it despite following the API guidelines
        filter_products = Observations.filter_products(data_products_by_obs, dataproduct_type = 'timeseries')
        manifest = Observations.download_products(filter_products)
            
        #get all the paths to lc.fits files
        filepaths = []
        for root, dirs, files in os.walk(fitspath):
            for name in files:
                print(name)
                if name.endswith(("lc.fits")):
                    filepaths.append(root + "/" + name)
                    #print("appended", name, "to filepaths")
        
        print(len(filepaths))
        
        if len(filepaths) == 0: #if no lc.fits were downloaded, move on
            print(targ, "no light curve available")
            time1 = 0
            i1 = 0
        else: #if there are lc.fits files, open them and get the goods
                #get the goods and then close it
            f = fits.open(filepaths[0], memmap=False)
            time1 = f[1].data['TIME']
            i1 = f[1].data['PDCSAP_FLUX']                
            f.close()
                  
        #then delete all downloads in the folder, no matter what type
        if os.path.isdir("mastDownload") == True:
            shutil.rmtree("mastDownload")
            print("folder deleted")
            
        #corrects for connnection errors
    except (ConnectionError, OSError, TimeoutError, RemoteServiceError):
        print(targ + "could not be accessed due to an error")
        i1 = 0
        time1 = 0
    
    return time1, i1



t1, i1 = get_lc_file_and_data(mypath, 71560002)
#%%


classifications = np.loadtxt("/Users/conta/UROP_Spring_2020/Sector20Cam1CCD1/classified_Sector20Cam1CCD1.txt", delimiter = ' ')
classes = classifications[:,1]


#%%
filedb = "/Users/conta/UROP_Spring_2020/plot_output/6-5/dbscan-confusion-matrices-scan.txt"
hand_classes = classifications[:,1] #there are no class 5s for this group!!



path = "/Users/conta/UROP_Spring_2020/plot_output/6-5"

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

acc, avgp, avgr = dbscan_param_scan(path, feats, 0.2, 3, 0.2, 2, 50, 4, hand_classes)

#%%
#color coded plotting of eps/min samples
#plotting for min samples
p = 0
for n in range(len(accuracies)):
    k = n % 14 #tells you what eps value you're on
    if 0 <= k <= 3:
        color = 'red'
    elif 3 < k <=6:
        color = 'pink'
    elif 6 < k <= 9:
        color = 'green'
    elif 9 < k <= 13:
        color = 'blue'
    else:
        color = 'black'
    
    if n == 0:
        plt.scatter(min_samps[0], avg_precision[n], c = color)
    else:
        if n % 14 == 0:
            p = p + 1
        
        plt.scatter(min_samps[p], avg_precision[n], c = color)

plt.xlabel("min samples")
plt.ylabel("avg precision")
plt.title("red: eps 0.2-0.8, pink 1-1.4, green 1.6-2, blue 2.2-2.8")

plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/6-1/dbscan-paramscan-avg-precision-samples-colored.pdf")

#%%
#plotting for eps value ranges

for n in range(len(accuracies)):
    k = n % 14 #what eps value is it
    
    if n <= 70: 
        color = 'red'
    elif 70 <n<= 140:
        color = 'pink'
    elif 140 < n <= 210:
        color = 'green'
    elif 210 < n <= 280:
        color = 'blue'
    elif n > 280:
        color = 'purple'
    
    #color = 'blue'
    plt.scatter(eps_values[k], avg_precision[n], c = color)

plt.xlabel("eps value")
plt.ylabel('avg precision')
plt.title("sample range by color: red: 2-18, pink 22-38, green 42-58, blue 62-78, purple 82-98")

plt.savefig("/Users/conta/UROP_Spring_2020/plot_output/6-1/dbscan-paramscan-avg-precision-eps-colored.pdf")



#%%
#plotting to indicate both actual class and the class predicted by dbscan
#eps 2.2, min samples 18
path = "/Users/conta/UROP_Spring_2020/plot_output/6-5/color-shape-2D-plots-kmeans"
hand_classes = classifications[:,1] #there are no class 5s for this group!!


s