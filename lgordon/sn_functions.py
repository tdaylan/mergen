# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 23:07:49 2020

@author: conta
"""

#Supernovae fitting functions
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
# rcParams['lines.color'] = 'k'
from scipy.signal import argrelextrema

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

import data_access as da
import data_functions as df

  


############ HELPER FUNCTIONS ####################
def preclean_mcmc(file):
    t,ints,error = da.load_lygos_csv(file)

    mean = np.mean(ints)
    
    sigclip = SigmaClip(sigma=5, maxiters=None, cenfunc='median')
    clipped_inds = np.nonzero(np.ma.getmask(sigclip(ints)))
    ints[clipped_inds] = mean #reset those values to the mean value (or remove??)
    
    t_sub = t - 2457000 #resets it
    
    return t_sub, ints, error

def crop_to_40(t, y, err):
    """ only fit first 40% of brightness of curve"""
    brightness40 = (y.max() - y.min()) * 0.4

    for n in range(len(y)):
        if y[n] > brightness40:
            cutoffindex = n
            break
                
    t_40 = t[0:cutoffindex]
    ints_40 = y[0:cutoffindex]
    err_40 = err[0:cutoffindex]
    return t_40, ints_40, err_40
    


def plot_chain(path, targetlabel, plotlabel, sampler, labels, ndim):
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = labels
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    
    axes[-1].set_xlabel("step number");
    plt.savefig(path + targetlabel+ plotlabel)
    plt.show()
    return

def bin_8_hours(t, i, e):
    n_points = 16
    binned_t = []
    binned_i = []
    binned_e = []
    n = 0
    m = n_points
        
    while m <= len(t):
        bin_t = t[n + 8] #get the midpoint of this data as the point to plot at
        binned_t.append(bin_t) #put into new array
        bin_i = np.mean(i[n:m]) #bin the stretch
        binned_i.append(bin_i) #put into new array
        bin_e = np.sqrt(np.sum(e[n:m]**2)) / n_points #error propagates as sqrt(sum of squares of error)
        binned_e.append(bin_e)
            
        n+= n_points
        m+= n_points
        
    return np.asarray(binned_t), np.asarray(binned_i), np.asarray(binned_e)

def retrieve_quaternions_bigfiles(savepath, quaternion_folder, sector, x):
        
    for root, dirs, files in os.walk(quaternion_folder):
        for name in files:
            if name.endswith(("sector"+sector+"-quat.fits")):
                filepath = root + "/" + name
                
    tQ, Q1, Q2, Q3, outliers = df.extract_smooth_quaterions(savepath, filepath, None, 
                                           31, x)
    return tQ, Q1, Q2, Q3, outliers 

def produce_discovery_time_dictionary(filename):
    discovery_dictionary = {}
    with open(filename, 'r') as f:
        lines = f.readlines()[1:]
        from astropy.time import Time
        for line in lines:
            line = line.replace('\n', '')
            name, ra, dec, discoverytime = line.split(',')
            discotime = Time(discoverytime, format='iso', scale='utc')
            discotime = discotime.jd - 2457000
            discovery_dictionary[name] = discotime
    return discovery_dictionary

############### BIG BOYS ##############
def mcmc_access_all(folderpath, savepath, withquats = False, plot = True):
    
    if not withquats:
        all_best_params = np.zeros((2,5))
    else:
        all_best_params = np.zeros((2,6))
    all_t = [] 
    all_i = []
    all_e = []
    all_labels = []
    sector_list = []
    discovery_dictionary = {}
    
    for root, dirs, files in os.walk(folderpath):
        for name in files:
            if name.startswith(("rflxtarg")):
                filepath = root + "/" + name 
                print(name)
                label = name.split("_")
                filelabel = label[4] + label[5]
                all_labels.append(filelabel)
                sector = label[5][0:2]
                sector_list.append(sector)
                t,i,e = preclean_mcmc(filepath)
                if not withquats:
                    bestparams = run_mcmc_fit(savepath, filelabel, t,i,e, sector, plot = plot)
                else:
                    bestparams = run_mcmc_fit_w_quats(savepath, filelabel, t, i, e, 
                                                      sector, plot = plot)
                
                all_best_params = np.concatenate((all_best_params, bestparams))
                #print(all_best_params)
                all_t.append(t)
                all_i.append(i)
                all_e.append(e)
            elif name.startswith(("SNe")):
                filepath = root + "/" + name 
                print(name)
                discovery_dictionary = produce_discovery_time_dictionary(filepath)
                
                                

    return all_best_params, all_t, all_i, all_e, all_labels, sector_list, discovery_dictionary


    


def run_mcmc_fit(path, targetlabel, t, intensity, error, sector, plot = True,
                 quaternion_folder = "/users/conta/urop/quaternions/"):
    
    def log_likelihood(theta, x, y, yerr):
        """ calculates the log likelihood function. theta = [c0, c1, c2, c3, c4]"""
        c0, c1, c2, c3, c4 = theta
        model = c0 + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4
        yerr2 = yerr ** 2
        return -0.5 * np.sum((y - model) ** 2 / yerr2 + np.log(yerr2))
    
    #YYY, this probably needs something from the user?
    def log_prior(theta):
        """ calculates the log prior value and -2 < c3 < 2 and -2 < c4 < 2"""
        c0, c1, c2, c3, c4= theta
        #if -2.0 < c0 < 2 and -2 < c1 < 2 and -2 < c2 < 2 and -2 < c3 < 2 and -2 < c4 < 2:
        if -2 < c1 < 2 and -2 < c2 < 2 and -2 < c3 < 2 and -2 < c4 < 2:
                #what the fuck does this do
            return 0.0
        return -np.inf
        
        #log probability
    def log_probability(theta, x, y, yerr):
        """ calculates log probabilty"""
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, x, y, yerr)
    
    import matplotlib.pyplot as plt
    import emcee
    rcParams['figure.figsize'] = 16,6
     
    x = t
    y = intensity
    yerr = error
    
    #create quaternions -- load from other file
    #tQ, Q1, Q2, Q3, outliers = retrieve_quaternions(savepath, quaternion_folder, sector, x)
    tQ, Q1, Q2, Q3, outliers = df.metafile_load_smooth_quaternions(sector, x)
    
    nwalkers = 32 
    ndim = 5
    p0 = np.random.rand(nwalkers, ndim)
    labels = ["c0", "c1", "c2", "c3", "c4"]
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
    #burn in stage
    state = sampler.run_mcmc(p0, 5000, progress=True)
    if plot:
        plot_chain(path, targetlabel, "-burn-in-plot.png", sampler, labels, ndim)
    
    #clear and run again 
    sampler.reset()
    sampler.run_mcmc(state, 5000, progress=True);
    
    if plot:
        plot_chain(path, targetlabel, "-after-burn-plot.png", sampler, labels, ndim)
    
    
    flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)
    print(flat_samples.shape)
    

    #print out the best fit params based on 16th, 50th, 84th percentiles
    best_mcmc = np.zeros((1,5))
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        #q = np.diff(mcmc)
        #print(labels[i], mcmc[1], -1 * q[0], q[1] )
        best_mcmc[0][i] = mcmc[1]
    
    if plot:
        #corner plot the samples
        import corner
        fig = corner.corner(
            flat_samples, labels=labels,
            quantiles = [0.16, 0.5, 0.84],
                           show_titles=True,title_fmt = ".4f", title_kwargs={"fontsize": 12}
        );
        fig.savefig(path + targetlabel + 'corner-plot-params.png')
        plt.close()
        #plotting mcmc stuff
        for ind in range(len(flat_samples)):
            sample = flat_samples[ind]
            plt.plot(x, sample[0] + sample[1] * x + sample[2] * x**2 + sample[3] * x**3 + sample[4] * x**4, color = 'blue', alpha = 0.1)

        plt.scatter(x, y, label = "FFI data", color = 'gray')
        best_fit_model = best_mcmc[0][0] + best_mcmc[0][1] * x + best_mcmc[0][2] * x**2 + best_mcmc[0][3] * x**3 + best_mcmc[0][4] * x**4
        residual = y - best_fit_model
        plt.scatter(x, best_fit_model, label="best fit model", color = 'red')
        plt.scatter(x, residual, label = "residual", color = "green")
        
        t_8, i_8, e_8 = bin_8_hours(x, y, yerr)
        plt.errorbar(t_8, i_8, e_8, fmt='.k', color='black', label="binned data")
        
        plt.legend(fontsize=8, loc="upper left")
        plt.title(targetlabel)
        plt.savefig(path + targetlabel + "-MCMCmodel.png")
    
    return best_mcmc

def run_mcmc_fit_w_quats(path, targetlabel, t, intensity, error, sector, plot = True,
                 quaternion_folder = "/users/conta/urop/quaternions/"):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import emcee
    rcParams['figure.figsize'] = 16,6
     
    def log_likelihood(theta, x, y, yerr, Qall):
        """ calculates the log likelihood function. theta = [c0, c1, c2, c3, c4]"""
        c0, c1, c2, c3, c4, cQ = theta
        model = c0 + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4 + cQ * Qall
        yerr2 = yerr ** 2
        return -0.5 * np.sum((y - model) ** 2 / yerr2 + np.log(yerr2))
    
    #YYY, this probably needs something from the user?
    def log_prior(theta):
        """ calculates the log prior value and -2 < c3 < 2 and -2 < c4 < 2"""
        c0, c1, c2, c3, c4, cQ = theta
        #if -2.0 < c0 < 2 and -2 < c1 < 2 and -2 < c2 < 2 and -2 < c3 < 2 and -2 < c4 < 2:
        if -2 < c1 < 2 and -2 < c2 < 2 and -2 < c3 < 2 and -2 < c4 < 2:
                #what the fuck does this do
            return 0.0
        return -np.inf
        
        #log probability
    def log_probability(theta, x, y, yerr, Qall):
        """ calculates log probabilty"""
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, x, y, yerr, Qall)

    x = t
    y = intensity
    yerr = error
    
    
    #create quaternions -- load from other file
    tQ, Q1, Q2, Q3, outliers = df.metafile_load_smooth_quaternions(sector, x)
    Qall = Q1 + Q2 + Q3
    
    # correct length differences between tQ and x
    if len(x) > len(tQ): #main is longer, truncate main
        x = x[:len(tQ)]
        y = y[:len(tQ)]
        yerr = yerr[:len(tQ)]
    elif len(tQ) > len(x): # if  tQ is longer, truncate tQ
        tQ = tQ[:len(x)]
        Q1 = Q1[:len(x)]
        Q2 = Q2[:len(x)]
        Q3 = Q3[:len(x)]
        Qall = Qall[:len(x)]
        
    print(len(x), len(tQ), len(Q1))    
    nwalkers = 32 
    ndim = 6
    p0 = np.random.rand(nwalkers, ndim)
    labels = ["c0", "c1", "c2", "c3", "c4", "cQ"]
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr, Qall))
    state = sampler.run_mcmc(p0, 5000, progress=True)
    if plot:
        plot_chain(path, targetlabel, "-burn-in-plot.png", sampler, labels, ndim)
    
    #clear and run again 
    sampler.reset()
    sampler.run_mcmc(state, 5000, progress=True);
    
    if plot:
        plot_chain(path, targetlabel, "-after-burn-plot.png", sampler, labels, ndim)
    
    
    flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)
    print(flat_samples.shape)
    

    #print out the best fit params based on 16th, 50th, 84th percentiles
    best_mcmc = np.zeros((1,6))
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        #q = np.diff(mcmc)
        #print(labels[i], mcmc[1], -1 * q[0], q[1] )
        best_mcmc[0][i] = mcmc[1]
    
    if plot:
        #corner plot the samples
        import corner
        fig = corner.corner(
            flat_samples, labels=labels,
            quantiles = [0.16, 0.5, 0.84],
                           show_titles=True,title_fmt = ".4f", title_kwargs={"fontsize": 12}
        );
        fig.savefig(path + targetlabel + 'corner-plot-params.png')
        plt.close()
        #plotting mcmc stuff
        #for ind in range(100):
         #   sample = flat_samples[ind]
          #  model = sample[0] + sample[1] * x + sample[2] * x**2 + sample[3] * x**3 + sample[4] * x**4 + sample[5] * Qall
           # plt.plot(x,model , color = 'blue', alpha = 0.1)

        plt.scatter(x, y, label = "FFI data", color = 'gray')
        best_fit_model = best_mcmc[0][0] + best_mcmc[0][1] * x + best_mcmc[0][2] * x**2 + best_mcmc[0][3] * x**3 + best_mcmc[0][4] * x**4 +best_mcmc[0][5] * Qall
        residual = y - best_fit_model
        plt.scatter(x, best_fit_model, label="best fit model", color = 'red')
        plt.scatter(x, residual, label = "residual", color = "green")
        
        t_8, i_8, e_8 = bin_8_hours(x, y, yerr)
        plt.errorbar(t_8, i_8, e_8, fmt='.k', color='black', label="binned data")
        
        plt.legend(fontsize=8, loc="upper left")
        plt.title(targetlabel)
        plt.savefig(path + targetlabel + "-MCMCmodel.png")
    
    return best_mcmc

def plot_mcmc_all(folderpath, savepath, all_t, all_i, all_e, all_labels, discovery_times, all_best_params, withquats = False):
    import math
    def plot_histogram_mcmc(parameters, savepath):
        labels = ["c0", "c1", "c2", "c3", "c4"]
        for i in range(5):
            fig, ax1 = plt.subplots(figsize=(10,10))
            data = parameters[:,i]
            n_in, bins, patches = ax1.hist(data, 5)
            
            y_range = np.abs(n_in.max() - n_in.min())
            x_range = np.abs(data.max() - data.min())
            ax1.set_ylabel('Number of light curves')
            ax1.set_xlabel(labels[i])
            plt.savefig(savepath + labels[i] + "-histogram.png")
            plt.close()
        return
    
    
    plot_histogram_mcmc(all_best_params, savepath)

    ncols = 4
    nrows = 4
    ntotal = len(all_labels)
    #print(ntotal)
    num_figs = int(np.ceil(ntotal / ncols))
    #print(num_figs)
    m = 0
    
    #plotlabels = ['best fit', 'residuals', 'quaternions', 'background']
    for i in range(num_figs): # for each figure
        fig, ax = plt.subplots(nrows, ncols, sharex=False,
                               figsize=(8*ncols, 3*nrows))
        fig.suptitle("Plotting MCMC fits (" +  str(i) + ")")
                        
        if i == num_figs - 1 and ntotal % ncols != 0:
            num_curves_to_plot = ntotal % ncols
        else:
            num_curves_to_plot = ncols
                    
        for j in range(num_curves_to_plot): #loop thru columns
            tQ, Q1, Q2, Q3, outliers = df.metafile_load_smooth_quaternions(all_labels[m][9:11], all_t[m])
            Qall = Q1 + Q2 + Q3
                
                # correct length differences between tQ and x
            if len(all_t[m]) > len(tQ): #main is longer, truncate main
                all_t[m] = all_t[m][:len(tQ)]
                all_i[m] = y[:len(tQ)]
                all_e[m] = yerr[:len(tQ)]
            elif len(tQ) > len(all_t[m]): # if  tQ is longer, truncate tQ
                tQ = tQ[:len(all_t[m])]
                Q1 = Q1[:len(all_t[m])]
                Q2 = Q2[:len(all_t[m])]
                Q3 = Q3[:len(all_t[m])]
                Qall = Qall[:len(all_t[m])]
                
            if withquats:
                best_fit_model = all_best_params[m][0] + all_best_params[m][1] * all_t[m] + all_best_params[m][2] * all_t[m]**2 + all_best_params[m][3] * all_t[m]**3 + all_best_params[m][4] * all_t[m]**4 + all_best_params[m][5] * Qall
                linquadfit =  all_best_params[m][1] * all_t[m] + all_best_params[m][2] + all_best_params[m][5] * Qall
            else:
                best_fit_model = all_best_params[m][0] + all_best_params[m][1] * all_t[m] + all_best_params[m][2] * all_t[m]**2 + all_best_params[m][3] * all_t[m]**3 + all_best_params[m][4] * all_t[m]**4
                linquadfit = all_best_params[m][1] * all_t[m] + all_best_params[m][2]
             
            nuisanceparamfit =  all_best_params[m][0] + all_best_params[m][3] * all_t[m]**3 + all_best_params[m][4] * all_t[m]**4
            #main plot of data and model
            ax[0,j].set_title(all_labels[m])
            ax[0,j].scatter(all_t[m],all_i[m], label="FFI data", color = 'gray')
            ax[0,j].scatter(all_t[m], best_fit_model, label="best fit model", color = 'red')
            ax[0,j].legend(loc="upper left")
            
            #plot residuals
            ax[1,j].set_title("residuals")
            residual = all_i[m] - best_fit_model
            ax[1,j].scatter(all_t[m], residual, label = "residual", color = "green")
            
            #plot quaternions
            ax[2,j].set_title("quaternions")
            #sector = all_labels[m][9:11]
            ax[2,j].scatter(tQ, Q1)
            ax[2,j].scatter(tQ, Q2)
            ax[2,j].scatter(tQ, Q3)
            
            #plot linear and quadratic fit to curve
            ax[3,j].set_title("Component fits")
            ax[3,j].scatter(all_t[m],all_i[m], label="FFI data", color = 'gray')
            ax[3,j].scatter(all_t[m], linquadfit, label = "linear + quadratic fit", color = "red")
            
            ax[3,j].scatter(all_t[m], nuisanceparamfit, color = "blue", label = "nuisance parameter fit")
            ax[3,j].legend(loc="upper left")
            ax[3,j].set_ylim(ax[0,j].get_ylim())
            
            
            ax[-1, j].set_xlabel('Time [BJD - 2457000]')
            
            #plot discovery times on all of them: 
            for n in range(4): 
                discotime = discovery_times[all_labels[m][0:9]]
                ax[n,j].axvline(discotime, color = 'green')
                
            m+=1 #go to next set of curves/parameters for the next one
                
        for l in range(nrows-1):
            ax[l, 0].set_ylabel('Relative flux')
                                    
        fig.tight_layout()
        fig.savefig(savepath + "mcmc-all-" + str(i) + '.png')
        plt.show(fig)
                

    return