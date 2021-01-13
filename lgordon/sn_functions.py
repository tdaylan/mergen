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
def preclean_mcmc_polynomial(file):
    """ opens the file and cleans the data for use in MCMC modeling"""
    #load data
    t,ints,error = da.load_lygos_csv(file)

    #sigma clip - set outliers to the mean value because idk how to remove them
    #is sigma clipping fucking up quaternions?
    mean = np.mean(ints)
    
    sigclip = SigmaClip(sigma=4, maxiters=None, cenfunc='median')
    clipped_inds = np.nonzero(np.ma.getmask(sigclip(ints)))
    ints[clipped_inds] = mean #reset those values to the mean value (or remove??)
    
    
    t_sub = t - t.min() #put the start at 0 for better curve fitting
    
    ints = df.normalize(ints, axis=0)
    
    return t_sub, ints, error, t.min()

def preclean_mcmc_powerlaw(file):
    """ opens the file and cleans the data for use in MCMC modeling"""
    #load data
    t,ints,error = da.load_lygos_csv(file)

    #sigma clip - set outliers to the mean value because idk how to remove them
    #is sigma clipping fucking up quaternions?
    mean = np.mean(ints)
    
    sigclip = SigmaClip(sigma=4, maxiters=None, cenfunc='median')
    clipped_inds = np.nonzero(np.ma.getmask(sigclip(ints)))
    ints[clipped_inds] = mean #reset those values to the mean value (or remove??)
    
    
    t -= 2457000
    
    ints = df.normalize(ints, axis=0)
    
    return t, ints, error

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

def produce_discovery_time_dictionary(filename, t_starts):
    """ returns the discovery time MINUS the start time of the sector"""
    discovery_dictionary = {}
    with open(filename, 'r') as f:
        lines = f.readlines()[1:]
        from astropy.time import Time
        for line in lines:
            line = line.replace('\n', '')
            name, ra, dec, discoverytime = line.split(',')
            discotime = Time(discoverytime, format='iso', scale='utc')
            try: 
                discotime = discotime.jd - t_starts[name[2:]]
            except KeyError:
                discotime = discotime.jd
            discovery_dictionary[name[2:]] = discotime
    return discovery_dictionary

def generate_clip_quats_cbvs(sector, x, y, yerr, targetlabel, CBV_folder):
    tQ, Q1, Q2, Q3, outliers = df.metafile_load_smooth_quaternions(sector, x)
    Qall = Q1 + Q2 + Q3
    #load CBVs
    camera = targetlabel[-2]
    ccd = targetlabel[-1]
    cbv_file = CBV_folder + "s00{sector}/cbv_components_s00{sector}_000{camera}_000{ccd}.txt".format(sector = sector,
                                                                                          camera = camera,
                                                                                          ccd = ccd)
    cbvs = np.genfromtxt(cbv_file)
    CBV1 = cbvs[:,0]
    CBV2 = cbvs[:,1]
    CBV3 = cbvs[:,2]
    #correct length differences:
    lengths = np.array((len(x), len(tQ), len(CBV1)))
    length_corr = lengths.min()
    x = x[:length_corr]
    y = y[:length_corr]
    yerr = yerr[:length_corr]
    tQ = tQ[:length_corr]
    Qall = Qall[:length_corr]
    CBV1 = CBV1[:length_corr]
    CBV2 = CBV2[:length_corr]
    CBV3 = CBV3[:length_corr]
    return x,y,yerr, tQ, Qall, CBV1, CBV2, CBV3

############### BIG BOYS ##############
def mcmc_access_all(folderpath, savepath, polynomial = True, newlabels = False):
    """ Opens all Lygos files and loads them in."""
    
    all_t = [] 
    t_starts = {}
    all_i = []
    all_e = []
    all_labels = []
    sector_list = []
    discovery_dictionary = {}
    
    for root, dirs, files in os.walk(folderpath):
        for name in files:
            if name.startswith(("SNe")):
                filepath = root + "/" + name 
                print(name)
                discovery_dictionary = produce_discovery_time_dictionary(filepath, t_starts)

            if newlabels: #new file labelling
                if name.startswith(("rflx")):
                    filepath = root + "/" + name 
                    print(name)
                    label = name.split("_")
                    filelabel = label[1] + label[2]
                    all_labels.append(filelabel)
                    sector = label[2][0:2]
                    sector_list.append(sector)
                    if polynomial:
                        t,i,e, t_start = preclean_mcmc_polynomial(filepath)
                        t_starts[label[1]] = t_start
                    else:
                        t,i,e = preclean_mcmc_powerlaw(filepath)
                        t_starts[label[1]] = 2457000
                    
                    all_t.append(t)
                    all_i.append(i)
                    all_e.append(e)
            
            else: #old file labelling
                if name.startswith(("rflx")):
                    filepath = root + "/" + name 
                    print(name)
                    label = name.split("_")
                    filelabel = label[4] + label[5]
                    all_labels.append(filelabel[2:])
                    sector = label[5][0:2]
                    sector_list.append(sector)
                    if polynomial:
                        t,i,e, t_start = preclean_mcmc_polynomial(filepath)
                        t_starts[label[4]] = t_start
                    else:
                        t,i,e = preclean_mcmc_powerlaw(filepath)
                        t_starts[label[4]] = 2457000
                    
                    all_t.append(t)
                    all_i.append(i)
                    all_e.append(e)
                    
    return all_t, all_i, all_e, all_labels, sector_list, discovery_dictionary, t_starts


def produce_all_best_params(savepath, all_labels, all_t, all_i, all_e,
                            sector_list, discovery_times, t_starts, 
                            plot = False,polynomial = True, 
                            savefile = None, sn_names = None):
    """ produces all best parameter sets for the given light curves. 
    returns parameters, upper errors, and lower errors"""
    num_curves = len(all_i)
    if polynomial: 
        ndim = 9
    else: 
        ndim = 7
    
    
    upper_error = np.zeros((num_curves, ndim))
    lower_error = np.zeros((num_curves, ndim))
    all_best_params = np.zeros((num_curves, ndim))
    
    
    for k in range(num_curves):
        
        t = all_t[k]
        i = all_i[k]
        e = all_e[k]
        filelabel = all_labels[k]
        sector = sector_list[k]
        
        if polynomial: 
            bestparams, uppererror, lowererror = mcmc_fit_polynomial_heaviside(savepath, filelabel, t, i, e, 
                                  sector, discovery_times, t_starts, plot = plot, 
                                  savefile = savefile, sn_names = sn_names)
            
        else:
            #offset the discovery time by like five days - if this pushes negative, do not?
            if discovery_times[all_labels[k][:-4]] - 5 > 0:
                
                t_start = discovery_times[all_labels[k][:-4]] - 5
            else:
                t_start = 0
            bestparams, uppererror, lowererror = run_mcmc_fit_stepped_powerlaw(savepath, filelabel, t,i,e, sector, t_start, 
                                                                              discovery_times, t_starts, plot = plot, savefile = savefile, sn_names = sn_names)
               
        all_best_params[k] = bestparams
        upper_error[k] = uppererror
        lower_error[k] = lowererror
     
    return all_best_params, upper_error, lower_error



def run_mcmc_fit_stepped_powerlaw(path, targetlabel, t, intensity, error, sector, t0,
                                  discovery_times, t_starts, plot = True, 
                                 quaternion_folder = "/users/conta/urop/quaternions/", 
                                 CBV_folder = "C:/Users/conta/.eleanor/metadata/", 
                                 savefile = None, sn_names = None):
    """ Runs MCMC fitting for stepped power law fit
    
    """
    
    
    def log_likelihood(theta, x, y, yerr, t0):
        """ calculates the log likelihood function. 
        constrain beta between 0.5 and 4.0
        A is positive
        need to add in cQ and CBVs!!
        only fit up to 40% of the flux"""
        A, beta, B, cQ, cbv1, cbv2, cbv3 = theta #, cQ, cbv1, cbv2, cbv3
        #print(A, beta, B)
        model = np.heaviside((x - t0), 1) * A *(x-t0)**beta + B + cQ * Qall + cbv1 * CBV1 + cbv2 * CBV2 + cbv3 * CBV3
        
        yerr2 = yerr**2.0
        returnval = -0.5 * np.nansum((y - model) ** 2 / yerr2 + np.log(yerr2))
        return returnval
    
    def log_prior(theta):
        """ calculates the log prior value """
        A, beta, B, cQ, cbv1, cbv2, cbv3 = theta #, cQ, cbv1, cbv2, cbv3
        #print(A, beta, B, cQ, cbv1, cbv2, cbv3)
        if 0.5 < beta < 6.0 and 0.0 < A < 5.0 and -10 < B < 10 and -5000 < cQ < 5000 and -5000 < cbv1 < 5000 and -5000 < cbv2 < 5000 and -5000 < cbv3 < 5000:
            return 0.0
        return -np.inf
        
        #log probability
    def log_probability(theta, x, y, yerr, t0):
        """ calculates log probabilty"""
        lp = log_prior(theta)
            
        if not np.isfinite(lp) or np.isnan(lp): #if lp is not 0.0
            return -np.inf
        
        return lp + log_likelihood(theta, x, y, yerr, t0)
    
    import matplotlib.pyplot as plt
    import emcee
    rcParams['figure.figsize'] = 16,6
     
    x = t
    y = intensity
    yerr = error
    
    #load quaternions and CBVs
    x,y,yerr, tQ, Qall, CBV1, CBV2, CBV3 = generate_clip_quats_cbvs(sector, x, y, yerr,targetlabel, CBV_folder)
        
    
    #running MCMC
    np.random.seed(42)   
    nwalkers = 32
    ndim = 7
    labels = ["A", "beta", "B", "cQ", "cbv1", "cbv2", "cbv3"] #, "cQ", "cbv1", "cbv2", "cbv3"
    p0 = np.ones((nwalkers, ndim)) + 1 * np.random.rand(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr, t0))
    
   # try:
    state = sampler.run_mcmc(p0, 15000, progress=True)
    if plot:
        plot_chain(path, targetlabel, "-burn-in-plot.png", sampler, labels, ndim)
    
    
    flat_samples = sampler.get_chain(discard=4000, thin=15, flat=True)
    print(flat_samples.shape)

    #print out the best fit params based on 16th, 50th, 84th percentiles
    best_mcmc = np.zeros((1,ndim))
    upper_error = np.zeros((1,ndim))
    lower_error = np.zeros((1,ndim))
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(labels[i], mcmc[1], -1 * q[0], q[1] )
        best_mcmc[0][i] = mcmc[1]
        upper_error[0][i] = q[1]
        lower_error[0][i] = q[0]
        
    print(best_mcmc)
    
    if plot:
        #corner plot the samples
        import corner
        fig = corner.corner(
            flat_samples, labels=labels,
            quantiles = [0.16, 0.5, 0.84],
                           show_titles=True,title_fmt = ".4f", title_kwargs={"fontsize": 12}
        );
        fig.savefig(path + targetlabel + 'corner-plot-params.png')
        plt.show()
        plt.close()
        

        plt.scatter(x, y, label = "FFI data", color = 'gray')
         
        #raw best fit model
        #print(best_mcmc[0][0])
        best_fit_model = np.heaviside((x - t0), 1) * best_mcmc[0][0] * ((x-t0)**(best_mcmc[0][1])) + best_mcmc[0][2]
        best_fit_model = best_fit_model + best_mcmc[0][3] * Qall + best_mcmc[0][4] * CBV1 + best_mcmc[0][5] * CBV2 + best_mcmc[0][6] * CBV3
        
        #residual = y - best_fit_model
        plt.scatter(x, best_fit_model, label="best fit model", s = 5, color = 'red')
        
        discotime = discovery_times[targetlabel[:-4]]
        plt.axvline(discotime, color = 'green')
        
        plt.legend(fontsize=8, loc="upper left")
        plt.title(targetlabel)
        plt.xlabel("BJD")
        #plt.show()
        plt.savefig(path + targetlabel + "-MCMCmodel.png")
        
        
    if savefile is not None:
        with open(savefile, 'a') as f:
            for i in range(ndim):
                f.write(str(best_mcmc[0][i]))
            f.write("\n")
        with open(sn_names, 'a') as f:
            f.write(targetlabel)
            f.write("\n")
    return best_mcmc, upper_error, lower_error

def run_mcmc_fit_stepped_powerlaw_nocbvs(path, targetlabel, t, intensity, error, sector, t0,
                                  discovery_times, t_starts, plot = True, 
                                 quaternion_folder = "/users/conta/urop/quaternions/", 
                                 CBV_folder = "C:/Users/conta/.eleanor/metadata/", 
                                 savefile = None, sn_names = None):
    """ Runs MCMC fitting for stepped power law fit
    
    """
    
    
    def log_likelihood(theta, x, y, yerr, t0):
        """ calculates the log likelihood function. 
        constrain beta between 0.5 and 4.0
        A is positive
        need to add in cQ and CBVs!!
        only fit up to 40% of the flux"""
        A, beta, B = theta #, cQ, cbv1, cbv2, cbv3
        #print(A, beta, B)
        model = np.heaviside((x - t0), 1) * A *(x-t0)**beta + B
        
        yerr2 = yerr**2.0
        returnval = -0.5 * np.nansum((y - model) ** 2 / yerr2 + np.log(yerr2))
        return returnval
    
    def log_prior(theta):
        """ calculates the log prior value """
        A, beta, B= theta #, cQ, cbv1, cbv2, cbv3
        #print(A, beta, B, cQ, cbv1, cbv2, cbv3)
        if 0.5 < beta < 6.0 and 0.0 < A < 5.0 and -10 < B < 10:
            return 0.0
        return -np.inf
        
        #log probability
    def log_probability(theta, x, y, yerr, t0):
        """ calculates log probabilty"""
        lp = log_prior(theta)
            
        if not np.isfinite(lp) or np.isnan(lp): #if lp is not 0.0
            return -np.inf
        
        return lp + log_likelihood(theta, x, y, yerr, t0)
    
    import matplotlib.pyplot as plt
    import emcee
    rcParams['figure.figsize'] = 16,6
     
    x = t
    y = intensity
    yerr = error
    
    #load quaternions and CBVs
    #x,y,yerr, tQ, Qall, CBV1, CBV2, CBV3 = generate_clip_quats_cbvs(sector, x, y, yerr,targetlabel, CBV_folder)
        
    
    #running MCMC
    np.random.seed(42)   
    nwalkers = 32
    ndim = 3
    labels = ["A", "beta", "B"] #, "cQ", "cbv1", "cbv2", "cbv3"
    p0 = np.ones((nwalkers, ndim)) + 1 * np.random.rand(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr, t0))
    
   # try:
    state = sampler.run_mcmc(p0, 15000, progress=True)
    if plot:
        plot_chain(path, targetlabel, "-burn-in-plot.png", sampler, labels, ndim)
    
    
    flat_samples = sampler.get_chain(discard=4000, thin=15, flat=True)
    print(flat_samples.shape)

    #print out the best fit params based on 16th, 50th, 84th percentiles
    best_mcmc = np.zeros((1,ndim))
    upper_error = np.zeros((1,ndim))
    lower_error = np.zeros((1,ndim))
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(labels[i], mcmc[1], -1 * q[0], q[1] )
        best_mcmc[0][i] = mcmc[1]
        upper_error[0][i] = q[1]
        lower_error[0][i] = q[0]
        
    print(best_mcmc)
    
    if plot:
        #corner plot the samples
        import corner
        fig = corner.corner(
            flat_samples, labels=labels,
            quantiles = [0.16, 0.5, 0.84],
                           show_titles=True,title_fmt = ".4f", title_kwargs={"fontsize": 12}
        );
        fig.savefig(path + targetlabel + 'corner-plot-params.png')
        plt.show()
        plt.close()
        

        plt.scatter(x, y, label = "FFI data", color = 'gray')
         
        #raw best fit model
        #print(best_mcmc[0][0])
        best_fit_model = np.heaviside((x - t0), 1) * best_mcmc[0][0] * ((x-t0)**(best_mcmc[0][1])) + best_mcmc[0][2]
        #best_fit_model = best_fit_model + best_mcmc[0][3] * Qall + best_mcmc[0][4] * CBV1 + best_mcmc[0][5] * CBV2 + best_mcmc[0][6] * CBV3
        
        #residual = y - best_fit_model
        plt.scatter(x, best_fit_model, label="best fit model", s = 5, color = 'red')
        
        discotime = discovery_times[targetlabel[:-4]]
        plt.axvline(discotime, color = 'green')
        
        plt.legend(fontsize=8, loc="upper left")
        plt.title(targetlabel)
        plt.xlabel("BJD")
        #plt.show()
        plt.savefig(path + targetlabel + "-MCMCmodel.png")
        
        
    if savefile is not None:
        with open(savefile, 'a') as f:
            for i in range(ndim):
                f.write(str(best_mcmc[0][i]))
            f.write("\n")
        with open(sn_names, 'a') as f:
            f.write(targetlabel)
            f.write("\n")
    return best_mcmc, upper_error, lower_error

def mcmc_fit_polynomial_heaviside(path, targetlabel, t, intensity, error, 
                                  sector, discovery_times,t_starts, plot = True,
                                  quaternion_folder = "/users/conta/urop/quaternions/",
                                  CBV_folder = "C:/Users/conta/.eleanor/metadata/", 
                                  savefile = None, sn_names = None):
    """ Runs MCMC fitting, mandatory quaternions AND CBV_folder"""
    
    
    def log_likelihood(theta, x, y, yerr):
        """ calculates the log likelihood function. """
        t0, c0, c1, c2, cQ, cbv1, cbv2, cbv3 = theta
        t1 = x - t0
        model = np.heaviside((c1 * t1 + c2 * t1 **2), 1) + c0 + cQ * Qall + cbv1 * CBV1 + cbv2 * CBV2 + cbv3 * CBV3
        yerr2 = yerr ** 2
        return -0.5 * np.nansum((y - model) ** 2 / yerr2 + np.log(yerr2))
    
    def log_prior(theta):
        """ calculates the log prior value """
        t0, c0, c1, c2, cQ, cbv1, cbv2, cbv3 = theta
        if -2.0 < c0 < 2 and -2 < c1 < 2 and -2 < c2 < 2 and 0 < t0 < 30:
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
    
    ndim = 8
    print(int(sector))
    if int(sector) > 26:
        print("sector out of range")
        return np.zeros((1,ndim))
    else: 
        x = t
        y = intensity
        yerr = error
        
        np.random.seed(42)
               
        nwalkers = 32 
        
        #load quaternions and CBVs
        x,y,yerr, tQ, Qall, CBV1, CBV2, CBV3 = generate_clip_quats_cbvs(sector, x, y, yerr,targetlabel, CBV_folder)
        
        #running MCMC
        labels = ["t0", "c0", "c1", "c2", "cQ", "cbv1", "cbv2", "cbv3"]
        p0 = np.zeros((nwalkers, ndim)) + 1e-2 * (np.random.rand(nwalkers, ndim) - 0.5)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
        state = sampler.run_mcmc(p0, 15000, progress=True)
        
        if plot:
            plot_chain(path, targetlabel, "-burn-in-plot.png", sampler, labels, ndim)
        
        
        flat_samples = sampler.get_chain(discard=5000, thin=15, flat=True)
        best_mcmc = np.zeros((1,ndim))
        upper_error = np.zeros((1,ndim))
        lower_error = np.zeros((1,ndim))
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            #print(labels[i], mcmc[1], -1 * q[0], q[1] )
            best_mcmc[0][i] = mcmc[1]
            upper_error[0][i] = q[1]
            lower_error[0][i] = q[0]
        print(best_mcmc)
    
        
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
            for ind in range(100):
                sample = flat_samples[ind]
                t1 = x - sample[0]
                c0 = sample[1]
                c1 = sample[2]
                c2 = sample[3]
                cQ = sample[4]
                cbv1 = sample[5]
                cbv2 = sample[6]
                cbv3 = sample[7]
                
                modeltoplot = np.heaviside((c1 * t1 + c2 * t1 **2), 1) + c0 + cQ * Qall + cbv1 * CBV1 + cbv2 * CBV2 + cbv3 * CBV3
                plt.plot(x, modeltoplot, color = 'blue', alpha = 0.1)
            
    
            plt.scatter(x, y, label = "FFI data", color = 'gray')
             
            #raw best fit model
            t1 = x - best_mcmc[0][0]
            c0 = best_mcmc[0][1]
            c1 = best_mcmc[0][2]
            c2 = best_mcmc[0][3]
            cQ = best_mcmc[0][4]
            cbv1 = best_mcmc[0][5]
            cbv2 = best_mcmc[0][6]
            cbv3 = best_mcmc[0][7]
            best_fit_model = np.heaviside((c1 * t1 + c2 * t1 **2), 1) + c0 + cQ * Qall + cbv1 * CBV1 + cbv2 * CBV2 + cbv3 * CBV3
            
            plt.scatter(x, best_fit_model, label = "best fit")
            
            discotime = discovery_times[targetlabel[:-4]] - t_starts["SN" + targetlabel[:-4]]
            
            plt.axvline(discotime, color = 'green')
            plt.legend(fontsize=8, loc="upper left")
            plt.title(targetlabel)
            plt.xlabel("BJD")
            plt.savefig(path + targetlabel + "-MCMC-polynomial-heaviside.png")
        
        if savefile is not None:
            with open(savefile, 'a') as f:
                for i in range(ndim):
                    f.write(str(best_mcmc[0][i]))
                f.write("\n")
            with open(sn_names, 'a') as f:
                f.write(targetlabel)
                f.write("\n")
            
            
        return best_mcmc, upper_error, lower_error

def run_mcmc_fit_polynomial(path, targetlabel, t, intensity, error, sector, discovery_times,
                 plot = True,
                 quaternion_folder = "/users/conta/urop/quaternions/",
                 CBV_folder = "C:/Users/conta/.eleanor/metadata/", savefile = None,
                 sn_names = None):
    """ Runs MCMC fitting, mandatory quaternions AND CBV_folder"""
    
    
    def log_likelihood(theta, x, y, yerr):
        """ calculates the log likelihood function. theta = [c0, c1, c2, c3, c4]"""
        c0, c1, c2, c3, c4, cQ, cbv1, cbv2, cbv3 = theta
        model = c0 + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4 + cQ * Qall + cbv1 * CBV1 + cbv2 * CBV2 + cbv3 * CBV3
        yerr2 = yerr ** 2
        return -0.5 * np.sum((y - model) ** 2 / yerr2 + np.log(yerr2))
    
    def log_prior(theta):
        """ calculates the log prior value """
        c0, c1, c2, c3, c4, cQ, cbv1, cbv2, cbv3 = theta
        if -2.0 < c0 < 2 and -2 < c1 < 2 and -2 < c2 < 2 and -2 < c3 < 2 and -2 < c4 < 2:
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
    
    ndim = 9
    print(int(sector))
    if int(sector) > 26:
        print("sector out of range")
        return np.zeros((1,ndim))
    else: 
        x = t
        y = intensity
        yerr = error
        
        np.random.seed(42)
               
        nwalkers = 32 
        
        #load quaternions and CBVs
        x,y,yerr, tQ, Qall, CBV1, CBV2, CBV3 = generate_clip_quats_cbvs(sector, x, y, yerr,targetlabel, CBV_folder)
        
        #running MCMC
        
        labels = ["c0", "c1", "c2", "c3", "c4", "cQ", "cbv1", "cbv2", "cbv3"]
        p0 = np.zeros((nwalkers, ndim)) + 1e-2 * (np.random.rand(nwalkers, ndim) - 0.5)
        #p0[:,5] += 10000
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
        
        try:
            state = sampler.run_mcmc(p0, 15000, progress=True)
            if plot:
                plot_chain(path, targetlabel, "-burn-in-plot.png", sampler, labels, ndim)
            
            
            flat_samples = sampler.get_chain(discard=5000, thin=15, flat=True)
            print(flat_samples.shape)
        
            #print out the best fit params based on 16th, 50th, 84th percentiles
            best_mcmc = np.zeros((1,ndim))
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
                for ind in range(100):
                    sample = flat_samples[ind]
                    modeltoplot = sample[0] + sample[1] * x + sample[2] * x**2 + sample[3] * x**3 + sample[4] * x**4 + sample[5] * Qall
                    modeltoplot = modeltoplot + sample[6] * CBV1 + sample[7] * CBV2 + sample[8] * CBV3
                    plt.plot(x, modeltoplot, color = 'blue', alpha = 0.1)
                
        
                plt.scatter(x, y, label = "FFI data", color = 'gray')
                 
                #raw best fit model
                best_fit_model = best_mcmc[0][0] + best_mcmc[0][1] * x + best_mcmc[0][2] * x**2 + best_mcmc[0][3] * x**3 + best_mcmc[0][4] * x**4 
                
                
                
                residual = y - best_fit_model
                #plt.scatter(x, best_fit_model, label="best fit model", color = 'red')
                #plt.scatter(x, residual, label = "residual", color = "green")
                
                
                #best fit model with cbv/quats in it
                best_fit_model = best_fit_model + best_mcmc[0][5] * Qall + best_mcmc[0][6] * CBV1 + best_mcmc[0][7] * CBV2 + best_mcmc[0][8] * CBV3
                plt.scatter(x, best_fit_model, label = "best fit with correction terms")
                
                discotime = discovery_times[targetlabel[0:9]]
                plt.axvline(discotime, color = 'green')
                plt.legend(fontsize=8, loc="upper left")
                plt.title(targetlabel)
                plt.xlabel("BJD")
                plt.savefig(path + targetlabel + "-MCMCmodel.png")
            
            if savefile is not None:
                with open(savefile, 'a') as f:
                    for i in range(ndim):
                        f.write(str(best_mcmc[0][i]))
                    f.write("\n")
                with open(sn_names, 'a') as f:
                    f.write(targetlabel)
                    f.write("\n")
                
                
            return best_mcmc
        except ValueError:
            print("Probability function returned Nan")
            return np.zeros((1,ndim))





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
                if n == 2: 
                    continue
                else:
                    discotime = discovery_times[all_labels[m][0:9]]
                    ax[n,j].axvline(discotime, color = 'green')
                
            m+=1 #go to next set of curves/parameters for the next one
                
        for l in range(nrows-1):
            ax[l, 0].set_ylabel('Relative flux')
                                    
        fig.tight_layout()
        fig.savefig(savepath + "mcmc-all-" + str(i) + '.png')
        plt.show(fig)
                

    return