# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 23:07:49 2020

@author: conta
"""

#Supernovae fitting functions

############ HELPER FUNCTIONS ####################
def preclean_mcmc(file):
    t,ints,error = da.load_lygos_csv(file)

    mean = np.mean(ints)
    
    sigclip = SigmaClip(sigma=5, maxiters=None, cenfunc='median')
    clipped_inds = np.nonzero(np.ma.getmask(sigclip(ints)))
    ints[clipped_inds] = mean #reset those values to the mean value (or remove??)
    
    t_sub = t - round(t.min()) #puts it at ~0 start
    
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
    
def log_likelihood(theta, x, y, yerr):
    """ calculates the log likelihood function. theta = [c1, c2, c3]"""
    c0, c1, c2, c3, c4 = theta
    model = c0 + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4
    yerr2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / yerr2 + np.log(yerr2))

#YYY, this probably needs something from the user?
def log_prior(theta):
        """ calculates the log prior value and -2 < c3 < 2 and -2 < c4 < 2"""
    c0, c1, c2, c3, c4= theta
    if -2.0 < c0 < 2 and -2 < c1 < 2 and -2 < c2 < 2 and -2 < c3 < 2 and -2 < c4 < 2:
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

def plot_chain(targetlabel, plotlabel, sampler, labels, ndim):
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
        bin_e = np.sqrt(np.sum(e[n:m]**2)) #error propagates as sqrt(sum of squares of error)
        binned_e.append(bin_e)
            
        n+= n_points
        m+= n_points
        
    return np.asarray(binned_t), np.asarray(binned_i), np.asarray(binned_e)

def retrieve_quaternions_bigfiles(savepath, quaternion_folder, sector, x):
        
    for root, dirs, files in os.walk(quaternion_folder):
        for name in files:
            if name.endswith(("sector"+sector+"-quat.fits")):
                filepath = root + "/" + name
                
    tQ, Q1, Q2, Q3, outliers = extract_smooth_quaterions(savepath, filepath, None, 
                                           31, x)
    return tQ, Q1, Q2, Q3, outliers 

############### BIG BOYS ##############
def run_mcmc_fit(path, targetlabel, t, intensity, error, sector, plot = True,
                 quaternion_folder = "/users/conta/urop/quaternions/"):
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    import emcee
    rcParams['figure.figsize'] = 16,6
     

    x = t
    y = intensity
    yerr = error
    
    
    #create quaternions -- load from other file
    #tQ, Q1, Q2, Q3, outliers = retrieve_quaternions(savepath, quaternion_folder, sector, x)
    
    nwalkers = 32 
    ndim = 5
    p0 = np.random.rand(nwalkers, ndim)
    labels = ["c0", "c1", "c2", "c3", "c4"]
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
    #burn in stage
    state = sampler.run_mcmc(p0, 5000, progress=True)
    if plot:
        plot_chain(targetlabel, "-burn-in-plot.png", sampler, labels, ndim)
    
    #clear and run again 
    sampler.reset()
    sampler.run_mcmc(state, 5000, progress=True);
    
    if plot:
        plot_chain(targetlabel, "-after-burn-plot.png", sampler, labels, ndim)
    
    #corner plot the samples
    flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)
    print(flat_samples.shape)
    

    #print out the best fit params based on 16th, 50th, 84th percentiles
    best_mcmc = np.zeros((1,5))
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        #print(labels[i], mcmc[1], -1 * q[0], q[1] )
        best_mcmc[0][i] = mcmc[1]
    
    if plot:
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