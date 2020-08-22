# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 22:19:31 2020

@author: Lindsey Gordon @lcgordon

All functions for accessing FFI light curves + producing their custom feature vectors

Functions: 
    * eleanor_lc  - pulls data from ra and dec list
    * open_eleanor_lc_files  - opens the saved light curves
    * create_save_featvec_different_timeaxes - produces featvecs and saves them
    * build_simbad_extragalactic_database  - produces text file list of the targets
"""
import numpy as np
import numpy.ma as ma 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)

import scipy.signal as signal
from scipy.stats import moment
from scipy import stats
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
rcParams["lines.markersize"] = 2

import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import fnmatch

import os
import shutil
from scipy.stats import moment, sigmaclip

import astropy
from astropy.io import fits
import scipy.signal as signal
from astropy.stats import SigmaClip
from astropy.utils import exceptions

import pdb
import fnmatch as fm

import plotting_functions as pf
import data_functions as df


def test_data():
    """make sure the module loads in"""
    print("FFI functions loaded in.")

def eleanor_lc(path, ra_declist, plotting = False):
    """ 
    retrieves + produces eleanor light curves from FFI files
    """
    import eleanor
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    import warnings
    warnings.filterwarnings('ignore')
    from eleanor.utils import SearchError
    
    gaia_ids = []
    filename = path + "eleanor_lightcurves_from_radeclist.fits"
    
    for n in range(len(ra_declist)):
        try:
            coords = SkyCoord(ra=ra_declist[n][0], dec=ra_declist[n][1], unit=(u.deg, u.deg))
            #try:
            files = eleanor.Source(coords=coords, tic=0) #by not providing a sector argument, will ONLY retrieve most recent sector
            print('Found TIC {0} (Gaia {1}), with TESS magnitude {2}, RA {3}, and Dec {4}'
                         .format(files.tic, files.gaia, files.tess_mag, files.coords[0], files.coords[1]))
            data = eleanor.TargetData(files)
            plt.figure(figsize=(16,6))
    
            q = data.quality == 0
            if plotting and n % 20 == 0: 
                    plt.scatter(data.time[q], data.raw_flux[q]/np.nanmedian(data.raw_flux[q])+0.06, c='black', s=0.5)
                    plt.scatter(data.time[q], data.corr_flux[q]/np.nanmedian(data.corr_flux[q]) + 0.03, c='red', s=0.5)
                    plt.ylabel('Normalized Flux')
                    plt.xlabel('Time [BJD - 2457000]')
                    plt.title("(" + str(ra_declist[n][0]) + str(ra_declist[n][1]) + ")")
                    plt.savefig(path + str(n) + 'lightcurveplotted.png')
                    plt.show()
                    plt.close()
                
            fluxandtime = [data.time[q], data.raw_flux[q]]
            lightcurve = np.asarray(fluxandtime)
                #print(lightcurve)
            if n == 0: #setting up fits file + save first one            
                hdr = fits.Header() # >> make the header
                hdu = fits.PrimaryHDU(lightcurve, header=hdr)
                hdu.writeto(filename)
                                            
            elif n != 0: #saving the rest
                fits.append(filename, lightcurve)
                print(int(n))
               
            gaia_ids.append(int(files.gaia))
        except (SearchError, ValueError):
            print("Some kind of error - either no TESS image exists, no GAIA ID exists, or there was a connection issue")
        
        if os.path.isdir("/Users/conta/.eleanor/tesscut") == True:
            shutil.rmtree("/Users/conta/.eleanor/tesscut")
            print("All files deleted")
    fits.append(filename, np.asarray(gaia_ids))
    print("All light curves saved into fits file")
    return gaia_ids

def open_eleanor_lc_files(path):
    """ opens the fits file that the eleanor light curves are saved into
    parameters:
        * path to the fits file
    returns:
        * list of gaia_ids
        * time indexes
        * intensities
    modified [lcg 08212020]"""
    f = fits.open(path, memmap=False)
    gaia_ids = f[-1].data
    target_nums = len(f) - 1
    all_timeindexes = []
    all_intensities = []
    for n in range(target_nums):
        all_timeindexes.append(f[n].data[0])
        all_intensities.append(f[n].data[1])
        
    f.close()
    
    return gaia_ids, np.asarray(all_timeindexes), np.asarray(all_intensities)

def create_save_featvec_different_timeaxes(yourpath, times, intensities, gaia_ids, filelabel, version=0, save=True):
    """Produces the feature vectors for each light curve and saves them all
    into a single fits file. all light curves have their OWN time axis
    this is set up to work on the eleanor light curves
    Parameters:
        * yourpath = folder you want the file saved into
        * times = all time axes
        * intensities = array of all light curves (NOT normalized)
        * sector, camera, ccd = integers 
        * version = what version of feature vector to calculate for all. 
            default is 0
        * save = whether or not to save into a fits file
    returns: list of feature vectors + fits file containing all feature vectors
    requires: featvec()
    modified: [lcg 08212020]"""
    

    fname_features = yourpath + "/"+ filelabel + "_features_v"+str(version)+".fits"
    feature_list = []
    
    if version == 0:
	#median normalize for the v0 features
        for n in range(len(intensities)):
            intensities[n] = normalize(intensities[n], axis=0)
    elif version == 1: 
        import transitleastsquares
        from transitleastsquares import transitleastsquares
        #mean normalize the intensity so goes to 1
        for n in range(len(intensities)):
            intensities[n] = mean_norm(intensities[n], axis=0)

    print("Begining Feature Vector Creation Now")
    for n in range(len(intensities)):
        feature_vector = df.featvec(times[n], intensities[n], v=version)
        feature_list.append(feature_vector)
        
        if n % 25 == 0: print(str(n) + " completed")
    
    feature_list = np.asarray(feature_list)
    
    if save == True:
        hdr = fits.Header()
        hdr["VERSION"] = version
        hdu = fits.PrimaryHDU(feature_list, header=hdr)
        hdu.writeto(fname_features)
        fits.append(fname_features, gaia_ids)
    else: 
        print("Not saving feature vectors to fits")
    
    return feature_list

def build_simbad_extragalactic_database(maglim, out='./simbad_v19galaxies.txt'):
    '''Object type follows format in:
    http://vizier.u-strasbg.fr/cgi-bin/OType?$1'''
    
    # -- querying object type -------------------------------------------------
    customSimbad = Simbad()
    customSimbad.TIMEOUT = 1000
    # customSimbad.get_votable_fields()
    customSimbad.add_votable_fields('otype')
    customSimbad.add_votable_fields('ra(:;A;ICRS;J2000)', 'dec(:;D;ICRS;2000)')
    table = customSimbad.query_criteria('Vmag <=' + str(maglim), otype='G')
    objects = list(table['MAIN_ID'])
    ras = list(table['RA___A_ICRS_J2000'])
    decs = list(table['DEC___D_ICRS_2000'])

    # >> now loop through all of the objects
    for i in range(len(objects)):
        # >> decode bytes object to convert to string
        obj = objects[i].decode('utf-8')
        ra = ras[i]
        dec = decs[i]
       
        with open(out, 'a') as f:
                f.write(obj + ',' + ra + ',' + dec + ',' + '\n')
                
def get_radecfromtext(directory):
    ''' pulls ra and dec from text file containing all targets
    '''
    ra_all = []
    dec_all = []
    
    # >> find all text files in directory
    fnames = fm.filter(os.listdir(directory), '*.txt')
    
    for fname in fnames:
        # >> read text file
        with open(directory + fname, 'r') as f:
            lines = f.readlines()
            for line in lines:
                name, ra, dec, empty = line.split(',')
                
                ra_all.append(ra)
                dec_all.append(dec)
                
    return np.asarray(ra_all), np.asarray(dec_all)