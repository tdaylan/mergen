# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 20:27:13 2020

mergen.py
@author: LG and EC
"""

import numpy as np
import numpy.ma as ma 
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)

from pylab import rcParams
rcParams['figure.figsize'] = 10,10

rcParams["lines.markersize"] = 2
from scipy.signal import argrelextrema, medfilt


import astropy
import astropy.units as u
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
from sklearn.cluster import KMeans, DBSCAN
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

import data_functions as df
import plotting_functions as pf
import ffi_hyperleda as fh
import model as ml
import data_access as da
import ENF_functions as enf
import sn_functions as sn


class mergen(object):
    """ Main mergen class. Initialize this to work with everything else conveniently. """
    def __init__(self, datapath, savepath, datatype, momentum_dump_csv, filelabel = None):
        self.datapath = datapath
        self.savepath = savepath
        self.datatype = datatype #SPOC or FFI
        self.mdumpcsv = momentum_dump_csv
        if filelabel is not None:
            self.filelabel = filelabel
        else:
            self.filelabel = "mergen"
        
        self.folder_initiate()
    
    def folder_initiate(self):
        """Makes all the big folders"""
        print("Setting up CAE folder")
        self.CAEpath = self.savepath + "CAE/"
        try:
            os.makedirs(self.CAEpath)
        except OSError:
            print ("Directory %s already exists" % self.CAEpath)
            
        print("Setting up ENF folder")
        self.ENFpath = self.savepath + "ENF/"
        try:
            os.makedirs(self.ENFpath)
        except OSError:
            print ("Directory %s already exists" % self.ENFpath)
        return
    
    def load_lightcurves_local(self):
        """Loads in data saved in metafiles on datapath"""
        #check for self.datatype to determine loading scheme. figure out consistent stuff for FFI original locations
        return 6
    
            
    def download_and_load_lightcurves(self):
        """ ??? this is just the other option for if you want to run batch downloads and then make metafiles"""
        return

    def data_clean(self):
        """ Cleans data up - just BASE cleanup of normalizing + sigma clipping. CAE additional cleans done later"""
        return
    
    def load_existing_features(self):
        """ Load in feature metafiles stored in the datapath"""
        return
    
    def generate_engineered(self):
        """Run engineered feature creation"""
        return
    
    def generate_CAE(self):
        """Run CAE feature creation """
        return


    s