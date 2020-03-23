# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 23:27:17 2020

@author: conta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

import lightkurve as lc

import astropy
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.utils.data import get_pkg_data_filename

from datetime import datetime

#borrowing largely from https://docs.lightkurve.org/tutorials/01-using-the-periodogram-class.html
filename = lc.open("/Users/conta/UROP_Spring_2020/tessdata_lc_sector20_1000/tess2019357164649-s0020-0000000004132133-0165-s_lc.fits")
file_pdcsap = filename.PDCSAP_FLUX
file_pdcsap.scatter()

now = datetime.now()
timestamp = datetime.timestamp(now)
timestamp = str(timestamp)

pdcsap_clean = file_pdcsap.remove_nans()
periodogram = pdcsap_clean.to_periodogram(method='lombscargle',oversample_factor=1)
periodogram.plot()
plt.savefig(timestamp + "periodogram_freq.png")

periodogram.plot(view='period', scale='log')
plt.savefig(timestamp + "periodogram_period.png")

periodogram.period
periodogram.period_at_max_power
