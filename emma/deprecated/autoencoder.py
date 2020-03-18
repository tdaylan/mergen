# !! deprecated !!
# autoencoder
# https://blog.keras.io/building-autoencoders-in-keras.html
# https://towardsdatascience.com/autoencoders-for-the-compression-of-stock-market-data-28e8c1a2da3e

import os
import fnmatch
import numpy as np
from astropy.io import fits
import pdb
import modellibrary ml
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSamping1D
# from keras.models import Model
# from keras import backend as K

lc_dir = './tessdata_lc'

fnames_all = os.listdir(lc_dir)
fnames = fnames_all[np.nonzero(fnmatch.fnmatch(file, '*.fits') for file in \
                               fnames_all)[0]]

training_size = 1000
test_size = 100
dim = 18954
kernel_size = 100

# :: build autoencoder  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::

autoencoder = ml.autoencoder()

# :: get training data :::::::::::::::::::::::::::::::::::::::::::::::::::::::::
for i in range(training_size + test_size):
    f = fits.open(lc_dir + '/' + fnames[i])
    time = f[1].data['TIME']
    x = f[1].data['PDCSAP_FLUX']
    
    


