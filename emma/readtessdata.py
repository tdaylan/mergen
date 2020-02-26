from astropy.io import fits
import numpy as np
import pdb
import matplotlib.pyplot as plt

# f = fits.open('./tessdata_lc/tess2019357164649-s0020-0000000004132133-0165-s_lc.fits')
f = fits.open('./tessdata_lc/tess2019357164649-s0020-0000000004244059-0165-s_lc.fits')
print(f.info())
# print(f[1].header)

# get data
time = f[1].data['TIME']
intensity = f[1].data['PDCSAP_FLUX']

# make plot of data
plt.ion()
plt.figure(0)
plt.plot(time, intensity, '.')
