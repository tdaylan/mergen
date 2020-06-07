import fnmatch
from astropy.io import fits
import os
import matplotlib.pyplot as plt
import modellibrary as ml
import numpy as np
import pdb

# >> testing num_sigma for sigma clipping
fitspath = './tessdata_lc/'
output_dir = './plots/plots060620/'

ticid = 428885434

fnames_all = os.listdir(fitspath)
fname = fnmatch.filter(fnames_all, '*'+str(ticid)+'*')[0]

f = fits.open(fitspath + fname)
time = f[1].data['TIME']
intensity = f[1].data['PDCSAP_FLUX']
addend=1.

fig, ax = plt.subplots()
ax.plot(time, intensity+1., '.k', markersize=2)
ax.set_ylabel('relative flux')
fig.suptitle('num_sigma=0')
ax.set_label('time [BJD - 2457000]')
ml.format_axes(ax)
ml.ticid_label(ax, ticid, title=True)
fig.savefig(output_dir+'sigma_clip-original.png',
            bbox_inches='tight')

pdb.set_trace()
fig, ax = plt.subplots(11, 1, sharex=True, figsize=(3, 3*11))
ax[0].plot(time, intensity+1., '.k', markersize=2)
ax[0].set_ylabel('original\nrelative flux')
ml.format_axes(ax[0])
ml.ticid_label(ax[0], ticid, title=True)
ax[-1].set_xlabel('time [BJD - 2457000]')

# >> try different values for sigma clip
intensity = np.array([intensity])
for i in range(1, 11):
    num_sigma = i
    intensity_interp, time_interp = ml.interpolate_lc(intensity, time,
                                                      num_sigma=num_sigma)
    intensity_interp = intensity_interp.reshape(np.shape(intensity_interp)[1])
    ax[i].plot(time_interp, intensity_interp+addend, '.k', markersize=2)
    ax[i].set_ylabel('num_sigma='+str(num_sigma)+'\nrelative flux')
    ml.format_axes(ax[i])
    
fig.savefig(output_dir + 'sigma_clip-'+str(ticid)+'.png', bbox_inches='tight')