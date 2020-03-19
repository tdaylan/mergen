# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
# read tess data (sector 20)
# time in TESS-truncated JD (BJD - 2457000)
# emma 03/16/2020
#
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

from astropy.io import fits
import numpy as np
import pdb
import os
import fnmatch
import matplotlib.pyplot as plt
from itertools import groupby

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

fitspath = './tessdata_lc/'
output_dir = './lightcurves031620/'

fnames_all = os.listdir(fitspath)
fnames = fnmatch.filter(fnames_all, '*fits*')
# fnames = ['./tessdata_lc/tess2019357164649-s0020-0000000004287518-0165-s_lc.fits']
# fnames = ['./tessdata_lc/tess2019357164649-s0020-0000000004132133-0165-s_lc.fits',
#           './tessdata_lc/tess2019357164649-s0020-0000000004244059-0165-s_lc.fits']

# >> remove buggy fits file (interrupted download)
fnames.pop(fnames.index('tess2019357164649-s0020-0000000156168236-0165-s_lc.fits'))

interp_tol = 20. / (24*60) # >> interpolate small gaps (less than 20 minutes)

intensity = []

for file in fnames:
    # -- open file -------------------------------------------------------------
    f = fits.open(fitspath + file)
    # print(f.info())
    # print(f[1].header)

    # >> get data
    time = f[1].data['TIME']
    i = f[1].data['PDCSAP_FLUX']

    # -- find small nan gaps ---------------------------------------------------
    # >> adapted from https://gist.github.com/alimanfoo/
    #    c5977e87111abe8127453b21204c1065
    # >> find run starts
    n = np.shape(i)[0]
    loc_run_start = np.empty(n, dtype=bool)
    loc_run_start[0] = True
    np.not_equal(np.isnan(i)[:-1], np.isnan(i)[1:], out=loc_run_start[1:])
    run_starts = np.nonzero(loc_run_start)[0]

    # >> find run lengths
    run_lengths = np.diff(np.append(run_starts, n))

    tdim = time[1] - time[0]
    interp_inds = run_starts[np.nonzero((run_lengths * tdim <= interp_tol) * \
                                        np.isnan(i[run_starts]))]
    interp_lens = run_lengths[np.nonzero((run_lengths * tdim <= interp_tol) * \
                                         np.isnan(i[run_starts]))]

    # -- interpolation ---------------------------------------------------------
    # >> interpolate small gaps
    i_interp = np.copy(i)
    for a in range(np.shape(interp_inds)[0]):
        start_ind = interp_inds[a]
        end_ind = interp_inds[a] + interp_lens[a]
        i_interp[start_ind:end_ind] = np.interp(time[start_ind:end_ind],
                                                time[np.nonzero(~np.isnan(i))],
                                                i[np.nonzero(~np.isnan(i))])
    intensity.append(i_interp)

    # # >> plot
    # fig, ax = plt.subplots(1,1)
    # ax.plot(time, i_interp, '.')
    # plt.savefig(output_dir + file[14:-5] + 'interpolated' + '.png')

# -- remove orbit nan gap ------------------------------------------------------
intensity = np.array(intensity)
nan_inds = np.nonzero(np.prod(np.isnan(np.array(intensity)), axis = 0))
time = np.delete(time, nan_inds)
intensity = np.delete(intensity, nan_inds, 1)

# -- plot ----------------------------------------------------------------------
for i in range(np.shape(intensity)[0]):
    fig, ax = plt.subplots(1,1)
    ax.plot(time, intensity[i], '.')
    plt.savefig(output_dir + fnames[i][14:-5] + 'interpolated' + '.png')
    plt.close(fig)