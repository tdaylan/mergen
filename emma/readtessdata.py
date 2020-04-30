# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
# read tess data (sector 20)
# time in TESS-truncated JD (BJD - 2457000)
# emma 0420
#
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

from astropy.io import fits
import numpy as np
import pdb
import os
import fnmatch
import matplotlib.pyplot as plt
from astropy.stats import SigmaClip

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

fitspath = './tessdata_lc/'
# fitspath = './tessdata_sector20/'
output_dir = './plots/lightcurves032920/'

fnames_all = os.listdir(fitspath)
fnames = fnmatch.filter(fnames_all, '*fits*')
# # >> remove buggy fits file (interrupted download)
fnames.pop(fnames.index('tess2019357164649-s0020-0000000156168236-0165-s_lc.fits'))



interp_tol = 20. / (24*60) # >> interpolate small gaps (less than 20 minutes)

intensity = []
# prefix = 'sector20-'
# prefix = 'supervised100-'
prefix = 's0020-1155-'

plot = False
before_orbit = False

classified = False
classes = np.array([1., 1., 2., 2., 3., 4., 4., 4., 1., 2., 3., 4.])
if classified:
    # with open('./labelled_100.csv', 'r') as f:
    #     lines = f.readlines()
    # tics = [line.split()[1][:7] for line in lines[1:]]
    # classes = [int(line.split()[2]) for line in lines[1:]]
    with open('./3class-labelled-ticid.txt', 'r') as f:
        tics = f.readlines()
        tics = [ticid.split()[0] for ticid in tics]
    # np.savetxt(prefix + 'classification.txt', classes, delimiter = ',')
    fnames_classified = []
    for i in range(len(tics)):
        fname = list(filter(lambda x: tics[i] in x, fnames))[0]
        fnames_classified.append(fname)
    fnames = fnames_classified
    
ticids = []
count = 0
print(len(fnames))
for file in fnames:
    print(count)
    # -- open file ------------------------------------------------------------
    f = fits.open(fitspath + file)
    # print(f.info())
    # print(f[1].header)

    # >> get data
    time = f[1].data['TIME']
    i = f[1].data['PDCSAP_FLUX']
    ticids.append(f[1].header['TICID'])

    # >> plot
    # fig, ax = plt.subplots(1,1)
    # ax.plot(time, i, '.')
    # plt.savefig(output_dir + file[14:-5] + '.png')
    # plt.close(fig)
    
    # >> sigmaclip
    sigclip = SigmaClip(sigma=5, maxiters=None, cenfunc='median')
    clipped_inds = np.nonzero(np.ma.getmask(sigclip(i, masked=True)))
    i[clipped_inds] = np.nan

    # -- find small nan gaps --------------------------------------------------
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

    # -- interpolation --------------------------------------------------------
    # >> interpolate small gaps
    i_interp = np.copy(i)
    for a in range(np.shape(interp_inds)[0]):
        start_ind = interp_inds[a]
        end_ind = interp_inds[a] + interp_lens[a]
        i_interp[start_ind:end_ind] = np.interp(time[start_ind:end_ind],
                                                time[np.nonzero(~np.isnan(i))],
                                                i[np.nonzero(~np.isnan(i))])
    intensity.append(i_interp)

    count += 1

# -- remove orbit nan gap -----------------------------------------------------
intensity = np.array(intensity)
nan_inds = np.nonzero(np.prod(np.isnan(intensity)==False, axis = 0) == False)

if before_orbit:
    orbit_gap_start = np.argmax(np.diff(time))
    intensity = intensity[:,:orbit_gap_start]
    time = time[:orbit_gap_start]

time = np.delete(time, nan_inds)
intensity = np.delete(intensity, nan_inds, 1)

# -- save as txt file ---------------------------------------------------------

np.savetxt(prefix + 'time.txt', time, delimiter = ',')
np.savetxt(prefix + 'flux.csv', intensity, delimiter = ',', fmt = '%d')
np.savetxt(prefix + 'ticid.txt', ticids, delimiter=',')

if classified:
    np.savetxt(prefix + 'classification.txt', classes, delimiter = ',')

# -- plot ---------------------------------------------------------------------
if plot:
    for i in range(np.shape(intensity)[0]):
        fig, ax = plt.subplots(1,1)
        ax.plot(time, intensity[i], '.')
        plt.savefig(output_dir + fnames[i][14:-5] + 'interpolated' + '.png')
        plt.close(fig)
    
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
    
# fnames = ['tess2019357164649-s0020-0000000004287518-0165-s_lc.fits',
#           'tess2019357164649-s0020-0000000004305219-0165-s_lc.fits',
#           'tess2019357164649-s0020-0000000009264019-0165-s_lc.fits',
#           'tess2019357164649-s0020-0000000011673333-0165-s_lc.fits',
#           'tess2019357164649-s0020-0000000015863853-0165-s_lc.fits',
#           'tess2019357164649-s0020-0000000051238317-0165-s_lc.fits',
#           'tess2019357164649-s0020-0000000099725941-0165-s_lc.fits',
#           'tess2019357164649-s0020-0000000130158361-0165-s_lc.fits',
#           'tess2019357164649-s0020-0000000004375248-0165-s_lc.fits',
#           'tess2019357164649-s0020-0000000081210712-0165-s_lc.fits',
#           'tess2019357164649-s0020-0000000053335431-0165-s_lc.fits',
#           'tess2019357164649-s0020-0000000160160991-0165-s_lc.fits']
