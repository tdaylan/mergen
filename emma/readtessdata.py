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
import modellibrary as ml

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
prefix = 's0020-before_orbit-1155-'

plot = False
before_orbit = True

classified = False
if classified:
    with open('./s0020-348-ticid.csv', 'r') as f:
        tics = f.readlines()
        tics = [ticid.split()[0] for ticid in tics]
        
    fnames_classified = []
    for i in range(len(tics)):
        fname = list(filter(lambda x: str(int(float(tics[i]))) in x, fnames))[0]
        fnames_classified.append(fname)
    fnames = fnames_classified

# -- interpolation ------------------------------------------------------------
    
ticids = []
count = 0
print(len(fnames))
for file in fnames:
    print(count)
    # -- open file ------------------------------------------------------------
    f = fits.open(fitspath + file)
    # print(f.info(), f[1].header)

    # >> get data
    time = f[1].data['TIME']
    i = f[1].data['PDCSAP_FLUX']
    ticids.append(f[1].header['TICID'])
    
    intensity.append(i)

    f.close()
    count+=1
    
intensity = np.array(intensity)
intensity, time = ml.interpolate_lc(intensity, time, interp_tol = interp_tol)

if before_orbit:
    orbit_gap_start = np.argmax(np.diff(time))
    intensity = intensity[:,:orbit_gap_start]
    time = time[:orbit_gap_start]

# -- save as txt file ---------------------------------------------------------

np.savetxt(prefix + 'time.txt', time, delimiter = ',')
np.savetxt(prefix + 'flux.csv', intensity, delimiter = ',')
np.savetxt(prefix + 'ticid.txt', ticids, delimiter=',', fmt='%d')

# if classified:
#     np.savetxt(prefix + 'classification.txt', classes, delimiter = ',')

# -- plot ---------------------------------------------------------------------
if plot:
    for i in range(np.shape(intensity)[0]):
        fig, ax = plt.subplots(1,1)
        ax.plot(time, intensity[i], '.')
        plt.savefig(output_dir + fnames[i][14:-5] + 'interpolated' + '.png')
        plt.close(fig)
    
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    
#     # >> sigmaclip
#     sigclip = SigmaClip(sigma=5, maxiters=None, cenfunc='median')
#     clipped_inds = np.nonzero(np.ma.getmask(sigclip(i, masked=True)))
#     i[clipped_inds] = np.nan

#     # -- find small nan gaps --------------------------------------------------
#     # >> adapted from https://gist.github.com/alimanfoo/
#     #    c5977e87111abe8127453b21204c1065
#     # >> find run starts
#     n = np.shape(i)[0]
#     loc_run_start = np.empty(n, dtype=bool)
#     loc_run_start[0] = True
#     np.not_equal(np.isnan(i)[:-1], np.isnan(i)[1:], out=loc_run_start[1:])
#     run_starts = np.nonzero(loc_run_start)[0]

#     # >> find run lengths
#     run_lengths = np.diff(np.append(run_starts, n))

#     tdim = time[1] - time[0]
#     interp_inds = run_starts[np.nonzero((run_lengths * tdim <= interp_tol) * \
#                                         np.isnan(i[run_starts]))]
#     interp_lens = run_lengths[np.nonzero((run_lengths * tdim <= interp_tol) * \
#                                          np.isnan(i[run_starts]))]

#     # -- interpolation --------------------------------------------------------
#     # >> interpolate small gaps
#     i_interp = np.copy(i)
#     for a in range(np.shape(interp_inds)[0]):
#         start_ind = interp_inds[a]
#         end_ind = interp_inds[a] + interp_lens[a]
#         i_interp[start_ind:end_ind] = np.interp(time[start_ind:end_ind],
#                                                 time[np.nonzero(~np.isnan(i))],
#                                                 i[np.nonzero(~np.isnan(i))])
#     intensity.append(i_interp)

#     count += 1

# # -- remove orbit nan gap -----------------------------------------------------
# intensity = np.array(intensity)
# nan_inds = np.nonzero(np.prod(np.isnan(intensity)==False, axis = 0) == False)


# time = np.delete(time, nan_inds)
# intensity = np.delete(intensity, nan_inds, 1)
