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

# fitspath = './tessdata_sector20/'
fitspath = './tessdata_lc/'
output_dir = './plots/plots061820/'

fnames_all = os.listdir(fitspath)
fnames = fnmatch.filter(fnames_all, '*fits*')
# fnames = fnmatch.filter(fnames_all, '*192980481*fits*')
# fnames = fnmatch.filter(fnames_all, '*130158361*fits*')

# # >> remove buggy fits file (interrupted download)
# fnames.pop(fnames.index('tess2019357164649-s0020-0000000156168236-0165-s_lc.fits'))
# pdb.set_trace()

interp_tol = 20. / (24*60) # >> interpolate small gaps (less than 20 minutes)
num_sigma=10
DEBUG = True
debug_ticid = 428885434

intensity = []
prefix = 's0020-'

plot = False
before_orbit = False

classified = False
if classified:
    with open('./s0020-before_orbit-348-ticid.txt', 'r') as f:
        tics = f.readlines()
        tics = [ticid.split()[0] for ticid in tics]
        
    fnames_classified = []
    for i in range(len(tics)):
        fname = list(filter(lambda x: str(int(tics[i])) in x, fnames))[0]
        fnames_classified.append(fname)
    fnames = fnames_classified

# -- interpolation ------------------------------------------------------------
    
ticids = []
count = 0
print(len(fnames))
for file in fnames:
    print(count)
    # -- open file ------------------------------------------------------------
    
    # print(f.info[1].header)

    # >> get data
    try:
        f = fits.open(fitspath + file)
        time = f[1].data['TIME']
        i = f[1].data['PDCSAP_FLUX']
        ticids.append(f[1].header['TICID'])
        f.close()
        # time = fits.getdata(fitspath+file, 1)['TIME']
        # i = fits.getdata(fitspath+file, 1)['PDCSAP_FLUX']
        # ticids.append(fits.getheader(fitspath+file, 1)['TICID'])
        
        i_interp = ml.interpolate_lc(i, time, DEBUG_INTERP=False,
                                     output_dir=output_dir,
                                     prefix=str(ticids[-1]),
                                     num_sigma=num_sigma)
        intensity.append(i_interp)
        # f.close()
        
        
    
    except:
        print('buggy! ' + file)
    

    count += 1
    
intensity = np.array(intensity)
# intensity_interp, time_interp = ml.nan_mask(intensity, time)
# intensity_interp, time_interp = ml.interpolate_lc(intensity, time,
#                                                   interp_tol = interp_tol,
#                                                   num_sigma=num_sigma,
#                                                   DEBUG=False,
#                                                   output_dir=output_dir,
#                                                   prefix='sigma_clip'+'-'+\
#                                                       str(num_sigma)+'-')

# if before_orbit:
#     orbit_gap_start = np.argmax(np.diff(time_interp))
#     intensity_interp = intensity_interp[:,:orbit_gap_start]
#     time_interp = time_interp[:orbit_gap_start]

# -- plot ---------------------------------------------------------------------
# if DEBUG:
#     ind = ticids.index(debug_ticid)
#     fig, ax = plt.subplots()
#     ax.plot(time_interp, intensity_interp[ind]+1., '.k', markersize=2)
#     ax.set_ylabel('relative flux')
#     fig.suptitle('num_sigma='+str(num_sigma))
#     ax.set_label('time [BJD - 2457000]')
#     ml.format_axes(ax)
#     ml.ticid_label(ax, debug_ticid, title=True)
#     fig.savefig(output_dir+'sigma_clip-'+str(num_sigma)+'.png',
#                 bbox_inches='tight')

if plot:
    for j in range(np.shape(intensity)[0]):
        fig, ax = plt.subplots(1,1)
        ax.plot(time, intensity[j], '.k', markersize=2)
        plt.savefig(output_dir + fnames[j][14:-5] + 'interpolated' + '.png')
        plt.close(fig)

# -- save as txt file ---------------------------------------------------------

np.savetxt(prefix + 'time.txt', time, delimiter = ',')
np.savetxt(prefix + 'flux.csv', intensity, delimiter = ',')
np.savetxt(prefix + 'ticid.txt', ticids, delimiter=',', fmt='%d')
# if classified:
#     np.savetxt(prefix + 'classification.txt', classes, delimiter = ',')


    
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

