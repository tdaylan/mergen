# >> generates 5 txt/csv files
# * ticid
# * time
# * flux
# * flux error
# * class
#   * 0: noise
#   * 1: sinusoidal
#   * 2: single transits
#   * 3: multiple transits
#   * 4: flares
#   * 5: idk

import os
import sys
import pdb
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import fnmatch

prefix = 's0020-'

download1 = False
buggy_fits = ['tess2019357164649-s0020-0000000156924740-0165-s_lc.fits']

def on_key(event):
    print(event.key)
    keys.append(event.key)
    sys.stdout.flush()

if download1:
    with open('./tessdata_lc/tesscurl_sector_20_lc.sh', 'r') as f:
        fnames = [line.split()[5] for line in f.readlines()[1:]]
else:
    fitspath = './tessdata_sector20/'
    fnames_all = os.listdir(fitspath)
    fnames = fnmatch.filter(fnames_all, '*fits*')
    
with open(prefix+'ticid.txt', 'r') as f:
    completed = [line.split()[0] for line in f.readlines()]

for fname in fnames:
    ticid = int(fname[24:40])
    if str(ticid) in completed:
        print('Skipping ' + str(ticid))
    elif fname in buggy_fits:
        print('Buggy ' + str(ticid))
    else:
        if download1:
            # >> download fits file
            os.system('curl -C - -L -o ' + fname +\
                      ' https://mast.stsci.edu/api/v0.1/Download/file/'+\
                          '?uri=mast:TESS/product/' + fname)                
            file = fits.open(fname)
        else:
            file = fits.open(fitspath + fname)
        time = file[1].data['TIME']
        flux = file[1].data['PDCSAP_FLUX']
        flux_err = file[1].data['PDCSAP_FLUX_ERR']
        
        # >> give light curve a class
        keys = []
        fig, ax = plt.subplots(figsize=(12,4.5))
        ax.plot(time, flux, '.', markersize=3)
        cid = fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()
        try: 
            if keys[-1] == ' ': # >> skip light curve if necessary
                plt.close(fig)
                fig.canvas.mpl_disconnect(cid)
            else:
                with open(prefix + 'class.txt', 'a') as f:
                    f.write(keys[-1]+'\n')
                fig.canvas.mpl_disconnect(cid)
                plt.close(fig)
        except:
            print(str(ticid))
            break
        
        # >> save to txt file
        with open(prefix + 'ticid.txt', 'a') as f:
            f.write(str(ticid)+'\n')
        with open(prefix + 'time.csv', 'a') as f:
            f.write(','.join(time.astype('str')) + '\n')
        with open(prefix + 'flux.csv', 'a') as f:
            f.write(','.join(flux.astype('str')) + '\n')
        with open(prefix + 'flux_err.csv', 'a') as f:
            f.write(','.join(flux_err.astype('str')) + '\n')

        # >> remove fits file
        if download1:
            os.system('rm ' + fname)
    

