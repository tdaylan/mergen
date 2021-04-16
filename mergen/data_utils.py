# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:00:13 2021

@author: Emma Chickles, Lindsey Gordon
data_utils.py
* data load
* data cleaning
* feature loading

To Do List:
    - HYPERLEDA LC load in
    - Loading fxns for CAE features
"""
import numpy as np

######## DATA LOADING (SPOC) ############

def combine_sectors(sectors, data_dir, custom_masks=None):
    all_flux = []
    all_ticid = []
    all_target_info = []
    all_x = []
    if type(custom_masks) == type(None):
        custom_masks = [[]*len(sectors)]
    for i in range(len(sectors)):
        flux, x, ticid, target_info = \
            load_data_from_metafiles(data_dir, sectors[i], nan_mask_check=True,
                                     custom_mask=custom_masks[i])
            
        mins = np.min(flux, axis = 1, keepdims=True)
        flux = flux - mins
        maxs = np.max(flux, axis=1, keepdims=True)
        flux = flux / maxs            
        all_flux.append(flux)
        all_ticid.append(ticid)
        all_target_info.append(target_info)
        all_x.append(x)
    # >> stitch together !! can only handle 2 sectors
    all_ticid1, comm1, comm2 = np.intersect1d(all_ticid[0], all_ticid[1],
                                             return_indices=True)
        
    flux = np.concatenate([all_flux[0][comm1], all_flux[1][comm2]], axis = -1)

    target_info = []
    for i in range(len(comm1)):
        target_info.append([','.join([all_target_info[0][comm1[i]][0],
                                     all_target_info[1][comm2[i]][0]]),
                           ','.join([all_target_info[0][comm1[i]][1],
                                     all_target_info[1][comm2[i]][1]]),
                           ','.join([all_target_info[0][comm1[i]][2],
                                     all_target_info[1][comm2[i]][2]]),
                           all_target_info[0][comm1[i]][3],
                           all_target_info[0][comm1[i]][4]])
    
    x = np.concatenate(all_x)
    
    ticid = all_ticid[0][comm1]
    
    return flux, x, ticid, np.array(target_info)

def combine_sectors_by_lc(sectors, data_dir, custom_mask=[],
                          output_dir='./', DEBUG=True):
    all_flux = []
    all_ticid = []
    all_target_info = []
    all_x = []
    flux_lengths = []
    for i in range(len(sectors)):
        flux, x, ticid, target_info = \
            load_data_from_metafiles(data_dir, sectors[i], nan_mask_check=False)
        all_flux.append(flux)
        all_ticid.append(ticid)
        all_target_info.append(target_info)
        all_x.append(x)
        flux_lengths.append(np.shape(flux)[1])
        
    # >> truncate
    new_length = np.min(flux_lengths)
    for i in range(len(sectors)):
        all_flux[i] = all_flux[i][:,:new_length]
        
    flux = np.concatenate([all_flux[0], all_flux[1]], axis = 0)
    x = all_x[0][:new_length]
    target_info = np.concatenate([all_target_info[0], all_target_info[1]],
                                 axis=0)
    ticid = np.concatenate([all_ticid[0], all_ticid[1]])    
    flux, x = nan_mask(flux, x, custom_mask=custom_mask, ticid=ticid,
                       target_info=target_info,
                       output_dir=output_dir, DEBUG=True)

    
    return flux, x, ticid, np.array(target_info)

def load_data_from_metafiles(data_dir, sector, cams=[1,2,3,4],
                             ccds=[[1,2,3,4]]*4, data_type='SPOC',
                             cadence='2-minute', DEBUG=False, fast=False,
                             output_dir='./', debug_ind=0,
                             nan_mask_check=True,
                             custom_mask=[]):
    '''Pulls light curves from fits files, and applies nan mask.
    
    Parameters:
        * data_dir : folder containing fits files for each group
        * sector : sector, given as int, or as a list
        * cams : list of cameras
        * ccds : list of CCDs
        * data_type : 'SPOC', 'FFI'
        * cadence : '2-minute', '20-second'
        * DEBUG : makes nan_mask debugging plots. If True, the following are
                  required:
            * output_dir
            * debug_ind
        * nan_mask_check : if True, applies NaN mask
    
    Returns:
        * flux : array of light curve PDCSAP_FLUX,
                 shape=(num light curves, num data points)
        * x : time array, shape=(num data points)
        * ticid : list of TICIDs, shape=(num light curves)
        * target_info : [sector, cam, ccd, data_type, cadence] for each light
                        curve, shape=(num light curves, 5)
    '''
    
    # >> get file names for each group
    fnames = []
    fname_info = []
    for i in range(len(cams)):
        cam = cams[i]
        for ccd in ccds[i]:
            if fast:
                s = 'Sector{sector}_20s/Sector{sector}Cam{cam}CCD{ccd}/' + \
                    'Sector{sector}Cam{cam}CCD{ccd}_lightcurves.fits'
            else:
                s = 'Sector{sector}/Sector{sector}Cam{cam}CCD{ccd}/' + \
                    'Sector{sector}Cam{cam}CCD{ccd}_lightcurves.fits'
            fnames.append(s.format(sector=sector, cam=cam, ccd=ccd))
            fname_info.append([sector, cam, ccd, data_type, cadence])
                
    # >> pull data from each fits file
    print('Pulling data')
    flux_list = []
    ticid = np.empty((0, 1))
    target_info = [] # >> [sector, cam, ccd, data_type, cadence]
    for i in range(len(fnames)):
        print('Loading ' + fnames[i] + '...')
        with fits.open(data_dir + fnames[i], memmap=False) as hdul:
            if i == 0:
                x = hdul[0].data
            flux = hdul[1].data
            ticid_list = hdul[2].data
    
        flux_list.append(flux)
        ticid = np.append(ticid, ticid_list)
        target_info.extend([fname_info[i]] * len(flux))

    # >> concatenate flux array         
    flux = np.concatenate(flux_list, axis=0)
        
    # >> apply nan mask
    if nan_mask_check:
        print('Applying nan mask')
        flux, x = nan_mask(flux, x, DEBUG=DEBUG, ticid=ticid,
                           debug_ind=debug_ind, target_info=target_info,
                           output_dir=output_dir, custom_mask=custom_mask)

    return flux, x, ticid, np.array(target_info)

def data_access_sector_by_bulk(yourpath, sectorfile, sector,
                               bulk_download_dir, custom_mask=[],
                               apply_nan_mask=False):
    '''Get interpolated flux array for each group, if you already have all the
    _lc.fits files downloaded in bulk_download_dir.
    Parameters:
        * yourpath : directory to store outputs in
        * sectorfile : txt file containing the camera and ccd of each light
          curve in the sector, from
          https://tess.mit.edu/observations/target-lists/
        * sector : int
        * bulk_download_dir : directory containing all the _lc.fits files,
          can be downloaded from 
          http://archive.stsci.edu/tess/bulk_downloads.html
          Also see bulk_download_helper()
    e.g. df.data_access_sector_by_bulk('../../',
                                       '../../all_targets_S020_v1.txt', 20,
                                       '../../tessdata_sector_20/')
    '''
    
    for cam in [1,2,3,4]:
        for ccd in [1,2,3,4]:
            data_access_by_group_fits(yourpath, sectorfile, sector, cam,
                                      ccd, bulk_download=True,
                                      bulk_download_dir=bulk_download_dir,
                                      custom_mask=custom_mask,
                                      apply_nan_mask=apply_nan_mask)
            
def bulk_download_helper(yourpath, shell_script):
    '''Downloads all the light curves for a sector. Can also be used to go back
    and check you have all the light curves from a sector.
    Parameters:
        * yourpath : directory to save .fits files in, contains shell_script
        * shell_script : file name for shell script (tesscurl_sector_*_lc.sh)
          from http://archive.stsci.edu/tess/bulk_downloads.html
    e.g. bulk_download_helper('./tessdata_sector_18/',
                              'tesscurl_sector_18_lc.sh')
    TODO:
    * modify to handle 30-min cadence data
    '''
    import fnmatch as fm
    with open(yourpath+shell_script, 'r') as f:
        sector_targets = f.readlines()[1:]
        
    downloaded_targets = os.listdir(yourpath)
    
    # >> loop through all the sector_targets
    for i in range(len(sector_targets)):
        
        # >> check if already downloaded
        fname = sector_targets[i].split()[5]
        matches = fm.filter(downloaded_targets, fname)
        
        # >> if not downloaded, download the light curve
        if len(matches) == 0:
            print(str(i) + '/' + str(len(sector_targets)))            
            print(fname)
            command = sector_targets[i].split()[:5] + [yourpath+fname] + \
                [sector_targets[i].split()[6]]
            os.system(' '.join(command))
        else:
            print('Already downloaded '+fname)
            
def get_lc_file_and_data(yourpath, target):
    """ goes in, grabs the data for the target, gets the time index, intensity,and TIC
    if connection error w/ MAST, skips it.
    Also masks any flagged data points according to the QUALITY column.
    parameters: 
        * yourpath, where you want the files saved to. must end in /
        * targets, target list of all TICs 
    modified [lcg 07082020] - fixed handling no results, fixed deleting download folder"""
    fitspath = yourpath + 'mastDownload/TESS/' # >> download directory
    targ = "TIC " + str(int(target))
    print(targ)
    try:
        #find and download data products for your target objectname='TIC '+str(int(target)),
        obs_table = Observations.query_criteria(obs_collection='TESS',
                                        dataproduct_type='timeseries',
                                        target_name=str(int(target)),
                                        objectname=targ)
        data_products_by_obs = Observations.get_product_list(obs_table[0:8])
            
        filter_products = Observations.filter_products(data_products_by_obs,
                                                       description = 'Light curves')
        if len(filter_products) != 0:
            manifest = Observations.download_products(filter_products, download_dir= yourpath, extension='fits')
        else: 
            print("Query yielded no matching data produts for ", targ)
            time1 = 0
            i1 = 0
            ticid = 0
            
        #get all the paths to lc.fits files
        filepaths = []
        for root, dirs, files in os.walk(fitspath):
            for name in files:
                #print(name)
                if name.endswith(("lc.fits")):
                    filepaths.append(root + "/" + name)
        #print(len(filepaths))
        #print(filepaths)
        
        if len(filepaths) == 0: #if no lc.fits were downloaded, move on
            print("No lc.fits files available for TIC ", targ)
            time1 = 0
            i1 = 0
            ticid = 0
        else: #if there are lc.fits files, open them and get the goods
                #get the goods and then close it
            f = fits.open(filepaths[0], memmap=False)
            time1 = f[1].data['TIME']
            i1 = f[1].data['PDCSAP_FLUX']
            ticid = f[1].header["TICID"]
            quality = f[1].data['QUALITY']
            f.close()
            
            # >> mask out any nonzero points
            flagged_inds = np.nonzero(quality)
            i1[flagged_inds] = np.nan # >> will be interpolated later
                  
        #then delete all downloads in the folder, no matter what type
        if os.path.isdir(yourpath + "mastDownload") == True:
            shutil.rmtree(yourpath + "mastDownload")
            print("Download folder deleted.")
            
        #corrects for connnection errors
    except (ConnectionError, OSError, TimeoutError, RemoteServiceError):
        print(targ, " could not be accessed due to an error.")
        i1 = 0
        time1 = 0
        ticid = 0
    
    return time1, i1, ticid

####### DATA LOADING (FFI) ##########

def load_lygos_csv(file):
    import pandas as pd
    data = pd.read_csv(file, sep = ' ', header = None)
    #print (data)
    t = np.asarray(data[0])
    ints = np.asarray(data[1])
    error = np.asarray(data[2])
    return t, ints, error

def load_all_lygos(datapath):
    
    all_t = [] 
    all_i = []
    all_e = []
    all_labels = []
    
    for root, dirs, files in os.walk(datapath):
        for name in files:
            if name.startswith(("rflx")):
                filepath = root + "/" + name 
                print(name)
                label = name.split("_")
                full_label = label[1] + label[2]
                all_labels.append(full_label)
                
                t,i,e = load_lygos_csv(name)
                mean = np.mean(i)
                sigclip = SigmaClip(sigma=4, maxiters=None, cenfunc='median')
                clipped_inds = np.nonzero(np.ma.getmask(sigclip(i)))
                i[clipped_inds] = mean #reset those values to the mean value (or remove??)
    
                all_t.append(t)
                all_i.append(i)
                all_e.append(e)
                
    return all_t, all_i, all_e, all_labels



######## DATA CLEANING ###########

def normalize(flux, axis=1):
    '''Dividing by median.
    !!Current method blows points out of proportion if the median is too close to 0?'''
    medians = np.median(flux, axis = axis, keepdims=True)
    flux = flux / medians
    return flux

def mean_norm(flux, axis=1): 
    """ normalizes by dividing by mean - necessary for TLS running 
    modified lcg 07192020"""
    means = np.mean(flux, axis = axis, keepdims=True)
    flux = flux / means
    return flux

def rms(x, axis=1):
    rms = np.sqrt(np.nanmean(x**2, axis = axis))
    return rms

def standardize(x, ax=1):
    means = np.nanmean(x, axis = ax, keepdims=True) # >> subtract mean
    x = x - means
    stdevs = np.nanstd(x, axis = ax, keepdims=True) # >> divide by standard dev
    
    # >> avoid dividing by 0.0
    stdevs[ np.nonzero(stdevs == 0.) ] = 1e-8
    
    x = x / stdevs
    return x

def interpolate_all(flux, time, ticid, flux_err=False, interp_tol=20./(24*60),
                    num_sigma=10, k=3, DEBUG_INTERP=False, output_dir='./',
                    apply_nan_mask=False, DEBUG_MASK=False, custom_mask=[]):
    '''Interpolates each light curves in flux array.'''
    
    flux_interp = []
    ticid_interp = []
    flagged = []
    ticid_flagged = []
    for i in range(len(flux)):
        i_interp, flag = interpolate_lc(flux[i], time, flux_err=flux_err,
                                        interp_tol=interp_tol,
                                        num_sigma=num_sigma, k=k,
                                        DEBUG_INTERP=DEBUG_INTERP,
                                        output_dir=output_dir,
                                        prefix=str(i)+'-')
        if not flag:
            flux_interp.append(i_interp)
            ticid_interp.append(ticid[i])
        else:
            flagged.append(i_interp)
            ticid_flagged.append(ticid[i])
            print('Spline interpolation failed!')
    
    if apply_nan_mask:
        flux_interp, time = nan_mask(flux_interp, time, DEBUG=DEBUG_MASK,
                                     output_dir=output_dir, ticid=ticid_interp,
                                     custom_mask=custom_mask)
    
    return np.array(flux_interp), time, np.array(ticid_interp), \
            np.array(flagged), np.array(ticid_flagged)

def interpolate_lc(i, time, flux_err=False, interp_tol=20./(24*60),
                   num_sigma=10, k=3, search_range=200, med_tol=2,
                   DEBUG_INTERP=False,
                   output_dir='./', prefix=''):
    '''Interpolation for one light curve. Linearly interpolates nan gaps less
    than 20 minutes long. Spline interpolates nan gaps more than 20 minutes
    long (and shorter than orbit gap)
    Parameters:
        * i : intensity array, shape=(n)
        * time : time array, shape=(n)
        * interp_tol : if nan gap is less than interp_tol days, then will
                       linear interpolate
        * num_sigma : number of sigma to clip
        * k : power of spline
        * search_range : number of data points around interpolate region to 
                         calculate the local standard deviation and median
        * med_tol : checks if median of interpolate region is between
                    med_tol*(local median) and (local median)/med_tol
    
    example code snippet
    import data_functions as df
    from astropy.io import fits
    f = fits.open('tess2019306063752-s0018-0000000005613228-0162-s_lc.fits')
    i_interp, flag = df.interpolate_lc(f[1].data['PDCSAP_FLUX'],
                                      f[1].data['TIME'], DEBUG_INTERP=True)
    '''
    from astropy.stats import SigmaClip
    from scipy import interpolate
    
    # >> plot original light curve
    if DEBUG_INTERP:
        fig, ax = plt.subplots(4, 1, figsize=(8, 3*5))
        ax[0].plot(time, i, '.k')
        ax[0].set_title('original')
        
    # >> get spacing in time array
    dt = np.nanmin( np.diff(time) )
    
    # -- sigma clip -----------------------------------------------------------
    sigclip = SigmaClip(sigma=num_sigma, maxiters=None, cenfunc='median')
    clipped_inds = np.nonzero(np.ma.getmask(sigclip(i, masked=True)))
    i[clipped_inds] = np.nan
    if DEBUG_INTERP:
        time_plot = np.linspace(np.nanmin(time), np.nanmax(time), len(time))
        ax[1].plot(time_plot, i, '.k')
        ax[1].set_title('clipped')
    
    # -- locate nan gaps ------------------------------------------------------
    # >> find all runs
    n = np.shape(i)[0]
    loc_run_start = np.empty(n, dtype=bool)
    loc_run_start[0] = True
    np.not_equal(np.isnan(i)[:-1], np.isnan(i)[1:], out=loc_run_start[1:])
    run_starts = np.nonzero(loc_run_start)[0]

    # >> find nan window lengths
    run_lengths = np.diff(np.append(run_starts, n))
    
    # >> find nan windows
    nan_inds = np.nonzero(np.isnan(i[run_starts]))
    run_starts = run_starts[ nan_inds ]
    run_lengths = run_lengths[ nan_inds ]
    
    # >> create x array
    # !! TODO remove end NaN window from x array
    x = np.arange(len(i))
    
    # >> remove nan windows at the beginning and end
    if run_starts[0] == 0:
        run_starts = np.delete(run_starts, 0)
        run_lengths = np.delete(run_lengths, 0)
    if run_starts[-1] + run_lengths[-1] == len(i):
        x = np.delete(x, range(run_starts[-1], run_starts[-1]+run_lengths[-1]))
        run_starts = np.delete(run_starts, -1)
        run_lengths = np.delete(run_lengths, -1)
        
    # >> remove orbit gap from list
    orbit_gap_ind = np.argmax(run_lengths)
    orbit_gap_start = run_starts[ orbit_gap_ind ]
    orbit_gap_end = orbit_gap_start + run_lengths[ orbit_gap_ind ]    
    run_starts = np.delete(run_starts, orbit_gap_ind)
    run_lengths = np.delete(run_lengths, orbit_gap_ind)
    
    # -- fit a spline ---------------------------------------------------------
    
    # >> get all non-nan points
    num_inds = np.nonzero( (~np.isnan(i)) * (~np.isnan(time)) )[0]
    
    # >> fit spline to non-nan points
    ius = interpolate.InterpolatedUnivariateSpline(num_inds, i[num_inds],
                                                   k=k)
    
    if DEBUG_INTERP:
        x_plot = np.delete(x, range(num_inds[-1], len(x)))
        x_plot = np.delete(x_plot, range(orbit_gap_start, orbit_gap_end))
        x_plot = np.delete(x_plot, range(0, num_inds[0]))
        ax[2].plot(x_plot, ius(x_plot), '.k')
        ax[2].set_title('spline')    
    
    # -- interpolate nan gaps -------------------------------------------------
    i_interp = np.copy(i)
    # rms_lc = np.sqrt(np.mean(i[num_inds]**2)) # >> RMS of entire light curve
    # avg_lc = np.mean(i[num_inds])
    # std_lc = np.std(i[num_inds])
    # >> loop through each orbit gap
    for a in range(len(run_starts)):
        
        flag=False
        if run_lengths[a] * dt > interp_tol: # >> spline interpolate
            start = run_starts[a]
            end = run_starts[a] + run_lengths[a]
            spline_interp = \
                ius(x[start : end])
               
            # >> compare std, median of interpolate region to local std, median
            std_local = np.mean([np.nanstd(i[start-search_range : start]),
                                 np.nanstd(i[end : end+search_range])])
            med_local = np.mean([np.nanmedian(i[start-search_range : start]),
                                 np.nanmedian(i[end : end+search_range])])
            
            if np.std(spline_interp) > std_local or \
                np.median(spline_interp) > med_tol*med_local or\
                    np.median(spline_interp) < med_local/med_tol:
                
            # # >> check if RMS of interpolated region is crazy
            # rms_interp = np.sqrt(np.mean(spline_interp**2))
            # avg_interp = np.mean(spline_interp)
            # # if rms_interp > 1.25*rms_lc: # !! factor
            # if avg_interp > avg_lc+std_lc or avg_interp < avg_lc-std_lc:
                flag=True
            else:
                i_interp[run_starts[a] : run_starts[a] + run_lengths[a]] =\
                    spline_interp
                
        if run_lengths[a] * dt < interp_tol or flag: # >> linear interpolate
            i_interp[run_starts[a] : run_starts[a] + run_lengths[a]] = \
                np.interp(x[run_starts[a] : run_starts[a] + run_lengths[a]],
                          x[num_inds], i[num_inds])
            flag=False
                
    if DEBUG_INTERP:
        ax[3].plot(time_plot, i_interp, '.k')
        ax[3].set_title('interpolated')
        fig.tight_layout()
        fig.savefig(output_dir + prefix + 'interpolate_debug.png',
                    bbox_inches='tight')
        plt.close(fig)        
        
    return i_interp, flag

def nan_mask(flux, time, flux_err=False, DEBUG=False, debug_ind=1042,
             ticid=False, target_info=False,
             output_dir='./', prefix='', tol1=0.05, tol2=0.1,
             custom_mask=[], use_tol2=True):
    '''Apply nan mask to flux and time array.
    Returns masked, homogenous flux and time array.
    If there are only a few (less than tol1 light curves) light curves that
    contribute (more than tol2 data points are NaNs) to NaN mask, then will
    remove those light curves.
    Parameters:
        * flux : shape=(num light curves, num data points)
        * time : shape=(num data points)
        * flux_err : shape=(num light curves, num data points)
        * ticid : shape=(num light curves)
        * target_info : shape=(num light curves, 5)
        * tol1 : given as fraction of num light curves, determines whether to
          remove the NaN-iest light curves, or remove NaN regions from all
          light curves
        * tol2 : given as fraction of num data points
        * custom_mask : list of indicies to remove from all light curves
    '''
    # >> apply custom NaN mask
    if len(custom_mask) > 0: print('Applying custom NaN mask')
    time = np.delete(time, custom_mask)
    flux = np.delete(flux, custom_mask, 1)

    mask = np.nonzero(np.prod(~np.isnan(flux), axis = 0) == False)
    # >> plot histogram of number of data points thrown out
    num_nan = np.sum( np.isnan(flux), axis=1 )

    def count_masked(x):
        '''Counts number of masked data points for one light curve.'''
        return np.count_nonzero( ~np.isin(mask, np.nonzero(x)) )
    num_masked = np.apply_along_axis(count_masked, axis=1, arr=np.isnan(flux))
    
    plt.figure()
    plt.hist(num_masked, bins=50)
    plt.ylabel('number of light curves')
    plt.xlabel('number of data points masked')
    plt.savefig(output_dir + 'nan_mask.png')
    plt.close()
    
    # >> debugging plots
    if DEBUG:
        fig, ax = plt.subplots()
        ax.plot(time, flux[debug_ind], '.k')
        ax.set_title('removed orbit gap')
        fig.tight_layout()
        fig.savefig(output_dir + prefix + 'nanmask_debug.png',
                    bbox_inches='tight')
        plt.close(fig) 
        
        # >> plot nan-y light curves
        sorted_inds = np.argsort(num_masked)
        for k in range(2): # >> plot top and lowest
            fig, ax = plt.subplots(nrows=10, figsize=(8, 3*10))
            for i in range(10):
                if k == 0:
                    ind = sorted_inds[i]
                else:
                    ind = sorted_inds[-i-1]
                ax[i].plot(time, flux[ind], '.k')
                pf.ticid_label(ax[i], ticid[ind], target_info[ind], title=True)
                num_nans = np.count_nonzero(np.isnan(flux[ind]))
                ax[i].text(0.98, 0.98, 'Num NaNs: '+str(num_nans)+\
                           '\nNum masked: '+str(num_masked[ind]),
                           transform=ax[i].transAxes,
                           horizontalalignment='right',
                           verticalalignment='top', fontsize='xx-small')
            if k == 0:
                fig.tight_layout()
                fig.savefig(output_dir + prefix + 'nanmask_top.png',
                            bbox_inches='tight')
            else:
                fig.tight_layout()
                fig.savefig(output_dir + prefix + 'nanmask_low.png',
                            bbox_inches='tight')
       
    # >> check if only a few light curves contribute to NaN mask
    num_nan = np.array(num_nan)
    worst_inds = np.nonzero( num_nan > tol2 )
    if len(worst_inds[0]) < tol1 * len(flux) and use_tol2: # >> only a few bad light curves
        np.delete(flux, worst_inds, 0)
        
        pdb.set_trace()
        # >> and calculate new mask
        mask = np.nonzero(np.prod(~np.isnan(flux), axis = 0) == False)    
        
    # >> apply NaN mask
    time = np.delete(time, mask)
    flux = np.delete(flux, mask, 1)
    
    # # >> will need to truncate if using multiple sectors
    # new_length = np.min([np.shape(i)[1] for i in flux])
    
    if type(flux_err) != bool:
        flux_err = np.delete(flux_err, mask, 1)
        flux_err = np.delete(flux_err, custom_mask, 1)
        return flux, time, flux_err
    else:
        return flux, time

######## FEATURE LOADING ###########

def load_ENF_feature_metafile(folderpath):
    filepaths = []
    for root, dirs, files in os.walk(folderpath):
        for file in files:
            if file.endswith("features_v0.fits"):
                filepaths.append(root + "/" + file)
                print(root + "/" + file)
            elif file.endswith("features_v1.fits"):
                filepaths.append(root + "/" + file)
                print(root + "/" + file)
        
    f = fits.open(filepaths[0], memmap=False)
    features = np.asarray(f[0].data)
    f.close()
    for n in range(len(filepaths) -1):
        f = fits.open(filepaths[n+1], memmap=False)
        features_new = np.asarray(f[0].data)
        features = np.column_stack((features, features_new))
        f.close()
    return features


######### QUATERNION HANDLING ##########
def convert_to_quat_metafile(file, fileoutput):
    f = fits.open(file, memmap=False)
    
    t = f[1].data['TIME']
    Q1 = f[1].data['C1_Q1']
    Q2 = f[1].data['C1_Q2']
    Q3 = f[1].data['C1_Q3']
    f.close()
    
    big_quat_array = np.asarray((t, Q1, Q2, Q3))
    np.savetxt(fileoutput, big_quat_array)

def metafile_load_smooth_quaternions(sector, maintimeaxis, 
                                     quaternion_folder = "/users/conta/urop/quaternions/"):
    
    def quaternion_binning(quaternion_t, q, maintimeaxis):
        sector_start = maintimeaxis[0]
        bins = 900 #30 min times sixty seconds/2 second cadence
                
        def find_nearest_values_index(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx
                
        binning_start = find_nearest_values_index(quaternion_t, sector_start)
        n = binning_start
        m = n + bins
        binned_Q = []
        binned_t = []
                
        while m <= len(quaternion_t):
            bin_t = quaternion_t[n]
            binned_t.append(bin_t)
            bin_q = np.mean(q[n:m])
            binned_Q.append(bin_q)
            n += 900
            m += 900
                
            
        standard_dev = np.std(np.asarray(binned_Q))
        mean_Q = np.mean(binned_Q)
        outlier_indexes = []
                
        for n in range(len(binned_Q)):
            if binned_Q[n] >= mean_Q + 5*standard_dev or binned_Q[n] <= mean_Q - 5*standard_dev:
                outlier_indexes.append(n)
                
                      
        return np.asarray(binned_t), np.asarray(binned_Q), outlier_indexes
        
    from scipy.signal import medfilt
    for root, dirs, files in os.walk(quaternion_folder):
            for name in files:
                if name.endswith(("S"+sector+"-quat.txt")):
                    print(name)
                    filepath = root + "/" + name
                    c = np.genfromtxt(filepath)
                    tQ = c[0]
                    Q1 = c[1]
                    Q2 = c[2]
                    Q3 = c[3]

    q = [Q1, Q2, Q3]

    
    for n in range(3):
        smoothed = medfilt(q[n], kernel_size = 31)
        if n == 0:
            Q1 = smoothed
            tQ_, Q1, Q1_outliers = quaternion_binning(tQ, Q1, maintimeaxis)
        elif n == 1:
            Q2 = smoothed
            tQ_, Q2, Q2_outliers = quaternion_binning(tQ, Q2, maintimeaxis)
        elif n == 2:
            Q3 = smoothed
            tQ_, Q3, Q3_outliers = quaternion_binning(tQ, Q3, maintimeaxis)
    
    outlier_indexes = np.unique(np.concatenate((Q1_outliers, Q2_outliers, Q3_outliers)))
    return tQ_, Q1, Q2, Q3, outlier_indexes  

def extract_smooth_quaterions(path, file, momentum_dump_csv, kernal, maintimeaxis, plot = False):

    from scipy.signal import medfilt
    f = fits.open(file, memmap=False)

    t = f[1].data['TIME']
    Q1 = f[1].data['C1_Q1']
    Q2 = f[1].data['C1_Q2']
    Q3 = f[1].data['C1_Q3']
    
    f.close()
    
    
    q = [Q1, Q2, Q3]
    
    if plot:
        with open(momentum_dump_csv, 'r') as f:
            lines = f.readlines()
            mom_dumps = [ float(line.split()[3][:-1]) for line in lines[6:] ]
            inds = np.nonzero((mom_dumps >= np.min(t)) * \
                              (mom_dumps <= np.max(t)))
            mom_dumps = np.array(mom_dumps)[inds]
    #q is a list of qs
    for n in range(3):
        
        smoothed = medfilt(q[n], kernel_size = kernal)
        
        if plot:
            plt.scatter(t, q[n], label = "original")
            plt.scatter(t, smoothed, label = "smoothed")
            
            for k in mom_dumps:
                plt.axvline(k, color='g', linestyle='--', alpha = 0.1)
            plt.legend(loc = "upper left")
            plt.title("Q" + str(n+1))
            plt.savefig(path + str(n + 1) + "-kernal-" + str(kernal) +"-both.png")
            plt.show()
            #plt.scatter(t, q[n], label = "original")
            plt.scatter(t, smoothed, label = "smoothed")
            for k in mom_dumps:
                plt.axvline(k, color='g', linestyle='--', alpha = 0.1)
            plt.legend(loc="upper left")
            plt.title("Q" + str(n+1) + "Smoothed")
            plt.savefig(path + str(n + 1) + "-kernal-" + str(kernal) +"-median-smoothed-only.png")
            plt.show()
            
        def quaternion_binning(quaternion_t, q, maintimeaxis):
            sector_start = maintimeaxis[0]
            bins = 900 #30 min times sixty seconds/2 second cadence
            
            def find_nearest_values_index(array, value):
                array = np.asarray(array)
                idx = (np.abs(array - value)).argmin()
                return idx
            binning_start = find_nearest_values_index(quaternion_t, sector_start)
            n = binning_start
            m = n + bins
            binned_Q = []
            binned_t = []
            
            while m <= len(t):
                bin_t = quaternion_t[n]
                binned_t.append(bin_t)
                bin_q = np.mean(q[n:m])
                binned_Q.append(bin_q)
                n += 900
                m += 900
            plt.scatter(binned_t, binned_Q)
            plt.show()
        
            standard_dev = np.std(np.asarray(binned_Q))
            mean_Q = np.mean(binned_Q)
            outlier_indexes = []
            
            for n in range(len(binned_Q)):
                if binned_Q[n] >= mean_Q + 5*standard_dev or binned_Q[n] <= mean_Q - 5*standard_dev:
                    outlier_indexes.append(n)
            
            print(outlier_indexes)      
            return outlier_indexes
        
        if n == 0:
            Q1 = smoothed
            Q1_outliers = quaternion_binning(t, Q1, maintimeaxis)
        elif n == 1:
            Q2 = smoothed
            Q2_outliers = quaternion_binning(t, Q2, maintimeaxis)
        elif n == 2:
            Q3 = smoothed
            Q3_outliers = quaternion_binning(t, Q3, maintimeaxis)
    
    outlier_indexes = np.unique(np.concatenate((Q1_outliers, Q2_outliers, Q3_outliers)))
    print(outlier_indexes)
    return t, Q1, Q2, Q3, outlier_indexes  



            









