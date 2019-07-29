from lion import main as lionmain
import tdpy
from tdpy.util import summgene

import numpy as np

import time as timemodu

import pickle

from sklearn.manifold import TSNE
from sklearn import manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection, neighbors
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
            
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

#from skimage import data
#from skimage.morphology import disk
#from skimage.filters.rank import median

#import pyod
#from pyod.models.abod import ABOD
#from pyod.models.cblof import CBLOF
#from pyod.models.feature_bagging import FeatureBagging
#from pyod.models.hbos import HBOS
#from pyod.models.iforest import IForest
#from pyod.models.knn import KNN
#from pyod.models.lof import LOF
#from pyod.models.loci import LOCI
#from pyod.models.mcd import MCD
#from pyod.models.ocsvm import OCSVM
#from pyod.models.pca import PCA
#from pyod.models.sos import SOS
#from pyod.models.lscp import LSCP

import scipy.signal
from scipy import interpolate

import os, sys, datetime, fnmatch

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from astroquery.mast import Catalogs
import astroquery

import astropy
from astropy.wcs import WCS
from astropy import units as u
from astropy.io import fits
import astropy.time
from astropy.coordinates import SkyCoord

import multiprocessing


def plot_embe(gdat, lcurflat, X_embedded, strg, titl):
    
    X_embedded = (X_embedded - np.amin(X_embedded, 0)) / (np.amax(X_embedded, 0) - np.amin(X_embedded, 0))

    figr, axis = plt.subplots(figsize=(12, 12))
    axis.scatter(X_embedded[:, 0], X_embedded[:, 1], s=5, marker='x', color='r', lw=0.5)
    shown_images = np.array([[1., 1.]])  # just something big
    for i in range(X_embedded.shape[0]):
        dist = np.sum((X_embedded[i] - shown_images) ** 2, 1)
        if np.min(dist) < 1e-3:
            continue
        shown_images = np.r_[shown_images, [X_embedded[i]]]
        axins3 = inset_axes(axis, width="100%", height="100%", \
                        bbox_to_anchor=(X_embedded[i, 0] - 0.02, X_embedded[i, 1] - 0.02, .04, .04), bbox_transform=axis.transData, loc='center', borderpad=0)
        axins3.plot(gdat.time, lcurflat[i, :], alpha=0.5, color='g')
        #axins3.set_ylim([0, 2])
        axins3.text(X_embedded[i, 0], X_embedded[i, 1] + 0.02, '%g %g' % (np.amin(lcurflat[i, :]), np.amax(lcurflat[i, :])), fontsize=12) 
        axins3.axis('off')
    axis.set_title(titl)
    plt.tight_layout()
    path = gdat.pathdata + '%s_%s.pdf' % (strg, gdat.strgcntp)
    print 'Writing to %s...' % path
    plt.savefig(path)
    plt.close()
    
            
def plot_anim(gdat, cntp, strgvarb, cmap='Greys_r', strgtitlbase='', boolresi=False, indxsideyposoffs=0, indxsidexposoffs=0):
    
    vmin = np.amin(cntp)
    vmax = np.amax(cntp)
    if boolresi:
        vmax = max(abs(vmax), abs(vmin))
        vmin = -vmax
    
    for t in gdat.indxtime:
        strgtitl = strgtitlbase + ', JD = %d' % gdat.time[t]
        path = gdat.pathdata + '%s_%s_%05d.pdf' % (strgvarb, gdat.strgcntp, t)
        plot_imag(gdat, cntp[:, :, t], path=path, strgvarb=strgvarb, cmap=cmap, strgtitl=strgtitl, \
                                        indxsideyposoffs=indxsideyposoffs, indxsidexposoffs=indxsidexposoffs, boolresi=boolresi, vmin=vmin, vmax=vmax)
    os.system('convert -density 300 -delay 10 %s%s_%s_*.pdf %s%s_%s.gif' % (gdat.pathdata, strgvarb, gdat.strgcntp, gdat.pathdata, strgvarb, gdat.strgcntp))
    ### delete the frame plots
    path = gdat.pathdata + '%s_%s_*.pdf' % (strgvarb, gdat.strgcntp)
    #os.system('rm %s' % path)


def plot_imag(gdat, cntp, strgvarb, path=None, cmap=None, indxsideyposoffs=0, indxsidexposoffs=0, \
                    strgtitl='', boolresi=False, xposoffs=None, yposoffs=None, indxpixlcolr=None, vmin=None, vmax=None):
    
    if cmap == None:
        if boolresi:
            cmap = 'RdBu'
        else:
            cmap = 'Greys_r'
    
    if vmin is None or vmax is None:
        vmax = np.amax(cntp)
        vmin = np.amin(cntp)
        if boolresi:
            vmax = max(abs(vmax), abs(vmin))
        vmin = -vmax
    
    if gdat.cntpscaltype == 'asnh':
        cntp = np.arcsinh(cntp)
        vmin = np.arcsinh(vmin)
        vmax = np.arcsinh(vmax)

    figr, axis = plt.subplots(figsize=(8, 6))
    objtimag = axis.imshow(cntp, origin='lower', interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    
    if indxpixlcolr is not None:
        temp = np.zeros_like(cntp).flatten()
        temp[indxpixlcolr[-1]] = 1.
        temp = temp.reshape((gdat.numbside, gdat.numbside))
        alph = np.zeros_like(cntp).flatten()
        alph[indxpixlcolr[-1]] = 1.
        alph = alph.reshape((gdat.numbside, gdat.numbside))
        alph = np.copy(temp)
        axis.imshow(temp, origin='lower', interpolation='nearest', alpha=0.5)
    
    # overplot catalog
    plot_catl(gdat, axis, indxsideyposoffs=indxsideyposoffs, indxsidexposoffs=indxsidexposoffs)
    
    # make color bar
    cax = figr.add_axes([0.83, 0.1, 0.03, 0.8])
    cbar = figr.colorbar(objtimag, cax=cax)
    if gdat.cntpscaltype == 'asnh':
        tick = cbar.get_ticks()
        print 'tick'
        print tick
        tick = np.sinh(tick)
        print 'tick'
        print tick
        labl = ['%d' % int(tick[k]) for k in range(len(tick))]
        print 'labl'
        print labl
        print 
        cbar.set_ticklabels(labl)

    if path is None:
        path = gdat.pathdata + '%s_%s.pdf' % (strgvarb, gdat.strgcntp)
    print 'Writing to %s...' % path
    #plt.tight_layout()
    plt.savefig(path)
    plt.close()
    

def init( \
         listisec=None, \
         icam=None, \
         iccd=None, \
         pathfile=None, \
         datatype='obsd', \
         rasctarg=None, \
         decltarg=None, \
         labltarg=None, \
         strgtarg=None, \
         numbside=None, \
         **args \
        ):
    
    # preliminary setup
    # construct the global object 
    gdat = tdpy.util.gdatstrt()
    for attr, valu in locals().iteritems():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)
    
    # copy all provided inputs to the global object
    for strg, valu in args.iteritems():
        setattr(gdat, strg, valu)

    # string for date and time
    gdat.strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print 'TTEP initialized at %s...' % gdat.strgtimestmp
   
    print 'gdat.datatype'
    print gdat.datatype

    # paths
    ## read PCAT path environment variable
    gdat.pathdata = os.environ['TESSTRAN_DATA_PATH'] + '/'
    ## define paths
    #gdat.pathdataorig = '/pdo/qlp-data/orbit-%d/ffi/cam%d/ccd%d/FITS/' % (isec, icam, iccd)
    gdat.pathdataorig = gdat.pathdata + 'ffis/'
    gdat.pathdatafilt = gdat.pathdata + 'filt/'
    gdat.pathdatainit = gdat.pathdata + 'init/'
    gdat.pathdatainitimag = gdat.pathdatainit + 'imag/'
    gdat.pathdatainitanim = gdat.pathdatainit + 'anim/'
    gdat.pathdatacomm = gdat.pathdata + 'comm/'
    ## make folders 
    os.system('mkdir -p %s' % gdat.pathdatafilt)
    
    if pathfile is None:
        print 'Cutout input data are not provided.'
        if numbside is None:
            numbsideyposfull = 2078
            numbsidexposfull = 2136
            numbpixloffsypos = 30
            numbpixloffsxpos = 44
            strgmode = 'full'
        else:
            numbpixloffsypos = 0
            numbpixloffsxpos = 0
            strgmode = 'targ'
    else:
        strgmode = 'file'
        numbsideyposfull = cntpmemo.shape[0]
        numbsidexposfull = cntpmemo.shape[1]
        numbpixloffsypos = 0
        numbpixloffsxpos = 0
        print 'Cutout input data are provided.'
        
    if not (pathfile is None and numbside is not None):
        numbside = numbsideyposfull - numbpixloffsypos
        print 'numbsideyposfull'
        print numbsideyposfull
        print 'numbsidexposfull'
        print numbsidexposfull
    print 'numbside'
    print numbside
    
    timeexpo = 1426.
    
    if strgmode == 'targ':
        from astroquery.mast import Tesscut
        from astropy.coordinates import SkyCoord
        cutout_coord = SkyCoord(rasctarg, decltarg, unit="deg")
        listhdundata = Tesscut.get_cutouts(cutout_coord, gdat.numbside)
        sector_table = Tesscut.get_sectors(SkyCoord(gdat.rasctarg, gdat.decltarg, unit="deg"))
        listisec = sector_table['sector'].data
        listicam = sector_table['camera'].data
        listiccd = sector_table['ccd'].data
    
        if len(listhdundata) == 0:
            raise Exception('TESSCut could not find any data.')
    
    numbsect = len(listisec)
    indxsect = np.arange(numbsect)
    for o in indxsect:

        # check inputs
        np.set_printoptions(precision=3, linewidth=180)
        print 'Sector: %d' % listisec[o]
        print 'Camera: %d' % listicam[o]
        print 'CCD: %d' % listiccd[o]
        
        isec = listisec[o]
        icam = listicam[o]
        iccd = listiccd[o]

        verbtype = 1
        
        np.random.seed(45)

        # fix the seed
        if gdat.datatype == 'mock':
            numbsour = 1000
            numbsupn = 10
        
        gdat.numbtimerebn = 30
        
        if strgmode == 'file':
            # get list of paths where FFIs live
            listrtag = fnmatch.filter(os.listdir(gdat.pathdataorig), 'tess*-s%04d-%d-%d-*-s_ffic.fits' % (isec, icam, iccd))
            listrtag.sort()
            #listrtag = listrtag#[:20]

            if gdat.datatype == 'mock':
                gdat.numbtime = gdat.numbtimerebn
            else:
                gdat.numbtime = len(listrtag)
        else:
            #if strgmode == 'full':
            #listhdundata = fits.open(pathfile)
            time = listhdundata[o][1].data['TIME']
            gdat.numbtime = time.size
            cntpmemo = np.swapaxes(np.swapaxes(listhdundata[o][1].data['FLUX'], 0, 1), 1, 2)
            listobjtwcss = WCS(listhdundata[o][1].header)
        
        # settings
        ## parameters
        numbsidecorr = 1
        numbneigaper = 1
        numbstrd = 1
        
        ## plotting
        gdat.cntpscaltype = 'asnh'
        
        gdat.offscorr = numbsidecorr / 2

        gdat.numbneigback = 8
        
        if pathfile is not None and gdat.datatype == 'mock':
            raise Exception('')
        
        numbsideback = 2 * gdat.numbneigback + 1
        gdat.indxtime = np.arange(gdat.numbtime)
        
        gdat.numbsideedge = gdat.numbneigback + gdat.offscorr
        numbmemo = (numbside - gdat.numbsideedge) // 256 + 1
        indxmemo = np.arange(numbmemo)
        numbsidesrch = (numbside - gdat.numbsideedge) / numbmemo
        numbsideshft = numbsidesrch + gdat.numbsideedge
        numbsidememo = numbsidesrch + 2 * gdat.numbneigback + 2 * gdat.offscorr
        indxsidememo = np.arange(numbsidememo)
        
        if (numbside - gdat.numbsideedge) % numbmemo != 0:
            print 'gdat.numbsideedge'
            print gdat.numbsideedge
            print 'numbside - gdat.numbsideedge'
            print numbside - gdat.numbsideedge
            print 'numbmemo'
            print numbmemo
            raise Exception('')

        print 'numbsidememo'
        print numbsidememo
        print 'numbsidesrch'
        print numbsidesrch
        
        numbdata = numbsidesrch**2
        indxdata = np.arange(numbdata)
        
        indxsidesrch = np.arange(numbsidesrch)
        numbsrch = numbsidesrch**2
        
        indxsideyposmemo, indxsidexposmemo = np.meshgrid(indxsidememo, indxsidememo)
        indxsideyposdata = indxsideyposmemo[gdat.numbsideedge:-gdat.numbsideedge, gdat.numbsideedge:-gdat.numbsideedge] - gdat.numbsideedge
        indxsidexposdata = indxsidexposmemo[gdat.numbsideedge:-gdat.numbsideedge, gdat.numbsideedge:-gdat.numbsideedge] - gdat.numbsideedge
        
        if numbdata != indxsideyposdata.size:
            print 'numbdata'
            print numbdata
            print 'indxsideyposdata'
            summgene(indxsideyposdata)
            raise Exception('')

        indxsideyposdatatemp = np.empty((numbsidesrch, numbsidesrch, numbsidecorr, numbsidecorr))
        indxsidexposdatatemp = np.empty((numbsidesrch, numbsidesrch, numbsidecorr, numbsidecorr))
        indxsideyposdatatemp[:, :, :, :] = indxsideyposdata[:, :, None, None]
        indxsidexposdatatemp[:, :, :, :] = indxsidexposdata[:, :, None, None]
        gdat.indxsideyposdataflat = indxsideyposdatatemp.flatten().astype(int)
        gdat.indxsidexposdataflat = indxsidexposdatatemp.flatten().astype(int)
        numbpixlmemo = numbsidememo**2
                
        if gdat.datatype == 'mock':
            listlabltrue = np.zeros(numbdata, dtype=int)
            numbinli = numbdata - numbsour
            numboutl = numbsour
        
        thrsrmss = 0.01
        thrsmaxm = 1.5
        thrsdiff = 0.5
        numbsideaper = 2 * numbneigaper + 1
        numbpixlaper = numbsideaper**2
        
        if float(numbside) % numbstrd != 0:
            raise Exception('')

        # grid of flux space
        minmproj = 0.1
        maxmproj = 2
        limtproj = [minmproj, maxmproj]
        arry = np.linspace(minmproj, maxmproj, 100)
        xx, yy = np.meshgrid(arry, arry)
        
        magtminm = 12.
        magtmaxm = 19.
        print 'gdat.numbtime'
        print gdat.numbtime
        print 'numbsideback'
        print numbsideback
        print 'numbsideaper'
        print numbsideaper
        print 'numbstrd'
        print numbstrd
        print 'numbsrch'
        print numbsrch
        
        indxsidememosrch = np.arange(gdat.numbneigback + gdat.offscorr, numbsidesrch + gdat.numbsideedge)
        
        liststrgpara = ['rmss', 'maxm', 'mean', 'diff']
        listlablpara = ['R', 'Max', 'Mean', 'Difference']
        dictpara = {}
        dictpara['rmss'] = np.zeros((numbsidememo, numbsidememo)) - 1.
        dictpara['maxm'] = np.zeros((numbsidememo, numbsidememo)) - 1.
        dictpara['mean'] = np.zeros((numbsidememo, numbsidememo)) - 1.
        dictpara['diff'] = np.zeros((numbsidememo, numbsidememo)) - 1.
        
        print 'Reading the quaternion...'
        path = gdat.pathdata + 'quat/' + fnmatch.filter(os.listdir(gdat.pathdata + 'quat/'), 'tess*_sector%02d-quat.fits' % isec)[0]
        print 'path'
        print path
        listhdunquat = fits.open(path)
        print 'listhdunquat[0].header'
        print listhdunquat[0].header
        quat = np.empty((2, gdat.numbtime))
        quat[:, :] = np.linspace(0., 0.1, gdat.numbtime)[None, :]
        listhdunquat.close()
        
        # i indexes the y-axis
        for i in indxmemo: 
            # j indexes the y-axis
            for j in indxmemo: 
        
                if (i != 0 or j != 0):# and gdat.datatype == 'mock':
                    continue

                print 'Memory region i=%d, j=%d' % (i, j)
                
                # plots
                ## file name string extension for image plots
                gdat.strgcntp = '%s_%02d%d%d_%d_%d' % (gdat.datatype, isec, icam, iccd, i, j)
               
                # determine the initial and final pixel indices of the FFI data to be copied
                indxsideyposdatainit = numbpixloffsypos + j * numbsidememo - gdat.numbsideedge
                indxsideyposdatafinl = numbpixloffsypos + (j + 1) * numbsidememo - gdat.numbsideedge
                indxsidexposdatainit = numbpixloffsxpos + i * numbsidememo - gdat.numbsideedge
                indxsidexposdatafinl = numbpixloffsxpos + (i + 1) * numbsidememo - gdat.numbsideedge
                if j == 0:
                    indxsideyposdatainit += gdat.numbneigback + gdat.offscorr
                    indxsideyposdatafinl += gdat.numbneigback + gdat.offscorr
                #if j == numbmemo - 1:
                #    indxsideyposdatainit -= gdat.numbneigback + gdat.offscorr
                #    indxsideyposdatafinl -= gdat.numbneigback + gdat.offscorr
                if i == 0:
                    indxsidexposdatainit += gdat.numbneigback + gdat.offscorr
                    indxsidexposdatafinl += gdat.numbneigback + gdat.offscorr
                #if i == numbmemo - 1:
                #    indxsidexposdatainit -= gdat.numbneigback + gdat.offscorr
                #    indxsidexposdatafinl -= gdat.numbneigback + gdat.offscorr
                
                print 'indxsideyposdatainit'
                print indxsideyposdatainit
                print 'indxsideyposdatafinl'
                print indxsideyposdatafinl
                print 'indxsidexposdatainit'
                print indxsidexposdatainit
                print 'indxsidexposdatafinl'
                print indxsidexposdatafinl
                print
                
                #if strgmode == 'targ':
                #    listhduncatl = listhdundata[o]
                #else:
                #    path = gdat.pathdataorig + listrtag[0]
                #    listhduncatl = fits.open(path)
                #if rasctarg is not None:
                #    posipixl = listobjtwcss[0].all_world2pix(np.array([[rasctarg, decltarg]]), 1)
                pathsavedata = gdat.pathdata + '%s_data.npz' % gdat.strgcntp
                if strgmode == 'targ':
                    listhdundatatemp = listhdundata[o]
                    hdundata = listhdundatatemp[1].data['FLUX'].astype(float) * timeexpo
                
                else:
                    if gdat.datatype == 'mock' or not os.path.exists(pathsavedata):
                        
                        listobjtwcss = []
                        if gdat.datatype == 'obsd':
                            gdat.time = []
                            cntpmemo = []
                        for t in gdat.indxtime:
                            if t == 0 and gdat.datatype == 'mock':
                                continue
                            if t % 100 == 0 and gdat.datatype == 'obsd':
                                print 'Loading the image into memory, t = %d' % t
                            path = gdat.pathdataorig + listrtag[t]
                            listhdundatatemp = fits.open(path, memmap=False)
                            
                            objtheadseco = listhdundatatemp[1].header
                            listobjtwcss.append(WCS(objtheadseco))
                            if gdat.datatype == 'obsd':
                                objtheadfrst = listhdundatatemp[0].header
                                #print 'objtheadfrst'
                                #print objtheadfrst
                                timetemp = (objtheadfrst['TSTOP'] + objtheadfrst['TSTART']) / 2
                                gdat.time.append(timetemp)
                                
                                hdundata = listhdundatatemp[1].data.astype(float) * timeexpo
                                cntpmemo.append(hdundata[indxsideyposdatainit:indxsideyposdatafinl, indxsidexposdatainit:indxsidexposdatafinl])
                            
                            listhdundatatemp.close()
                        
                        if gdat.datatype == 'obsd':
                            gdat.time = np.array(gdat.time)
                            cntpmemo = np.stack(cntpmemo, axis=-1)
                            listtemp = [listobjtwcss, gdat.time, cntpmemo]

                            objtfile = open(pathsavedata, 'wb')
                            
                            print 'Writing to %s...' % pathsavedata
                            pickle.dump(listtemp, objtfile)
                    else:
                        print 'Reading from %s...' % pathsavedata
                        objtfile = open(pathsavedata, "rb" )
                        listtemp = pickle.load(objtfile)
                        if gdat.datatype == 'obsd':
                            listobjtwcss, gdat.time, cntpmemo = listtemp
                        else:
                            listobjtwcss, gdat.time = listtemp
                       
                if gdat.datatype == 'mock':
                    # Data generation
                    ## image
                    gdat.time = np.concatenate((np.linspace(1., 12.7, gdat.numbtime / 2), np.linspace(1., 12.7, gdat.numbtime / 2)))
                    arrytime = np.empty((2, gdat.numbtime))
                    arrytime[:, :] = np.linspace(-0.5, 0.5, gdat.numbtime)[None, :]
                    gdat.indxsour = np.arange(numbsour)
                    gdat.indxtime = np.arange(gdat.numbtime)
                    posiquat = 5e-2 * np.random.randn(2 * gdat.numbtime).reshape((2, gdat.numbtime)) + arrytime * 0.1
                    gdat.trueypos = numbsidememo * np.random.random(numbsour)[None, :] + posiquat[0, :, None]
                    gdat.truexpos = numbsidememo * np.random.random(numbsour)[None, :] + posiquat[1, :, None]
                    
                    cntpmemo = np.ones((numbsidememo, numbsidememo, gdat.numbtime)) * 6.
                    
                    indxsideyposmemocent = (j + 1.5) * numbsideshft + numbpixloffsypos + gdat.numbsideedge
                    indxsidexposmemocent = (i + 1.5) * numbsideshft + numbpixloffsxpos + gdat.numbsideedge
                    posiskyy = listobjtwcss[0].all_pix2world(indxsideyposmemocent, indxsidexposmemocent, 0)
                    strgsrch = '%g %g' % (posiskyy[0], posiskyy[1])
                    try:
                        catalogData = Catalogs.query_region(strgsrch, radius='0.1m', catalog = "TIC")
                        if len(catalogData) > 0:
                            tici = int(catalogData[0]['ID'])
                            titl += ', TIC %d' % tici
                    except:
                        pass

                    # inject signal
                    indxsupn = np.arange(numbsupn)
                    truecntpsour = np.empty((gdat.numbtime, numbsour))
                    truemagt = np.empty((gdat.numbtime, numbsour))
                    gdat.indxsoursupn = np.random.choice(gdat.indxsour, size=numbsupn, replace=False)
                    for n in gdat.indxsour:
                        #print 'n'
                        #print n
                        if n in gdat.indxsoursupn:
                            timenorm = -0.5 + (gdat.time / np.amax(gdat.time)) + 2. * (np.random.random(1) - 0.5)
                            objtrand = scipy.stats.skewnorm(10.).pdf(timenorm)
                            objtrand /= np.amax(objtrand)
                            truemagt[:, n] = 8. + 6. * (2. - objtrand)
                            print 'n'
                            print n
                            for t in gdat.indxtime:
                                print t, truemagt[t, n]
                            print
                        
                        else:
                            truemagt[:, n] = np.random.rand() * 5 + 15.
                        truecntpsour[:, n] = 10**((20.424 - truemagt[:, n]) / 2.5)
                        #print 'truemagt[:, n]'
                        #summgene(truemagt[:, n])
                        #print 'truecntpsour[:, n]'
                        #summgene(truecntpsour[:, n])
                        #print
                    gdat.truemagtmean = np.mean(truemagt, 0)
                    gdat.truemagtstdv = np.std(truemagt, 0)

                    indxsideypossour = np.round(np.mean(gdat.trueypos, 0)).astype(int)
                    indxsidexpossour = np.round(np.mean(gdat.truexpos, 0)).astype(int)
                    
                    sigmpsfn = 1.

                    for k in gdat.indxsour:
                        temp = 1. / np.sqrt(sigmpsfn**2 * (2. * np.pi)**2) * \
                            truecntpsour[None, None, :, k] * np.exp(-0.5 * ((indxsidexposmemo[:, :, None] - gdat.truexpos[None, None, :, k])**2 + \
                                                                    (indxsideyposmemo[:, :, None] - gdat.trueypos[None, None, :, k])**2) / sigmpsfn**2)
                        cntpmemo[:, :, :] += temp
                    
                    indxsideypossour[np.where(indxsideypossour == numbsidememo)] = numbsidememo - 1
                    indxsidexpossour[np.where(indxsidexpossour == numbsidememo)] = numbsidememo - 1
                    indxsideypossour[np.where(indxsideypossour < 0)] = 0
                    indxsidexpossour[np.where(indxsidexpossour < 0)] = 0
                    
                    indxsourinsd = np.where((indxsideypossour > gdat.numbneigback + gdat.offscorr) & \
                                   (indxsidexpossour > gdat.numbneigback + gdat.offscorr) & (indxsideypossour < numbsidememo - gdat.numbsideedge) & \
                                   (indxsidexpossour < numbsidememo - gdat.numbsideedge))[0]
                    
                    indxsupninsd = np.where((indxsideypossour[gdat.indxsoursupn] > gdat.numbneigback + gdat.offscorr) & \
                                   (indxsidexpossour[gdat.indxsoursupn] > gdat.numbneigback + gdat.offscorr) & \
                                   (indxsideypossour[gdat.indxsoursupn] < numbsidememo - gdat.numbsideedge) & \
                                   (indxsidexpossour[gdat.indxsoursupn] < numbsidememo - gdat.numbsideedge))[0]
                    
                    indxdatasour = (indxsideypossour - gdat.numbsideedge) * numbsidesrch + indxsidexpossour - gdat.numbsideedge
                    indxdatasupn = (indxsideypossour[gdat.indxsoursupn] - gdat.numbsideedge) * numbsidesrch + \
                                                                        indxsidexpossour[gdat.indxsoursupn] - gdat.numbsideedge
                    
                    indxdataback = np.setdiff1d(indxdata, indxdatasour)
                    listlabltrue[indxdatasour[indxsourinsd]] = 1
                    cntpmemo *= timeexpo
                    cntpmemo = np.random.poisson(cntpmemo).astype(float)
                
                # plot the quaternion
                if i == 0 and j == 0:
                    figr, axis = plt.subplots()
                    axis.plot(gdat.time, quat[0, :], color='black', ls='', marker='o', markersize=3, label='q_x')
                    axis.plot(gdat.time, quat[1, :], color='black', ls='', marker='o', markersize=3, label='q_y')
                    axis.plot(gdat.time, np.sqrt(quat[0, :]**2 + quat[1, :]**2), \
                                        color='black', ls='', marker='o', markersize=3, label=r'$\sqrt{q_x^2 + q_y^2}$')
                    axis.set_xlabel('Time [BJD]')
                    axis.set_ylabel('Quaternion [px]')
                    path = gdat.pathdata + 'quat_%s.pdf' % (gdat.strgcntp)
                    plt.savefig(path)
                    plt.close()
                
                print 'Shifting the images...'
                x = np.linspace(0.5, numbsidememo - 0.5, numbsidememo)
                y = np.linspace(0.5, numbsidememo - 0.5, numbsidememo)
                print 'cntpmemo'
                summgene(cntpmemo)
                print 'x'
                summgene(x)
                f = interpolate.interp2d(x, y, cntpmemo[:, :, 0], kind='cubic')
                print 'cntpmemo[:, :, 0]'
                summgene(cntpmemo[:, :, 0])
                for t in gdat.indxtime:
                    cntpmemo[:, :, t] = f(x + quat[0, t], y + quat[1, t])
                    print 't'
                    print t
                    print 'quat[:, t]'
                    print quat[:, t]
                    print 'cntpmemo[:, :, t]'
                    summgene(cntpmemo[:, :, t])
                cntpmemospatmedi = np.median(cntpmemo, (0, 1))
                    
                cntpcutt = 100 * timeexpo

                # plot the spatial median
                figr, axis = plt.subplots()
                temp = np.arcsinh(cntpmemospatmedi)
                axis.plot(gdat.time, temp, color='black', ls='', marker='o', markersize=3)
                tick = np.linspace(np.amin(temp), np.amax(temp), 10)
                axis.set_yticks(tick)
                axis.set_yticklabels(['%.3g' % np.sinh(tick[k]) for k in range(10)])
                axis.axhline(cntpcutt, ls='', alpha=0.5)
                axis.set_xlabel('Time [BJD]')
                axis.set_ylabel('Flux [e$^{-}$/s]')
                path = gdat.pathdata + 'cntpmemospatmedi_%s.pdf' % (gdat.strgcntp)
                plt.savefig(path)
                plt.close()
                
                # mask the data
                print 'Masking out the data...'
                gdat.indxtimeorbt = np.argmax(np.diff(gdat.time))
                gdat.indxtimegood = np.where((gdat.time > np.amin(gdat.time) + 2.) & (gdat.time < np.amax(gdat.time) - 0.) & (cntpmemospatmedi < cntpcutt) & \
                                        ((gdat.time < gdat.time[gdat.indxtimeorbt] - 0.) | (gdat.time > gdat.time[gdat.indxtimeorbt+1] + 2.)))[0]
                print 'Number of pixels that pass the mask: %d' % gdat.indxtimegood.size
                listobjtwcss = np.array(listobjtwcss)[gdat.indxtimegood]
                gdat.time = gdat.time[gdat.indxtimegood]
                cntpmemo = cntpmemo[:, :, gdat.indxtimegood]
                gdat.numbtime = gdat.time.size
                gdat.indxtime = np.arange(gdat.numbtime)
                cntpmemospatmedi = cntpmemospatmedi[gdat.indxtimegood]
                    
                # plots
                ## random pixel light curves
                figr, axis = plt.subplots(10, 4)
                indxsideyposrand = np.random.choice(indxsidememo, size=40)
                indxsidexposrand = np.random.choice(indxsidememo, size=40)
                for a in range(10):
                    for b in range(4):
                        p = a * 4 + b
                        if p >= numbdata:
                            continue
                        axis[a][b].plot(gdat.time, cntpmemo[indxsideyposrand[p], indxsidexposrand[p], :], color='black', ls='', marker='o', markersize=3)
                        if a != 9:
                            axis[a][b].set_xticks([])
                        if b != 0:
                            axis[a][b].set_yticks([])
                path = gdat.pathdata + 'lcurrand_%s.pdf' % (gdat.strgcntp)
                plt.savefig(path)
                plt.close()

                # spatial median
                print 'Performing the spatial median filter...'
                #cntpmemo = cntpmemo - scipy.signal.medfilt(cntpmemo, (11, 11, 1))

                # temporal median filter
                numbtimefilt = min(9, gdat.numbtime)
                if numbtimefilt % 2 == 0:
                    numbtimefilt -= 1
                print 'Performing the temporal median filter...'
                cntpmemo = scipy.signal.medfilt(cntpmemo, (1, 1, numbtimefilt))
                
                if cntpmemo.shape[0] != numbsidememo or cntpmemo.shape[1] != numbsidememo:
                    print 'cntpmemo'
                    summgene(cntpmemo)
                    print 'numbsidememo'
                    print numbsidememo
                    raise Exception('')

                # rebin in time
                if gdat.numbtime > gdat.numbtimerebn:
                    print 'Rebinning in time...'
                    numbtimeoldd = gdat.numbtime
                    gdat.numbtime = gdat.numbtimerebn
                    numbtimebins = numbtimeoldd / gdat.numbtime
                    cntpmemoneww = np.zeros((numbsidememo, numbsidememo, gdat.numbtime)) - 1.
                    timeneww = np.zeros(gdat.numbtime)
                    for t in range(gdat.numbtime):
                        if t == gdat.numbtime - 1:
                            cntpmemoneww[:, :, t] = np.mean(cntpmemo[:, :, (gdat.numbtime-1)*numbtimebins:], axis=2)
                            timeneww[t] = np.mean(gdat.time[(gdat.numbtime-1)*numbtimebins:])
                        else:
                            cntpmemoneww[:, :, t] = np.mean(cntpmemo[:, :, t*numbtimebins:(t+1)*numbtimebins], axis=2)
                            timeneww[t] = np.mean(gdat.time[t*numbtimebins:(t+1)*numbtimebins])
                    gdat.indxtimegood = np.isfinite(timeneww)
                    cntpmemo = cntpmemoneww[:, :, gdat.indxtimegood]
                    gdat.time = timeneww[gdat.indxtimegood]
                    gdat.numbtime = gdat.indxtimegood.size
                    gdat.indxtime = np.arange(gdat.numbtime)
                
                # check cntpmemo
                if not np.isfinite(cntpmemo).all():
                    raise Exception('')

                # calculate derived maps
                ## RMS image
                cntpmemotimemedi = np.median(cntpmemo, 2)
                cntptemp = np.std(cntpmemo - cntpmemotimemedi[:, :, None], 2) / cntpmemotimemedi
                plot_imag(gdat, cntptemp, 'cntpstdv', cmap='Reds', strgtitl='RMS')

                strgtype = 'tsne'

                if i == 0 and j == 0:
                    lcurarry = np.empty((numbsidesrch, numbsidesrch, gdat.numbtime, numbsidecorr, numbsidecorr)) 
                    medilcur = np.empty((numbsidesrch, numbsidesrch)) 
               
                # normalize by the temporal median

                lcuravgd = np.empty(gdat.numbtime)
                cntr = 0
                prevfrac = -1
                k = 0
                
                # machine learning
                ## get neighboring pixels as additional features
                for a in np.arange(numbsidecorr):
                    for b in np.arange(numbsidecorr):
                        if numbsidecorr == 3:
                            offs = -1
                        else:
                            offs = 0
                        indx = gdat.numbneigback + gdat.offscorr + a + offs
                        lcurarry[:, :, :, a, b] = cntpmemo[indx:indx+numbsidesrch, indx:indx+numbsidesrch, :]
                lcurflat = lcurarry.reshape((numbsidesrch**2, gdat.numbtime, numbsidecorr, numbsidecorr))
                lcurflat = lcurflat.reshape((numbsidesrch**2, gdat.numbtime * numbsidecorr**2))

                if False:
                            
                    indxsideypostemp = k * numbstrd
                    indxsidexpostemp = l * numbstrd
                    if not np.isfinite(cntpmemo).all():
                        raise Exception('')

                    if strgtype == 'tsne' or strgtype == 'tmpt':
                        pass
                    else:
                        indxsideyposaperinit = indxsideypostemp - numbneigaper
                        indxsidexposaperinit = indxsidexpostemp - numbneigaper
                        indxsideyposaperfinl = indxsideypostemp + numbneigaper + 1
                        indxsidexposaperfinl = indxsidexpostemp + numbneigaper + 1
                        indxsideyposbackinit = indxsideypostemp - gdat.numbneigback
                        indxsidexposbackinit = indxsidexpostemp - gdat.numbneigback
                        indxsideyposbackfinl = indxsideypostemp + gdat.numbneigback + 1
                        indxsidexposbackfinl = indxsidexpostemp + gdat.numbneigback + 1
                        
                        imagbackmedi = np.median(cntpmemo[indxsideyposbackinit:indxsideyposbackfinl, \
                                                                indxsidexposbackinit:indxsidexposbackfinl, :], axis=(0, 1))
                        for t in gdat.indxtime:
                            lcur[t] = np.sum(cntpmemo[indxsideyposaperinit:indxsideyposaperfinl, indxsidexposaperinit:indxsidexposaperfinl, t]) - \
                                                                                                        imagbackmedi[t] * numbpixlaper
                        if not np.isfinite(lcur).all():
                            print 'imagbackmedi'
                            print imagbackmedi
                            raise Exception('')
                        
                        # normalize
                        meanlcur = np.mean(lcur)
                        lcur /= meanlcur
                        #lcurmedi = scipy.signal.medfilt(lcur, 11)
                        dictpara['mean'][k, l] = meanlcur
                        
                        lcurdiff = lcur - lcuravgd
                        gdat.indxtimediff = np.argsort(lcurdiff)[::-1]
                        for t in gdat.indxtimediff:
                            if t < 0.2 * gdat.numbtime or (t >= 0.5 * gdat.numbtime and t <= 0.7 * gdat.numbtime):
                                continue
                            if lcurdiff[t] > thrsdiff and lcurdiff[t-1] > thrsdiff:
                                break
                        gdat.indxtimediffaccp = t
                        dictpara['diff'][k, l] = lcurdiff[gdat.indxtimediffaccp]

                        # acceptance condition
                        boolgood = False
                        if dictpara['diff'][k, l] > thrsdiff:
                        #if dictpara['maxm'][k, l] > thrsmaxm and (lcurtest[gdat.indxtimemaxm+1] > thrsmaxm or lcurtest[gdat.indxtimemaxm-1] > thrsmaxm):
                            boolgood = True
                        
                        if abs(k - 50) < 4 and abs(l - 50) < 4:
                            boolgood = True
                            
                        if boolgood:# or (abs(k - 53) < 5 and abs(l - 53) < 5):
                            
                            indxsideyposaccp.append(k * numbstrd)
                            indxsidexposaccp.append(l * numbstrd)
                    
                            # plot
                            figr, axis = plt.subplots(figsize=(12, 6))
                            axis.plot(time, lcur, ls='', marker='o', markersize=3, label='Raw')
                            #axis.plot(time, lcurmedi, ls='', marker='o', markersize=3, label='Median')
                            #axis.plot(time, lcurtest, ls='', marker='o', markersize=3, label='Cleaned')
                            axis.plot(time, lcurdiff + 1., ls='', marker='o', markersize=3, label='Diff')
                            
                            axis.set_xlabel('Time [days]')
                            axis.set_ylabel('Relative Flux')
                            axis.legend()
                            
                            axis.axhline(thrsdiff + 1., ls='--', alpha=0.3, color='gray')
                            axis.axvline(time[int(0.2*gdat.numbtime)], ls='--', alpha=0.3, color='red')
                            axis.axvline(time[int(0.5*gdat.numbtime)], ls='--', alpha=0.3, color='red')
                            axis.axvline(time[int(0.7*gdat.numbtime)], ls='--', alpha=0.3, color='red')
                            
                            posisili = np.empty((1, 2))
                            posisili[0, 0] = indxsideyposdatainit + k
                            posisili[0, 1] = indxsidexposdatainit + l
                            
                            titl = 'Diff: %g' % (dictpara['diff'][k, l])
                            if gdat.datatype == 'obsd':
                                if pathfile is None:
                                    posiskyy = listobjtwcss[t].all_pix2world(posisili, 0)
                                else:
                                    posiskyy = listobjtwcss.all_pix2world(posisili, 0)
                                rasc = posiskyy[:, 0]
                                decl = posiskyy[:, 1]
                                strgsrch = '%g %g' % (rasc, decl)
                                titl += 'k = %d, l = %d' % (k, l)
                                #catalogData = Catalogs.query_region(strgsrch, radius='0.1m', catalog = "TIC")
                                #if len(catalogData) > 0:
                                #    tici = int(catalogData[0]['ID'])
                                #    titl += ', TIC %d' % tici
                            
                                #axis.axvspan(1345, 1350, alpha=0.5, color='red')
                            
                            axis.set_title(titl)
                            plt.tight_layout()
                            path = gdat.pathdata + 'lcur_%s_%04d_%04d.pdf' % (gdat.strgcntp, k, l)
                            print 'Writing to %s...' % path
                            plt.savefig(path)
                            plt.close()
                    
                n_neighbors = 30
               
                X = lcurflat

                indxdata = np.arange(numbdata)
                
                outliers_fraction = 0.15
                
                # define outlier/anomaly detection methods to be compared
                listobjtalgoanom = [
                                    #("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1)), \
                                    #("Isolation Forest", IsolationForest(contamination=outliers_fraction)), \
                                    ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction))
                                   ]
                
                numbmeth = len(listobjtalgoanom)
                indxmeth = np.arange(numbmeth)
                
                listindxsideyposaccp = []
                listindxsidexposaccp = []
                listscor = []
                listlablmodl = []
                
                numbtimeplotscat = min(6, gdat.numbtime)
                limt = [np.amin(X), np.amax(X)]
                
                c = 0
                print 'Running anomaly-detection algorithms...'
                for name, objtalgoanom in listobjtalgoanom:
                    t0 = timemodu.time()
                    print 'name'
                    print name
                    print 'c'
                    print c

                    objtalgoanom.fit(X)
                    t1 = timemodu.time()
                
                    # fit the data and tag outliers
                    if name == 'Local Outlier Factor':
                        scor = objtalgoanom.negative_outlier_factor_
                    else:
                        scor = objtalgoanom.decision_function(X)
                    if name == "Local Outlier Factor":
                        lablmodl = np.zeros(numbdata)
                        lablmodl[np.where(scor < -2)[0]] = 1.
                    else:
                        lablmodl = objtalgoanom.fit(X).predict(X)
                    
                    indxdataposi = np.where(lablmodl == 1)[0]
                    indxdatanega = np.setdiff1d(indxdata, indxdataposi)
                    numbposi = indxdataposi.size
                    gdat.numbpositext = min(200, numbposi)

                    indxsideyposaccp = indxdataposi // numbsidememo
                    indxsidexposaccp = indxdataposi % numbsidememo
                    
                    listindxsideyposaccp.append(indxsideyposaccp)
                    listindxsidexposaccp.append(indxsidexposaccp)
                    listscor.append(scor)
                    listlablmodl.append(lablmodl)
                    
                    gdat.indxdatascorsort = np.argsort(listscor[c])
                    
                    # make plots
                    ## animation of regions
                    plot_anim(gdat, cntpmemo, 'cntpmemo')
                
                    ## animation of targets
                    numbsideedgeplot = 100
                    numbposiplot = 2
                    for n in range(numbposiplot):
                        
                        indxsideyposoffs = gdat.indxsideyposdataflat[gdat.indxdatascorsort[n]] - numbsideedgeplot / 2 + gdat.numbsideedge
                        indxsidexposoffs = gdat.indxsidexposdataflat[gdat.indxdatascorsort[n]] - numbsideedgeplot / 2 + gdat.numbsideedge
                        
                        indxyposmemoinit = max(indxsideyposoffs, 0)
                        indxyposmemofinl = min(indxsideyposoffs + numbsideedgeplot, numbsidememo)
                        indxxposmemoinit = max(indxsidexposoffs, 0)
                        indxxposmemofinl = min(indxsidexposoffs + numbsideedgeplot, numbsidememo)
                        
                        indxypostarginit = max(indxsideyposoffs, 0) - indxsideyposoffs
                        indxypostargfinl = numbsideedgeplot + min(indxyposmemoinit + numbsideedgeplot, numbsidememo) - indxyposmemoinit - numbsideedgeplot
                        indxxpostarginit = max(indxsidexposoffs, 0) - indxsidexposoffs
                        indxxpostargfinl = numbsideedgeplot + min(indxxposmemoinit + numbsideedgeplot, numbsidememo) - indxxposmemoinit - numbsideedgeplot
                        
                        cntptarg = np.zeros((numbsideedgeplot, numbsideedgeplot, gdat.numbtime))
                        
                        cntptarg[indxypostarginit:indxypostargfinl, indxxpostarginit:indxxpostargfinl, :] = \
                                                    cntpmemo[indxyposmemoinit:indxyposmemofinl, indxxposmemoinit:indxxposmemofinl, :]
                        plot_anim(gdat, cntptarg, 'cntpmemo_posi%04d' % n, indxsideyposoffs=indxsideyposoffs, indxsidexposoffs=indxsidexposoffs)
                
                    ## labeled marginal distributions
                    figr, axis = plt.subplots(numbtimeplotscat - 1, numbtimeplotscat - 1, figsize=(10, 10))
                    for t in gdat.indxtime[:numbtimeplotscat-1]:
                        for tt in gdat.indxtime[:numbtimeplotscat-1]:
                            if t < tt:
                                axis[t][tt].axis('off')
                                continue
                            axis[t][tt].scatter(X[indxdatanega, t+1], X[indxdatanega, tt], s=20, color='r', alpha=0.3)#*listscor[c])
                            axis[t][tt].scatter(X[indxdataposi, t+1], X[indxdataposi, tt], s=20, color='b', alpha=0.3)#*listscor[c])
                            axis[t][tt].set_ylim(limt)
                            axis[t][tt].set_xlim(limt)
                            #axis[t][tt].set_xticks(())
                            #axis[t][tt].set_yticks(())
                    path = gdat.pathdata + 'pmar_%s_%04d.pdf'% (gdat.strgcntp, c)
                    plt.savefig(path)
                    plt.close()
                    
                    ## median image with the labels
                    plot_imag(gdat, cntpmemotimemedi, 'cntpmemotimemedi', strgtitl='Median')

                    # plot data with colors based on predicted class
                    figr, axis = plt.subplots(10, 4)
                    for a in range(10):
                        for b in range(4):
                            p = a * 4 + b
                            if p >= numbdata:
                                continue
                            if False and gdat.datatype == 'mock':
                                if listlablmodl[c][p] == 1 and listlabltrue[p] == 1:
                                    colr = 'g'
                                elif listlablmodl[c][p] == 0 and listlabltrue[p] == 0:
                                    colr = 'r'
                                elif listlablmodl[c][p] == 0 and listlabltrue[p] == 1:
                                    colr = 'b'
                                elif listlablmodl[c][p] == 1 and listlabltrue[p] == 0:
                                    colr = 'orange'
                            else:
                                if listlablmodl[c][p] == 1:
                                    colr = 'b'
                                else:
                                    colr = 'r'
                            axis[a][b].plot(gdat.time, X[p, :].reshape((gdat.numbtime, numbsidecorr, numbsidecorr))[:, gdat.offscorr, gdat.offscorr], \
                                                                                                color=colr, alpha=0.1, ls='', marker='o', markersize=3)
                            if a != 9:
                                axis[a][b].set_xticks([])
                            if b != 0:
                                axis[a][b].set_yticks([])
                    path = gdat.pathdata + 'lcurpred_%s_%04d.pdf' % (gdat.strgcntp, c)
                    plt.savefig(path)
                    plt.close()

                    # plot a histogram of decision functions evaluated at the samples
                    figr, axis = plt.subplots()
                    axis.hist(listscor[c])
                    axis.set_xlabel('Score')
                    axis.set_yscale('log')
                    path = gdat.pathdata + 'histscor_%s_%04d.pdf' % (gdat.strgcntp, c)
                    plt.savefig(path)
                    plt.close()
                    
                    # plot data with the least and highest scores
                    figr, axis = plt.subplots(20, 2, figsize=(12, 24))

                    for l in range(2):
                        for k in range(20):
                            if l == 0:
                                indx = gdat.indxdatascorsort[k]
                            else:
                                indx = gdat.indxdatascorsort[numbdata-k-1]
                            
                            #strg = 'y, x = %d %d' % (indxsideyposdata[indx], indxsidexposdata[indx])
                            #axis[k][l].text(.9, .9, strg, transform=plt.gca().transAxes, size=15)
                            #horizontalalignment='right')
                            
                            if not isinstance(indx, int):
                                indx = indx[0]
                            axis[k][l].plot(gdat.time, X[indx, :].reshape((gdat.numbtime, numbsidecorr, numbsidecorr))[:, gdat.offscorr, gdat.offscorr], \
                                                                                                    color='black', ls='', marker='o', markersize=3)
                    path = gdat.pathdata + 'lcursort_%s_%04d.pdf' % (gdat.strgcntp, c)
                    plt.savefig(path)
                    plt.close()
        
                    numbposi = indxsidexposaccp.size
                    numbbins = 10
                    numbpositrue = np.zeros(numbbins)
                    binsmagt = np.linspace(magtminm, magtmaxm, numbbins + 1)
                    meanmagt = (binsmagt[1:] + binsmagt[:-1]) / 2.
                    reca = np.empty(numbbins)
                    numbsupnmagt = np.zeros(numbbins)
                    if gdat.datatype == 'mock':
                        for n in indxsupn:
                            indxmagt = np.digitize(np.amax(truemagt[:, n]), binsmagt) - 1
                            numbsupnmagt[indxmagt] += 1
                            indxpixlposi = np.where((abs(indxsidexpossour[n] - indxsidexposaccp) < 2) & (abs(indxsideypossour[n] - indxsideyposaccp) < 2))[0]
                            if indxpixlposi.size > 0:
                                numbpositrue[indxmagt] += 1
                        recamagt = numbpositrue.astype(float) / numbsupnmagt
                        prec = sum(numbpositrue).astype(float) / numbposi
                        figr, axis = plt.subplots(figsize=(12, 6))
                        axis.plot(meanmagt, recamagt, ls='', marker='o')
                        axis.set_ylabel('Recall')
                        axis.set_xlabel('Tmag')
                        plt.tight_layout()
                        path = gdat.pathdata + 'reca_%s_%04d.pdf' % (gdat.strgcntp, c)
                        print 'Writing to %s...' % path
                        plt.savefig(path)
                        plt.close()
        
                    c += 1
                
                
                #continue

                ## clustering with pyod
                ## fraction of outliers
                #fracoutl = 0.25
                #
                ## initialize a set of detectors for LSCP
                #detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
                #                 LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
                #                 LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
                #                 LOF(n_neighbors=50)]
                #
                ## Show the statics of the data
                ## Define nine outlier detection tools to be compared
                #classifiers = {
                #    'Angle-based Outlier Detector (ABOD)':
                #        ABOD(contamination=fracoutl),
                #    'Cluster-based Local Outlier Factor (CBLOF)':
                #        CBLOF(contamination=fracoutl,
                #              check_estimator=False, random_state=random_state),
                #    'Feature Bagging':
                #        FeatureBagging(LOF(n_neighbors=35),
                #                       contamination=fracoutl,
                #                       random_state=random_state),
                #    #'Histogram-base Outlier Detection (HBOS)': HBOS(
                #    #    contamination=fracoutl),
                #    'Isolation Forest': IForest(contamination=fracoutl,
                #                                random_state=random_state),
                #    'K Nearest Neighbors (KNN)': KNN(
                #        contamination=fracoutl),
                #    'Average KNN': KNN(method='mean',
                #                       contamination=fracoutl),
                #    # 'Median KNN': KNN(method='median',
                #    #                   contamination=fracoutl),
                #    'Local Outlier Factor (LOF)':
                #        LOF(n_neighbors=35, contamination=fracoutl),
                #    # 'Local Correlation Integral (LOCI)':
                #    #     LOCI(contamination=fracoutl),
                #    
                #    #'Minimum Covariance Determinant (MCD)': MCD(
                #    #    contamination=fracoutl, random_state=random_state),
                #    
                #    'One-class SVM (OCSVM)': OCSVM(contamination=fracoutl),
                #    'Principal Component Analysis (PCA)': PCA(
                #        contamination=fracoutl, random_state=random_state, standardization=False),
                #    # 'Stochastic Outlier Selection (SOS)': SOS(
                #    #     contamination=fracoutl),
                #    'Locally Selective Combination (LSCP)': LSCP(
                #        detector_list, contamination=fracoutl,
                #        random_state=random_state)
                #}
                #
                #return
                #raise Exception('')

                ## Fit the model
                #plt.figure(figsize=(15, 12))
                #for i, (clf_name, clf) in enumerate(classifiers.items()):
                #    print(i, 'fitting', clf_name)

                #    # fit the data and tag outliers
                #    clf.fit(X)
                #    scores_pred = clf.decision_function(X) * -1
                #    y_pred = clf.predict(X)
                #    threshold = np.percentile(scores_pred, 100 * fracoutl)
                #    n_errors = np.where(y_pred != listlabltrue)[0].size
                #    # plot the levels lines and the points
                #    #if i == 1:
                #    #    continue
                #    #Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
                #    #Z = Z.reshape(xx.shape)
                #    Z = np.zeros((100, 100))
                #    subplot = plt.subplot(3, 4, i + 1)
                #    subplot.contourf(xx, yy, Z, #levels=np.linspace(Z.min(), threshold, 7),
                #                     cmap=plt.cm.Blues_r)
                #    subplot.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
                #    a = subplot.contour(xx, yy, Z, levels=[threshold],
                #                        linewidths=2, colors='red')
                #    subplot.contourf(xx, yy, Z, #levels=[threshold, Z.max()],
                #                     colors='orange')
                #    b = subplot.scatter(X[:-numboutl, 0], X[:-numboutl, 1], c='green', s=20, edgecolor='k')
                #    c = subplot.scatter(X[-numboutl:, 0], X[-numboutl:, 1], c='purple', s=20, edgecolor='k')
                #    subplot.axis('tight')
                #    subplot.legend(
                #        [a.collections[0], b, c],
                #        ['learned decision function', 'true inliers', 'true outliers'],
                #        prop=matplotlib.font_manager.FontProperties(size=10),
                #        loc='lower right')
                #    subplot.set_xlabel("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))
                #    subplot.set_xlim(limtproj)
                #    subplot.set_ylim(limtproj)
                #plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
                #plt.suptitle("Outlier detection")
                #plt.savefig(pathplot + 'pyod.png', dpi=300)
                #plt.close()

                #
                #default_base = {'quantile': .3,
                #                'eps': .3,
                #                'damping': .9,
                #                'preference': -200,
                #                'n_neighbors': 10,
                #                'n_clusters': 3,
                #                'min_samples': 20,
                #                'xi': 0.05,
                #                'min_cluster_size': 0.1}
                #
                ## update parameters with dataset-specific values
                #
                #algo_params = {'damping': .77, 'preference': -240,
                #     'quantile': .2, 'n_clusters': 2,
                #     'min_samples': 20, 'xi': 0.25}

                #params = default_base.copy()
                #params.update(algo_params)
                #
                ## normalize dataset for easier parameter selection
                #X = StandardScaler().fit_transform(X)
                #
                ## estimate bandwidth for mean shift
                #bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])
                #
                ## connectivity matrix for structured Ward
                #connectivity = kneighbors_graph(
                #    X, n_neighbors=params['n_neighbors'], include_self=False)
                ## make connectivity symmetric
                #connectivity = 0.5 * (connectivity + connectivity.T)
                #
                #ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
                #two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
                #ward = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='ward', connectivity=connectivity)
                #spectral = cluster.SpectralClustering(n_clusters=params['n_clusters'], eigen_solver='arpack', affinity="nearest_neighbors")
                #dbscan = cluster.DBSCAN(eps=params['eps'])
                #
                ##optics = cluster.OPTICS(min_samples=params['min_samples'],
                ##                        xi=params['xi'],
                ##                        min_cluster_size=params['min_cluster_size'])
                #
                #affinity_propagation = cluster.AffinityPropagation(damping=params['damping'], preference=params['preference'])
                #average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock", \
                #                                                                    n_clusters=params['n_clusters'], connectivity=connectivity)
                #birch = cluster.Birch(n_clusters=params['n_clusters'])
                #gmm = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')
                #
                #clustering_algorithms = (
                #    ('MiniBatchKMeans', two_means),
                #    ('AffinityPropagation', affinity_propagation),
                #    ('MeanShift', ms),
                #    ('SpectralClustering', spectral),
                #    ('Ward', ward),
                #    ('AgglomerativeClustering', average_linkage),
                #    ('DBSCAN', dbscan),
                #    #('OPTICS', optics),
                #    ('Birch', birch),
                #    ('GaussianMixture', gmm)
                #)
                #
                #figr, axis = plt.subplots(1, numbmeth)
                #k = 0
                #for name, algorithm in clustering_algorithms:
                #    t0 = timemodu.time()
                #    
                #    print 'name'
                #    print name
                #    print

                #    # catch warnings related to kneighbors_graph
                #    with warnings.catch_warnings():
                #        #warnings.filterwarnings(
                #        #    "ignore",
                #        #    message="the number of connected components of the " +
                #        #    "connectivity matrix is [0-9]{1,2}" +
                #        #    " > 1. Completing it to avoid stopping the tree early.",
                #        #    category=UserWarning)
                #        #warnings.filterwarnings(
                #        #    "ignore",
                #        #    message="Graph is not fully connected, spectral embedding" +
                #        #    " may not work as expected.",
                #        #    category=UserWarning)
                #        algorithm.fit(X)
                #
                #    t1 = timemodu.time()
                #    if hasattr(algorithm, 'labels_'):
                #        print 'Has labels_'
                #        lablmodl = algorithm.labels_.astype(np.int)
                #    else:
                #        lablmodl = algorithm.predict(X)
                #    
                #    print 'lablmodl'
                #    summgene(lablmodl)
                #    print ''

                #    axis[k].set_title(name, size=18)
                #
                #    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                #                                         '#f781bf', '#a65628', '#984ea3',
                #                                         '#999999', '#e41a1c', '#dede00']),
                #                                  int(max(lablmodl) + 1))))
                #    # add black color for outliers (if any)
                #    colors = np.append(colors, ["#000000"])
                #    axis[k].scatter(X[:, 0], X[:, 1], s=10, color=colors[lablmodl])
                #
                #    axis[k].set_xlim(-2.5, 2.5)
                #    axis[k].set_ylim(-2.5, 2.5)
                #    axis[k].set_xticks(())
                #    axis[k].set_yticks(())
                #    axis[k].text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                #             transform=plt.gca().transAxes, size=15,
                #             horizontalalignment='right')
                #    k += 1
                #    listlablmodl.append(lablmodl)
                #path = gdat.pathdata + 'clus.pdf'
                #plt.savefig(path)
                #plt.close()


                ## Random 2D projection using a random unitary matrix
                #print("Computing random projection")
                #rp = random_projection.SparseRandomProjection(n_components=2)
                #X_projected = rp.fit_transform(lcurflat)
                #print 'X_projected'
                #summgene(X_projected)
                #plot_embe(gdat, lcurflat, X_projected, 'rand', "Random Projection")
                #
                ## Projection on to the first 2 principal components
                #print("Computing PCA projection")
                #t0 = timemodl.time()
                #print 'lcurflat'
                #summgene(lcurflat)
                #X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(lcurflat)
                #plot_embe(gdat, lcurflat, X_pca, 'pcaa', "Principal Components projection (time %.2fs)" % (timemodl.time() - t0))
                #
                ## Projection on to the first 2 linear discriminant components
                ##print("Computing Linear Discriminant Analysis projection")
                ##X2 = lcurflat.copy()
                ##X2.flat[::lcurflat.shape[1] + 1] += 0.01  # Make X invertible
                ##t0 = timemodl.time()
                ##X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X2, y)
                ##plot_embe(gdat, lcurflat, X_lda, 'ldap', "Linear Discriminant projection (time %.2fs)" % (timemodl.time() - t0))
                #
                ## t-SNE embedding dataset
                #print("Computing t-SNE embedding")
                #tsne = manifold.TSNE(n_components=2, random_state=0, perplexity=30)
                #t0 = timemodl.time()
                #X_tsne = tsne.fit_transform(lcurflat)
                #plot_embe(gdat, lcurflat, X_tsne, 'tsne0030', "t-SNE embedding with perplexity 30")
                #
                ## t-SNE embedding dataset
                #print("Computing t-SNE embedding")
                #tsne = manifold.TSNE(n_components=2, random_state=0, perplexity=5)
                #t0 = timemodl.time()
                #X_tsne = tsne.fit_transform(lcurflat)
                #plot_embe(gdat, lcurflat, X_tsne, 'tsne0005', "t-SNE embedding with perplexity 5")
                #
                ## Isomap projection dataset
                #print("Computing Isomap projection")
                #t0 = timemodl.time()
                #X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(lcurflat)
                #print("Done.")
                #plot_embe(gdat, lcurflat, X_iso, 'isop', "Isomap projection (time %.2fs)" % (timemodl.time() - t0))
                #
                ## Locally linear embedding dataset
                #print("Computing LLE embedding")
                #clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='standard')
                #t0 = timemodl.time()
                #X_lle = clf.fit_transform(lcurflat)
                #print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
                #plot_embe(gdat, lcurflat, X_lle, 'llep', "Locally Linear Embedding (time %.2fs)" % (timemodl.time() - t0))
                #
                ## Modified Locally linear embedding dataset
                #print("Computing modified LLE embedding")
                #clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='modified')
                #t0 = timemodl.time()
                #X_mlle = clf.fit_transform(lcurflat)
                #print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
                #plot_embe(gdat, lcurflat, X_mlle, 'mlle', "Modified Locally Linear Embedding (time %.2fs)" % (timemodl.time() - t0))
                #
                ## HLLE embedding dataset
                #print("Computing Hessian LLE embedding")
                #clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='hessian')
                #t0 = timemodl.time()
                #X_hlle = clf.fit_transform(lcurflat)
                #print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
                #plot_embe(gdat, lcurflat, X_hlle, 'hlle', "Hessian Locally Linear Embedding (time %.2fs)" % (timemodl.time() - t0))
                #
                ## LTSA embedding dataset
                #print("Computing LTSA embedding")
                #clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='ltsa')
                #t0 = timemodl.time()
                #X_ltsa = clf.fit_transform(lcurflat)
                #print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
                #plot_embe(gdat, lcurflat, X_ltsa, 'ltsa', "Local Tangent Space Alignment (time %.2fs)" % (timemodl.time() - t0))
                #
                ## MDS  embedding dataset
                #print("Computing MDS embedding")
                #clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
                #t0 = timemodl.time()
                #X_mds = clf.fit_transform(lcurflat)
                #print("Done. Stress: %f" % clf.stress_)
                #plot_embe(gdat, lcurflat, X_mds, 'mdse', "MDS embedding (time %.2fs)" % (timemodl.time() - t0))
                #
                ## Random Trees embedding dataset
                #print("Computing Totally Random Trees embedding")
                #hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0, max_depth=5)
                #t0 = timemodl.time()
                #X_transformed = hasher.fit_transform(lcurflat)
                #pca = decomposition.TruncatedSVD(n_components=2)
                #X_reduced = pca.fit_transform(X_transformed)
                #plot_embe(gdat, lcurflat, X_reduced, 'rfep', "Random forest embedding (time %.2fs)" % (timemodl.time() - t0))
                #
                ## Spectral embedding dataset
                #print("Computing Spectral embedding")
                #embedder = manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack")
                #t0 = timemodl.time()
                #X_se = embedder.fit_transform(lcurflat)
                #plot_embe(gdat, lcurflat, X_se, 'csep', "Spectral embedding (time %.2fs)" % (timemodl.time() - t0))
                #
                ## NCA projection dataset
                ##print("Computing NCA projection")
                ##nca = neighbors.NeighborhoodComponentsAnalysis(n_components=2, random_state=0)
                ##t0 = timemodl.time()
                ##X_nca = nca.fit_transform(lcurflat, y)
                ##plot_embe(gdat, lcurflat, X_nca, 'ncap', "NCA embedding (time %.2fs)" % (timemodl.time() - t0))

                #indxsidexposaccp = np.array(indxsidexposaccp)
                #indxsideyposaccp = np.array(indxsideyposaccp)
                #figr, axis = plt.subplots(figsize=(12, 6))
                #objtimag = axis.imshow(np.std(cntpmemo, axis=2), interpolation='nearest', cmap='Reds')
                #
                #if gdat.datatype == 'mock':
                #    for n in indxsupn:
                #        axis.scatter(indxsidexpossour[n], indxsideypossour[n], s=50, marker='o', color='g')
                #
                #for indxsideyposaccptemp, indxsidexposaccptemp in zip(indxsideyposaccp, indxsidexposaccp):
                #    axis.scatter(indxsidexposaccptemp, indxsideyposaccptemp, s=50, marker='x', color='b')

                #for strgvarb in ['diff']:
                #    figr, axis = plt.subplots(figsize=(12, 6))
                #    #if strgvarb == 'diff':
                #    #    varbtemp = np.arcsinh(dictpara[strgvarb])
                #    #else:
                #    #    varbtemp = dictpara[strgvarb]
                #    varbtemp = dictpara[strgvarb]
                #    vmin = -1
                #    vmax = 1
                #    objtimag = axis.imshow(varbtemp, interpolation='nearest', cmap='Greens', vmin=vmin, vmax=vmax)
                #    for indxsideyposaccptemp, indxsidexposaccptemp in zip(indxsideyposaccp, indxsidexposaccp):
                #        axis.scatter(indxsideyposaccptemp, indxsidexposaccptemp, s=5, marker='x', color='b', lw=0.5)
                #    plt.colorbar(objtimag)
                #    plt.tight_layout()
                #    path = gdat.pathdata + '%s_%s.pdf' % (strgvarb, gdat.strgcntp)
                #    print 'Writing to %s...' % path
                #    plt.savefig(path)
                #    plt.close()
    

def retr_timeexec():
    # input PCAT speed per 100x100 pixel region
    timeregi = 30. # [min]
    
    # number of time frames in each region
    numbtser = 13.7 * 4 * 24 * 60. / 30.
    
    timeregitser = numbtser * timeregi / 60. / 24 # [day]
    timeffim = 16.8e6 / 1e4 * timeregi # [day]
    timesegm = 4. * timeffim / 7. # [week]
    timefsky = 26 * timesegm / 7. # [week]
    
    print 'Full frame, full sky: %d weeks per 1000 cores' % (timefsky / 1000.) 


def plot_peri(): 
    ## plot Lomb Scargle periodogram
    figr, axis = plt.subplots(figsize=(12, 4))
    axis.set_ylabel('Power')
    axis.set_xlabel('Frequency [1/day]')
    arryfreq = np.linspace(0.1, 10., 2000)
    for a in range(2):
        indxtemp = np.arange(arryseco.shape[0])
        if a == 0:
            colr = 'g'
        if a == 1:
            colr = 'r'
            for k in range(1400, 1500):
                indxtemp = np.setdiff1d(indxtemp, np.where(abs(arryseco[:, 0] - k * peri - epoc) < dura * 2)[0])
        ydat = scipy.signal.lombscargle(arryseco[indxtemp, 0], arryseco[indxtemp, 1], arryfreq)
        axis.plot(arryfreq * 2. * np.pi, ydat, ls='', marker='o', markersize=5, alpha=0.3, color=colr)
    for a in range(4):
        axis.axvline(a / peri, ls='--', color='black')
    plt.tight_layout()
    path = pathimag + 'lspd_%s.pdf' % (strgmask)
    print 'Writing to %s...' % path
    plt.savefig(path)
    plt.close()

def plot_catl(gdat, axis, indxsideyposoffs=0, indxsidexposoffs=0):

    try:
        for k in range(gdat.numbpositext):
            axis.text(gdat.indxsideyposdataflat[gdat.indxdatascorsort[k]] - indxsideyposoffs + gdat.numbsideedge, \
                      gdat.indxsidexposdataflat[gdat.indxdatascorsort[k]] - indxsidexposoffs + gdat.numbsideedge, '%d' % k, size=7, color='b', alpha=0.3)
    except:
        pass

    if gdat.datatype == 'mock':

        for k in gdat.indxsour:
            axis.text(np.mean(gdat.trueypos[:, k]) - indxsideyposoffs, \
                      np.mean(gdat.truexpos[:, k]) - indxsidexposoffs, '*', alpha=0.1, size=15, color='y', ha='center', va='center')
            #axis.text(np.mean(gdat.trueypos[:, k]) - indxsideyposoffs + 0.5, \
            #          np.mean(gdat.truexpos[:, k]) - indxsidexposoffs + 0.5, '%.3g, %.3g' % (gdat.truemagtmean[k], gdat.truemagtstdv[k]), \
            #                                                        alpha=0.3, size=5, color='y', ha='center', va='center')

        for k in gdat.indxsoursupn:
            axis.text(np.mean(gdat.trueypos[:, k]) - indxsideyposoffs, \
                      np.mean(gdat.truexpos[:, k]) - indxsidexposoffs, '*', alpha=0.1, size=15, color='g', ha='center', va='center')
            axis.text(np.mean(gdat.trueypos[:, k]) - indxsideyposoffs + 0.5, \
                      np.mean(gdat.truexpos[:, k]) - indxsidexposoffs + 0.5, '%.3g, %.3g' % (gdat.truemagtmean[k], gdat.truemagtstdv[k]), \
                                                                                                alpha=0.1, size=5, color='g', ha='center', va='center')

def cnfg_tici():

    # 272551828
    isec = 3
    icam = 4
    iccd = 3
    rasctarg = 121.865609
    decltarg = -76.533524
    init(isec, icam, iccd, rasctarg=rasctarg, decltarg=decltarg)


def cnfg_mock():
   
    init(9, 1, 1, datatype='mock')


def cnfg_tdie():
   
    pathdata = '/Users/tdaylan/Documents/work/data/tesstran/tdie/tesscut/'
    listpath = fnmatch.filter(os.listdir(pathdata), 'tess*')
    for p in range(len(listpath)):
        isec = int(listpath[p][6:10])
        icam = int(listpath[p][11])
        iccd = int(listpath[p][13])
        pathfile = pathdata + listpath[p]
        if isec == 7 or isec == 8:
            init(isec, icam, iccd, pathfile=pathfile)


def cnfg_defa():
    
    jobs = []
    isec = 1
    for i in range(1, 5):
        for j in range(1, 5):
            if i == 4 and  j == 1:
                init(isec, i, j)
                
            #p = multiprocessing.Process(target=work, args=(isec, i, j))
            #jobs.append(p)
            #p.start()


def cnfg_supn():
    
    print 'Type Ia SN analysis started.'

    path = os.environ['TCAT_DATA_PATH'] + '/tns_search.csv'
    numbside = 30
    objtfile = open(path, 'r')
    
    # get the list of SN
    objtfilelist = open(os.environ['TCAT_DATA_PATH'] + '/supnlist.dat', 'r')
    listsupn = []
    for line in objtfilelist:
        linesplt = line.split('&')
        listsupn.append(linesplt[1][4:-2])

    for k, line in enumerate(objtfile):
        linesplt = line.split(',')
        labltarg = linesplt[1][1:-1]
        strgtarg = labltarg[3:]
        rasctarg = linesplt[2][1:-1]
        decltarg = linesplt[3][1:-1]
        
        if not strgtarg in listsupn:
            continue
        
        objt = SkyCoord(rasctarg, decltarg, frame='icrs', unit=(u.hourangle, u.deg))  # passing in string format
        rasctarg = objt.ra.degree
        decltarg = objt.dec.degree
        init( \
             rasctarg=rasctarg, \
             decltarg=decltarg, \
             labltarg=labltarg, \
             strgtarg=strgtarg, \
             numbside=numbside, \
            )
        

def cnfg_sect():
    
    for isec in range(9, 10):
        for icam in range(4, 5):
            for iccd in range(2, 3):
                init(isec, icam, iccd)

globals().get(sys.argv[1])()

