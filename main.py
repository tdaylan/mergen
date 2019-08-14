from lion import main as lionmain
import tdpy
from tdpy.util import summgene

import numpy as np

import h5py

import astroquery.mast
from astroquery.mast import Tesscut
from astropy.coordinates import SkyCoord
import tcat

import time as timemodu

#import pickle

#from sklearn.manifold import TSNE
#from sklearn import manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection, neighbors
#from sklearn import svm
#from sklearn.covariance import EllipticEnvelope
#from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
            
#from sklearn import cluster, datasets, mixture
#from sklearn.neighbors import kneighbors_graph
#from sklearn.preprocessing import StandardScaler

#from skimage import data
#from skimage.morphology import disk
#from skimage.filters.rank import median

#import pyod
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.loci import LOCI
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.sos import SOS
from pyod.models.lscp import LSCP

import scipy.signal
from scipy import interpolate

import os, sys, datetime, fnmatch

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#from astroquery.mast import Catalogs
import astroquery
#
#import astropy
#from astropy.wcs import WCS
#from astropy import units as u
#from astropy.io import fits
#import astropy.time
#from astropy.coordinates import SkyCoord

import multiprocessing

#import eleanor



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
    path = gdat.pathimag + '%s_%s.pdf' % (strg, gdat.strgcntp)
    print('Writing to %s...' % path)
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
        tick = np.sinh(tick)
        labl = ['%d' % int(tick[k]) for k in range(len(tick))]
        cbar.set_ticklabels(labl)

    if path is None:
        path = gdat.pathimag + '%s_%s.pdf' % (strgvarb, gdat.strgcntp)
    print('Writing to %s...' % path)
    #plt.tight_layout()
    plt.savefig(path)
    plt.close()
    

def init( \
         listisec=None, \
         listicam=None, \
         listiccd=None, \
         extrtype='qlop', \
         targtype='slen', \
         pathfile=None, \
         datatype='obsd', \
         rasctarg=None, \
         decltarg=None, \
         labltarg=None, \
         strgtarg=None, \
         numbside=None, \
         **args \
        ):
 
    # inputs:
    # 1) TIC IDs
    # 2) One sector, One Cam, One CCD

    # preliminary setup
    # construct the global object 
    gdat = tdpy.util.gdatstrt()
    #for attr, valu in locals().iteritems():
    #    if '__' not in attr and attr != 'gdat':
    #        setattr(gdat, attr, valu)
    
    # copy all provided inputs to the global object
    #for strg, valu in args.iteritems():
    #    setattr(gdat, strg, valu)
    
    gdat.datatype = datatype

    # string for date and time
    gdat.strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print('CTLC initialized at %s...' % gdat.strgtimestmp)
    
    #if ((listicam is not None or listiccd is not None) and listtici is not None):
    #    raise Exception('')
    
    if gdat.datatype == 'mock':
        gdat.trueclastype = 'catg'

    #star = eleanor.Source(tic=38846515, sector=1, tc=True)

    gdat.strgcntp = gdat.datatype

    # paths
    ## read PCAT path environment variable
    gdat.pathbase = os.environ['CTLC_DATA_PATH'] + '/'
    gdat.pathdata = gdat.pathbase + 'data/'
    gdat.pathimag = gdat.pathbase + 'imag/'
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
        if numbside is None:
            strgmode = 'full'
        else:
            strgmode = 'targ'
    else:
        strgmode = 'file'
        
    random_state = 42

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
        strgsecc = '%02d%d%d' % (listisec[o], listicam[o], listiccd[o])
        print('Sector: %d' % listisec[o])
        print('Camera: %d' % listicam[o])
        print('CCD: %d' % listiccd[o])
        
        isec = listisec[o]

        verbtype = 1
        
        np.random.seed(45)

        gdat.numbfeat = 1

        gdat.numbtimerebn = None#30
        
        # settings
        ## plotting
        ### number of pixels along a postage stamp plot
        gdat.numbsideplot = 11
        gdat.cntpscaltype = 'asnh'
        if pathfile is not None and gdat.datatype == 'mock':
            raise Exception('')
        
        # grid of flux space
        minmproj = 0.1
        maxmproj = 2
        limtproj = [minmproj, maxmproj]
        arry = np.linspace(minmproj, maxmproj, 100)
        xx, yy = np.meshgrid(arry, arry)
        
        magtminm = 12.
        magtmaxm = 19.
        
        # get data
        if gdat.datatype == 'obsd':
            print('Reading files...')
            path = gdat.pathdata + 'qlop/'
            liststrgfile = fnmatch.filter(os.listdir(path), '*.h5')
            liststrgfile = liststrgfile[:100]
            
            liststrgtici = []
            for strgfile in liststrgfile:
                liststrgtici.append(strgfile[:-3])

            gdat.numbdata = len(liststrgfile)
            fracdatanann = np.empty(gdat.numbdata)
            listindxtimebadd = []
            for k, strgfile in enumerate(liststrgfile):
                with h5py.File(path + strgfile, 'r') as objtfile:
                    if k == 0:
                        gdat.time = objtfile['LightCurve/BJD'][()]
                        gdat.numbtime = gdat.time.size
                        lcur = np.empty((gdat.numbdata, gdat.numbtime, gdat.numbfeat))
                    tmag = objtfile['LightCurve/AperturePhotometry/Aperture_002/RawMagnitude'][()]
                    if k == 0:
                        gdat.indxtime = np.arange(gdat.numbtime)
                    indxtimegood = np.where(np.isfinite(tmag))[0]
                    indxtimenann = np.setdiff1d(gdat.indxtime, indxtimegood)
                    lcur[k, :, 0] = 10**(-(tmag - np.median(tmag[indxtimegood])) / 2.5)
                    listindxtimebadd.append(indxtimenann)
                    fracdatanann[k] = indxtimenann.size / float(gdat.numbtime)

            listindxtimebadd = np.concatenate(listindxtimebadd)
            listindxtimebadd = np.unique(listindxtimebadd)
            listindxtimebadd = np.concatenate((listindxtimebadd, np.arange(100)))
            listindxtimebadd = np.concatenate((listindxtimebadd, gdat.numbtime / 2 + np.arange(100)))
            listindxtimegood = np.setdiff1d(gdat.indxtime, listindxtimebadd)
            #listhdundata = fits.open(pathfile)
            print('Filtering the data...')
            # filter the data
            gdat.time = gdat.time[listindxtimegood]
            lcur = lcur[:, listindxtimegood, :]
        
        # inject signal
        if gdat.datatype == 'mock':
            gdat.numbtime = 50
            gdat.numbfeat = 1
            gdat.numbdata = 100
            gdat.truenumbclas = 2
            gdat.trueindxclas = np.arange(gdat.truenumbclas)
            gdat.numbobjtclas = 100
            gdat.numbdata = gdat.numbobjtclas * gdat.truenumbclas
            gdat.time = np.arange(gdat.numbtime).astype(float)
            lcur = np.random.randn(gdat.numbdata * gdat.numbtime * gdat.numbfeat).reshape((gdat.numbdata, gdat.numbtime, gdat.numbfeat))
            gdat.numbtime = gdat.time.size
            gdat.indxobjtclas = np.arange(gdat.numbobjtclas)
            truemagt = np.empty((gdat.numbdata, gdat.numbtime))
            truecntp = np.empty((gdat.numbdata, gdat.numbtime))
            for l in gdat.trueindxclas:
                for n in gdat.indxobjtclas:
                    a = n + l * gdat.numbobjtclas
                    if l == 0:
                        truemagt[a, :] = np.random.rand() * 5 + 15.
                    if l == 1:
                        timenorm = -0.5 + (gdat.time / np.amax(gdat.time)) + 2. * (np.random.random() - 0.5)
                        objtrand = scipy.stats.skewnorm.pdf(timenorm, 4)
                        objtrand /= np.amax(objtrand)
                        truemagt[a, :] = 8. + 6. * (2. - objtrand)
            truecntp = 10**((20.424 - truemagt) / 2.5)
            truecntp *= 1. + 1e-3 * np.random.randn(gdat.numbdata * gdat.numbtime).reshape((gdat.numbdata, gdat.numbtime))
            truecntp /= np.mean(truecntp, 1)[:, None]
            lcur[:, :, 0] = truecntp
            gdat.truemagtmean = np.mean(truemagt, 0)
            gdat.truemagtstdv = np.std(truemagt, 0)
            gdat.indxtime = np.arange(gdat.numbtime)
            if gdat.trueclastype == 'outl':
                gdat.trueindxdataoutl = gdat.numbobjtoutl

        gdat.numbtime = gdat.time.size
        
        if (~np.isfinite(lcur)).any():
            raise Exception('')

        if gdat.datatype == 'obsd':
            # plot the fraction of data that is NaN
            figr, axis = plt.subplots(figsize=(6, 4))
            axis.hist(fracdatanann)
            axis.set_xlabel('$f_{nan}$')
            axis.set_ylabel('$N(f_{nan})$')
            path = gdat.pathimag + 'histfracdatanann_%s.pdf' % (gdat.strgcntp)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()

        gdat.numbdata = lcur.shape[0]
        if gdat.datatype == 'mock':
            listlabltrue = np.zeros(gdat.numbdata, dtype=int)
            if gdat.trueclastype == 'outl':
                numbinli = gdat.numbdata - gdat.numbsour
                numboutl = gdat.numbsour
        
        # plot the data
        figr, axis = plt.subplots(10, 4)
        indxdata = np.arange(gdat.numbdata)
        gdat.numbdataplot = min(40, gdat.numbdata)
        indxdataplot = np.random.choice(indxdata, size=gdat.numbdataplot, replace=False)
        for a in range(10):
            for b in range(4):
                p = a * 4 + b
                if p >= gdat.numbdata:
                    continue
                axis[a][b].plot(gdat.time, lcur[indxdataplot[p], :, 0], color='black', ls='', marker='o', markersize=1)
                if a != 9:
                    axis[a][b].set_xticks([])
                if b != 0:
                    axis[a][b].set_yticks([])
        plt.subplots_adjust(hspace=0, wspace=0)
        path = gdat.pathimag + 'lcurrand_%s.pdf' % (gdat.strgcntp)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()

        # temporal median filter
        numbtimefilt = min(9, gdat.numbtime)
        if numbtimefilt % 2 == 0:
            numbtimefilt -= 1
        print('Performing the temporal median filter...')
        
        # rebin in time
        if gdat.numbtimerebn is not None and gdat.numbtime > gdat.numbtimerebn:
            print('Rebinning in time...')
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
            gdat.time = timeneww[gdat.indxtimegood]
            gdat.numbtime = gdat.indxtimegood.size
            gdat.indxtime = np.arange(gdat.numbtime)
        
        # calculate derived maps
        ## RMS image

        strgtype = 'tsne'

        lcuravgd = np.empty(gdat.numbtime)
        cntr = 0
        prevfrac = -1
        k = 0
        
        scorthrs = -2.

        # machine learning

        n_neighbors = 30
        
        X = lcur[:, :, 0]

        indxdata = np.arange(gdat.numbdata)
        
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
        print('Running anomaly-detection algorithms...')
        for name, objtalgoanom in listobjtalgoanom:
            t0 = timemodu.time()
            
            objtalgoanom.fit(X)
            t1 = timemodu.time()
        
            # fit the data and tag outliers
            if name == 'Local Outlier Factor':
                scor = objtalgoanom.negative_outlier_factor_
            else:
                scor = objtalgoanom.decision_function(X)
            if name == "Local Outlier Factor":
                lablmodl = np.zeros(gdat.numbdata)
                lablmodl[np.where(scor < scorthrs)[0]] = 1.
            else:
                lablmodl = objtalgoanom.fit(X).predict(X)
            
            indxdataposi = np.where(lablmodl == 1)[0]
            indxdatanega = np.setdiff1d(indxdata, indxdataposi)
            numbposi = indxdataposi.size
            gdat.numbpositext = min(200, numbposi)

            listscor.append(scor)
            listlablmodl.append(lablmodl)
            
            gdat.indxdatascorsort = np.argsort(listscor[c])

            # make plots
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
            path = gdat.pathimag + 'pmar_%s_%04d.pdf'% (gdat.strgcntp, c)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
            
            # plot data with colors based on predicted class
            figr, axis = plt.subplots(10, 4)
            for a in range(10):
                for b in range(4):
                    p = a * 4 + b
                    if p >= gdat.numbdata:
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
                    axis[a][b].plot(gdat.time, X[p, :], color=colr, alpha=0.1, ls='', marker='o', markersize=3)
                    if a != 9:
                        axis[a][b].set_xticks([])
                    if b != 0:
                        axis[a][b].set_yticks([])
            plt.subplots_adjust(hspace=0, wspace=0)
            path = gdat.pathimag + 'lcurpred_%s_%04d.pdf' % (gdat.strgcntp, c)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()

            # plot a histogram of decision functions evaluated at the samples
            figr, axis = plt.subplots()
            axis.hist(listscor[c], color='k')
            if gdat.datatype == 'mock':
                if gdat.trueclastype == 'outl':
                    axis.hist(listscor[c][gdat.trueindxdataoutl], color='g', label='True')
            axis.axvline(scorthrs)
            axis.set_xlabel('$S$')
            axis.set_ylabel('$N(S)$')
            axis.set_yscale('log')
            path = gdat.pathimag + 'histscor_%s_%04d.pdf' % (gdat.strgcntp, c)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
            
            # plot data with the least and highest scores
            gdat.numbdataplotsort = min(gdat.numbdata, 40)
            figr, axis = plt.subplots(gdat.numbdataplotsort, 2, figsize=(12, 2 * gdat.numbdataplotsort))
            for l in range(2):
                for k in range(gdat.numbdataplotsort):
                    
                    if l == 0:
                        indx = gdat.indxdatascorsort[k]
                    else:
                        indx = gdat.indxdatascorsort[gdat.numbdata-k-1]
                    if not isinstance(indx, int):
                        indx = indx[0]
                    
                    # plot the postage stamp animation
                    if gdat.datatype == 'obsd':
                        if l == 0:
                            catalogData = astroquery.mast.Catalogs.query_object(int(liststrgtici[indx]), catalog="TIC")
                            rasctarg = catalogData[0]['ra']
                            decltarg = catalogData[0]['dec']
                            cutout_coord = SkyCoord(rasctarg, decltarg, unit="deg")
                            listhdundata = Tesscut.get_cutouts(cutout_coord, gdat.numbsideplot)
                            #gdat.listtime = listhdundata[o][1].data['TIME']
                            cntpdata = 30. * 60. * listhdundata[o][1].data['FLUX'].swapaxes(0, 2).swapaxes(0, 1)[None, :, :, :]
                            cntpresi = cntpdata - np.median(cntpdata, 2)
                            #plot_cntpwrap(gdat, cntp, indxtimework, o, strgsecc, boolresi=False, strgtemp='data', indxpixlcolr=None, indxstarcolr=None)
                            print('hey')
                            for a in range(2):
                                if a == 0:
                                    cntptemp = cntpdata
                                    boolresi = False
                                else:
                                    cntptemp = cntpresi
                                    boolresi = True
                                plot_cntpwrap(gdat, cntptemp, gdat.indxtime, o, strgsecc, boolresi=boolresi)
                    
                    axis[k][l].plot(gdat.time, X[indx, :], color='black', ls='', marker='o', markersize=1)
                    if gdat.datatype == 'obsd':
                        axis[k][l].text(.9, .9, 'TIC: %s' % liststrgtici[indx], transform=axis[k][l].transAxes, size=15, ha='right', va='center')
                    if l == 1:
                        axis[k][l].yaxis.set_label_position('right')
                        axis[k][l].yaxis.tick_right()
            plt.subplots_adjust(hspace=0, wspace=0)
            path = gdat.pathimag + 'lcursort_%s_%04d.pdf' % (gdat.strgcntp, c)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
        
            numbbins = 10
            numbpositrue = np.zeros(numbbins)
            binsmagt = np.linspace(magtminm, magtmaxm, numbbins + 1)
            meanmagt = (binsmagt[1:] + binsmagt[:-1]) / 2.
            reca = np.empty(numbbins)
            numbsupnmagt = np.zeros(numbbins)
            #if gdat.datatype == 'mock':
            #    for n in indxsupn:
            #        indxmagt = np.digitize(np.amax(truemagt[:, n]), binsmagt) - 1
            #        numbsupnmagt[indxmagt] += 1
            #        if indxpixlposi.size > 0:
            #            numbpositrue[indxmagt] += 1
            #    recamagt = numbpositrue.astype(float) / numbsupnmagt
            #    prec = sum(numbpositrue).astype(float) / numbposi
            #    figr, axis = plt.subplots(figsize=(12, 6))
            #    axis.plot(meanmagt, recamagt, ls='', marker='o')
            #    axis.set_ylabel('Recall')
            #    axis.set_xlabel('Tmag')
            #    plt.tight_layout()
            #    path = gdat.pathimag + 'reca_%s_%04d.pdf' % (gdat.strgcntp, c)
            #    print('Writing to %s...' % path)
            #    plt.savefig(path)
            #    plt.close()
        
            c += 1
                
                
            # clustering with pyod
            # fraction of outliers
            fracoutl = 0.25
            
            # initialize a set of detectors for LSCP
            detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
                             LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
                             LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
                             LOF(n_neighbors=50)]
            
            # Show the statics of the data
            # Define nine outlier detection tools to be compared
            classifiers = {
                'Angle-based Outlier Detector (ABOD)':
                    ABOD(contamination=fracoutl),
                'Cluster-based Local Outlier Factor (CBLOF)':
                    CBLOF(contamination=fracoutl,
                          check_estimator=False, random_state=random_state),
                'Feature Bagging':
                    FeatureBagging(LOF(n_neighbors=35),
                                   contamination=fracoutl,
                                   random_state=random_state),
                #'Histogram-base Outlier Detection (HBOS)': HBOS(
                #    contamination=fracoutl),
                'Isolation Forest': IForest(contamination=fracoutl,
                                            random_state=random_state),
                'K Nearest Neighbors (KNN)': KNN(
                    contamination=fracoutl),
                'Average KNN': KNN(method='mean',
                                   contamination=fracoutl),
                # 'Median KNN': KNN(method='median',
                #                   contamination=fracoutl),
                'Local Outlier Factor (LOF)':
                    LOF(n_neighbors=35, contamination=fracoutl),
                # 'Local Correlation Integral (LOCI)':
                #     LOCI(contamination=fracoutl),
                
                #'Minimum Covariance Determinant (MCD)': MCD(
                #    contamination=fracoutl, random_state=random_state),
                
                'One-class SVM (OCSVM)': OCSVM(contamination=fracoutl),
                'Principal Component Analysis (PCA)': PCA(
                    contamination=fracoutl, random_state=random_state, standardization=False),
                # 'Stochastic Outlier Selection (SOS)': SOS(
                #     contamination=fracoutl),
                'Locally Selective Combination (LSCP)': LSCP(
                    detector_list, contamination=fracoutl,
                    random_state=random_state)
            }
            
            return
            raise Exception('')

            # Fit the model
            plt.figure(figsize=(15, 12))
            for i, (clf_name, clf) in enumerate(classifiers.items()):

                # fit the data and tag outliers
                clf.fit(X)
                scores_pred = clf.decision_function(X) * -1
                y_pred = clf.predict(X)
                threshold = np.percentile(scores_pred, 100 * fracoutl)
                n_errors = np.where(y_pred != listlabltrue)[0].size
                # plot the levels lines and the points
                #if i == 1:
                #    continue
                #Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
                #Z = Z.reshape(xx.shape)
                Z = np.zeros((100, 100))
                subplot = plt.subplot(3, 4, i + 1)
                subplot.contourf(xx, yy, Z, #levels=np.linspace(Z.min(), threshold, 7),
                                 cmap=plt.cm.Blues_r)
                subplot.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
                a = subplot.contour(xx, yy, Z, levels=[threshold],
                                    linewidths=2, colors='red')
                subplot.contourf(xx, yy, Z, #levels=[threshold, Z.max()],
                                 colors='orange')
                b = subplot.scatter(X[:-numboutl, 0], X[:-numboutl, 1], c='green', s=20, edgecolor='k')
                c = subplot.scatter(X[-numboutl:, 0], X[-numboutl:, 1], c='purple', s=20, edgecolor='k')
                subplot.axis('tight')
                subplot.legend(
                    [a.collections[0], b, c],
                    ['learned decision function', 'true inliers', 'true outliers'],
                    prop=matplotlib.font_manager.FontProperties(size=10),
                    loc='lower right')
                subplot.set_xlabel("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))
                subplot.set_xlim(limtproj)
                subplot.set_ylim(limtproj)
            plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
            plt.suptitle("Outlier detection")
            path = gdat.pathimag + 'pyod.png'
            print('Writing to %s...' % path)
            plt.savefig(path, dpi=300)
            plt.close()

            
            default_base = {'quantile': .3,
                            'eps': .3,
                            'damping': .9,
                            'preference': -200,
                            'n_neighbors': 10,
                            'n_clusters': 3,
                            'min_samples': 20,
                            'xi': 0.05,
                            'min_cluster_size': 0.1}
            
            # update parameters with dataset-specific values
            
            algo_params = {'damping': .77, 'preference': -240,
                 'quantile': .2, 'n_clusters': 2,
                 'min_samples': 20, 'xi': 0.25}

            params = default_base.copy()
            params.update(algo_params)
            
            # normalize dataset for easier parameter selection
            X = StandardScaler().fit_transform(X)
            
            # estimate bandwidth for mean shift
            bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])
            
            # connectivity matrix for structured Ward
            connectivity = kneighbors_graph(
                X, n_neighbors=params['n_neighbors'], include_self=False)
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)
            
            ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
            two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
            ward = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='ward', connectivity=connectivity)
            spectral = cluster.SpectralClustering(n_clusters=params['n_clusters'], eigen_solver='arpack', affinity="nearest_neighbors")
            dbscan = cluster.DBSCAN(eps=params['eps'])
            
            #optics = cluster.OPTICS(min_samples=params['min_samples'],
            #                        xi=params['xi'],
            #                        min_cluster_size=params['min_cluster_size'])
            
            affinity_propagation = cluster.AffinityPropagation(damping=params['damping'], preference=params['preference'])
            average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock", \
                                                                                n_clusters=params['n_clusters'], connectivity=connectivity)
            birch = cluster.Birch(n_clusters=params['n_clusters'])
            gmm = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')
            
            clustering_algorithms = (
                ('MiniBatchKMeans', two_means),
                ('AffinityPropagation', affinity_propagation),
                ('MeanShift', ms),
                ('SpectralClustering', spectral),
                ('Ward', ward),
                ('AgglomerativeClustering', average_linkage),
                ('DBSCAN', dbscan),
                #('OPTICS', optics),
                ('Birch', birch),
                ('GaussianMixture', gmm)
            )
            
            figr, axis = plt.subplots(1, numbmeth)
            k = 0
            for name, algorithm in clustering_algorithms:
                t0 = timemodu.time()
                
                # catch warnings related to kneighbors_graph
                with warnings.catch_warnings():
                    #warnings.filterwarnings(
                    #    "ignore",
                    #    message="the number of connected components of the " +
                    #    "connectivity matrix is [0-9]{1,2}" +
                    #    " > 1. Completing it to avoid stopping the tree early.",
                    #    category=UserWarning)
                    #warnings.filterwarnings(
                    #    "ignore",
                    #    message="Graph is not fully connected, spectral embedding" +
                    #    " may not work as expected.",
                    #    category=UserWarning)
                    algorithm.fit(X)
            
                t1 = timemodu.time()
                if hasattr(algorithm, 'labels_'):
                    lablmodl = algorithm.labels_.astype(np.int)
                else:
                    lablmodl = algorithm.predict(X)
                

                axis[k].set_title(name, size=18)
            
                colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                     '#f781bf', '#a65628', '#984ea3',
                                                     '#999999', '#e41a1c', '#dede00']),
                                              int(max(lablmodl) + 1))))
                # add black color for outliers (if any)
                colors = np.append(colors, ["#000000"])
                axis[k].scatter(X[:, 0], X[:, 1], s=10, color=colors[lablmodl])
            
                axis[k].set_xlim(-2.5, 2.5)
                axis[k].set_ylim(-2.5, 2.5)
                axis[k].set_xticks(())
                axis[k].set_yticks(())
                axis[k].text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                         transform=plt.gca().transAxes, size=15,
                         horizontalalignment='right')
                k += 1
                listlablmodl.append(lablmodl)
            path = gdat.pathimag + 'clus.pdf'
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()


            # Random 2D projection using a random unitary matrix
            rp = random_projection.SparseRandomProjection(n_components=2)
            X_projected = rp.fit_transform(lcurflat)
            plot_embe(gdat, lcurflat, X_projected, 'rand', "Random Projection")
            
            # Projection on to the first 2 principal components
            t0 = timemodl.time()
            X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(lcurflat)
            plot_embe(gdat, lcurflat, X_pca, 'pcaa', "Principal Components projection (time %.2fs)" % (timemodl.time() - t0))
            
            # Projection on to the first 2 linear discriminant components
            #X2 = lcurflat.copy()
            #X2.flat[::lcurflat.shape[1] + 1] += 0.01  # Make X invertible
            #t0 = timemodl.time()
            #X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X2, y)
            #plot_embe(gdat, lcurflat, X_lda, 'ldap', "Linear Discriminant projection (time %.2fs)" % (timemodl.time() - t0))
            
            # t-SNE embedding dataset
            tsne = manifold.TSNE(n_components=2, random_state=0, perplexity=30)
            t0 = timemodl.time()
            X_tsne = tsne.fit_transform(lcurflat)
            plot_embe(gdat, lcurflat, X_tsne, 'tsne0030', "t-SNE embedding with perplexity 30")
            
            # t-SNE embedding dataset
            tsne = manifold.TSNE(n_components=2, random_state=0, perplexity=5)
            t0 = timemodl.time()
            X_tsne = tsne.fit_transform(lcurflat)
            plot_embe(gdat, lcurflat, X_tsne, 'tsne0005', "t-SNE embedding with perplexity 5")
            
            # Isomap projection dataset
            t0 = timemodl.time()
            X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(lcurflat)
            plot_embe(gdat, lcurflat, X_iso, 'isop', "Isomap projection (time %.2fs)" % (timemodl.time() - t0))
            
            # Locally linear embedding dataset
            clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='standard')
            t0 = timemodl.time()
            X_lle = clf.fit_transform(lcurflat)
            plot_embe(gdat, lcurflat, X_lle, 'llep', "Locally Linear Embedding (time %.2fs)" % (timemodl.time() - t0))
            
            # Modified Locally linear embedding dataset
            clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='modified')
            t0 = timemodl.time()
            X_mlle = clf.fit_transform(lcurflat)
            plot_embe(gdat, lcurflat, X_mlle, 'mlle', "Modified Locally Linear Embedding (time %.2fs)" % (timemodl.time() - t0))
            
            # HLLE embedding dataset
            clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='hessian')
            t0 = timemodl.time()
            X_hlle = clf.fit_transform(lcurflat)
            plot_embe(gdat, lcurflat, X_hlle, 'hlle', "Hessian Locally Linear Embedding (time %.2fs)" % (timemodl.time() - t0))
            
            # LTSA embedding dataset
            clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='ltsa')
            t0 = timemodl.time()
            X_ltsa = clf.fit_transform(lcurflat)
            plot_embe(gdat, lcurflat, X_ltsa, 'ltsa', "Local Tangent Space Alignment (time %.2fs)" % (timemodl.time() - t0))
            
            # MDS  embedding dataset
            clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
            t0 = timemodl.time()
            X_mds = clf.fit_transform(lcurflat)
            plot_embe(gdat, lcurflat, X_mds, 'mdse', "MDS embedding (time %.2fs)" % (timemodl.time() - t0))
            
            # Random Trees embedding dataset
            hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0, max_depth=5)
            t0 = timemodl.time()
            X_transformed = hasher.fit_transform(lcurflat)
            pca = decomposition.TruncatedSVD(n_components=2)
            X_reduced = pca.fit_transform(X_transformed)
            plot_embe(gdat, lcurflat, X_reduced, 'rfep', "Random forest embedding (time %.2fs)" % (timemodl.time() - t0))
            
            # Spectral embedding dataset
            embedder = manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack")
            t0 = timemodl.time()
            X_se = embedder.fit_transform(lcurflat)
            plot_embe(gdat, lcurflat, X_se, 'csep', "Spectral embedding (time %.2fs)" % (timemodl.time() - t0))
            
            # NCA projection dataset
            #nca = neighbors.NeighborhoodComponentsAnalysis(n_components=2, random_state=0)
            #t0 = timemodl.time()
            #X_nca = nca.fit_transform(lcurflat, y)
            #plot_embe(gdat, lcurflat, X_nca, 'ncap', "NCA embedding (time %.2fs)" % (timemodl.time() - t0))

            figr, axis = plt.subplots(figsize=(12, 6))
            
            for strgvarb in ['diff']:
                figr, axis = plt.subplots(figsize=(12, 6))
                #if strgvarb == 'diff':
                #    varbtemp = np.arcsinh(dictpara[strgvarb])
                #else:
                #    varbtemp = dictpara[strgvarb]
                varbtemp = dictpara[strgvarb]
                vmin = -1
                vmax = 1
                objtimag = axis.imshow(varbtemp, interpolation='nearest', cmap='Greens', vmin=vmin, vmax=vmax)
                plt.colorbar(objtimag)
                plt.tight_layout()
                path = gdat.pathimag + '%s_%s.pdf' % (strgvarb, gdat.strgcntp)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
    

def retr_timeexec():
    # input PCAT speed per 100x100 pixel region
    timeregi = 30. # [min]
    
    # number of time frames in each region
    numbtser = 13.7 * 4 * 24 * 60. / 30.
    
    timeregitser = numbtser * timeregi / 60. / 24 # [day]
    timeffim = 16.8e6 / 1e4 * timeregi # [day]
    timesegm = 4. * timeffim / 7. # [week]
    timefsky = 26 * timesegm / 7. # [week]
    
    print('Full frame, full sky: %d weeks per 1000 cores' % (timefsky / 1000.))


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
    path = gdat.pathimag + 'lspd_%s.pdf' % (strgmask)
    print('Writing to %s...' % path)
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


def cnfg_defa():
   
    listisec = [9]
    listicam = [1]
    listiccd = [1]
    init( \
         listisec=listisec, \
         listicam=listicam, \
         listiccd=listiccd, \
         extrtype='qlop')


def cnfg_mock():
   
    listisec = [9]
    listicam = [1]
    listiccd = [1]
    init( \
         listisec=listisec, \
         listicam=listicam, \
         listiccd=listiccd, \
         datatype='mock', \
         extrtype='qlop')


def cnfg_sect():
    
    for isec in range(9, 10):
        for icam in range(4, 5):
            for iccd in range(2, 3):
                init(isec, icam, iccd)

globals().get(sys.argv[1])()

