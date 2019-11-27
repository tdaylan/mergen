from lion import main as lionmain
import tdpy
from tdpy.util import summgene

import numpy as np

import h5py

import astroquery.mast
from astroquery.mast import Tesscut
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
import tcat.main

import time as timemodu

#import pickle

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

# own 
import tesstarg.util

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
import astropy.wcs
#from astropy.wcs import WCS
#from astropy import units as u
#from astropy.io import fits
#import astropy.time

import multiprocessing


def plot_embe(gdat, datatranembe, strg, titl):
    
    datatranembe = (datatranembe - np.amin(datatranembe, 0)) / (np.amax(datatranembe, 0) - np.amin(datatranembe, 0))

    figr, axis = plt.subplots(figsize=(12, 12))
    axis.scatter(datatranembe[:, 0], datatranembe[:, 1], s=5, marker='x', color='r', lw=0.5)
    shown_images = np.array([[1., 1.]])  # just something big
    for i in range(datatranembe.shape[0]):
        dist = np.sum((datatranembe[i] - shown_images) ** 2, 1)
        if np.min(dist) < 1e-3:
            continue
        shown_images = np.r_[shown_images, [datatranembe[i]]]
        axins3 = inset_axes(axis, width="100%", height="100%", \
                        bbox_to_anchor=(datatranembe[i, 0] - 0.02, datatranembe[i, 1] - 0.02, .04, .04), bbox_transform=axis.transData, loc='center', borderpad=0)
        axins3.plot(gdat.time, gdat.datatran[i, :], alpha=0.5, color='g')
        #axins3.set_ylim([0, 2])
        axins3.text(datatranembe[i, 0], datatranembe[i, 1] + 0.02, '%g %g' % (np.amin(gdat.datatran[i, :]), np.amax(gdat.datatran[i, :])), fontsize=12) 
        axins3.axis('off')
    axis.set_title(titl)
    plt.tight_layout()
    path = gdat.pathimag + 'embe_%s_%s.png' % (strg, gdat.strgcntp)
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
        path = gdat.pathdata + '%s_%s_%05d.%s' % (strgvarb, gdat.strgcntp, t, gdat.strgfext)
        plot_imag(gdat, cntp[:, :, t], path=path, strgvarb=strgvarb, cmap=cmap, strgtitl=strgtitl, \
                                        indxsideyposoffs=indxsideyposoffs, indxsidexposoffs=indxsidexposoffs, boolresi=boolresi, vmin=vmin, vmax=vmax)
    os.system('convert -density 300 -delay 50 %s%s_%s_*.%s %s%s_%s.gif' % (gdat.pathdata, strgvarb, gdat.strgcntp, gdat.strgfext, \
                                                                                                            gdat.pathdata, strgvarb, gdat.strgcntp))
    ### delete the frame plots
    path = gdat.pathdata + '%s_%s_*.%s' % (strgvarb, gdat.strgcntp, gdat.strgfext)
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
        path = gdat.pathimag + '%s_%s.%s' % (strgvarb, gdat.strgcntp, gdat.strgfext)
    print('Writing to %s...' % path)
    #plt.tight_layout()
    plt.savefig(path)
    plt.close()
    

def init( \
         listisec=None, \
         listicam=None, \
         listiccd=None, \
         # light curve extraction type
         extrtype='spoc', \
         targtype='slen', \
         pathfile=None, \
         datatype='obsd', \
         rasctarg=None, \
         decltarg=None, \
         verbtype=1, \
         labltarg=None, \
         numbside=None, \
         **args \
        ):
 
    # inputs:
    # 1) TIC IDs
    # 2) One sector, One Cam, One CCD

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
    
    print('CTLC initialized at %s...' % gdat.strgtimestmp)
    
    #if ((icam is not None or iccd is not None) and listtici is not None):
    #    raise Exception('')
    
    if gdat.datatype == 'mock':
        gdat.trueclastype = 'catg'
    

    #star = eleanor.Source(tic=38846515, sector=1, tc=True)

    gdat.strgcntp = gdat.datatype

    # paths
    ## read PCAT path environment variable
    gdat.pathbase = os.environ['CTLC_DATA_PATH'] + '/'
    gdat.pathdata = gdat.pathbase + 'data/'
    gdat.pathimagbase = gdat.pathbase + 'imag/'
    ## define paths
    gdat.pathdataorig = gdat.pathdata + 'ffis/'
    gdat.pathdatainit = gdat.pathdata + 'init/'
    gdat.pathdatainitimag = gdat.pathdatainit + 'imag/'
    gdat.pathdatainitanim = gdat.pathdatainit + 'anim/'
    gdat.pathdatacomm = gdat.pathdata + 'comm/'
    
    gdat.boolsupe = False

    if pathfile is None:
        if numbside is None:
            strgmode = 'full'
        else:
            strgmode = 'targ'
    else:
        strgmode = 'file'
    
    # reference catalog
    gdat.numbrefr = 1

    random_state = 42

    timeexpo = 1426.
    
    np.random.seed(45)

    gdat.numbfeat = 1

    gdat.numbtimerebn = None#30
    
    # settings
    ## plotting
    gdat.strgffim = 'ffim'
    gdat.strgfext = 'png'
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

    gdat.numbsect = len(gdat.listisec)
    gdat.indxsect = np.arange(gdat.numbsect)
    for o in gdat.indxsect:
        strgsecc = '%02d%d%d' % (gdat.listisec[o], gdat.listicam[o], gdat.listiccd[o])
        print('Sector: %d' % gdat.listisec[o])
        print('Camera: %d' % gdat.listicam[o])
        print('CCD: %d' % gdat.listiccd[o])
        print('strgsecc')
        print(strgsecc)
        
        gdat.pathimag = gdat.pathimagbase + 'sector-%d/cam%d/ccd%d/' % (gdat.listisec[o], gdat.listicam[o], gdat.listiccd[o])
        cmnd = 'mkdir -p %s' % gdat.pathimag
        os.system(cmnd)

        if gdat.datatype == 'obsd':
            print('Reading files...')
            gdat.pathsect = gdat.pathdata + 'lcur/%s/sector-%d/cam%d/ccd%d/' % (gdat.extrtype, gdat.listisec[o], gdat.listicam[o], gdat.listiccd[o])
            print('gdat.pathsect')
            print(gdat.pathsect)
            liststrgfile = fnmatch.filter(os.listdir(gdat.pathsect), '*.fits')
            numbfile = len(liststrgfile)
            print('Number of light curves: %s' % numbfile)
            if numbfile == 0:
                raise Exception('')
            liststrgfile = np.array(liststrgfile)
    
        numbchun = np.round(numbfile / 2000.).astype(int)
        numbfilechun = np.empty(numbchun, dtype=int)
        numbfilechun[:-1] = numbfile / numbchun
        numbfilechun[-1] = numbfile - numbfilechun[0] * (numbchun - 1)
        print('numbfilechun')
        print(numbfilechun)
        print('np.sum(numbfilechun)')
        print(np.sum(numbfilechun))

        indxchun = np.arange(numbchun)
        cntr = 0
        for j in indxchun: 
                
            indxfilechun = np.arange(cntr, np.sum(numbfilechun[:j+1]))
            
            print('Chunck number %d' % j)

            pathlcursort = gdat.pathimag + 'lcursort_outd_%s_%s_%04d.%s' % (gdat.strgcntp, 'louf', j, gdat.strgfext)
            cntr += np.sum(numbfilechun[j])
            if os.path.exists(pathlcursort):
                continue
            print('indxfilechun')
            summgene(indxfilechun)
            if gdat.datatype == 'obsd':
                
                liststrgtici = []
                for strgfile in liststrgfile[indxfilechun]:
                    liststrgtici.append(strgfile[:-3])

                gdat.numbdata = len(liststrgfile[indxfilechun])
                fracdatanann = np.empty(gdat.numbdata)
                listindxtimebadd = []
                for k, strgfile in enumerate(liststrgfile[indxfilechun]):
                    if k % 1000 == 0:
                        print('k')
                        print(k)
                    if gdat.extrtype == 'spoc':
                        pathlcur = gdat.pathsect + strgfile
                        arrypdcc, indxtimequalgood, indxtimenanngood = tesstarg.util.read_tesskplr_file(pathlcur, \
                                                                        typeinst='tess', strgtype='PDCSAP_FLUX', boolmaskqual=False, boolmasknann=False)
                        arrypdcc = arrypdcc[indxtimequalgood, :] = np.nan
                        gdat.time = arrypdcc[:, 0]
                        print('gdat.time')
                        summgene(gdat.time)
                        if k == 0:
                            gdat.numbtime = gdat.time.size
                            lcur = np.empty((gdat.numbdata, gdat.numbtime, gdat.numbfeat))
                        print('lcur')
                        summgene(lcur)
                        lcur[k, :, 0] = arrypdcc[:, 1]
                    if gdat.extrtype == 'qlop':
                        with h5py.File(gdat.pathsect + strgfile, 'r') as objtfile:
                            gdat.time = objtfile['LightCurve/BJD'][()]
                            if k == 0:
                                gdat.numbtime = gdat.time.size
                                lcur = np.empty((gdat.numbdata, gdat.numbtime, gdat.numbfeat))
                                print('gdat.numbtime')
                                print(gdat.numbtime)
                            tmag = objtfile['LightCurve/AperturePhotometry/Aperture_002/RawMagnitude'][()]
                            if tmag.size != gdat.numbtime:
                                print('k')
                                print(k)
                                print('strgfile')
                                print(strgfile)
                                print('Different number of time bins.')
                                print
                                tmag = np.ones(gdat.numbtime)
                            if k == 0:
                                gdat.indxtimetemp = np.arange(gdat.numbtime)
                            indxtimegood = np.where(np.isfinite(tmag))[0]
                            indxtimenann = np.setdiff1d(gdat.indxtimetemp, indxtimegood)
                            lcur[k, :, 0] = 10**(-(tmag - np.nanmedian(tmag[indxtimegood])) / 2.5)
                            listindxtimebadd.append(indxtimenann)
                            fracdatanann[k] = indxtimenann.size / float(gdat.numbtime)
                

                listindxtimebadd = np.concatenate(listindxtimebadd)
                listindxtimebadd = np.unique(listindxtimebadd)
                
                listindxtimebadd = np.concatenate((listindxtimebadd, np.arange(100)))
                listindxtimebadd = np.concatenate((listindxtimebadd, gdat.numbtime / 2 + np.arange(100)))
                listindxtimegood = np.setdiff1d(gdat.indxtimetemp, listindxtimebadd)
                
                #print('Masking out the data...')
                # filter the data
                #gdat.time = gdat.time[listindxtimegood]
                lcur = lcur[:, listindxtimegood, :]
            
            gdat.numbtime = gdat.time.size
            gdat.indxtime = np.arange(gdat.numbtime) 
            if (~np.isfinite(lcur)).any():
                raise Exception('')

            if gdat.datatype == 'obsd':
                # plot the fraction of data that is NaN
                figr, axis = plt.subplots(figsize=(6, 4))
                axis.hist(fracdatanann, bins=100)
                axis.set_yscale('log')
                axis.set_xlabel('$f_{nan}$')
                axis.set_ylabel('$N(f_{nan})$')
                plt.tight_layout()
                path = gdat.pathimag + 'histfracdatanann_%s.%s' % (gdat.strgcntp, gdat.strgfext)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()

            gdat.numbdata = lcur.shape[0]
            if gdat.datatype == 'mock':
                listlabltrue = np.zeros(gdat.numbdata, dtype=int)
                if gdat.trueclastype == 'outl':
                    numbinli = gdat.numbdata - gdat.numbsour
                    gdat.truenumboutl = gdat.numbsour
            
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
            path = gdat.pathimag + 'lcurrand_%s.%s' % (gdat.strgcntp, gdat.strgfext)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()

            # temporal median filter
            numbtimefilt = min(9, gdat.numbtime)
            if numbtimefilt % 2 == 0:
                numbtimefilt -= 1
            
            # rebin in time
            if False and gdat.numbtimerebn is not None and gdat.numbtime > gdat.numbtimerebn:
                print('Rebinning in time...')
                numbtimeoldd = gdat.numbtime
                gdat.numbtime = gdat.numbtimerebn
                numbtimebins = numbtimeoldd / gdat.numbtime
                gdat.facttimerebn = numbtimebins
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
            else:
                gdat.facttimerebn = 1
            # calculate derived maps
            ## RMS image

            strgtype = 'tsne'

            lcuravgd = np.empty(gdat.numbtime)
            prevfrac = -1
            k = 0
            
            scorthrs = -2.

            # machine learning

            gdat.numbneig = 30
            
            gdat.datatran = lcur[:, :, 0]

            print('gdat.datatran')
            summgene(gdat.datatran)
                    
            indxdata = np.arange(gdat.numbdata)
            
            fracoutl = 0.15
            
            print('Perform outlier detection using sklearn...')
            # define outlier/anomaly detection methods to be compared
            listobjtalgooutd = [
                                #('osvm', "One-Class SVM", svm.OneClassSVM(nu=fracoutl, kernel="rbf", gamma=0.1)), \
                                #('ifor', "Isolation Forest", IsolationForest(contamination=fracoutl)), \
                                ('louf', "Local Outlier Factor", LocalOutlierFactor(n_neighbors=gdat.numbneig, contamination=fracoutl))
                               ]
            
            numbmeth = len(listobjtalgooutd)
            indxmeth = np.arange(numbmeth)
            
            listindxsideyposaccp = []
            listindxsidexposaccp = []
            listscor = []
            listlablmodl = []
            
            numbtimeplotscat = min(6, gdat.numbtime)
            limt = [np.amin(gdat.datatran), np.amax(gdat.datatran)]
            
            c = 0
            for strg, name, objtalgooutd in listobjtalgooutd:
                t0 = timemodu.time()
                
                objtalgooutd.fit(gdat.datatran)
                t1 = timemodu.time()
            
                # fit the data and tag outliers
                if name == 'Local Outlier Factor':
                    scor = objtalgooutd.negative_outlier_factor_
                else:
                    scor = objtalgooutd.decision_function(gdat.datatran)
                if name == "Local Outlier Factor":
                    lablmodl = np.zeros(gdat.numbdata)
                    lablmodl[np.where(scor < scorthrs)[0]] = 1.
                else:
                    lablmodl = objtalgooutd.fit(gdat.datatran).predict(gdat.datatran)
                
                indxdataposi = np.where(lablmodl == 1)[0]
                indxdatanega = np.setdiff1d(indxdata, indxdataposi)
                numbposi = indxdataposi.size
                gdat.numbpositext = min(200, numbposi)

                numboutl = indxdataposi.size

                listscor.append(scor)
                listlablmodl.append(lablmodl)
                
                gdat.indxdatascorsort = np.argsort(listscor[c])
                
                if False:
                    # make plots
                    ## labeled marginal distributions
                    figr, axis = plt.subplots(numbtimeplotscat - 1, numbtimeplotscat - 1, figsize=(10, 10))
                    for t in gdat.indxtime[:numbtimeplotscat-1]:
                        for tt in gdat.indxtime[:numbtimeplotscat-1]:
                            if t < tt:
                                axis[t][tt].axis('off')
                                continue
                            axis[t][tt].scatter(gdat.datatran[indxdatanega, t+1], gdat.datatran[indxdatanega, tt], s=20, color='r', alpha=0.3)#*listscor[c])
                            axis[t][tt].scatter(gdat.datatran[indxdataposi, t+1], gdat.datatran[indxdataposi, tt], s=20, color='b', alpha=0.3)#*listscor[c])
                            axis[t][tt].set_ylim(limt)
                            axis[t][tt].set_xlim(limt)
                            #axis[t][tt].set_xticks(())
                            #axis[t][tt].set_yticks(())
                    path = gdat.pathimag + 'pmar_outd_%s_%s.%s' % (gdat.strgcntp, strg, gdat.strgfext)
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
                        axis[a][b].plot(gdat.time, gdat.datatran[p, :], color=colr, alpha=0.1, ls='', marker='o', markersize=3)
                        if a != 9:
                            axis[a][b].set_xticks([])
                        if b != 0:
                            axis[a][b].set_yticks([])
                plt.subplots_adjust(hspace=0, wspace=0)
                path = gdat.pathimag + 'lcurpred_outd_%s_%s.%s' % (gdat.strgcntp, strg, gdat.strgfext)
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
                path = gdat.pathimag + 'histscor_outd_%s_%s.%s' % (gdat.strgcntp, strg, gdat.strgfext)
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
                        if not np.isscalar(indx):
                            indx = indx[0]
                        
                        boolplotfram = False
                        print('lk')
                        print(l, k)
                        # plot the postage stamp animation
                        if gdat.datatype == 'obsd':
                            labltarg = 'TIC: %s' % liststrgtici[indx]
                            if l == 0:
                                gdat.strgtarg = '%012d' % int(liststrgtici[indx])
                                
                                # get the catalog near the target
                                catalogData = astroquery.mast.Catalogs.query_object(int(liststrgtici[indx]), radius='0.1m', catalog='TIC')
                                rasctarg = catalogData[0]['ra']
                                decltarg = catalogData[0]['dec']
                                    
                                try:
                                    result_table = Simbad.query_region('%g %g' % (rasctarg, decltarg), radius='2m')
                                    print('TIC')
                                    print(int(liststrgtici[indx]))
                                    lablobjt = result_table[0]['MAIN_ID']
                                    print
                                except:
                                    print('Failed to query simbad')
                                    lablobjt = ''
                                
                                if boolplotfram:
                                    # get cut-out data
                                    cutout_coord = SkyCoord(rasctarg, decltarg, unit="deg")
                                    listhdundata = Tesscut.get_cutouts(cutout_coord, gdat.numbsideplot)
                                    #gdat.listtime = listhdundata[o][1].data['TIME']
                                    cntpdata = 30. * 60. * listhdundata[o][1].data['FLUX'].swapaxes(0, 2).swapaxes(0, 1)[None, :, :, :]
                                    
                                    # transform RA and DEC to pixel coordinates
                                    skyyfitttemp = np.empty((catalogData[:]['ra'].size, 2))
                                    skyyfitttemp[:, 0] = catalogData[:]['ra']
                                    skyyfitttemp[:, 1] = catalogData[:]['dec']
                                    objtwcss = astropy.wcs.WCS(listhdundata[o][2].header)
                                    posifitttemp = objtwcss.all_world2pix(skyyfitttemp, 0)
                                    posifitttemp = posifitttemp[np.where((posifitttemp[:, 0] < gdat.numbsideplot - 0.5) & (posifitttemp[:, 0] > 0) & \
                                                                         (posifitttemp[:, 1] < gdat.numbsideplot - 0.5) & (posifitttemp[:, 1] > 0))[0], :]
                                    # make a reference catalog
                                    gdat.catlrefr = [{}]
                                    gdat.catlrefr[0]['xpos'] = posifitttemp[:, 0]
                                    gdat.catlrefr[0]['ypos'] = posifitttemp[:, 1]
                                        
                                    #except:
                                    #    rasctarg = np.random.random(10) * gdat.numbsideplot
                                    #    decltarg = np.random.random(10) * gdat.numbsideplot
                                    #    cntpdata = np.random.randn(gdat.numbtime * gdat.numbsideplot**2).reshape((1, \
                                    #                                                                gdat.numbsideplot, gdat.numbsideplot, gdat.numbtime))
                                    #    print('MAST catalog retriieval failed. Using dummy images and catalogs.')
                                    cntpresi = cntpdata - np.median(cntpdata, 2)
                                    for a in range(2):
                                        if a == 0:
                                            cntptemp = cntpdata
                                            boolresi = False
                                        else:
                                            cntptemp = cntpresi
                                            boolresi = True
                                        #tcat.util.plot_cntpwrap(gdat, cntptemp, gdat.indxtime[::4], o, strgsecc, lcur=gdat.datatran[indx, :], time=gdat.time, \
                                        #                                                                                    boolresi=boolresi, labltarg=labltarg)
                        else:
                            lablobjt = ''
                        print('lablobjt')
                        print(lablobjt)
                        axis[k][l].plot(gdat.time, gdat.datatran[indx, :], color='black', ls='', marker='o', markersize=1)
                        if gdat.datatype == 'obsd':
                            axis[k][l].set_title('TIC: %s, %s' % (liststrgtici[indx], lablobjt))
                            #axis[k][l].text(.9, .9, 'TIC: %s, %s' % (liststrgtici[indx], lablobjt), \
                            #                                                    transform=axis[k][l].transAxes, size=10, color='r', ha='right', va='center')
                        if l == 1:
                            axis[k][l].yaxis.set_label_position('right')
                            axis[k][l].yaxis.tick_right()
                        if k != gdat.numbdataplotsort - 1:
                            axis[k][l].set_xticks([])
                plt.subplots_adjust(hspace=0.2, wspace=0)
                print('Writing to %s...' % pathlcursort)
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
                #    path = gdat.pathimag + 'reca_%s_%04d.%s' % (gdat.strgcntp, c, gdat.strgfext)
                #    print('Writing to %s...' % path)
                #    plt.savefig(path)
                #    plt.close()
            
                c += 1
                
        # clustering with pyod
        # fraction of outliers
        
        # initialize a set of detectors for LSCP
        detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
                         LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
                         LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
                         LOF(n_neighbors=50)]
        
        # Show the statics of the data
        # Define nine outlier detection tools to be compared
        listobjtclas = {
            'Angle-based Outlier Detector (ABOD)':
            ABOD(contamination=fracoutl),
            
            'Cluster-based Local Outlier Factor (CBLOF)':
            CBLOF(contamination=fracoutl, check_estimator=False, random_state=random_state), 
            
            'Feature Bagging':
            FeatureBagging(LOF(n_neighbors=gdat.numbneig), contamination=fracoutl, random_state=random_state),
            
            #'Histogram-base Outlier Detection (HBOS)': 
            #HBOS(contamination=fracoutl),
            
            'Isolation Forest': 
            IForest(contamination=fracoutl, random_state=random_state),
            
            'K Nearest Neighbors (KNN)': 
            KNN(contamination=fracoutl),
            
            'Average KNN':
            KNN(method='mean', contamination=fracoutl),
            
            # 'Median KNN': KNN(method='median',
            #                   contamination=fracoutl),
            
            'Local Outlier Factor (LOF)':
            LOF(n_neighbors=gdat.numbneig, contamination=fracoutl),
            
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
        
        print('Fitting outlier detection models using Pyod...')

        # Fit the model
        plt.figure(figsize=(15, 12))
        for i, (lablmethoutdpyod, objtmethoutdpyod) in enumerate(listobjtclas.items()):

            print('i')
            print(i)
            # fit the data and tag outliers
            objtmethoutdpyod.fit(gdat.datatran)
            scores_pred = objtmethoutdpyod.decision_function(gdat.datatran) * -1
            y_pred = objtmethoutdpyod.predict(gdat.datatran)
            threshold = np.percentile(scores_pred, 100 * fracoutl)
            if gdat.boolsupe:
                n_errors = np.where(y_pred != listlabltrue)[0].size
            # plot the levels lines and the points
            if i == 4:
                break
            #Z = objtmethoutdpyod.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
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
            b = subplot.scatter(gdat.datatran[:-numboutl, 0], gdat.datatran[:-numboutl, 1], c='green', s=20, edgecolor='k')
            c = subplot.scatter(gdat.datatran[-numboutl:, 0], gdat.datatran[-numboutl:, 1], c='purple', s=20, edgecolor='k')
            subplot.axis('tight')
            subplot.legend(
                [a.collections[0], b, c],
                ['learned decision function', 'true inliers', 'true outliers'],
                prop=matplotlib.font_manager.FontProperties(size=10),
                loc='lower right')
            if gdat.boolsupe:
                subplot.set_xlabel("%d. %s (errors: %d)" % (i + 1, lablmethoutdpyod, n_errors))
            subplot.set_xlim(limtproj)
            subplot.set_ylim(limtproj)
        plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
        plt.suptitle("Outlier detection")
        path = gdat.pathimag + 'pyod_%s.png' % gdat.strgcntp
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
        gdat.datatran = (gdat.datatran - np.mean(gdat.datatran)) / np.std(gdat.datatran)
        
        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(gdat.datatran, quantile=params['quantile'])
        
        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(
            gdat.datatran, n_neighbors=params['n_neighbors'], include_self=False)
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
        
        print('Fitting clustering models...')

        figr, axgr = plt.subplots(numbmeth, 1, figsize=(6, numbmeth * 6))
        if numbmeth == 1:
            axgr = [axgr]
        for k, axis in enumerate(axgr):
            print('k')
            print(k)
            name, algorithm = clustering_algorithms[k]
            t0 = timemodu.time()
            
            algorithm.fit(gdat.datatran)
        
            t1 = timemodu.time()
            if hasattr(algorithm, 'labels_'):
                lablmodl = algorithm.labels_.astype(np.int)
            else:
                lablmodl = algorithm.predict(gdat.datatran)

            axis.set_title(name, size=18)
        
            axis.scatter(gdat.datatran[:, 0], gdat.datatran[:, 1], s=10)
        
            axis.set_xlim(-2.5, 2.5)
            axis.set_ylim(-2.5, 2.5)
            axis.set_xticks(())
            axis.set_yticks(())
            axis.text(.9, .1, '%.3g sec' % (t1 - t0), transform=axis.transAxes, size=15, ha='center')
            listlablmodl.append(lablmodl)
        path = gdat.pathimag + 'clus_%s.%s' % (gdat.strgcntp, gdat.strgfext)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()


        print('Fitting dimensional reduction models...')

        # Random 2D projection using a random unitary matrix
        rp = random_projection.SparseRandomProjection(n_components=2)
        datatranproj = rp.fit_transform(gdat.datatran)
        plot_embe(gdat, datatranproj, 'rand', "Random Projection")
        
        # Projection on to the first 2 principal components
        #t0 = timemodu.time()
        #datatranproj = decomposition.TruncatedSVD(n_components=2).fit_transform(gdat.datatran)
        #plot_embe(gdat, datatranproj, 'pcaa', "Principal Components projection (time %.2fs)" % (timemodu.time() - t0))
        
        # Projection on to the first 2 linear discriminant components
        #X2 = lcurflat.copy()
        #X2.flat[::lcurflat.shape[1] + 1] += 0.01  # Make X invertible
        #t0 = timemodu.time()
        #X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X2, y)
        #plot_embe(gdat, X_lda, 'ldap', "Linear Discriminant projection (time %.2fs)" % (timemodu.time() - t0))
        
        # t-SNE embedding dataset
        tsne = manifold.TSNE(n_components=2, random_state=0, perplexity=30)
        t0 = timemodu.time()
        datatranproj = tsne.fit_transform(gdat.datatran)
        plot_embe(gdat, datatranproj, 'tsne0030', "t-SNE embedding with perplexity 30")
        
        # t-SNE embedding dataset
        tsne = manifold.TSNE(n_components=2, random_state=0, perplexity=5)
        t0 = timemodu.time()
        datatranproj = tsne.fit_transform(gdat.datatran)
        plot_embe(gdat, datatranproj, 'tsne0005', "t-SNE embedding with perplexity 5")
        
        # Isomap projection dataset
        t0 = timemodu.time()
        X_iso = manifold.Isomap(gdat.numbneig, n_components=2).fit_transform(gdat.datatran)
        plot_embe(gdat, X_iso, 'isop', "Isomap projection (time %.2fs)" % (timemodu.time() - t0))
        
        # Locally linear embedding dataset
        clf = manifold.LocallyLinearEmbedding(gdat.numbneig, n_components=2, method='standard')
        t0 = timemodu.time()
        X_lle = clf.fit_transform(gdat.datatran)
        plot_embe(gdat, X_lle, 'llep', "Locally Linear Embedding (time %.2fs)" % (timemodu.time() - t0))
        
        # Modified Locally linear embedding dataset
        clf = manifold.LocallyLinearEmbedding(gdat.numbneig, n_components=2, method='modified')
        t0 = timemodu.time()
        X_mlle = clf.fit_transform(gdat.datatran)
        plot_embe(gdat, X_mlle, 'mlle', "Modified Locally Linear Embedding (time %.2fs)" % (timemodu.time() - t0))
        
        # HLLE embedding dataset
        clf = manifold.LocallyLinearEmbedding(gdat.numbneig, n_components=2, method='hessian')
        t0 = timemodu.time()
        X_hlle = clf.fit_transform(gdat.datatran)
        plot_embe(gdat, X_hlle, 'hlle', "Hessian Locally Linear Embedding (time %.2fs)" % (timemodu.time() - t0))
        
        # LTSA embedding dataset
        clf = manifold.LocallyLinearEmbedding(gdat.numbneig, n_components=2, method='ltsa')
        t0 = timemodu.time()
        X_ltsa = clf.fit_transform(gdat.datatran)
        plot_embe(gdat, X_ltsa, 'ltsa', "Local Tangent Space Alignment (time %.2fs)" % (timemodu.time() - t0))
        
        # MDS  embedding dataset
        clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
        t0 = timemodu.time()
        X_mds = clf.fit_transform(gdat.datatran)
        plot_embe(gdat, X_mds, 'mdse', "MDS embedding (time %.2fs)" % (timemodu.time() - t0))
        
        # Random Trees embedding dataset
        #hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0, max_depth=5)
        #t0 = timemodu.time()
        #X_transformed = hasher.fit_transform(gdat.datatran)
        #pca = decomposition.TruncatedSVD(n_components=2)
        #X_reduced = pca.fit_transform(X_transformed)
        #plot_embe(gdat, X_reduced, 'rfep', "Random forest embedding (time %.2fs)" % (timemodu.time() - t0))
        
        # Spectral embedding dataset
        embedder = manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack")
        t0 = timemodu.time()
        X_se = embedder.fit_transform(gdat.datatran)
        plot_embe(gdat, X_se, 'csep', "Spectral embedding (time %.2fs)" % (timemodu.time() - t0))
        
        # NCA projection dataset
        #nca = neighbors.NeighborhoodComponentsAnalysis(n_components=2, random_state=0)
        #t0 = timemodu.time()
        #X_nca = nca.fit_transform(gdat.datatran, y)
        #plot_embe(gdat, X_nca, 'ncap', "NCA embedding (time %.2fs)" % (timemodu.time() - t0))

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
            path = gdat.pathimag + '%s_%s.%s' % (strgvarb, gdat.strgcntp, gdat.strgfext)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
    

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
    path = gdat.pathimag + 'lspd_%s.' % (strgmask, gdat.strgfext)
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


def cnfg_obsdsing():
   
    listisec = [16]
    listicam = [1]
    listiccd = [1]
    init( \
         listisec=listisec, \
         listicam=listicam, \
         listiccd=listiccd, \
         #extrtype='qlop', \
         )


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


globals().get(sys.argv[1])()

