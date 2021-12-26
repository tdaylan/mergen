# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:50:41 2021

@author: Lindsey Gordon, Emma Chickles
learn_utils.py

Clustering algorithms
* run_kmeans
* run_dbscan
* run_hdbscan
* run_GMM
* run_LOF
* KNN_plotting
* dbscan_param_search
* load_paramscan_txt
* quick_hdbscan_param_search
* hdbscan_param_search
* lof_param_scan
* make_confusion_matrix
* optimize_confusion_matrix

Deep learning functions
* autoencoder_preprocessing
* bottleneck_preprocessing
* post_process
* param_summary
* model_summary_txt

Autoencoder
* pretrain
* convolutional_autoencoder
  * cae_encoder
  * cae_decoder
* deep_autoencoder
* variational_autoencoder
  * sampling
* vae_gan

Iterative training scheme
* split_reconstruction
* split_segments
* split_cae
* iterative_cae
* iterative_cae_clustering

Partitioning training and testing
* split_data_features
* split_data

Mock data
* gaussian
* signal_data
* no_signal_data
* get_high_freq_mock_data

Helper functions
* get_activations
* get_bottleneck
* compile_model
* Conv1DTranspose
* swish
* mean_cubic_loss


To Do List: 
    - Remove redundancy
    - Better function documentation
    - Clean up usability of param searches
    - Kmeans param scan?
    - Move imports to init
"""

import sklearn
import numpy as np
from sklearn.manifold import TSNE
import os
import pdb
import matplotlib.pyplot as plt
import numpy as np
# import plotting_functions as pf
from . import plot_utils as pt
# import data_functions as df
from . import data_utils as dt
from astropy.io import fits
from astropy.timeseries import LombScargle
import random
import time
from sklearn.cluster import KMeans    
import fnmatch as fm
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model

from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

#import talos

def run_kmeans(features, n_clusters = 2):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    labels = kmeans.labels_
    return kmeans, labels
    
def run_dbscan(features, eps=0.2, min_samples=5, metric= 'minkowski', 
               algorithm = 'auto', leaf_size = '30', p='2'):
    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=eps,min_samples=min_samples,metric=metric,
                algorithm=algorithm,leaf_size=leaf_size,p=p).fit(features)
    return db
    
def run_hdbscan(features, min_cluster_size = 10, metric = 'minkowski', min_samples = 10,
                p = '2', algorithm = 'best'):
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size),
                                metric=metric, min_samples=min_samples,
                                p=p, algorithm=algorithm)
    clusterer.fit(features)
    labels = clusterer.labels_
    return clusterer, labels
    
def run_gmm(ticid, feats, numclstr=100, runiter=False, numiter=1, save=True,
            savepath='./'):
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=numclstr, random_state=0).fit(feats)
    clstr = gmm.predict(feats)

    if save:
        if runiter:
            prefix = 'iteration'+str(numiter-1)+'-all-'
            suffix = '-n'+str(numclstr)+'.txt'
        else:
            prefix = ''
            suffix = '.txt'
        np.savetxt(savepath+prefix+'gmm_labels'+suffix, np.array([ticid, clstr]),
                   header='TICID,ClusterNumber')
    return clstr
    
def run_LOF(features, n_neighbors = 20, p = 2, metric = 'minkowski', contamination = 0.1,
            algorithm = 'auto'):
    from sklearn.neighbors import LocalOutlierFactor
    
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, p=p, metric=metric,
                             contamination=contamination, algorithm=algorithm)
    clf.fit_predict(features)
    negative_factor = clf.negative_outlier_factor_
    
    lof = -1 * negative_factor
    return lof
    
def run_tsne(features, n_components=2, perplexity=30, early_exaggeration=12,
             save=True, savepath='./'):
    '''Returns low-dimensional t-sidtributed Stochastic Neighbor Embedding to
    visualize high-deimsnional feature spaces. Using PCA to initially reduce
    the dimensionality will suppress some noise.'''
    from sklearn.manifold import TSNE
    print('Training tSNE...')
    X = TSNE(n_components=n_components).fit_transform(features)

    if save:
        hdr=fits.Header()
        hdu=fits.PrimaryHDU(X, header=hdr)
        hdu.writeto(savepath+'tsne.fits', overwrite=True)

    return X
    
def load_tsne_from_fits(ensbpath):
    with fits.open(ensbpath+'tsne.fits') as hdul:
        X = hdul[0].data
    return X

def label_clusters(ensbpath, sectors, ticid, clstr, totype, numtot, otdict):

    # >> classified inds
    inds = np.nonzero(totype != 'NONE')
    cm  = confusion_matrix(clstr[inds], numtot[inds])

    # >> make the matrix square so that we can apply linear_sum_assignment
    unqpot = np.unique(clstr[inds])         # >> unique predicted otypes
    unqtot = np.array(list(otdict.values()))  # >> unique true otypes
    unqtot = np.delete(unqtot, np.nonzero(unqtot=='NONE')[0][0])
    while len(unqpot) < len(cm):
        unqpot = np.append(unqpot, 'NONE')     
    while len(unqtot) < len(cm):
        unqtot = np.append(unqtot, 'NONE')

    # >> make confusion matrix diagonal by re-ordering columns
    row_ind, col_ind = linear_sum_assignment(-1*cm)
    cm = cm[:,col_ind]
    unqpot = unqpot[col_ind]

    # >> create a dictionary [cluster number, variability type]
    potd = {} # >> predicted otype dictionary
    for i in range(len(unqpot)):
        # >> check if there is a real label assigned
        if unqpot[i] != 'NONE':
            potd[int(unqpot[i])] = unqtot[i]
    for i in np.unique(clstr):
        if i not in list(potd.keys()):
            potd[i] = 'NONE'

    # >> create list of predicted otypes (potype)
    potype = []
    # fname = ensbpath+'Sector'+str(sector)+'-ticid_to_label.txt'
    fname = ensbpath+'s-'+'-'.join(np.unique(sectors).astype('str'))+\
            '-ticid_to_label.txt'
    with open(fname, 'w') as f:
        f.write('TICID,OTYPE,SECTOR\n')
        for i in range(len(ticid)):
            otype = potd[clstr[i]]
            potype.append(otype)
            f.write(str(ticid[i])+','+otype+','+str(sectors[i])+'\n')
    print('Saved '+fname)

    return potype


##### PARAM SCANS #####
def KNN_plotting(savepath, features, k_values):
    """ This is based on a metric for finding the best possible eps/minsamp
    value from the original DBSCAN paper (Ester et al 1996). By calculating the
    average distances to the k-nearest neighbors and plotting
    those values sorted, you can determine heuristically the best eps 
    value. It should be eps value = yaxis value of first valley, and min_samp = k.
    
    Parameters:
        * savepath: location for folder of plots to be saved into
        * features
        * k_values: list of k-values (min_samples) to test
            ie, [2,3,4,10]
    Returns: nothing"""
    folderpath = savepath + "/KNN_plotting/"
    try:
        os.makedirs(folderpath)
    except OSError:
        print ("Directory %s already exists" % folderpath)
    
    from sklearn.neighbors import NearestNeighbors
    for n in range(len(k_values)):
        neigh = NearestNeighbors(n_neighbors=k_values[n])
        neigh.fit(features)
        k_dist, k_ind = neigh.kneighbors(features, return_distance=True)

        avg_kdist_sorted = np.sort(np.mean(k_dist, axis=1))[::-1]
        
        plt.scatter(np.arange(len(features)), avg_kdist_sorted)
        plt.xlabel("Points")
        plt.ylabel("Average K-Neighbor Distance")
        plt.ylim((0, 30))
        plt.title("K-Neighbor plot for k=" + str(k_values[n]))
        plt.savefig(folderpath + "kneighbors-" +str(k_values[n]) +"-plot-sorted.png")
        plt.close()    
    return
def dbscan_param_search(bottleneck, time, flux, ticid, target_info,
                        eps=list(np.arange(0.1,1.5,0.1)),
                        min_samples=[5],
                        metric=['euclidean', 'manhattan', 'minkowski'],
                        algorithm = ['auto', 'ball_tree', 'kd_tree',
                                     'brute'],
                        leaf_size = [30, 40, 50],
                        p = [1,2,3,4],
                        output_dir='./', DEBUG=False, single_file=False,
                        simbad_database_txt='./simbad_database.txt',
                        database_dir='./databases/', pca=True, tsne=True,
                        confusion_matrix=True, tsne_clustering=True):
    '''Performs a grid serach across parameter space for DBSCAN. Calculates
    
    Parameters:
        * bottleneck : array with shape=(num light curves, num features)
            ** is this just the array of features? ^
        * eps, min_samples, metric, algorithm, leaf_size, p : all DBSCAN
          parameters
        * success metric : !!
        * output_dir : output directory, ending with '/'
        * DEBUG : if DEBUG, plots first 5 light curves in each class
        
    TODO : only loop over p if metric = 'minkowski'
    '''
   
    classes = []
    num_classes = []
    counts = []
    num_noisy= []
    parameter_sets=[]
    silhouette_scores=[]
    ch_scores = []
    db_scores = []
    accuracy = []
    param_num = 0
    p0=p

    with open(output_dir + 'dbscan_param_search.txt', 'a') as f:
        f.write('{} {} {} {} {} {} {} {} {} {} {}\n'.format("eps\t\t", "samp\t\t", "metric\t\t", 
                                                         "alg\t\t", "leaf\t", "p\t",
                                                         "classes\t",
                                                         "silhouette\t\t\t", 'ch\t\t\t', 
                                                         'db\t\t\t', 'acc\t'))

    for i in range(len(eps)):
        for j in range(len(min_samples)):
            for k in range(len(metric)):
                for l in range(len(algorithm)):
                    for m in range(len(leaf_size)):
                        if metric[k] == 'minkowski':
                            p = p0
                        else:
                            p = [None]

                        for n in range(len(p)):
                            db = DBSCAN(eps=eps[i],
                                        min_samples=min_samples[j],
                                        metric=metric[k],
                                        algorithm=algorithm[l],
                                        leaf_size=leaf_size[m],
                                        p=p[n]).fit(bottleneck)
                            #print(db.labels_)
                            print(np.unique(db.labels_, return_counts=True))
                            classes_1, counts_1 = \
                                np.unique(db.labels_, return_counts=True)
                                
                            #param_num = str(len(parameter_sets)-1)
                            title='Parameter Set '+str(param_num)+': '+'{} {} {} {} {} {}'.format(eps[i],
                                                                                        min_samples[j],
                                                                                        metric[k],
                                                                                        algorithm[l],
                                                                                        leaf_size[m],
                                                                                        p[n])
                            
                            prefix='dbscan-p'+str(param_num)                            
                                
                            if confusion_matrix:
                                print('Plotting confusion matrix')
                                acc = pt.plot_confusion_matrix(ticid, db.labels_,
                                                               database_dir=database_dir,
                                                               single_file=single_file,
                                                               output_dir=output_dir,
                                                               prefix=prefix)
                            else:
                                acc = np.nan
                            accuracy.append(acc)
                                
                            if len(classes_1) > 1:
                                classes.append(classes_1)
                                num_classes.append(len(classes_1))
                                counts.append(counts_1)
                                num_noisy.append(counts_1[0])
                                parameter_sets.append([eps[i], min_samples[j],
                                                       metric[k],
                                                       algorithm[l],
                                                       leaf_size[m],
                                                       p[n]])
                                
                                # >> compute silhouette
                                print('Computing silhouette score')
                                silhouette = silhouette_score(bottleneck,db.labels_)
                                silhouette_scores.append(silhouette)
                                
                                # >> compute calinski harabasz score
                                print('Computing calinski harabasz score')
                                ch_score = calinski_harabasz_score(bottleneck,
                                                                db.labels_)
                                ch_scores.append(ch_score)
                                
                                # >> compute davies-bouldin score
                                print('Computing davies-bouldin score')
                                dav_boul_score = davies_bouldin_score(bottleneck,
                                                             db.labels_)
                                db_scores.append(dav_boul_score)
                                
                            else:
                                silhouette, ch_score, dav_boul_score = \
                                    np.nan, np.nan, np.nan
                                
                            print('Saving results to text file')
                            with open(output_dir + 'dbscan_param_search.txt', 'a') as f:
                                f.write('{}\t\t {}\t\t {}\t\t {}\t {}\t \
                                        {}\t {}\t\t\t {}\t\t\t {}\t\t\t {}\t {}\n'.format(eps[i],
                                                                   min_samples[j],
                                                                   metric[k],
                                                                   algorithm[l],
                                                                   leaf_size[m],
                                                                   p[n],
                                                                   len(classes_1),
                                                                   silhouette,
                                                                   ch_score,
                                                                   dav_boul_score,
                                                                   acc))
                                
                            if DEBUG and len(classes_1) > 1:

                                print('Plotting classification results')
                                pt.quick_plot_classification(time, flux,
                                                             ticid,
                                                             target_info, bottleneck,
                                                             db.labels_,
                                                             path=output_dir,
                                                             prefix=prefix,
                                                             simbad_database_txt=simbad_database_txt,
                                                             title=title,
                                                             database_dir=database_dir,
                                                             single_file=single_file)
                                
                                
                                if pca:
                                    print('Plot PCA...')
                                    pt.plot_pca(bottleneck, db.labels_,
                                                output_dir=output_dir,
                                                prefix=prefix)
                                
                                if tsne:
                                    print('Plot t-SNE...')
                                    pt.plot_tsne(bottleneck, db.labels_,
                                                 output_dir=output_dir,
                                                 prefix=prefix)
                                # if tsne_clustering:
                                    
                                    
                            plt.close('all')
                            param_num +=1
    print("Plot paramscan metrics...")
    pt.plot_paramscan_metrics(output_dir+'dbscan-', parameter_sets, 
                              silhouette_scores, db_scores, ch_scores)
    #print(len(parameter_sets), len(num_classes), len(num_noisy), num_noisy)

    pt.plot_paramscan_classes(output_dir+'dbscan-', parameter_sets, 
                                  np.asarray(num_classes), np.asarray(num_noisy))

        
    return parameter_sets, num_classes, silhouette_scores, db_scores, ch_scores, accuracy
def load_paramscan_txt(path):
    """ load in the paramscan stuff from the text file
    returns: parameter sets, number of classes, metric scores (in order: silhouettte, db, ch)
    modified [lcg 07292020 - created]"""
    params = np.genfromtxt(path, dtype=(float, int, 'S10', 'S10', int, int, int, np.float32, np.float32, np.float32), names=['eps', 'minsamp', 'metric', 'algorithm', 'leafsize', 'p', 'numclasses', 'silhouette', 'ch', 'db'])
    
    params = np.asarray(params)
    nan_indexes = []
    for n in range(len(params)):
        if np.isnan(params[n][8]):
            nan_indexes.append(int(n))
        
    nan_indexes = np.asarray(nan_indexes)
    
    cleaned_params = np.delete(params, nan_indexes, axis=0)   

    number_classes = np.asarray(cleaned_params['numclasses'])
    metric_scores = np.asarray(cleaned_params[['silhouette', 'db', 'ch']].tolist())
    
    return cleaned_params, number_classes, metric_scores

def gmm_param_search(features, output_dir='./', n_components=[50, 100, 150],
                     tsne=None):
    
    from datetime import datetime

    out = output_dir+'gmm_param_search.txt'
    with open(out, 'w') as f:
        f.write('n_components,comp_time,Silhouette,Calinski-Harabasz,'+
                'Davies-Bouldin\n')
    print('Touch '+out)
    scores = []
    for i in range(len(n_components)): 
        start = datetime.now() # >> start timer

        gmm = GaussianMixture(n_components=n_components[i])
        labels = gmm.fit_predict(features) # >> assigns clusters

        end = datetime.now() # >> end timer
        dur_sec = (end-start).total_seconds()

        # >> compute silhouette score
        silhouette = sklearn.metrics.silhouette_score(features, labels)

        # >> compute calinski harabasz score
        ch_score = sklearn.metrics.calinski_harabasz_score(features, labels)

        # >> compute davies-bouldin score
        db_score = sklearn.metrics.davies_bouldin_score(features, labels)

        scores.append([silhouette, ch_score, db_score])

        with open(out, 'a') as f:
            f.write('{},{},{},{},{}\n'.format(n_components[i], dur_sec, silhouette,
                                              ch_score, db_score))


        if type(tsne) != type(None):
            pt.plot_tsne(features, labels, X=tsne, output_dir=output_dir,
                         prefix='gmm-ncomp'+str(n_components[i])+'-')

    print('Wrote '+out)

    scores = np.array(scores)
    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(n_components, scores[:,0], '.')
    ax[0,0].set_ylabel('Silhouette') # >> higher = better
    ax[0,0].set_xlabel('Number of Components')
    ind = np.argmax(scores[:,0])
    ax[0,0].plot([n_components[ind]], [scores[:,0][ind]], 'xr', label='best')
    ax[0,0].legend()

    ax[1,0].plot(n_components, scores[:,1], '.')
    ax[1,0].set_ylabel('Calinski-Harabasz') # >> higher = better
    ax[1,0].set_xlabel('Number of Components')
    ind = np.argmax(scores[:,1])
    ax[1,0].plot([n_components[ind]], [scores[:,1][ind]], 'xr', label='best')
    ax[1,0].legend()

    ax[0,1].plot(n_components, scores[:,2], '.')
    ax[0,1].set_ylabel('Davies-Bouldin') # >> lower = better
    ax[0,1].set_xlabel('Number of Components')
    ind = np.argmin(scores[:,2])
    ax[0,1].plot([n_components[ind]], [scores[:,2][ind]], 'xr', label='best')
    ax[0,1].legend()

    ax[1,1].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_dir+'gmm_performance.png')
    plt.close()
    print('Wrote '+output_dir+'gmm_performance.png')

    return n_components[np.argmax(scores[:,0])]


def quick_hdbscan_param_search(features, min_samples=[2,3,4,5,6,7,8,15,50],
                               min_cluster_size=[50,100,500,1000],
                               metric=['all'], p0=[1,2,3,4], output_dir='./',
                               tsne=None):
    
    import hdbscan
    from datetime import datetime

    with open(output_dir + 'hdbscan_param_search.txt', 'w') as f:
        f.write('count,num_clusters,comp_time,Silhouette,Calinski-Harabasz,'+\
                'Davies-Bouldin,min_cluster_size,min_samples,metric,p,'+\
                'num_noise\n')
    if metric[0] == 'all':
        metric = list(hdbscan.dist_metrics.METRIC_MAPPING.keys())
        metric.remove('seuclidean')
        metric.remove('mahalanobis')
        metric.remove('wminkowski')
        metric.remove('haversine')
        metric.remove('cosine')
        metric.remove('arccos')
        metric.remove('pyfunc')        
        
    count = 0
    scores = []

    for i in range(len(min_cluster_size)):
        for j in range(len(metric)):
            if metric[j] == 'minkowski':
                p = p0
            else:
                p = [None]
            for n in range(len(p)):
                for k in range(len(min_samples)):    
                    start = datetime.now() # >> start timer
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=\
                                                int(min_cluster_size[i]),
                                                metric=metric[j],
                                                min_samples=min_samples[k],
                                                p=p[n], algorithm='best')
                    clusterer.fit(features)
                    labels = clusterer.labels_

                    end = datetime.now() # >> end timer
                    dur_sec = (end-start).total_seconds()

                    classes, counts = np.unique(clusterer.labels_,
                                                return_counts=True)
                    
                    # >> compute silhouette score
                    silhouette = sklearn.metrics.silhouette_score(features,
                                                                  labels)
                    # >> compute calinski harabasz score
                    ch_score = sklearn.metrics.calinski_harabasz_score(features,
                                                                       labels)
                    # >> compute davies-bouldin score
                    db_score = sklearn.metrics.davies_bouldin_score(features,
                                                                    labels)
                    scores.append([silhouette, ch_score, db_score])
                    line = [str(count),str(len(np.unique(classes)-1)),
                            str(dur_sec),str(silhouette),str(ch_score),
                            str(db_score),str(min_cluster_size[i]),
                            str(min_samples[k]),str(metric[j]),str(p[n]),
                            str(counts[0])]
                    with open(output_dir + 'hdbscan_param_search.txt', 'a') as f:
                        f.write(','.join(line))

                    if type(tsne) != type(None):
                        pt.plot_tsne(features, clusterer.labels_, X=tsne,
                                     output_dir=output_dir,
                                     prefix='hdbscan-'+str(count)+'-')

                    count += 1
    

def hdbscan_param_search(features, time, flux, ticid, target_info,
                            min_cluster_size=list(np.arange(5,30,2)),
                            min_samples = [5,10,15],
                            metric=['euclidean', 'manhattan', 'minkowski'],
                            p0 = [1,2,3,4],
                            output_dir='./', DEBUG=False,
                            simbad_database_txt='./simbad_database.txt',
                            database_dir='./databases/',
                            pca=False, tsne=False, confusion_matrix=True,
                            single_file=False,
                            data_dir='./data/', save=False,
                            parents=[], labels=[]):
    '''Performs a grid serach across parameter space for HDBSCAN. 
    
    Parameters:
        * features
        * time/flux/ticids/target information
        * min cluster size, metric, p (only for minkowski)
        * output_dir : output directory, ending with '/'
        * DEBUG : if DEBUG, plots first 5 light curves in each class
        * optional to plot pca & tsne coloring for it
        
    '''
    import hdbscan         
    classes = []
    num_classes = []
    counts = []
    num_noisy= []
    parameter_sets=[]
    silhouette_scores=[]
    ch_scores = []
    db_scores = []    
    param_num = 0
    accuracy = []
    
    if metric[0] == 'all':
        metric = list(hdbscan.dist_metrics.METRIC_MAPPING.keys())
        metric.remove('seuclidean')
        metric.remove('mahalanobis')
        metric.remove('wminkowski')
        metric.remove('haversine')
        metric.remove('cosine')
        metric.remove('arccos')
        metric.remove('pyfunc')    

    with open(output_dir + 'hdbscan_param_search.txt', 'a') as f:
        f.write('{} {} {} {}\n'.format("min_cluster_size", "min_samples",
                                       "metric", "p", 'num_classes', 
                                       'silhouette', 'db', 'ch', 'acc'))

    for i in range(len(min_cluster_size)):
        for j in range(len(metric)):
            if metric[j] == 'minkowski':
                p = p0
            else:
                p = [None]
            for n in range(len(p)):
                for k in range(len(min_samples)):
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size[i]),
                                                metric=metric[j], min_samples=min_samples[k],
                                                p=p[n], algorithm='best')
                    clusterer.fit(features)
                    labels = clusterer.labels_
                    
                    if save:
                        hdr=fits.Header()
                        hdu=fits.PrimaryHDU(labels, header=hdr)
                        hdu.writeto(output_dir + 'HDBSCAN_res'+str(param_num)+'.fits',
                                    overwrite=True)
                    
                    print(np.unique(labels, return_counts=True))
                    classes_1, counts_1 = np.unique(labels, return_counts=True)
                            
                                    
                    
                    title='Parameter Set '+str(param_num)+': '+'{} {} {} {}'.format(min_cluster_size[i],
                                                                                 min_samples[k],
                                                                                 metric[j],p[n])
                                
                    prefix='hdbscan-p'+str(param_num)                            
                                    
                    if len(classes_1) > 1:
                        classes.append(classes_1)
                        num_classes.append(len(classes_1))
                        counts.append(counts_1)
                        num_noisy.append(counts_1[0])
                        parameter_sets.append([min_cluster_size[i],metric[j],p[n]])
                        print('Computing silhouette score')
                        silhouette = silhouette_score(features, labels)
                        silhouette_scores.append(silhouette)
                        
                        # >> compute calinski harabasz score
                        print('Computing calinski harabasz score')
                        ch_score = calinski_harabasz_score(features, labels)
                        ch_scores.append(ch_score)
                        
                        # >> compute davies-bouldin score
                        print('Computing davies-bouldin score')
                        dav_boul_score = davies_bouldin_score(features, labels)
                        db_scores.append(dav_boul_score)                        
                                    
                        if confusion_matrix:
                            print('Computing accuracy')
                            acc = pt.plot_confusion_matrix(ticid, labels,
                                                           database_dir=database_dir,
                                                           single_file=single_file,
                                                           output_dir=output_dir,
                                                           prefix=prefix)       
                            
                        else:
                            acc=None
                        accuracy.append(acc)
                                  
                                    
                    with open(output_dir + 'hdbscan_param_search.txt', 'a') as f:
                        f.write(' \t'.join(map(str, [min_cluster_size[i],
                                                     min_samples[k],
                                                     metric[j], p[n],
                                                     len(classes_1),
                                                     silhouette, ch_score,
                                                     dav_boul_score, acc])) + '\n')
                        # s = '{}\t {}\t {}\t {}\t {}\t {}\t {}\t {}\n'
                        # f.write(s.format(min_cluster_size[i], min_samples[k],
                        #                  metric[j], p[n], len(classes_1),
                        #                  silhouette, ch_score,
                        #                  dav_boul_score, acc))
                                    
                    if DEBUG and len(classes_1) > 1:
                        pt.quick_plot_classification(time, flux,ticid,target_info, 
                                                     features, labels,path=output_dir,
                                                     prefix=prefix,
                                                     title=title,
                                                     database_dir=database_dir,
                                                     single_file=single_file)
                    
                        pt.plot_cross_identifications(time, flux, ticid,
                                                      target_info, features,
                                                      labels, path=output_dir,
                                                      database_dir=database_dir,
                                                      data_dir=data_dir)
                        pt.plot_confusion_matrix(ticid, labels,
                                                  database_dir=database_dir,
                                                  single_file=single_file,
                                                  output_dir=output_dir,
                                                  prefix=prefix+'merge', merge_classes=True,
                                                  labels=[], parents=parents) 
                    
                        if pca:
                            print('Plot PCA...')
                            pt.plot_pca(features, labels,
                                        output_dir=output_dir,
                                        prefix=prefix)
                                    
                        if tsne:
                            print('Plot t-SNE...')
                            pt.plot_tsne(features,labels,
                                         output_dir=output_dir,
                                         prefix=prefix)                
                    plt.close('all')
                    param_num +=1

        
    return parameter_sets, num_classes, acc         
                  
def lof_param_scan(ticid_feat, features,n_neighbors=list(range(10,40,10)),
                   metric=['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
                            'manhattan', 'braycurtis', 'canberra', 'chebyshev',
                            'correlation', 'dice', 'hamming', 'jaccard',
                            'kulinski', 'minkowski',
                            'rogerstanimoto', 'russellrao', 
                            'sokalmichener', 'sokalsneath', 'sqeuclidean',
                            'yule'],
                   p0=[2,4], algorithm=['auto'],
                   contamination=list(np.arange(0.1, 0.5, 0.1)),
                   rare_classes=['BY', 'rot'],
                   output_dir='./', database_dir='./databases/'):
    
    # >> want to find LOF of rare stuff
    class_info = get_true_classifications(ticid_feat, database_dir=database_dir)
    ticid_rare = {}
    inds_rare = {}
    for label in rare_classes:
        ticid_rare[label] = []
    for i in range(len(class_info)):
        for label in rare_classes:
            if label in class_info[i][1]:
                ticid_rare[label].append(int(class_info[i][0]))
    for label in rare_classes:
        intersection, comm1, comm2 = np.intersect1d(ticid_feat, ticid_rare[label],
                                                    return_indices=True)
        inds_rare[label] = comm1
    
    with open(output_dir + 'lof_param_search.txt', 'a') as f:
        f.write('n_neighbors metric p algorithm contamination rare_LOF\n')
    for i in range(len(n_neighbors)):
        for j in range(len(metric)):
            if metric[j] == 'minkowski':
                p = p0
            else:
                p = [None]
            for k in range(len(p)):
                for l in range(len(algorithm)):
                    for m in range(len(contamination)):
                        clf = LocalOutlierFactor(n_neighbors=int(n_neighbors[i]),
                                                 metric=metric[j],
                                                 p=p[k], algorithm=algorithm[l],
                                                 contamination=contamination[m])
                        fit_predictor = clf.fit_predict(features)
                        negative_factor = clf.negative_outlier_factor_
                        lof = -1 * negative_factor
                        
                        with open(output_dir + 'lof_param_search.txt', 'a') as f:
                            f.write('{} {} {} {} {} '.format(int(n_neighbors[i]),
                                                            metric[j], p[k],
                                                            algorithm[l],
                                                            contamination[m]))                        
                        
                        # >> calculate average lof for the rare things
                        for label in rare_classes:
                            avg_lof = np.mean(lof[comm1])
                            with open(output_dir + 'lof_param_search.txt', 'a') as f:     
                                f.write(str(avg_lof) + ' ')
                        with open(output_dir + 'lof_param_search.txt', 'a') as f:         
                            f.write('\n')
    

def make_confusion_matrix(ticid_pred, ticid_true, y_true_labels, y_pred,
                          debug=False, output_dir='./'):
    from sklearn.metrics import confusion_matrix
    from scipy.optimize import linear_sum_assignment   
    import seaborn as sn
    
    # >> find intersection
    intersection, comm1, comm2 = np.intersect1d(ticid_pred, ticid_true,
                                                return_indices=True)
    ticid_pred = ticid_pred[comm1]
    y_pred = y_pred[comm1]
    ticid_true = ticid_true[comm2]
    y_tru_labels = y_true_labels[comm2]           
        
    columns = np.unique(y_pred).astype('str')

    y_true = []
    for i in range(len(ticid_true)):
        class_num = np.nonzero(y_true_labels == y_true_labels[i])[0][0]
        y_true.append(class_num)
    y_true = np.array(y_true).astype('int')    
    
    cm = confusion_matrix(y_true, y_pred)
    while len(columns) < len(cm):
        columns = np.append(columns, 'X')       
    while len(y_true_labels) < len(cm):
        y_true_labels = np.append(y_true_labels, 'X')     
        
    row_ind, col_ind = linear_sum_assignment(-1*cm)
    cm = cm[:,col_ind]       
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    return cm, accuracy
  
       
def optimize_confusion_matrix(ticid_pred, y_pred, database_dir='./',
                              num_classes=[10,15], num_iter=10):
    from itertools import permutations
    import random
    class_info = get_true_classifications(ticid_pred,
                                          database_dir=database_dir,
                                          single_file=False)  
    ticid_true = class_info[:,0].astype('int')
    classes = []
    for i in range(len(class_info)):
        for label in class_info[i][1].split('|'):
            if label not in classes:
                classes.append(label)
                
    accuracy = []
    for n in num_classes:
        combinations = list(permutations(classes))
        print('Number of combinations: ' + str(len(combinations)))
        for i in range(num_iter):
            labels = random.choice(combinations)
            
            
            ticid_new = []
            y_true = []
            for i in range(len(ticid_true)):
                for j in range(len(labels)):
                    if labels[j] in class_info[i][1] and \
                        ticid_true[i] not in ticid_new:
                        y_true.append(labels[j])
                        ticid_new.append(ticid_true[i])
                        
            y_true = np.array(y_true)
            ticid_true_new = np.array(ticid_new)  
            
            cm, acc = make_confusion_matrix(ticid_pred, ticid_true_new,
                                            labels, y_pred)
            print(labels)
            print('accuracy: ' + str(acc))
            accuracy.append(acc)
            

# def DAE_preprocessing(x, train_test_ratio=1, norm_type=None, ax=0):
#     '''Preprocesses data in preparation for
#     training a deep autoencoder.
#     Parameters:
#         * p : parameter dictionary
#         * ticid : list of TICIDs, shape=(num light curves)
#         * target_info : meta data (sector, cam, ccd, data_type, cadence) for each light
#                         curve, shape=(num light curves, 5)
#         * train_test_ratio : partition ratio. If 1, then no partitioning.
#     '''


#     if train_test_ratio < 1:
#         print('Partitioning data...')
#         x_train, x_test, y_train, y_test, flux_train, flux_test,\
#         ticid_train, ticid_test, target_info_train, target_info_test, time =\
#             split_data_features(flux, features, time, ticid, target_info,
#                                 train_test_ratio=train_test_ratio)
#     else:
#         x_train = x

#     if type(norm_type) == type(None):
#         print('No normalization performed...')
#     elif norm_type == 'standardization':
#         print('Standardizing feature vectors...')
#         x_train = dt.standardize(x_train, ax=ax)
#         if train_test_ratio < 1: x_test = dt.standardize(x_test, ax=ax)


#     if train_test_ratio < 1:
#         return x_train, x_test, flux_train, flux_test, \
#             ticid_train, ticid_test, target_info_train, target_info_test, freq, time

#     else:
#         return x_train
            

def autoencoder_preprocessing(flux, time, p, ticid=None, target_info=None,
                              sector=1, mock_data=False,
                              validation_targets=[219107776],
                              DAE=False, features=False,
                              norm_type='standardization', input_rms=True,
                              input_psd = True, load_psd=False, n_pgram=1000,
                              train_test_ratio=0.9, data_dir='./',
                              split=False, output_dir='./', prefix='',
                              use_tess_features=True,
                              use_tls_features=True,
                              use_rms=True, flux_plot=None,
                              concat_ext_feats=False):
    '''Preprocesses output from dt.load_data_from_metafiles
    Shuffles array.
    Parameters:
        * flux : array of light curves, shape=(num light curves, num points)
        * ticid : list of TICIDs, shape=(num light curves)
        * p : parameter dictionary
        * target_info : [sector, cam, ccd, data_type, cadence] for each light
                        curve, shape=(num light curves, 5)
        * validation_targets : list of TICIDs to move from the training set to
                               testing set [deprecated]
        * DAE : preprocessing for deep autoencoder. if True, the following is
          required:
          * features : feature vector, shape=(num light curves, num features)
          If DAE is False, then performs preprocessing for convolutional
          autoencoder, and uses the folllowing parameters:
          * norm_type : either standardization, median_normalization,
                        minmax_normalization, none
          * input_rms : calculate RMS before normalizing
    '''

    # -- shuffle array ---------------------------------------------------------
    print('Shuffling data...')
    inds = np.arange(len(flux))
    random.Random(4).shuffle(inds)
    flux = flux[inds]
    ticid = ticid[inds]
    target_info = np.array(target_info)[inds]
    if DAE:
        features = features[inds]
                
    # -- calculate RMS -----------------------------------------------------
    if input_rms:
        print('Calculating RMS..')
        rms = dt.rms(flux)
    else: rms_train, rms_test = False, False


    # -- normalize ---------------------------------------------------------
    if norm_type == 'standardization':
        print('Standardizing fluxes...')
        x = dt.standardize(flux)

    elif norm_type == 'median_normalization':
        print('Normalizing fluxes (dividing by median)...')
        x = dt.normalize(flux)

    elif norm_type == 'minmax_normalization':
        print('Normalizing fluxes (changing minimum and range)...')
        x = dt.normalize_minmax(flux)

    else:
        print('Light curves were not normalized!')
        x = flux

    # -- partitioning training and testing set -----------------------------
    print('Partitioning data...')
    flux_train, flux_test, x_train, x_test, y_train, y_test, ticid_train, \
        ticid_test, target_info_train, target_info_test, time, time_plot  = \
        split_data(flux, x, ticid, target_info, time, p,
                   train_test_ratio=train_test_ratio,
                   supervised=False)             

    if input_rms:
        rms_train = rms[:np.shape(x_train)[0]]
        rms_test = rms[-1 * np.shape(x_test)[0]:]

        # >> save the RMS and TICIDs to a fits file
        hdr = fits.Header()
        hdu = fits.PrimaryHDU(rms_train, header=hdr)
        hdu.writeto(output_dir+prefix+'rms_train.fits', overwrite=True)
        fits.append(output_dir+prefix+'rms_train.fits', ticid_train)
        hdr = fits.Header()
        hdu = fits.PrimaryHDU(rms_test, header=hdr)
        hdu.writeto(output_dir+prefix+'rms_test.fits', overwrite=True)
        fits.append(output_dir+prefix+'rms_test.fits', ticid_test)

    if split:
        orbit_gap = np.argmax(np.diff(time))
        # >> split x_train and x_test at orbit gap
        x_train = np.split(x_train, [orbit_gap], axis=1)
        x_test = np.split(x_test, [orbit_gap], axis=1)

    # -- get other external features ---------------------------------------

    if concat_ext_feats:
        external_features_train, flux_train, ticid_train, target_info_train = \
            bottleneck_preprocessing(sector, x_train, ticid_train,
                                     target_info_train, rms_train,
                                     data_dir=data_dir,
                                     output_dir=output_dir,
                                     use_learned_features=False,
                                     use_tess_features=use_tess_features,
                                     use_engineered_features=False,
                                     use_tls_features=use_tls_features,
                                     log=False)  
        x_train = [x_train, external_features_train]
        external_features_test, flux_test, ticid_test, target_info_test = \
            bottleneck_preprocessing(sector, x_test, ticid_test,
                                     target_info_test, rms_test,
                                     data_dir=data_dir,
                                     output_dir=output_dir,
                                     use_learned_features=False,
                                     use_tess_features=use_tess_features,
                                     use_engineered_features=False,
                                     use_tls_features=use_tls_features,
                                     use_rms=use_rms,
                                     log=False)  
        x_test = [x_test, external_features_test]

    return flux_train, flux_test, x_train, x_test, ticid_train, ticid_test,\
        target_info_train, target_info_test, rms_train, rms_test, time, time_plot



def bottleneck_preprocessing(sector, flux, ticid, target_info,
                             rms=None,
                             output_dir='./SectorX/', prefix='',
                             data_dir='./', bottleneck_dir='./',
                             use_learned_features=False,
                             use_engineered_features=True,
                             use_tess_features=True,
                             use_tls_features=True,
                             use_rms=True, norm=False,
                             cams=[1,2,3,4], ccds=[1,2,3,4], log=False):
    '''Concatenates features (assumes features are already calculated and
    saved).
    
    Parameters:
        * sector : given as int TOOD: list of sector numbers
        * flux : array of light curves, shape=(num light curves, num data points)
        * ticid : list of TICIDs (given as int) for sector
        * target_info : list of [sector, cam, ccd, data_type cadence] for each
                        light curve
        * output_dir : output directory, containing CAE and DAE dirs
        * data_dir : directory containing _lightcurves.fits and _features*.fits
        * learned_features, engineered_features, tess_features : feature
          vectors with shape=(num light curves, num features). Will
          concatenate feature vectors if more than one is True.
              * learned_features : bottleneck of the convolutional autoencoder
              * engineered_features : custom features (e.g. kurtosis, skew)
              * tess_features : Teff, mass, rad, GAIAmag, d
              
    Returns feature vector for every light curve in TICID.
    '''
    
    features = []
    
    if use_engineered_features:
        fname = data_dir + 'Sector'+str(sector)+'_features_v0_all.fits'
        with fits.open(fname) as hdul:        
            engineered_feature_vector = hdul[0].data
            engineered_feature_ticid = hdul[1].data
        # >> re-arrange so that engineered_feature_ticid[i] = ticid[i]
        tmp = []
        for i in range(len(ticid)):
            ind = np.nonzero(engineered_feature_ticid == ticid[i])
            tmp.append(engineered_feature_vector[ind])
        engineered_feature_vector = np.array(tmp).reshape((len(ticid), 16))
        features.append(engineered_feature_vector)
            
    if use_learned_features:
        with fits.open(bottleneck_dir+prefix+'bottleneck_train.fits') as hdul:
            bottleneck_train = hdul[0].data
            ticid_bottleneck_train = hdul[1].data
        with fits.open(bottleneck_dir+prefix+'bottleneck_test.fits') as hdul:
            bottleneck_test = hdul[0].data
            ticid_bottleneck_test = hdul[1].data
        learned_feature_vector = np.concatenate([bottleneck_train,
                                                 bottleneck_test], axis=0)
        ticid_learned = np.concatenate([ticid_bottleneck_train,
                                        ticid_bottleneck_test])
        
        intersection, comm1, comm2 = np.intersect1d(ticid_learned, ticid,
                                                    return_indices=True)
        
        flux = flux[comm2]
        ticid = ticid[comm2]
        target_info = target_info[comm2]
        learned_feature_vector = learned_feature_vector[comm1]
        features.append(learned_feature_vector)
        
    if use_tls_features:
        # !! TODO: simplify function, by calling helper function
        # >> load all data
        engineered_features_v1 = []
        for cam in cams:
            for ccd in ccds:
                fname = data_dir+'Sector'+str(sector)+'/'+'Sector'+\
                    str(sector)+'Cam'+str(cam)+'CCD'+\
                    str(ccd)+ '/'+'Sector'+str(sector)+'Cam'+str(cam)+'CCD' + \
                        str(ccd) + '_features_v1.fits'
                with fits.open(fname) as hdul:        
                    engineered_features_v1.append(hdul[0].data)
                    
        # >> concatenate
        engineered_features_v1 = np.concatenate(engineered_features_v1, axis=0)
            
        # >> take out any light curves with nans
        inds = np.nonzero(np.prod(~np.isnan(engineered_features_v1), axis=1))
        engineered_features_v1 = engineered_features_v1[inds]
        target_info_v1 = target_info[inds]
        flux_v1 = flux[inds]
        ticid_v1 = ticid[inds]   
        
        features.append(engineered_features_v1)
        
        if use_learned_features:
            intersection, comm1, comm2 = np.intersect1d(ticid_v1, ticid,
                                                        return_indices=True)
            target_info = target_info_v1[comm1]
            ticid = ticid_v1[comm1]
            
            flux = flux_v1[comm1]
            learned_feature_vector = learned_feature_vector[comm2]
            features = []
            features.append(learned_feature_vector)
            features.append(engineered_features_v1)
        
        
    if use_tess_features:
        tess_features = pd.read_csv(data_dir+'Sector'+str(sector)+\
                                  '/Sector'+str(sector)+'tic_cat_all.csv')
        ticid_tess = tess_features['ID'].to_numpy()
        columns = ['Teff', 'rad', 'mass', 'GAIAmag', 'd']
        tess_features = tess_features[columns].to_numpy()
        
        # tess_features = np.loadtxt(data_dir + 'Sector'+str(sector)+\
        #                            '/tess_features_sector'+str(sector)+'.txt',
        #                            delimiter=' ', usecols=[1,2,3,4,5,6,8])

        # >> take out any light curves with nans
        inds = np.nonzero(np.prod(~np.isnan(tess_features), axis=1))
        tess_features = tess_features[inds]
        ticid_tess = ticid_tess[inds]
        intersection, comm1, comm2 = np.intersect1d(ticid_tess, ticid,
                                                    return_indices=True)
        # >> take intersection, and get rid of TICID column
        tess_features = tess_features[comm1]
        if log:
            tess_features = np.log(tess_features)        
        features = []
        features.append(tess_features)
        
        target_info = target_info[comm2]
        flux = flux[comm2]
        ticid = intersection
        
        if use_engineered_features:
            engineered_feature_vector = engineered_feature_vector[comm2]
            features.append(engineered_feature_vector)
        if use_learned_features:
            learned_feature_vector = learned_feature_vector[comm2]
            features.append(learned_feature_vector)
        if use_rms:
            rms = rms.reshape(-1, 1)
            rms = rms[comm2]
            if log:
                rms = np.log(rms)            
            features.append(rms)
        if use_tls_features:
            engineered_features_v1 = engineered_features_v1[comm2]
            features.append(engineered_features_v1)
        
    if use_rms and not use_tess_features:
        if log:
            rms = np.log(rms)
        rms = rms.reshape(-1, 1)
        features.append(rms)
    
    # >> concatenate features
    features = np.concatenate(features, axis=1)
        
    # >> standardize each feature
    if norm:
        features = dt.standardize(features, ax=0)
            
    return features, flux, ticid, target_info

def load_bottleneck_from_fits(bottleneck_dir, ticid, runIter=False, numIter=1):
    print("Loading CAE-learned features...")

    if runIter: # !! TODO: deal with cases numIter > 1
        prefix = 'iteration'+str(numIter-1)+'-'
    else:
        prefix = ''
    with fits.open(bottleneck_dir+prefix+'bottleneck_train.fits') as hdul:
        bottleneck_train = hdul[0].data
        ticid_bottleneck_train = hdul[1].data
    with fits.open(bottleneck_dir+prefix+'bottleneck_test.fits') as hdul:
        bottleneck_test = hdul[0].data
        ticid_bottleneck_test = hdul[1].data
    learned_feature_vector = np.concatenate([bottleneck_train,
                                             bottleneck_test], axis=0)
    ticid_learned = np.concatenate([ticid_bottleneck_train,
                                    ticid_bottleneck_test])

    sorted_inds = np.argsort(ticid)
    # >> intersect1d returns sorted arrays, so
    # >> ticid == ticid[sorted_inds][np.argsort(sorted_inds)]
    new_inds = np.argsort(sorted_inds)
    _, comm1, comm2 = np.intersect1d(ticid, ticid_learned, return_indices=True)
    learned_feature_vector = learned_feature_vector[comm2][new_inds]

    return learned_feature_vector
        
def load_DAE_bottleneck(savepath,):
    print("Loading DAE-learned features...")
    fnames = [f for f in os.listdir(savepath) if '_bottleneck_train.npy' in f]
    fnames.sort()
    bottleneck_train = []
    for fname in fnames:
        print('Loading '+fname)
        bottleneck_train.extend(np.load(savepath+fname))    
    return np.array(bottleneck_train)

def load_reconstructions(output_dir, ticid):
    filo = fits.open(output_dir + 'x_predict_train.fits')
    rcon = filo[0].data
    ticid_filo = filo[1].data

    sorted_inds = np.argsort(ticid)
    new_inds = np.argsort(sorted_inds)
    _, comm1, comm2 = np.intersect1d(ticid, ticid_filo, return_indices=True)
    rcon = rcon[comm2][new_inds]

    return rcon

def load_gmm_from_txt(output_dir, ticid, runIter=False, numIter=1,
                      numClusters=100):
    print('Loading GMM clustering results...')

    if runIter: # !! TODO: deal with cases numIter > 1
        prefix = 'iteration'+str(numIter-1)+'-all-'
        suffix = '-n'+str(numClusters)+'.txt'
    else:
        prefix = ''
        suffix = '.txt'
    txt = np.loadtxt(output_dir+prefix+'gmm_labels'+suffix)
    ticid_cluster = txt[0].astype('int')
    clusters = txt[1]

    # sorted_inds = np.argsort(ticid)
    # # >> intersect1d returns sorted arrays, so
    # # >> ticid == ticid[sorted_inds][np.argsort(sorted_inds)]
    # new_inds = np.argsort(sorted_inds)
    # _, comm1, comm2 = np.intersect1d(ticid, ticid_cluster, return_indices=True)
    # clusters = clusters[comm2][new_inds]

    match = []
    for i in range(len(ticid)):
        match.append(ticid_cluster[i] == ticid[i])
    if len(ticid) != np.count_nonzero(np.array(match)):
        print('!!! Missing '+str(len(ticid)-np.count_nonzero(np.array(match)))+\
             ' TICIDs')

    return clusters.astype('int')

def post_process(x, x_train, x_test, ticid_train, ticid_test, target_info_train,
                 target_info_test,
                 p, output_dir, sectors, prefix='', data_dir='./data/',
                 database_dir='./', bottleneck_dir='',
                 cams=[1,2,3,4], ccds=[1,2,3,4],
                 use_learned_features=True,
                 use_tess_features=False, use_engineered_features=False,
                 use_tls_features=False, log=False,
                 momentum_dump_csv='./Table_of_momentum_dumps.csv',
                 run_hdbscan=False, min_cluster_size=3, min_samples=3, power=4,
                 algorithm='auto', leaf_size=30, run_dbscan=False,
                 metric='canberra', run_gmm=True, classification_param_search=False,
                 plot_feat_space=False, DAE=False, DAE_hyperparam_opt=False,
                 VAE=True,
                 novelty_detection=True, classification=True,
                 features=None, flux_feat=None, ticid_feat=None, info_feat=None,
                 x_predict=None, do_diagnostic_plots=True, do_summary=True,
                 use_rms=False, n_components=100, n_tot=20, p_DAE={}):
    if type(features) == type(None):
        features, flux_feat, ticid_feat, info_feat = \
            bottleneck_preprocessing(sectors, np.concatenate([x_train, x_test], axis=0),
                                     np.concatenate([ticid_train, ticid_test]),
                                     np.concatenate([target_info_train,
                                                     target_info_test]),
                                     data_dir=data_dir, prefix=prefix,
                                     bottleneck_dir=bottleneck_dir,
                                     output_dir=output_dir,
                                     use_learned_features=use_learned_features,
                                     use_tess_features=use_tess_features,
                                     use_engineered_features=use_engineered_features,
                                     use_tls_features=use_tls_features,
                                     norm=True, use_rms=use_rms,
                                     cams=cams, ccds=ccds, log=log)
        print('Created feature space with dimensions: '+str(features.shape))

    if plot_feat_space:
        print('Plotting feature space')
        pt.latent_space_plot(features, output_dir+prefix+'feature_space.png')

    # -- deep autoencoder ------------------------------------------------------

    if DAE or VAE:
        if DAE_hyperparam_opt:
            t = talos.Scan(x=features,
                            y=features,
                            params=p_DAE,
                            model=deep_autoencoder,
                            experiment_name='DAE', 
                            reduction_metric = 'val_loss',
                            minimize_loss=True,
                            reduction_method='correlation',
                            fraction_limit = 0.1)            
            analyze_object = talos.Analyze(t)
            data_frame, best_param_ind,p_best = \
                pt.hyperparam_opt_diagnosis(analyze_object, output_dir,
                                            supervised=False) 
            p_DAE=p_best
            p_DAE['epochs'] = 100

        else:
            p_DAE = {'max_dim': 50, 'step': 4, 'latent_dim': 30,
                     'activation': 'elu', 'last_activation': 'elu',
                     'optimizer': 'adam', 'batch_norm': True,
                     'lr':0.001, 'epochs': 20, 'losses': 'mean_squared_error',
                     'batch_size': 128, 'initializer': 'glorot_uniform',
                     'fully_conv': False}    

        if DAE:
            suffix = '_DAE'

            if os.path.exists(output_dir+prefix+'feature_space_DAE.fits'):
                with fits.open(output_dir+prefix+'feature_space_DAE.fits') as f:
                    features = f[0].data
            else:
                history_DAE, model_DAE = deep_autoencoder(features, features,
                                                          None, None, p_DAE)
                new_features = get_bottleneck(model_DAE, features, p_DAE)
                features=new_features
                hdu = fits.PrimaryHDU(features)
                hdu.writeto(output_dir+prefix+'feature_space_DAE.fits',
                            overwrite=True)
                pt.epoch_plots(history_DAE, p_DAE, output_dir)
        elif VAE:
            suffix = '_VAE'
            if os.path.exists(output_dir+prefix+'feature_space_VAE.fits'):
                with fits.open(output_dir+prefix+'feature_space_VAE.fits') as f:
                    features = f[0].data
            else:
                history_DAE, model_DAE, encoder = \
                    variational_autoencoder(features, features, None, None, p_DAE)
                new_features = encoder.predict(features)
                features = new_features[2]
                hdu = fits.PrimaryHDU(features)
                hdu.writeto(output_dir+prefix+'feature_space'+suffix+'.fits',
                            overwrite=True)
                pt.epoch_plots(history_DAE, p_DAE, output_dir)

        if plot_feat_space:
            print('Plotting feature space')
            pt.latent_space_plot(features, output_dir+prefix+'feature_space'+\
                                 suffix+'.png')

    # -- novelty detection -----------------------------------------------------

    if novelty_detection:
        momentum_dump_csv=data_dir+'Table_of_momentum_dumps.csv' # !!
        pt.plot_lof(x, flux_feat, ticid_feat, features, 20,
                    output_dir,
                    n_tot=n_tot, target_info=info_feat, prefix=prefix,
                    database_dir=database_dir, debug=True,
                    log=True, n_pgram=1000,
                    plot_psd=True, momentum_dump_csv=momentum_dump_csv)       

        # !! 
        # pt.plot_lof_summary(x, flux_feat, ticid_feat, features, 20,
        #                     output_dir+prefix, target_info=info_feat,
        #                     database_dir=database_dir, 
        #                     momentum_dump_csv=momentum_dump_csv)

    # -- classification --------------------------------------------------------

    if classification:

        # -- dbscan ------------------------------------------------------------
        if run_dbscan:
            if classification_param_search:
                dt.KNN_plotting(output_dir+prefix, features, [10, 20, 100])

                print('DBSCAN parameter search')
                parameter_sets, num_classes, silhouette_scores, db_scores, \
                ch_scores, acc = \
                dt.dbscan_param_search(features, x, flux_feat, ticid_feat,
                                       info_feat, DEBUG=True, 
                                       output_dir=output_dir+prefix, 
                                       leaf_size=[30], algorithm=['auto'],
                                       min_samples=[5],
                                       metric=['minkowski'], p=[3,4],
                                       database_dir=database_dir,
                                       eps=list(np.arange(1.5, 4., 0.1)),
                                       confusion_matrix=True, pca=True,
                                       tsne=True,
                                       tsne_clustering=False)      

                print('Classification with best parameter set')
                best_ind = np.argmax(silhouette_scores)
                eps, min_samples, metric, algorithm, leaf_size, power = \
                    parameter_sets[best_ind]
        elif run_dbscan:
            from sklearn.cluster import DBSCAN
            db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
                        algorithm=algorithm, leaf_size=leaf_size,
                        power=power).fit(features)
            
        # -- hdbscan -----------------------------------------------------------
        if run_hdbscan:
            if classification_param_search:
                print('HDBSCAN parameter search')
                acc = dt.hdbscan_param_search(features, x, flux_feat, ticid_feat,
                                              info_feat, output_dir=output_dir,
                                              p0=[3,4], single_file=single_file,
                                              database_dir=database_dir, metric=['all'],
                                              min_samples=[3], min_cluster_size=[3],
                                              data_dir=data_dir)

            if os.path.exists(output_dir+prefix+'hdbscan_labels.txt'):
                _, labels = np.loadtxt(output_dir+prefix+'hdbscan_labels.txt')
            else:
                import hdbscan
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                                            min_samples=min_samples,
                                            metric=metric).fit(features)
                labels = clusterer.labels_
                np.savetxt(output_dir+prefix+'hdbscan_labels.txt',
                           np.array([ticid_feat, labels]))
                
        # -- GMM ---------------------------------------------------------------
        if run_gmm:
            print('Training GMM with '+str(n_components)+' components...')
            gmm = GaussianMixture(n_components=n_components)
            labels = gmm.fit_predict(features)
            np.savetxt(output_dir+prefix+'gmm_labels.txt',
                       np.array([ticid_feat, labels]))
        
        pt.classification_plots(features, x, flux_feat, ticid_feat, info_feat,
                                labels, output_dir=output_dir, prefix=prefix,
                                data_dir=data_dir,
                                x_predict=x_predict, do_summary=do_summary,
                                do_diagnostic_plots=do_diagnostic_plots)
            
def param_summary(history, x_train, x_test, x_predict_train, x_predict, p,
                  output_dir, param_set_num,
                  title, supervised=False, y_test=False):
    '''Saves text file *param_summary.txt with model parameters and metrics.'''
    with open(output_dir + 'param_summary.txt', 'a') as f:
        f.write('parameter set ' + str(param_set_num) + ' - ' + title +'\n')
        f.write(str(p.items()) + '\n')
        if supervised:
            label_list = ['loss', 'accuracy', 'precision', 'recall']
            key_list =['loss', 'accuracy', list(history.history.keys())[-2],
                    list(history.history.keys())[-1]]
        else:
            label_list = ['loss']
            key_list = ['loss']

        for j in range(len(label_list)):
            f.write(label_list[j]+' '+str(history.history[key_list[j]][-1])+\
                    '\n')
        if supervised:
            y_predict = np.argmax(x_predict, axis=-1)
            y_true = np.argmax(y_test, axis=-1)
            cm = confusion_matrix(y_predict, y_true)
            chi_2 = np.average((x_predict-x_test)**2 / 0.02)
            f.write('confusion matrix\n')
            f.write(str(cm))
            f.write('\ny_true\n')
            f.write(str(y_true)+'\n')
            f.write('y_predict\n')
            f.write(str(y_predict)+'\n')
        else:
            if type(x_predict) != type(None):
                mse = np.average((x_predict - x_test)**2)
                f.write('testing mse '+ str(mse) + '\n')
            if type(x_predict_train) != type(None):
                mse = np.average((x_predict_train - x_train)**2)
                f.write('training mse '+ str(mse) + '\n')

        f.write('\n')
    
def model_summary_txt(output_dir, model):
    with open(output_dir + 'model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda line: f.write(line + '\n'))

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Autoencoders ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def save_autoencoder_products(model, params, batch_fnames=None, output_dir='',
                              prefix='', x_train=None, x_test=None, 
                              ticid_train=None, ticid_test=None):

    from datetime import datetime

    if type(x_train) == type(None): # >> load x_train in batches
        bottleneck_train = []
        for i in range(len(batch_fnames)):
            print('Loading '+batch_fnames[i])
            chunk = np.load(batch_fnames[i])

            n_batch = chunk.shape[0] // params['batch_size'] # !! 'batch'
            bottleneck_chunk = []
            x_predict_chunk = []
            for n in range(n_batch):
                start=datetime.now()

                if n == n_batch-1:
                    batch = chunk[n*params['batch_size']:]
                else:
                    batch = chunk[n*params['batch_size']:(n+1)*params['batch_size']]

                print('Retrieving bottlneck...')
                bottleneck_chunk.extend(get_bottleneck(model, batch, params,
                                                       save=False))

                print('Retrieving reconstructions...')
                x_predict_chunk.extend(model.predict(batch))

                end = datetime.now()
                print((end-start).total_seconds())

            np.save(output_dir+prefix+'chunk%02d'%i+'_bottleneck_train.npy',
                    np.array(bottleneck_chunk))
            np.save(output_dir+prefix+'chunk%02d'%i+'_x_predict_train.npy',
                    np.array(x_predict_chunk))
            bottleneck_train.extend(bottleneck_chunk)

        return np.array(bottleneck_train)

    else: # >> x_train already in memory
        print('Retrieving bottlneck...')
        bottleneck_train = \
            get_bottleneck(model, x_train, params, save=True, ticid=ticid_train,
                           out=output_dir+prefix+'bottleneck_train.fits')

        print('Retrieving reconstructions...')
        x_predict_train = model.predict(x_train)      
        hdr = fits.Header()
        hdu = fits.PrimaryHDU(x_predict_train, header=hdr)
        hdu.writeto(output_dir+prefix+'x_predict_train.fits', overwrite=True)
        fits.append(output_dir+prefix+'x_predict_train.fits', ticid_train)
        return model, history, bottleneck_train, x_predict_train

    if type(x_test) != type(None): # >> if data patitioned 
        bottleneck_test = \
        get_bottleneck(model, x_test, params, save=True, ticid=ticid_test,
                       out=output_dir+prefix+'bottleneck_test.fits')    
        x_predict_test = model.predict(x_test)     
        hdr = fits.Header()
        if concat_ext_feats:
            hdu = fits.PrimaryHDU(x_predict_test[0], header=hdr)
        else:
            hdu = fits.PrimaryHDU(x_predict_test, header=hdr)
        hdu.writeto(output_dir+prefix+'x_predict.fits', overwrite=True)

        fits.append(output_dir+prefix+'x_predict.fits', ticid_test)
        return model, history, bottleneck_train, bottleneck_test,\
            x_predict_train, x_predict_test



def read_hyperparameters_from_txt(parampath):

    with open(parampath, 'r') as f:
        lines = f.readlines()
        params = {}
        for line in lines[1:]:
            key = line.split(': ')[0]
            val = line.split(': ')[1][:-1]
            try:
                val = float(val)
                try:
                    if int(val) == float(val):
                        val = int(val)
                except: pass
            except: 
                if val == 'None':
                    val = None
                elif val == 'True':
                    val = True
                elif val == 'False':
                    val = False
            params[key] = val
    return params

def truncate(params):
    if 'pool_size' in params.keys(): # >> CNN
        reduction_factor = np.max(params['pool_size'])*\
                           np.max(params['strides'])**\
                           np.max(params['num_consecutive']) 
        num_iter = np.max(params['num_conv_layers'])/2
        tot_reduction_factor = reduction_factor**num_iter
        new_length = int(params['n_features'] / tot_reduction_factor)*\
                     int(tot_reduction_factor)
    else: new_length = params['n_features']
    return new_length

def generate_batches(files, params):
    '''Run with
    train_files = [train_bundle_loc + "bundle_" + cb.__str__() for cb \
                   in range(nb_train_bundles)]
    gen = generate_batches(files=train_files, batch_size=batch_size)
    history = model.fit_generator(gen, samples_per_epoch=samples_per_epoch,
                                  nb_epoch=num_epoch,verbose=1,
                                  class_weight=class_weights)
    '''


    counter = 0
    new_length = truncate(params)
    while True:
        fname = files[counter]
        print(fname)
        counter = (counter + 1) % len(files)
        X_train = np.load(fname)
        for cbatch in range(0, X_train.shape[0], params['batch_size']):
            # pdb.set_trace()
            yield (X_train[cbatch:(cbatch + params['batch_size']),:new_length],
                   X_train[cbatch:(cbatch + params['batch_size']),:new_length])

# class My_Custom_Generator(keras.utils.Sequence) :
  
#     def __init__(self, image_filenames, labels, batch_size) :
#         self.image_filenames = image_filenames
#         self.labels = labels
#         self.batch_size = batch_size
    
#     def __len__(self) :
#         return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
#     def __getitem__(self, idx) :
#         batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
#         batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
#         return np.array([
#                 resize(imread('/content/all_images/' + str(file_name)), (80, 80, 3))
#                    for file_name in batch_x])/255.0, np.array(batch_y)

# def pretrain(p, output_dir, input_dim=18688, input_psd=True, input_rms=False,
#              dataset_size=10000, f_mean=2., truncate=True, reshape=False,
#              hyperparam_opt=False):
#     # !! unfinished
#     # >> make some mock data
#     x, x_train, x_test = \
#         get_high_freq_mock_data(p=p, dataset_size=dataset_size,
#                                 input_dim=input_dim, f_mean=f_mean,
#                                 truncate=truncate, reshape=reshape,
#                                 hyperparam_opt=hyperparam_opt)    

class TimeHistory(keras.callbacks.Callback):
    '''https://stackoverflow.com/questions/43178668/record-the-computation-time-
    for-each-epoch-in-keras-during-model-fit'''
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def conv_autoencoder(x_train, y_train, x_test=None, y_test=None, params=None, 
                     validation=False, split=False, input_features=False,
                     features=None, input_psd=False, save_model_epoch=False,
                     model_init=None, save_model=True, save_bottleneck=True,
                     predict=True, output_dir='./', prefix='',
                     input_rms=False, rms_train=None, rms_test=None,
                     ticid_train=None, ticid_test=None,
                     train=True, weights_path='./best_model.hdf5',
                     concat_ext_feats=False,
                     batch_fnames=None, report_time=True):
    
    if type(params) == str:
        with open(params, 'r') as f:
            lines = f.readlines()
            params = {}
            for line in lines[1:]:
                key = line.split(': ')[0]
                val = line.split(': ')[1][:-1]
                try:
                    val = float(val)
                    try:
                        if int(val) == float(val):
                            val = int(val)
                    except: pass
                except: 
                    if val == 'None':
                        val = None
                    elif val == 'True':
                        val = True
                    elif val == 'False':
                        val = False
                params[key] = val
    if 'n_features' not in params.keys() and type(batch_fnames) == type(None):
        params['n_features'] = x_train.shape[1]
    else:
        x_train = np.load(batch_fnames[0])
        params['n_features'] = x_train.shape[1]
        x_train = None
    
    if report_time:
        from datetime import datetime
        start = datetime.now()
        start_tot = datetime.now()

    # -- encoding -------------------------------------------------------------
    encoded = cae_encoder(x_train, params)

    if report_time:
        end = datetime.now()
        dur_sec = (end-start).total_seconds()
        with open(output_dir+prefix+'model_time.txt', 'w') as f:
            f.write('Time to add encoder model: '+str(dur_sec)+' (s)\n')
        start = end

    # -- decoding -------------------------------------------------------------
    decoded = cae_decoder(x_train, encoded.output, params)

    if report_time:
        end = datetime.now()
        dur_sec = (end-start).total_seconds()
        with open(output_dir+prefix+'model_time.txt', 'a') as f:
            f.write('Time to add decoder model: '+str(dur_sec)+' (s)\n')
        start = end
        
    model = Model(encoded.input, decoded)
    print(model.summary())
    model_summary_txt(output_dir, model)
    
    # -- initialize weights ---------------------------------------------------
    if type(model_init) != type(None):
        print('Re-initializing weights')
        model1 = keras.models.load_model(model_init, custom_objects={'tf': tf}) 
        conv_inds1 = np.nonzero(['conv' in x.name for x in model1.layers])[0]
        conv_inds2 = np.nonzero(['conv' in x.name for x in model.layers])[0]
        dense_inds1 = np.nonzero(['dense' in x.name for x in model1.layers])[0]
        dense_inds2 = np.nonzero(['dense' in x.name for x in model.layers])[0]        
        for i in range(len(conv_inds1)):
            model.layers[conv_inds2[i]].set_weights(model1.layers[conv_inds1[i]].get_weights())
        for i in range(len(dense_inds1)):
            model.layers[conv_inds2[i]].set_weights(model1.layers[conv_inds1[i]].get_weights())
        

    # # !! tmp
    # input_dim = params['n_features']
    # input_img = Input(shape = (input_dim,))
    # x = Reshape((input_dim, 1))(input_img)
    # x = MaxPooling1D(params['pool_size'], padding='same')(x)
    # x = MaxPooling1D(params['pool_size'], padding='same')(x)
    # x = Conv1D(16, 5, padding='same')(x)
    # x = MaxPooling1D(params['pool_size'], padding='same')(x)
    # x = Flatten()(x)
    # x = Dense(params['latent_dim'])(x)
    # x = Dense(200000)(x)
    # x = Reshape((12500, 16))(x)
    # x = UpSampling1D(params['pool_size'])(x)
    # x = Conv1D(1, 5, padding='same')(x)
    # x = UpSampling1D(params['pool_size'])(x)
    # x = UpSampling1D(params['pool_size'])(x)
    # x = Activation(params['last_activation'])(x)
    # decoded = Reshape((input_dim,))(x)
    # model = Model(input_img, decoded)
    # model.summary()
    # # !! tmp
    
    # -- compile model --------------------------------------------------------
    print('Compiling model...')
    compile_model(model, params)

    if report_time:
        end = datetime.now()
        dur_sec = (end-start).total_seconds()
        with open(output_dir+prefix+'model_time.txt', 'a') as f:
            f.write('Time to compile model: '+str(dur_sec)+' (s)\n')
        dur_sec = (end-start_tot).total_seconds()
        with open(output_dir+prefix+'model_time.txt', 'a') as f:
            f.write('Total time to create model: '+str(dur_sec)+' (s)\n')

    # -- train model ----------------------------------------------------------
    if train:
        print('Training model...')
        # tf.keras.backend.clear_session()
        time_callback = TimeHistory()
        callbacks=[time_callback]
        if save_model_epoch:
            tensorboard_callback = keras.callbacks.TensorBoard(histogram_freq=0)

            checkpoint = keras.callbacks.ModelCheckpoint(output_dir+"model.hdf5",
                                                         monitor='loss', verbose=1,
                                                         save_best_only=True, mode='auto',
                                                         save_freq='epoch')
            callbacks.append(checkpoint)
            callbacks.append(tensorboard_callback)
        
        if type(batch_fnames) == type(None):
            if validation:
                history = model.fit(x_train, x_train, epochs=params['epochs'],
                                    batch_size=params['batch_size'], shuffle=True,
                                    validation_data=(x_test, x_test),
                                    callbacks=callbacks)
            else:
                history = model.fit(x_train, x_train, epochs=params['epochs'],
                            batch_size=params['batch_size'], shuffle=True,
                                    callbacks=callbacks)

        else:
            gen = generate_batches(batch_fnames, params)
            history = model.fit(gen, epochs=params['epochs'],
                                batch_size=params['batch_size'], shuffle=True,
                                callbacks=callbacks)
        time = time_callback.times
        print('Training time: ' + str(time))
        with open(output_dir+prefix+'training_time.txt', 'w') as f:
            f.write(str(time[0]))
            
        if save_model:
            model.save(output_dir + prefix + 'model.hdf5')      
            
    else:
        print('Loading weights...')
        model.load_weights(weights_path)
        history=None
    
    # -- save model weights, bottleneck, reconstructions -----------------------
        
    print('Saving model...')
    model.save(output_dir + prefix + 'model.hdf5') 
    model_summary_txt(output_dir+prefix, model)
    pt.epoch_plots(history, params, output_dir+prefix)

    feats = save_autoencoder_products(model, params, batch_fnames, output_dir,
                                      prefix, x_train, x_test, ticid_train,
                                      ticid_test)
    return model, history, feats
    
    # --------------------------------------------------------------------------

    # res = [model, history]
    # pt.epoch_plots(history, params, output_dir)
    # if save_bottleneck:
    #     print('Getting bottlneck...')
    #     bottleneck_train = \
    #         get_bottleneck(model, x_train, params, save=True, ticid=ticid_train,
    #                        out=output_dir+prefix+'bottleneck_train.fits')
    #     if validation:
    #         bottleneck = get_bottleneck(model, x_test, params, save=True,
    #                                     ticid=ticid_test,
    #                                     out=output_dir+prefix+'bottleneck_test.fits')    
    #     else:
    #         bottleneck=np.empty((0,params['latent_dim']))
    #         hdr=  fits.Header()
    #         hdu = fits.PrimaryHDU(bottleneck, header=hdr)
    #         hdu.writeto(output_dir+prefix+'bottleneck_test.fits', overwrite=True)
    #         fits.append(output_dir+prefix+'bottleneck_test.fits', ticid_test)
            
    #     res.append(bottleneck_train)
    #     res.append(bottleneck) 
    # if predict:
    #     print('Getting x_predict...')
    #     if validation:
    #     # if len(x_test) > 0:
    #         x_predict = model.predict(x_test)      
    #         hdr = fits.Header()
    #         if concat_ext_feats:
    #             hdu = fits.PrimaryHDU(x_predict[0], header=hdr)
    #         else:
    #             hdu = fits.PrimaryHDU(x_predict, header=hdr)
    #         hdu.writeto(output_dir+prefix+'x_predict.fits', overwrite=True)
    #         fits.append(output_dir+prefix+'x_predict.fits', ticid_test)
    #         model_summary_txt(output_dir+prefix, model)
    #     else:
    #         x_predict = None
    #     res.append(x_predict)

    #     x_predict_train = model.predict(x_train)      
    #     hdr = fits.Header()
    #     if concat_ext_feats:
    #         hdu = fits.PrimaryHDU(x_predict_train[0], header=hdr)
    #     else:
    #         hdu = fits.PrimaryHDU(x_predict_train, header=hdr)
    #     hdu.writeto(output_dir+prefix+'x_predict_train.fits', overwrite=True)
    #     fits.append(output_dir+prefix+'x_predict_train.fits', ticid_train)
    #     model_summary_txt(output_dir+prefix, model)         
    #     res.append(x_predict_train)
    
    #     if train:
    #         if concat_ext_feats:
    #             param_summary(history, x_train[0], x_test[0], x_predict_train[0],
    #                           x_predict[0], params, output_dir+prefix, 0,'')
    #         else:
    #             param_summary(history, x_train, x_test, x_predict_train, x_predict,
    #                           params, output_dir+prefix, 0,'')            

    
    # # res = [model, history, bottleneck_train, bottleneck, x_predict, x_predict_train]

    # return res

def cae_encoder(x_train, params, reshape=False):
    '''x_train is an array with shape (num light curves, num data points, 1).
    params is a dictionary with keys:
        * kernel_size : 3, 5
        * latent_dim : dimension of bottleneck/latent space
        * strides : 1
        * epochs
        * dropout
        * num_filters : 8, 16, 32, 64...
        * num_conv_layers : number of convolutional layers in entire
          autoencoder (number of conv layers in encoder is num_conv_layers/2)
        * num_consecutive : number of consecutive convolutional layers (can
          currently only handle 1 or 2)
        * batch_size : 128
        * activation : 'elu'
        * last_activation : 'linear'        
        * optimizer : 'adam'
        * losses : 'mean_squared_error', 'custom'
        * lr : learning rate (e.g. 0.01)
        * initializer: 'random_normal', 'random_uniform', ...
    '''
    
    # input_dim = params['n_features']
    input_dim = truncate(params)
    num_iter = int(params['num_conv_layers']/2)
    
    if type(params['num_filters']) == np.int:
        params['num_filters'] = list(np.repeat(params['num_filters'], num_iter))
    if type(params['num_consecutive']) == np.int:
        params['num_consecutive'] = list(np.repeat(params['num_consecutive'], num_iter))

    input_img = Input(shape = (input_dim,))
    x = Reshape((input_dim, 1))(input_img)
    
    for i in range(num_iter):
        
        for j in range(params['num_consecutive'][i]):
            x = Conv1D(params['num_filters'][i], int(params['kernel_size']),
                    padding='same',
                    kernel_initializer=params['initializer'],
                    strides=params['strides'],
                    kernel_regularizer=params['kernel_regularizer'],
                    bias_regularizer=params['bias_regularizer'],
                    activity_regularizer=params['activity_regularizer'])(x)
            
            if params['batch_norm']:
                x = BatchNormalization()(x)     
            
            x = Activation(params['activation'])(x)
            
        x = MaxPooling1D(params['pool_size'], padding='same')(x)
        x = Dropout(params['dropout'])(x)
        
    if params['cvae']:
        x = Flatten()(x)
        z_mean = Dense(params['latent_dim'], activation=params['activation'],
                        kernel_initializer=params['initializer'],
                        kernel_regularizer=params['kernel_regularizer'],
                        bias_regularizer=params['bias_regularizer'],
                        activity_regularizer=params['activity_regularizer'])(x)
        z_log_var = Dense(params['latent_dim'], activation=params['activation'],
                          kernel_initializer=params['initializer'],
                          kernel_regularizer=params['kernel_regularizer'],
                          bias_regularizer=params['bias_regularizer'],
                          activity_regularizer=params['activity_regularizer'])(x)   
        z = Lambda(sampling, output_shape=(params['latent_dim'],),
                   name='bottleneck')([z_mean, z_log_var])         
        
    else:
        x = Flatten()(x)
        
        # x = Dense(256, activation=params['activation'],
        #                 kernel_initializer=params['initializer'])(x)
        # x = Dense(128, activation=params['activation'],
        #                 kernel_initializer=params['initializer'])(x)
        # x = Dense(64, activation=params['activation'],
        #                 kernel_initializer=params['initializer'])(x)
    
        encoded = Dense(params['latent_dim'], activation=params['activation'],
                        kernel_initializer=params['initializer'],
                        kernel_regularizer=params['kernel_regularizer'],
                        bias_regularizer=params['bias_regularizer'],
                        activity_regularizer=params['activity_regularizer'],
                        name='bottleneck')(x)
    
    if params['cvae']:
        encoder = Model(input_img, [z_mean, z_log_var, z])
    else:
        encoder = Model(input_img, encoded)
    return encoder


def cae_decoder(x_train, bottleneck, params):
    import tensorflow as tf
    
    # input_dim = params['n_features']
    input_dim = truncate(params)

    num_iter = int(params['num_conv_layers']/2)
    reduction_factor = params['pool_size'] * params['strides']**params['num_consecutive'][0] 
    tot_reduction_factor = reduction_factor**num_iter
    
    if type(params['num_filters']) == np.int:
        params['num_filters'] = list(np.repeat(params['num_filters'], num_iter))    
    if type(params['num_consecutive']) == np.int:
        params['num_consecutive'] = list(np.repeat(params['num_consecutive'], num_iter))
        

    x = bottleneck
    # reduction_factor = params['pool_size'] * params['strides']

    x = Dense(int(input_dim*params['num_filters'][-1]/tot_reduction_factor),
              kernel_initializer=params['initializer'],
              kernel_regularizer=params['kernel_regularizer'],
              bias_regularizer=params['bias_regularizer'],
              activity_regularizer=params['activity_regularizer'])(x) 
    x = Reshape((int(input_dim/tot_reduction_factor),
                  params['num_filters'][-1]))(x)


    for i in range(num_iter):
        if params['dropout'] > 0:
            x = Dropout(params['dropout'])(x)
            
        x = UpSampling1D(params['pool_size'])(x)

        for j in range(params['num_consecutive'][-1*i - 1]):
            
            # >> last layer
            if i == num_iter-1 and j == params['num_consecutive'][-1*i - 1]-1 \
                and not params['fully_conv']:
                
                if params['strides'] == 1: # >> faster than Conv1Dtranspose
                    x = Conv1D(1, int(params['kernel_size']),
                                      padding='same', strides=params['strides'],
                                      kernel_initializer=params['initializer'],
                                      kernel_regularizer=params['kernel_regularizer'],
                                      bias_regularizer=params['bias_regularizer'],
                                      activity_regularizer=params['activity_regularizer'])(x)  
                    
                
                else:
                    x = Conv1DTranspose(x, 1, int(params['kernel_size']),
                                padding='same',
                                strides=params['strides'],
                                kernel_initializer=params['initializer'],
                                kernel_regularizer=params['kernel_regularizer'],
                                bias_regularizer=params['bias_regularizer'],
                          activity_regularizer=params['activity_regularizer'])
                    
                x = BatchNormalization()(x)
                decoded = Activation(params['last_activation'])(x)
                decoded = Reshape((input_dim,))(decoded)
                    
            else: # >> intermediate layer
                
                if params['strides'] == 1:
                    x = Conv1D(params['num_filters'][-1*i - 1],
                                int(params['kernel_size']),padding='same',
                                strides=params['strides'],
                                kernel_initializer=params['initializer'],
                                kernel_regularizer=params['kernel_regularizer'],
                                bias_regularizer=params['bias_regularizer'],
                                activity_regularizer=params['activity_regularizer'])(x)   
                else:
                    x = Conv1DTranspose(x, params['num_filters'][-1*i - 1],
                                int(params['kernel_size']), padding='same',
                                strides=params['strides'],
                                kernel_initializer=params['initializer'],
                                kernel_regularizer=params['kernel_regularizer'],
                                bias_regularizer=params['bias_regularizer'],
                                activity_regularizer=params['activity_regularizer'])
                x = BatchNormalization()(x)
                x = Activation(params['activation'])(x)

    return decoded

# :: Deep fully-connected autoencoder ::::::::::::::::::::::::::::::::::::::::::

def deep_autoencoder(x_train, y_train, x_test=None, y_test=None, params=None,
                     parampath=None, batch_norm=True, ticid_train=None,
                     ticid_test=None, resize=False, output_dir='', prefix='',
                     report_time=True, batch_fnames=None):
    '''The y_train and y_test arguments are place-holders in order to use the
    Talos hyperparameter optimization library.'''

    if report_time:
        from datetime import datetime
        start = datetime.now()
        start_tot = datetime.now()

    if type(parampath) != type(None):
        params = read_hyperparameters_from_txt(parampath)

    # num_classes = np.shape(y_train)[1]
    # input_dim = np.shape(x_train)[1]
    if 'n_features' not in params.keys():
        if type(batch_fnames) == type(None):
            params['n_features'] = x_train.shape[1]
        else:
            x_train = np.load(batch_fnames[0])
            params['n_features'] = x_train.shape[1]
            x_train = None
    if 'n_samples' not in params.keys():
        if type(batch_fnames) == type(None):
            params['n_samples'] = x_train.shape[0]
        else:
            n_samples = 0
            for i in range(len(batch_fnames)):
                x_train = np.load(batch_fnames[i])
                n_samples += x_train.shape[0]
            x_train = None
            params['n_samples'] = n_samples
            

    input_dim = params['n_features']
    
    hidden_units = list(range(params['max_dim'],
                              params['latent_dim'],
                              -params['step']))    
    if hidden_units[-1] != params['latent_dim']:
        hidden_units.append(params['latent_dim'])

    # -- encoder ---------------------------------------------------------------

    if resize:
        input_img = Input(shape = (input_dim,1))
        x = Flatten()(input_img)
    else:
        input_img = Input(shape = (input_dim,))
        x = input_img

    if report_time:
        end = datetime.now()
        dur_sec = (end-start).total_seconds()
        with open(output_dir+prefix+'model_time.txt', 'w') as f:
            f.write('Time to add Input layer: '+str(dur_sec)+'\n')
        start = end

    for i in range(len(hidden_units)):
        x = Dense(hidden_units[i], activation=params['activation'],
                  kernel_initializer=params['initializer'])(x)
        if batch_norm: x = BatchNormalization()(x)

        if report_time:
            end = datetime.now()
            dur_sec = (end-start).total_seconds()
            with open(output_dir+prefix+'model_time.txt', 'a') as f:
                f.write('Time to add encoder Dense'+str(i)+' layer: '+\
                        str(dur_sec)+'\n')
            start = end
        
    # -- bottleneck ------------------------------------------------------------
    x = Dense(params['latent_dim'], activation=params['activation'],
              kernel_initializer=params['initializer'], name='bottleneck')(x)

    # -- decoder ---------------------------------------------------------------
    for i in np.arange(len(hidden_units)-1, -1, -1):
        if batch_norm: x = BatchNormalization()(x)        
        x = Dense(hidden_units[i], activation=params['activation'],
                  kernel_initializer=params['initializer'])(x)
        if report_time:
            end = datetime.now()
            dur_sec = (end-start).total_seconds()
            with open(output_dir+prefix+'model_time.txt', 'a') as f:
                f.write('Time to add decoder Dense'+str(i)+' layer: '+\
                        str(dur_sec)+'\n')
            start = end

    if batch_norm: x = BatchNormalization()(x)    
    x = Dense(input_dim, activation=params['last_activation'],
              kernel_initializer=params['initializer'])(x)
    if resize:
        x = Reshape((input_dim, 1))(x)
        
    # -- build model -----------------------------------------------------------

    model = Model(input_img, x)
    model.summary()

    compile_model(model, params)

    if type(x_test)==type(None):
        validation_data=None
    else:
        validation_data=(x_test,x_test)

    if report_time:
        end = datetime.now()
        dur_sec = (end-start).total_seconds()
        with open(output_dir+prefix+'model_time.txt', 'a') as f:
            f.write('Time to compile model: '+str(dur_sec)+'\n')
        dur_sec = (end-start_tot).total_seconds()
        with open(output_dir+prefix+'model_time.txt', 'a') as f:
            f.write('Total time to create model: '+str(dur_sec)+'\n')

    # -- train model -----------------------------------------------------------

    if type(batch_fnames) == type(None):
        history = model.fit(x_train, x_train, epochs=params['epochs'],
                            batch_size=params['batch_size'], shuffle=True,
                            validation_data=validation_data)
    else:
        gen = generate_batches(batch_fnames, params)
        history = model.fit(gen, epochs=params['epochs'],
                            batch_size=params['batch_size'], shuffle=True,
                            validation_data=validation_data,
                            steps_per_epoch=params['n_samples']\
                            //params['batch_size'])

    # -- save model weights, bottleneck, reconstructions -----------------------
        
    print('Saving model...')
    dt.create_dir(output_dir+prefix)
    model.save(output_dir + prefix + 'model.hdf5') 
    model_summary_txt(output_dir+prefix, model)
    pt.epoch_plots(history, params, output_dir+prefix)

    feats = save_autoencoder_products(model, params, batch_fnames, output_dir,
                                      prefix, x_train, x_test, ticid_train,
                                      ticid_test)
    return model, history, feats

# :: Variational Autoencoder :::::::::::::::::::::::::::::::::::::::::::::::::::

# def variational_autoencoder(x_train, y_train, x_test, y_test, params):
#     '''https://blog.keras.io/building-autoencoders-in-keras.html'''    
#     input_dim = np.shape(x_train)[1]    
#     hidden_units = list(range(params['max_dim'], params['latent_dim'],
#                               -params['step']))    
#     if hidden_units[-1] != params['latent_dim']:
#         hidden_units.append(params['latent_dim'])

#     # -- encoder ---------------------------------------------------------------
#     inputs = Input(shape = (input_dim,))
#     x = inputs
#     for i in range(len(hidden_units)):
#         x = Dense(hidden_units[i], activation=params['activation'],
#                   kernel_initializer=params['initializer'])(x)
#         if params['batch_norm']: x = BatchNormalization()(x)
        
#     x = Dense(params['latent_dim'], activation=params['activation'],
#               kernel_initializer=params['initializer'])(x)
#     z_mean = Dense(params['latent_dim'])(x)
#     z_log_sigma = Dense(params['latent_dim'])(x)
#     z = Lambda(sampling, name='bottleneck')([z_mean, z_log_sigma, params])
#     encoder = Model(inputs, [z_mean, z_log_sigma, z])
#     encoder.summary()

#     # -- decoder ---------------------------------------------------------------
#     latent_inputs = Input(shape=(params['latent_dim'],))
#     x = latent_inputs
#     for i in np.arange(len(hidden_units)-1, -1, -1):
#         if params['batch_norm']: x = BatchNormalization()(x)        
#         x = Dense(hidden_units[i], activation=params['activation'],
#                   kernel_initializer=params['initializer'])(x)
#     if params['batch_norm']: x = BatchNormalization()(x)    
#     decoder_outputs = Dense(input_dim, activation=params['last_activation'],
#                             kernel_initializer=params['initializer'])(x)
#     decoder = Model(latent_inputs, decoder_outputs)
#     decoder.summary()    

#     # -- instantiate VAE model -------------------------------------------------
#     outputs = decoder(encoder(inputs)[2])
#     vae = Model(inputs, outputs)
#     reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)
#     reconstruction_loss *= input_dim
#     kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
#     kl_loss = K.sum(kl_loss, axis=-1)
#     kl_loss *= -0.5
#     vae_loss = K.mean(reconstruction_loss + kl_loss)
#     vae.add_loss(vae_loss)
#     vae.compile(optimizer='adam')

#     if type(x_test)==type(None):
#         validation_data=None
#     else:
#         validation_data=(x_test,x_test)
#     history = vae.fit(x_train, x_train, epochs=params['epochs'],
#                       batch_size=params['batch_size'], shuffle=True,
#                       validation_data=validation_data)

#     return history, vae, encoder

def run_cvae(x_train, y_train, x_test, y_test, params, save_model=True,
             predict=False, output_dir='', prefix='', ticid_train=None,
             ticid_test=None, target_info_train=None,
             target_info_test=None):
    history, model = conv_variational_autoencoder(x_train, y_train, x_test,
                                                  y_test, params)

    if save_model: model.save(output_dir + prefix + 'model.hdf5')      

    res = [model, history]

    if save_bottleneck:
        print('Getting bottlneck...')
        bottleneck_train = \
            get_bottleneck(model, x_train, params, save=True, ticid=ticid_train,
                           out=output_dir+prefix+'bottleneck_train.fits')
        if len(x_test) > 0:
            bottleneck = get_bottleneck(model, x_test, params, save=True,
                                        ticid=ticid_test,
                                        out=output_dir+prefix+'bottleneck_test.fits')    
        else:
            bottleneck=np.empty((0,params['latent_dim']))
            hdr=  fits.Header()
            hdu = fits.PrimaryHDU(bottleneck, header=hdr)
            hdu.writeto(output_dir+prefix+'bottleneck_test.fits', overwrite=True)
            fits.append(output_dir+prefix+'bottleneck_test.fits', ticid_test)
            
        res.append(bottleneck_train)
        res.append(bottleneck)

    if predict:
        print('Getting x_predict...')
        if len(x_test) > 0:
            x_predict = model.predict(x_test)      
            hdr = fits.Header()
            hdu = fits.PrimaryHDU(x_predict, header=hdr)
            hdu.writeto(output_dir+prefix+'x_predict.fits', overwrite=True)
            fits.append(output_dir+prefix+'x_predict.fits', ticid_test)
            model_summary_txt(output_dir+prefix, model)
        else:
            x_predict = None
        res.append(x_predict)

        x_predict_train = model.predict(x_train)      
        hdr = fits.Header()
        hdu = fits.PrimaryHDU(x_predict_train, header=hdr)
        hdu.writeto(output_dir+prefix+'x_predict_train.fits', overwrite=True)
        fits.append(output_dir+prefix+'x_predict_train.fits', ticid_train)
        model_summary_txt(output_dir+prefix, model)         
        res.append(x_predict_train)
        param_summary(history, x_train, x_test, x_predict_train, x_predict,
                      params, output_dir+prefix, 0,'')            



def conv_variational_autoencoder(x_train, y_train, x_test, y_test, params):
    
    input_dim = np.shape(x_train)[1]    
    num_iter = int(params['num_conv_layers']/2)
    
    if type(params['num_filters']) == np.int:
        params['num_filters'] = list(np.repeat(params['num_filters'], num_iter))
    if type(params['num_consecutive']) == np.int:
        params['num_consecutive'] = list(np.repeat(params['num_consecutive'],
                                                   num_iter))

    # -- encoder ---------------------------------------------------------------
    inputs = Input(shape = (input_dim,))
    x = Reshape((input_dim, 1))(inputs)
    
    for i in range(num_iter):
        for j in range(params['num_consecutive'][i]):
            x = Conv1D(params['num_filters'][i], int(params['kernel_size']),
                    padding='same',
                    kernel_initializer=params['initializer'],
                    strides=params['strides'],
                    kernel_regularizer=params['kernel_regularizer'],
                    bias_regularizer=params['bias_regularizer'],
                    activity_regularizer=params['activity_regularizer'])(x)        

        if params['batch_norm']: x = BatchNormalization()(x)

        x = Activation(params['activation'])(x)            
        x = MaxPooling1D(params['pool_size'], padding='same')(x)
        x = Dropout(params['dropout'])(x)

    x = Flatten()(x)

    x = Dense(params['latent_dim'], activation=params['activation'],
              kernel_initializer=params['initializer'])(x)
    z_mean = Dense(params['latent_dim'])(x)
    z_log_sigma = Dense(params['latent_dim'])(x)
    z = Lambda(sampling, name='bottleneck')([z_mean, z_log_sigma, params])
    encoder = Model(inputs, [z_mean, z_log_sigma, z])
    encoder.summary()

    # -- decoder ---------------------------------------------------------------
    latent_inputs = Input(shape=(params['latent_dim'],))
    x = latent_inputs


    reduction_factor = params['pool_size'] * params['strides']**params['num_consecutive'][0] 
    tot_reduction_factor = reduction_factor**num_iter
    x = Dense(int(input_dim*params['num_filters'][-1]/tot_reduction_factor),
              kernel_initializer=params['initializer'],
              kernel_regularizer=params['kernel_regularizer'],
              bias_regularizer=params['bias_regularizer'],
              activity_regularizer=params['activity_regularizer'])(x) 
    x = Reshape((int(input_dim/tot_reduction_factor),
                  params['num_filters'][-1]))(x)

    for i in np.arange(num_iter):
        if params['batch_norm']: x = BatchNormalization()(x)        
        x = UpSampling1D(params['pool_size'])(x)
        for j in range(params['num_consecutive'][-1*i - 1]):
            
            # >> last layer
            if i == num_iter-1 and j == params['num_consecutive'][-1*i - 1]-1 \
                and not params['fully_conv']:
                
                if params['strides'] == 1: # >> faster than Conv1Dtranspose
                    x = Conv1D(1, int(params['kernel_size']),
                                      padding='same', strides=params['strides'],
                                      kernel_initializer=params['initializer'],
                                      kernel_regularizer=params['kernel_regularizer'],
                                      bias_regularizer=params['bias_regularizer'],
                                      activity_regularizer=params['activity_regularizer'])(x)  
                    
                
                else:
                    x = Conv1DTranspose(x, 1, int(params['kernel_size']),
                                padding='same',
                                strides=params['strides'],
                                kernel_initializer=params['initializer'],
                                kernel_regularizer=params['kernel_regularizer'],
                                bias_regularizer=params['bias_regularizer'],
                          activity_regularizer=params['activity_regularizer'])
                    
                x = BatchNormalization()(x)
                decoded = Activation(params['last_activation'])(x)
                decoded = Reshape((input_dim,))(decoded)

            else:
                
                if params['strides'] == 1:
                    x = Conv1D(params['num_filters'][-1*i - 1],
                                int(params['kernel_size']),padding='same',
                                strides=params['strides'],
                                kernel_initializer=params['initializer'],
                                kernel_regularizer=params['kernel_regularizer'],
                                bias_regularizer=params['bias_regularizer'],
                                activity_regularizer=params['activity_regularizer'])(x)   
                else:
                    x = Conv1DTranspose(x, params['num_filters'][-1*i - 1],
                                int(params['kernel_size']), padding='same',
                                strides=params['strides'],
                                kernel_initializer=params['initializer'],
                                kernel_regularizer=params['kernel_regularizer'],
                                bias_regularizer=params['bias_regularizer'],
                                activity_regularizer=params['activity_regularizer'])
                x = BatchNormalization()(x)
                x = Activation(params['activation'])(x)


    x = Reshape((input_dim,))(x)

    if params['batch_norm']: x = BatchNormalization()(x)    
    decoder_outputs = x
    # decoder_outputs = Dense(input_dim, activation=params['last_activation'],
    #                         kernel_initializer=params['initializer'])(x)
    decoder = Model(latent_inputs, decoder_outputs)
    decoder.summary()    

    # -- instantiate VAE model -------------------------------------------------
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs)
    reconstruction_loss = tf.keras.losses.mean_squared_error(inputs, outputs)
    reconstruction_loss *= input_dim
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    # vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae_loss = K.mean(reconstruction_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    if type(x_test)==type(None):
        validation_data=None
    else:
        validation_data=(x_test,x_test)
    history = vae.fit(x_train, x_train, epochs=params['epochs'],
                      batch_size=params['batch_size'], shuffle=True,
                      validation_data=validation_data)

    return history, vae


def sampling(args):
    z_mean, z_log_sigma, params = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], params['latent_dim']),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

# :: VAEGAN ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def vae_gan(x_train, y_train, x_test, y_test, params, 
            save_model=True, save_bottleneck=True,
            predict=True, output_dir='./',
            input_rms=False, rms_train=None, rms_test=None,
            ticid_train=None, ticid_test=None):
    # >> encoder
    E, feature_maps, pool_masks = encoder(x_train, params)
    compile_model(E, params)
    E.summary()
    model_summary_txt(output_dir+'VAEGAN_encoder_', E)  

    z_mean, z_log_var, z = E.output    
    
    # >> generator / decoder
    G = decgen(x_train, params)
    compile_model(G, params)
    G.summary()
    model_summary_txt(output_dir+'VAEGAN_generator_', G)
    
    # >> discriminator
    D = discriminator(x_train, params)
    compile_model(D, params)
    D.summary()
    model_summary_txt(output_dir+'VAEGAN_discriminator_', D)
    D_fixed = discriminator(E.input, params)
    compile_model(D_fixed, params)
    
    # >> VAE
    input_dim = np.shape(x_train)[1]
    X = Input(shape = (input_dim,))
    E_mean, E_logsigma, Z = E(X)
    output = G(Z)
    G_dec = G(E_mean + E_logsigma)
    D_fake, F_fake = D(output)
    D_fromGen, F_fromGen = D(G_dec)
    D_true, F_true = D(X)
    
    VAE = Model(X, output)
    kl = - 0.5 * K.sum(1 + E_logsigma - K.square(E_mean) - K.exp(E_logsigma),
                       axis=-1)
    crossent = 64 * metrics.mse(K.flatten(X), K.flatten(output))
    VAEloss = K.mean(crossent + kl)
    VAE.add_loss(VAEloss)
    opt = optimizers.adam(lr = params['lr'])        
    VAE.compile(optimizer=opt)
    VAE.summary()
    model_summary_txt(output_dir+'VAEGAN_', VAE)
    
    batch_size = params['batch_size']
    noise = np.random.normal(0, 1, (batch_size, params['latent_dim']))
    for epoch in range(params['epochs']):
        print('Epoch : '+str(epoch))
        latent_vect = E.predict(x_train)[0]
        encImg = G.predict(latent_vect)
        fakeImg = G.predict(noise)
    
        DlossTrue = D_true.train_on_batch(x_train, np.ones((batch_size, 1)))
        DlossEnc = D_fromGen.train_on_batch(encImg, np.ones((batch_size, 1)))
        DlossFake = D_fake.train_on_batch(fakeImg, np.zeros((batch_size, 1)))
    
        cnt = epoch
        while cnt > 3:
            cnt = cnt - 4
    
        if cnt == 0:
            GlossEnc = G.train_on_batch(latent_vect, np.ones((batch_size, 1)))
            GlossGen = G.train_on_batch(noise, np.ones((batch_size, 1)))
            Eloss = VAE.train_on_batch(x_train, None)
    
        chk = epoch
    
        while chk > 50:
            chk = chk - 51
    
        if chk == 0:
            D.save_weights('discriminator.h5')
            G.save_weights('generator.h5')
            E.save_weights('encoder.h5')
    
        print("epoch number", epoch + 1)
        print("loss:")
        print("D:", DlossTrue, DlossEnc, DlossFake)
        print("G:", GlossEnc, GlossGen)
        print("VAE:", Eloss)
    
    print('Training done,saving weights')
    D.save_weights('discriminator.h5')
    G.save_weights('generator.h5')
    E.save_weights('encoder.h5')
    print('end')    
   
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Iterative training ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::;::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 
def split_reconstruction(x, flux_train, flux_test,
                         x_train, x_test, x_predict_train, x_predict_test, 
                         ticid_train, ticid_test, target_info_train, 
                         target_info_test, features,
                         n_split=2, input_psd=False,
                         concat_ext_feats=False, train_psd_only=False,
                         error_threshold=0.5, 
                         return_highest_error_ticid=True, output_dir='./', prefix=''):
    if len(x_test) == 0:
        x_predict_test = np.empty((0, x_train.shape[-1]))
    
    # if concat_ext_feats or input_psd: 
    #     err_train = (x_train[0] - x_predict_train[0])**2
    #     err_test = (x_test[0] - x_predict_test[0])**2
    # else:      
    #     err_train = (x_train - x_predict_train)**2
    #     err_test = (x_test - x_predict_test)**2
        
    if concat_ext_feats or input_psd: 
        err_train = np.abs(x_train[0] - x_predict_train[0])**3
        err_test = np.abs(x_test[0] - x_predict_test[0])**3
    else:      
        err_train = np.abs(x_train - x_predict_train)**3
        err_test = np.abs(x_test - x_predict_test)**3
        
    err_train = np.mean(err_train, axis=1)
    err_train = err_train.reshape(np.shape(err_train)[0])
    ranked_train = np.argsort(err_train)
    err_test = np.mean(err_test, axis=1)
    err_test = err_test.reshape(np.shape(err_test)[0])
    ranked_test = np.argsort(err_test)    
    del err_train
    del err_test
    
    # >> re-order arrays
    flux_train = flux_train[ranked_train]
    ticid_train = ticid_train[ranked_train]
    info_train = target_info_train[ranked_train]
    flux_test = flux_test[ranked_test]
    ticid_test = ticid_test[ranked_test]
    info_test = target_info_test[ranked_test]
    if concat_ext_feats or input_psd:
        x_train = x_train[0][ranked_train]
        x_train_feat = x_train[1][ranked_train]
        x_test = x_test[0][ranked_test]
        x_test_feat = x_test[1][ranked_test]
    else:
        x_train = x_train[ranked_train]
        x_test = x_test[ranked_test]
        
    # >> now split
    train_len = len(x_train)
    test_len = len(x_test)
    # split_ind = int(train_len/n_split)
    split_ind = int(error_threshold*train_len)
    flux_train = np.split(flux_train, [split_ind])
    flux_test = np.split(flux_test, [split_ind])
    x_train = np.split(x_train, [split_ind])
    x_test = np.split(x_test, [split_ind])
    if concat_ext_feats or input_psd:
        x_train_feat = np.split(x_train_feat, [split_ind])
        x_train[0] = np.concatenate([x_train[0], x_train_feat[0]], axis=0)
        x_train[1] = np.concatenate([x_train[1], x_train_feat[1]], axis=0)        
        x_test_feat = np.split(x_test_feat, [split_ind])
        x_test[0] = np.concatenate([x_test[0], x_test_feat[0]], axis=0)
        x_test[1] = np.concatenate([x_test[1], x_test_feat[1]], axis=0)
    ticid_train = np.split(ticid_train, [split_ind])
    info_train = np.split(info_train, [split_ind])
    ticid_test = np.split(ticid_test, [split_ind])
    info_test = np.split(info_test, [split_ind])    
    features = np.split(features, [split_ind])
    x_predict_train = np.split(x_predict_train, [split_ind])
    x_predict_test = np.split(x_predict_test, [split_ind])
    
    if train_psd_only:
        x_train = x_train_feat
        x_test = x_test_feat
        x = x[1]

    np.savetxt(output_dir+prefix+'ticid_highest_error_train.txt', ticid_train[1])
    np.savetxt(output_dir+prefix+'ticid_lowest_error_train.txt', ticid_train[0])
    np.savetxt(output_dir+prefix+'ticid_highest_error_test.txt', ticid_test[1])
    np.savetxt(output_dir+prefix+'ticid_lowest_error_test.txt', ticid_test[0])
    

    if return_highest_error_ticid:
        return x, flux_train[-1], flux_test[-1], x_train[-1], x_test[-1],\
            ticid_train[-1], ticid_test[-1], info_train[-1], info_test[-1],\
            features[-1], x_predict_train[-1], x_predict_test[-1]
    return x, flux_train, flux_test, x_train, x_test, ticid_train, ticid_test,\
        info_train, info_test, features, x_predict_train, x_predict_test

def split_segments(x, x_train, x_test, p, target_info_train, target_info_test,
                   ticid_train, ticid_test, sectors, n_split=4, len_var=0.1,
                   output_dir='./', prefix='', debug=False):
    '''Split each light curve into n_split segments. Args:
        * len_var: determines possible range of segment lengths'''

    # >> first find the lengths of each segment
    original_len = np.shape(x_train)[-1]
    avg_len = original_len / n_split
    segment_lengths = np.random.randint(low=int(avg_len-avg_len*len_var),
                                        high=(avg_len+avg_len*len_var),
                                        size=n_split-1)
    last_segment_len = original_len - np.sum(segment_lengths)
    segment_lengths = np.append(segment_lengths, last_segment_len)

    # >> now make x_train have shape (n_split, num_objs, segment_length)
    new_x_train = []
    new_x_test = []
    new_x = []
    rms_train = [] # >> shape = (n_split, num_objs)
    rms_test = [] # >> shape = (n_split, num_objs)
    for i in range(n_split):
        segment_start = np.sum(segment_lengths[:i])

        # >> truncate
        reduction_factor = np.max(p['pool_size'])*\
                           np.max(p['strides'])**np.max(p['num_consecutive']) 
        num_iter = np.max(p['num_conv_layers'])/2
        tot_reduction_factor = reduction_factor**num_iter
        new_length = int(segment_lengths[i] / tot_reduction_factor)*\
                     int(tot_reduction_factor)
        segment_end = segment_start + new_length

        # >> normalize
        segment = x_train[:,segment_start:segment_end]
        segment_rms = dt.rms(segment)
        rms_train.append(segment_rms)
        segment = dt.standardize(segment)
        new_x_train.append(segment)

        segment = x_test[:,segment_start:segment_end]
        segment_rms = dt.rms(segment)
        rms_test.append(segment_rms)
        segment = dt.standardize(segment)
        new_x_test.append(segment)

        new_x.append(x[segment_start:segment_end])

    x_train = new_x_train
    x_test = new_x_test
    x = new_x
    del new_x_train
    del new_x_test
    del new_x

    if debug:
        nrows=6
        fig, ax = plt.subplots(6, n_split)
        for row in range(nrows):
            for col in range(n_split):
                ax[row, col].plot(x[col], x_train[col][row], '.k')
                pt.format_axes(ax[row,col], xlabel=True, ylabel=True)
        fig.tight_layout()
        fig.savefig(output_dir+prefix+'light_curve_segments.png')
        plt.close('all')

    return x, x_train, x_test, rms_train, rms_test


def split_cae(x, flux_train, flux_test, p, target_info_train, target_info_test,
              ticid_train, ticid_test, sectors, n_split=4, len_var=0.1, 
              data_dir='./', database_dir='./', output_dir='./', prefix0='',
              momentum_dump_csv='./Table_of_momentum_dumps.csv', debug=True,
              save_model_epoch=False, plot=False, hyperparam_opt=False, p_opt={}):
    # -- split x_train into n_split segments randomly --------------------------

    print('Splitting light curves into '+str(n_split)+' segments...')
    x, x_train, x_test, rms_train, rms_test = \
        split_segments(x, flux_train, flux_test, p, target_info_train, target_info_test,
                       ticid_train, ticid_test, sectors, n_split=n_split,
                       len_var=len_var, output_dir=output_dir, prefix=prefix0,
                       debug=debug)

    # -- train each segment ----------------------------------------------------
    features = np.empty((len(x_train[0])+len(x_test[0]), 0))
    for i in range(n_split):
        print('Segment ' + str(i) + '...')
        prefix = prefix0+'segment'+str(i)+'-'

        if hyperparam_opt:
            print('Starting hyperparameter optimization...')
            p_opt['latent_dim'] = [14,16]
            t = talos.Scan(x=x_train, y=x_train, params=p_opt, model=conv_autoencoder,
                           experiment_name=prefix, reduction_metric='val_loss',
                           minimize_loss=True, reduction_method='correlation',
                           fraction_limit=0.001)      
            analyze_object = talos.Analyze(t)
            data_frame, best_param_ind,p = \
                pt.hyperparam_opt_diagnosis(analyze_object, output_dir+prefix,
                                           supervised=False)        
            p['epochs'] = p['epochs']*3



        model, history, bottleneck_train, bottleneck_test, x_predict, x_predict_train = \
            conv_autoencoder(x_train[i], x_train[i], x_test[i], x_test[i], p,
                             ticid_train=ticid_train, ticid_test=ticid_test,
                             val=False, save_model=True, predict=True, 
                             save_bottleneck=True, prefix=prefix,
                             output_dir=output_dir,
                             save_model_epoch=save_model_epoch)
        features = np.append(features, bottleneck_train, axis=1)

        if plot:
            pt.diagnostic_plots(history, model, p, output_dir, x[i], x_train[i], x_test[i],
                                x_predict, x_predict_train=x_predict_train,
                                target_info_test=target_info_test,
                                target_info_train=target_info_train, prefix=prefix,
                                ticid_train=ticid_train, ticid_test=ticid_test, 
                                bottleneck_train=bottleneck_train, bottleneck=bottleneck_test,
                                plot_epoch = True,
                                plot_in_out = False,
                                plot_in_out_train = True,
                                plot_in_bottle_out=False,
                                plot_latent_test = False,
                                plot_latent_train = True,
                                plot_kernel=False,
                                plot_intermed_act=False,
                                make_movie = False,
                                plot_lof_test=False,
                                plot_lof_train=False,
                                plot_lof_all=False,
                                plot_reconstruction_error_test=False,
                                plot_reconstruction_error_train=True,
                                plot_reconstruction_error_all=False,
                                load_bottleneck=True)

    # -- novelty detection & classification ------------------------------------


    if plot:
        if len(x_test) > 0:
            flux_feat = np.concatenate([x_train, x_test], axis=0)
            ticid_feat = np.concatenate([ticid_train, ticid_test])
            info_feat = np.concatenate([target_info_train, target_info_test])
        else:
            flux_feat, ticid_feat, info_feat = x_train, ticid_train, target_info_train
            flux_feat = np.empty((len(x_train[0]), 0))
            for i in range(n_split):
                flux_feat = np.append(flux_feat, x_train[i], axis=-1)
            for i in range(n_split):
                with fits.open(output_dir+'segment'+str(i)+'-'+'bottleneck_train.fits') as hdul:
                    bottleneck_train = hdul[0].data
                    features=np.append(features, bottleneck_train, axis=-1)

        post_process(x[i], x_train[i], x_test[i], ticid_train, ticid_test,
                     target_info_train, target_info_test, p,
                     output_dir, sectors, prefix=prefix,
                     data_dir=data_dir, database_dir=database_dir,
                     momentum_dump_csv=momentum_dump_csv, features=features, 
                     flux_feat=flux_feat, ticid_feat=ticid_feat, info_feat=info_feat)

    # >> concatenate segments for x_train, x_test, x_predict_train, ...
    x_train = np.concatenate(x_train, axis=1)
    x_test = np.concatenate(x_test, axis=1)
    x_predict_train = np.empty((np.shape(x_train)[0], 0))
    x_predict_test = np.empty((np.shape(x_test)[0], 0))
    prefix = prefix.split('-')[0]+'-'
    for i in range(n_split):
        with fits.open(output_dir+prefix+'segment'+str(i)+'-x_predict_train.fits') as hdul:
            x_predict_train=np.append(x_predict_train, hdul[0].data, axis=1)
        if len(x_test) > 0:
            with fits.open(output_dir+prefix+'segment'+str(i)+'-x_predict_test.fits') as hdul:
                x_predict_test=np.append(x_predict_test, hdul[0].data, axis=1)
    if len(x_predict_test) == 0:
        x_predict_test=np.empty((0, x_predict_train.shape[1]))

    return x_train, x_test, x_predict_train, x_predict_test
    
def iterative_cae(flux_train, flux_test, x, p, ticid_train, 
                  ticid_test, target_info_train, target_info_test, iterations=2,
                  error_threshold=0.5, latent_dim=[20, 16],
                  n_split=[4,8], len_var=0.1, save_model_epoch=False,
                  output_dir='./', split=False, input_psd=False, 
                  database_dir='./', data_dir='./', train_psd_only=True,
                  momentum_dump_csv='./Table_of_momentum_dumps.csv', sectors=[],
                  concat_ext_feats=False, use_rms=False, do_diagnostic_plots=True,
                  do_iteration_summary=True, do_ensemble_summary=True,
                  novelty_detection=True, 
                  run=True, hyperparam_opt=False, p_opt={}):
    '''len(n_split)=iterations'''

    # -- first iteration -------------------------------------------------------
    print('-'*17)
    print('-- iteration 0 --')
    print('-'*17)

    prefix='iteration0-'
    rms_train = dt.rms(flux_train)
    x_train = dt.standardize(flux_train)
    rms_test = dt.rms(flux_test)
    x_test = dt.standardize(flux_test)
    
    if hyperparam_opt:
        print('Starting hyperparameter optimization...')
        t = talos.Scan(x=x_train, y=x_train, params=p_opt, model=conv_autoencoder,
                       experiment_name=prefix, reduction_metric='val_loss',
                       minimize_loss=True, reduction_method='correlation',
                       fraction_limit=0.001)      
        analyze_object = talos.Analyze(t)
        data_frame, best_param_ind,p = \
            pt.hyperparam_opt_diagnosis(analyze_object, output_dir+prefix,
                                       supervised=False)        
        p['epochs'] = p['epochs']*3

    if run:
        model, history, bottleneck_train, bottleneck_test, x_predict_test, x_predict_train = \
            conv_autoencoder(x_train, x_train, x_test, x_test, p,
                             ticid_train=ticid_train, ticid_test=ticid_test,
                             val=False, save_model=True, predict=True, 
                             save_bottleneck=True, prefix=prefix,
                             output_dir=output_dir,
                             save_model_epoch=save_model_epoch)
        plot_epoch=True
    else:
        history=None
        plot_epoch=False,
        model = load_model(output_dir+prefix+'model.hdf5')
        with fits.open(output_dir+prefix+'bottleneck_test.fits') as hdul:
            bottleneck_test = hdul[0].data
        with fits.open(output_dir+prefix+'bottleneck_train.fits') as hdul:
            bottleneck_train = hdul[0].data
        with fits.open(output_dir+prefix+'x_predict_train.fits') as hdul:
            x_predict_train = hdul[0].data
        if len(x_test) > 0:
            with fits.open(output_dir+prefix+'x_predict_test.fits') as hdul:
                x_predict_test = hdul[0].data
        else:
            x_predict_test = x_test

    if do_diagnostic_plots:
        pt.diagnostic_plots(history, model, p, output_dir, x, x_train, x_test,
                            x_predict_test, x_predict_train=x_predict_train,
                            target_info_test=target_info_test,
                            target_info_train=target_info_train, prefix=prefix,
                            ticid_train=ticid_train, ticid_test=ticid_test, 
                            bottleneck_train=bottleneck_train, bottleneck=bottleneck_test,
                            plot_epoch=plot_epoch, load_bottleneck=False)

    if len(x_test) > 0:
        flux_feat = np.concatenate([flux_train, flux_test], axis=0)
        ticid_feat = np.concatenate([ticid_train, ticid_test])
        info_feat = np.concatenate([target_info_train, target_info_test])
        features=None
    else:
        flux_feat, ticid_feat, info_feat = flux_train, ticid_train, target_info_train
        features = bottleneck_train

    # -- additional iterations -------------------------------------------------
    for i in [2]: # range(1,iterations+1):
        # >> split by reconstruction error
        x, flux_train, flux_test,  x_train, x_test, ticid_train, ticid_test,\
            info_train, info_test, features, x_predict_train, x_predict_test =\
            split_reconstruction(x, flux_train, flux_test, x_train, x_test,
                                 x_predict_train, x_predict_test,
                                 ticid_train, ticid_test, target_info_train,
                                 target_info_test, features,
                                 error_threshold=error_threshold,
                                 return_highest_error_ticid=False,
                                 output_dir=output_dir, prefix=prefix)

        # >> do postprocessing on lowest reconstruction error
        post_process(x, x_train[0], x_test[0], ticid_train[0], ticid_test[0],
                     info_train[0], info_test[0], p, output_dir, sectors,
                     data_dir=data_dir, database_dir=database_dir, prefix=prefix,
                     momentum_dump_csv=momentum_dump_csv, use_rms=use_rms,
                     features=features[0],  novelty_detection=novelty_detection,
                     flux_feat=np.concatenate([flux_train[0],
                                               flux_test[0]], axis=0),
                     ticid_feat=np.concatenate([ticid_train[0], ticid_test[0]]),
                     info_feat=np.concatenate([info_train[0],
                                               info_test[0]], axis=0),
                     x_predict=np.concatenate([x_predict_train[0],
                                               x_predict_test[0]], axis=0),
                     do_summary=do_iteration_summary,
                     do_diagnostic_plots=do_diagnostic_plots)


        # >> do split_cae on highest reconstruction error
        print('-'*17)
        print('-- iteration '+str(i)+' --')
        print('-'*17)

        prefix='iteration'+str(i)+'-'
        p['latent_dim']=latent_dim[i-1]
        flux_train, flux_test, info_train, info_test, ticid_train, ticid_test =\
            flux_train[1], flux_test[1], info_train[1], info_test[1], \
            ticid_train[1], ticid_test[1]
        if run:
            x_train, x_test, x_predict_train, x_predict_test =\
                split_cae(x, flux_train, flux_test, p, info_train, info_test,
                          ticid_train, ticid_test, sectors, n_split=n_split[i-1],
                          len_var=len_var, data_dir=data_dir, database_dir=database_dir,
                          output_dir=output_dir, prefix0=prefix,
                          momentum_dump_csv=momentum_dump_csv, debug=True,
                          save_model_epoch=False, plot=do_diagnostic_plots,
                          hyperparam_opt=hyperparam_opt)


        x_predict_test = np.empty((len(flux_test), 0))
        x_predict_train = np.empty((len(flux_train), 0))
        x_train = np.empty((len(flux_train), 0))
        x_test = np.empty((len(flux_test), 0))
        features = np.empty((len(x_train)+len(x_test), 0))
        for j in range(n_split[i-1]):
            fname=output_dir+prefix+'segment'+str(j)+'-x_predict_train.fits'
            with fits.open(fname) as hdul:
                segment_predict_train = hdul[0].data
            segment_len = segment_predict_train.shape[1]
            if len(flux_test) > 0:
                fname=output_dir+prefix+'segment'+str(j)+'-x_predict_test.fits'              
                with fits.open(fname) as hdul:
                    segment_predict_test = hdul[0].data
            else:
                segment_predict_test = np.empty((0,segment_len))
            fname=output_dir+prefix+'segment'+str(j)+'-bottleneck_train.fits'                
            with fits.open(fname) as hdul:
                bottleneck_train = hdul[0].data
            fname=output_dir+prefix+'segment'+str(j)+'-bottleneck_test.fits'                
            with fits.open(fname) as hdul:
                bottleneck_test = hdul[0].data

            start = x_predict_train.shape[1]
            end = start+segment_len
            segment_train = dt.standardize(flux_train[:,start:end])
            segment_test = dt.standardize(flux_test[:,start:end])

            if do_diagnostic_plots:
                pt.diagnostic_plots(history, model, p, output_dir,
                                    x[start:start+segment_len],
                                    segment_train, segment_test,
                                    segment_predict_test,
                                    segment_predict_train,
                                    target_info_test=info_test,
                                    target_info_train=info_train,
                                    prefix=prefix+'segment'+str(j)+'-',
                                    ticid_train=ticid_train, ticid_test=ticid_test, 
                                    bottleneck_train=bottleneck_train,
                                    bottleneck=bottleneck_test,
                                    plot_epoch=plot_epoch, load_bottleneck=False)

            x_predict_train = np.append(x_predict_train, segment_predict_train, axis=1)
            x_predict_test = np.append(x_predict_test, segment_predict_test, axis=1)
            x_train = np.append(x_train, segment_train, axis=1)
            x_test = np.append(x_test, segment_test, axis=1)
            bottleneck = np.concatenate([bottleneck_train, bottleneck_test], axis=0)
            features = np.append(features, bottleneck, axis=1)


    # -- plots for last iteration ----------------------------------------------
    post_process(x, x_train, x_test, ticid_train, ticid_test,
                 info_train, info_test, p, output_dir, sectors,
                 data_dir=data_dir, database_dir=database_dir, prefix=prefix,
                 momentum_dump_csv=momentum_dump_csv, use_rms=use_rms,
                 features=features, 
                 flux_feat=np.concatenate([flux_train, flux_test], axis=0),
                 ticid_feat=np.concatenate([ticid_train, ticid_test]),
                 info_feat=np.concatenate([info_train, info_test], axis=0),
                 x_predict=np.concatenate([x_predict_train, x_predict_test], axis=0),
                 do_summary=do_iteration_summary,
                 do_diagnostic_plots=do_diagnostic_plots)


    # -- summary plots for entire ensemble -------------------------------------
    if do_ensemble_summary:

        print('Making ensemble summary plots...')
        # >> get science label for every light curve in ensemble
        ticid_label = np.empty((0,2)) # >> list of [ticid, science label]
        for i in [0, 2]: #range(iterations+1):
            fname=output_dir+'iteration'+str(i)+'-ticid_to_label.txt'
            filo = np.loadtxt(fname, dtype='str', delimiter=',')
            ticid_label = np.append(ticid_label, filo[:,:2], axis=0)


        prefix='Sector'+str(sectors[0])+'-'

        np.savetxt(output_dir+prefix+'ticid_to_label.txt', ticid_label,
                   fmt='%s', delimiter=',')

        ticid = ticid_label[:,0].astype('float')
        labels = ticid_label[:,1]

        pt.ensemble_summary_plots(ticid, labels, output_dir, data_dir,
                                  sectors, prefix)

        # # >> before making a confusion matrix, we need to assign each science 
        # # >> label a number
        # underlying_classes  = np.unique(labels)
        # assignments = []
        # for i in range(len(underlying_classes)):
        #     assignments.append([i, underlying_classes[i]])
        # assignments = np.array(assignments)

        # # >> get the predicted labels (in numbers)
        # y_pred = []
        # for i in range(len(ticid)):
        #     ind = np.nonzero(assignments[:,1] == labels[i])
        #     y_pred.append(float(assignments[ind][0][0]))
        # y_pred = np.array(y_pred)


        # # >> create confusion matrix
        # cm, assignments, ticid_true, y_true, class_info_new, recalls,\
        # false_discovery_rates, counts_true, counts_pred, precisions, accuracy,\
        # label_true, label_pred=\
        #     pt.assign_real_labels(ticid, y_pred, database_dir, data_dir,
        #                           output_dir=output_dir)

        # # >> create summary pie charts
        # pt.ensemble_summary(ticid, labels, cm, assignments, label_true, label_pred,
        #                     database_dir=database_dir, output_dir=output_dir, 
        #                     prefix=prefix, data_dir=data_dir)
        # pt.ensemble_summary_tables(assignments, recalls, false_discovery_rates,
        #                         precisions, accuracy, counts_true, counts_pred,
        #                         output_dir+prefix)
        # pt.ensemble_summary_tables(assignments, recalls, false_discovery_rates,
        #                         precisions, accuracy, counts_true, counts_pred,
        #                         output_dir+prefix, target_labels=[])

        # # >> distribution plots
        # inter, comm1, comm2 = np.intersect1d(ticid_feat, ticid_true,
        #                                      return_indices=True)
        # y_pred = labels[comm1]
        # x_true = flux_feat[comm1]

        # classes, counts = np.unique(y_true, return_counts=True)
        # classes = classes[np.argsort(counts)]
        # for class_label in classes[-20:]:
        #     pt.plot_class_dists(assignments, ticid_true, y_pred, y_true,
        #                         data_dir, sectors, true_label=class_label,
        #                         output_dir=output_dir+'Sector'+str(sectors[0]))

def iterative_cae_clustering(ensemble_dir, data_dir, sectors=[], num_iter=2,
                             n_clusters=[100,100,100], first_iter_only=False):
    '''Do clustering'''
    for sector in sectors:
        print('Sector '+ str(sector))
        output_dir=ensemble_dir+'Ensemble-Sector_'+str(sector)+'/'

        flux, time, ticid, target_info = \
            dt.load_data_from_metafiles(data_dir, sector,
                                        output_dir=output_dir)
        flux = dt.normalize(flux)

        iterations = [0,2] # list(range(num_iter))
        if first_iter_only:
            iterations = [0]
        files = []
        for i in iterations: # reverse range(num_iter)

            if first_iter_only:
                with fits.open(output_dir+'iteration0-bottleneck_train.fits') as f:
                    ticid = f[1].data
            else:
                if i == iterations[-1]:
                    ticid=np.loadtxt(output_dir+'iteration'+str(iterations[-2])+\
                                     '-ticid_highest_error_train.txt')
                else:
                    ticid=np.loadtxt(output_dir+'iteration'+str(i)+\
                                     '-ticid_lowest_error_train.txt')

            prefix = 'iteration'+str(i)+'-'
            
            if first_iter_only:
                out_file = output_dir+prefix+'all-gmm_labels-n'+\
                           str(n_clusters[i])+'.txt'
            else:
                out_file = output_dir+prefix+'gmm_labels-n'+\
                           str(n_clusters[i])+'.txt'
   
            if os.path.exists(out_file):
                _, y_pred = np.loadtxt(out_file)
            else:

                fnames = fm.filter(os.listdir(output_dir),
                                   prefix+'*bottleneck_train.fits')

                features = []
                ticid_feat = []
                for fname in fnames:
                    with fits.open(output_dir+fname) as f:
                        features.append(f[0].data)
                        ticid_feat.extend(f[1].data)
                features = np.concatenate(features, axis=1)

                # if use_tess_features

                if first_iter_only:
                    ticid = np.array(ticid_feat)
                else:
                    inter, comm1, comm2 = np.intersect1d(ticid_feat, ticid,
                                                         return_indices=True)
                    features = features[comm1]
                    ticid = ticid[comm2]
                
                gmm = GaussianMixture(n_components=n_clusters[i])
                y_pred = gmm.fit_predict(features)
                np.savetxt(out_file,
                           np.array([ticid, y_pred]))

            assignments, ticid_label = \
                pt.assign_real_labels(ticid, y_pred, data_dir=data_dir,
                                        output_dir=output_dir, prefix=prefix)
            # with open(output_dir+prefix+'ticid_to_label.txt', 'w') as f:
            #     for i in range(len(y_pred)):
            #         ind = np.nonzero(assignments[:,0].astype('float') == y_pred[i])
            #         if len(ind[0]) == 0:
            #             f.write(str(ticid[i])+',NONE\n')
            #         else:
            #             f.write(str(ticid[i])+','+str(assignments[:,1][ind][0])+'\n')
            # print('Saved '+output_dir+prefix+'ticid_to_label.txt')
            # files.append(output_dir+prefix+'ticid_to_label.txt')

        os.system('cat '+' '.join(files)+' > '+\
                  output_dir+'Sector'+str(sector)+'-ticid_to_label.txt')
        print('Saved '+output_dir+'Sector'+str(sector)+'-ticid_to_label.txt')

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::    
# :: Partitioning training and and testing sets ::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def split_data_features(flux, features, time, ticid, target_info,
                        classes=False, train_test_ratio = 0.9,
                        cutoff=16336, supervised=False, interpolate=False,
                        resize_arr=False, truncate=False):

    # # >> truncate (must be a multiple of 2**num_conv_layers)
    # if truncate:
    #     new_length = int(np.shape(flux)[1] / \
    #                  (2**(np.max(p['num_conv_layers'])/2)))*\
    #                  int((2**(np.max(p['num_conv_layers'])/2)))
    #     flux=np.delete(flux,np.arange(new_length,np.shape(flux)[1]),1)
    #     time = time[:new_length]

    # >> shuffle array
    inds = np.arange(len(flux))
    random.Random().shuffle(inds)
    flux = flux[inds]
    ticid = ticid[inds]
    target_info = np.array(target_info)[inds]
    features = features[inds]

    # >> split test and train data
    if supervised:
        # >> partitions so that there are approximately equal amount of
        # >> examples from each class in both training and testing sets
                   
        train_inds = []
        test_inds = []
        class_types, counts = np.unique(classes, return_counts=True)
        num_classes = len(class_types)
        #  = min(counts)
        y_train = []
        y_test = []
        for i in range(len(class_types)):
            inds = np.nonzero(classes==i)[0]
            num_train = int(len(inds)*train_test_ratio)
            train_inds.extend(inds[:num_train])
            test_inds.extend(inds[num_train:])
            labels = np.zeros((len(inds), num_classes))
            labels[:,i] = 1.
            y_train.extend(labels[:num_train])
            y_test.extend(labels[num_train:])

        y_train = np.array(y_train)
        y_test = np.array(y_test)
        x_train = np.copy(features[train_inds])
        x_test = np.copy(features[test_inds])
        flux_train = np.copy(flux[train_inds])
        flux_test = np.copy(flux[test_inds])
        ticid_train = np.copy(ticid[train_inds])
        ticid_test = np.copy(ticid[test_inds])
        target_info_train = np.copy(target_info[train_inds])
        target_info_test = np.copy(target_info[test_inds])
    else: # >> unsupervised
        split_ind = int(train_test_ratio*np.shape(flux)[0])
        x_train = np.copy(features[:split_ind])
        x_test = np.copy(features[split_ind:])
        flux_train = np.copy(flux[:split_ind])
        flux_test = np.copy(flux[split_ind:])
        ticid_train = np.copy(ticid[:split_ind])
        ticid_test = np.copy(ticid[split_ind:])
        target_info_train = np.copy(target_info[:split_ind])
        target_info_test = np.copy(target_info[split_ind:])
        y_test, y_train = [False, False]
        
        
    if resize_arr:
        x_train =  np.resize(x_train, (np.shape(x_train)[0],
                                       np.shape(x_train)[1], 1))
        x_test =  np.resize(x_test, (np.shape(x_test)[0],
                                       np.shape(x_test)[1], 1))
    return x_train, x_test, y_train, y_test, flux_train, flux_test,\
        ticid_train, ticid_test, target_info_train, target_info_test, time

def split_data(flux, x, ticid, target_info, time, p,
               train_test_ratio = 0.9,
               supervised=False, classes=False, interpolate=False,
               resize_arr=False, truncate=True):

    if truncate:
        # >> dim reduced each iteration
        reduction_factor = np.max(p['pool_size'])* np.max(p['strides'])**np.max(p['num_consecutive'] )
        # reduction_factor = np.max(p['pool_size'])* np.max(p['strides'])
        
        num_iter = np.max(p['num_conv_layers'])/2
        tot_reduction_factor = reduction_factor**num_iter
        if p['fully_conv']:
            # >> 1 more conv layer
            tot_reduction_factor = tot_reduction_factor * np.max(p['strides'])
        new_length = int(x.shape[1] / tot_reduction_factor)*\
                     int(tot_reduction_factor)
        x = np.delete(x,np.arange(new_length,x.shape[1]),1)
        time_plot = time
        time = time[:new_length]         

    # >> split test and train data
    if supervised:
        train_inds = []
        test_inds = []
        class_types, counts = np.unique(classes, return_counts=True)
        num_classes = len(class_types)
        #  = min(counts)
        y_train = []
        y_test = []
        for i in range(len(class_types)):
            inds = np.nonzero(classes==i)[0]
            num_train = int(len(inds)*train_test_ratio)
            train_inds.extend(inds[:num_train])
            test_inds.extend(inds[num_train:])
            labels = np.zeros((len(inds), num_classes))
            labels[:,i] = 1.
            y_train.extend(labels[:num_train])
            y_test.extend(labels[num_train:])

        y_train = np.array(y_train)
        y_test - np.array(y_test)
        x_train = np.copy(flux[train_inds])
        x_test = np.copy(flux[test_inds])
        ticid_train = np.copy(ticid[train_inds])
        ticid_test = np.copy(ticid[test_inds])
        target_info_train = np.copy(target_info[train_inds])
        target_info_test = np.copy(target_info[test_inds])        
    else:
        split_ind = int(train_test_ratio*np.shape(flux)[0])
        flux_train = flux[:split_ind]
        flux_test = flux[split_ind:]
        x_train = x[:split_ind]
        x_test = x[split_ind:]
        ticid_train = ticid[:split_ind]
        ticid_test = ticid[split_ind:]
        target_info_train = target_info[:split_ind]
        target_info_test = target_info[split_ind:]    
        y_test, y_train = [None, None]
        
    if resize_arr:
        x_train =  np.resize(x_train, (np.shape(x_train)[0],
                                       np.shape(x_train)[1], 1))
        x_test =  np.resize(x_test, (np.shape(x_test)[0],
                                       np.shape(x_test)[1], 1))
    return flux_train, flux_test, x_train, x_test, y_train, y_test, ticid_train,\
        ticid_test, target_info_train, target_info_test, time, time
    
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Mock data :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def gaussian(x, a, b, c):
    '''a = height, b = position of center, c = stdev'''
    import numpy as np
    return a * np.exp(-(x-b)**2 / (2*c**2))

def signal_data(training_size = 10000, test_size = 100, input_dim = 100,
                 time_max = 30., noise_level = 0.0, height = 1., center = 15.,
                 stdev = 0.8, h_factor = 0.2, center_factor = 5.,
                 reshape=True):
    '''Generate training data set with flat light curves and gaussian light
    curves, with variable height, center, noise level as a fraction of gaussian
    height)
    '''

    x = np.empty((training_size + test_size, input_dim))
    # y = np.empty((training_size + test_size))
    y = np.zeros((training_size + test_size, 2))
    l = int(np.shape(x)[0]/2)
    
    # >> no peak data
    x[:l] = np.zeros((l, input_dim))
    # y[:l] = 0.
    y[:l, 0] = 1.
    

    # >> with peak data
    time = np.linspace(0, time_max, input_dim)
    for i in range(l):
        a = height + h_factor*np.random.normal()
        b = center + center_factor*np.random.normal()
        x[l+i] = gaussian(time, a = a, b = b, c = stdev)
    # y[l:] = 1.
    y[l:, 1] = 1.

    # >> add noise
    x += np.random.normal(scale = noise_level, size = np.shape(x))
    
    # >> normalize
    # x = x / np.median(x, axis = 1, keepdims=True) - 1.

    # >> partition training and test datasets
    x_train = np.concatenate((x[:int(training_size/2)], 
                              x[l:-int(test_size/2)]))
    y_train = np.concatenate((y[:int(training_size/2)], 
                              y[l:-int(test_size/2)]))
    x_test = np.concatenate((x[int(training_size/2):l], 
                             x[-int(test_size/2):]))
    y_test = np.concatenate((y[int(training_size/2):l], 
                             y[-int(test_size/2):]))

    if reshape:
        x_train = np.reshape(x_train, (np.shape(x_train)[0],
                                       np.shape(x_train)[1], 1))
        x_test = np.reshape(x_test, (np.shape(x_test)[0],
                                     np.shape(x_test)[1], 1))

    return time, x_train, y_train, x_test, y_test

def no_signal_data(training_size = 10000, test_size = 100, input_dim = 100,
                   noise_level = 0., min0max1=True, reshape=False):
    '''Generating flat light curves with some noise.'''
    import numpy as np

    x = np.empty((training_size + test_size, input_dim))
    y = np.empty((training_size + test_size))
    l = int(np.shape(x)[0]/2)
    
    # >> no peak data
    if min0max1:
        x = np.zeros(np.shape(x))
    else:
        x = np.ones(np.shape(x))
    y = 0.

    # >> add noise
    x += np.random.normal(scale = noise_level, size = np.shape(x))

    # >> partition training and test datasets
    x_train = np.concatenate((x[:int(training_size/2)], 
                              x[l:-int(test_size/2)]))
    y_train = np.concatenate((y[:int(training_size/2)], 
                              y[l:-int(test_size/2)]))
    x_test = np.concatenate((x[int(training_size/2):l], 
                             x[-int(test_size/2):]))
    y_test = np.concatenate((y[int(training_size/2):l], 
                             y[-int(test_size/2):]))

    if reshape:
        x_train = np.reshape(x_train, (np.shape(x_train)[0],
                                       np.shape(x_train)[1], 1))
        x_test = np.reshape(x_test, (np.shape(x_test)[0],
                                     np.shape(x_test)[1], 1))
    
    return x_train, y_train, x_test, y_test

       
        
def get_high_freq_mock_data(p=None, dataset_size=10000, train_test_ratio=0.9,
                            input_dim = 18688, xmax=20, f_mean=3.5, f_std=0.5,
                            noise_level=0.1, reshape=False, truncate=False,
                            hyperparam_opt=False):
    '''Generate training and testing datasets with high frequency sine curves.
    * f_min : in days^-1
    * f_max : in days^-1
    '''

    x = np.linspace(0, 20, input_dim)
    
    # >> get frequencies of each light curve
    random_values = np.random.randn(dataset_size)
    freq = f_mean + random_values*f_std

    flux = []
    for i in range(dataset_size):        
        flux.append(np.sin(2*np.pi*freq[i] * x))
    flux = np.array(flux)

    # >> add a little noise
    flux += np.random.normal(scale = noise_level, size = np.shape(flux))
    
    # >> truncate
    if truncate:
        # >> dim reduced each iteration
        if hyperparam_opt:
            reduction_factor = 2
        else: # !!
            reduction_factor = p['pool_size']* p['strides']**p['num_consecutive'] 
        num_iter = np.max(p['num_conv_layers'])/2
        tot_reduction_factor = reduction_factor**num_iter
        new_length = int(np.shape(flux)[1] / \
                     tot_reduction_factor)*\
                     int(tot_reduction_factor)
        flux=np.delete(flux,np.arange(new_length,np.shape(flux)[1]),1)
        x = x[:new_length]           

    # >> standardize
    flux = dt.standardize(flux)
    
    # >> partition data
    split_ind = int(train_test_ratio*np.shape(flux)[0])
    x_train = flux[:split_ind]
    x_test = flux[split_ind:]
    
    if reshape:
        x_train = np.reshape(x_train, (np.shape(x_train)[0],
                                       np.shape(x_train)[1], 1))
        x_test = np.reshape(x_test, (np.shape(x_test)[0],
                                     np.shape(x_test)[1], 1))

    return x, x_train, x_test     

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Miscellaneous ML helper functions :::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def load_model(path):
    # !! TODO save history dictionary into txt file
    from tensorflow import keras
    model = keras.models.load_model(path)
    return model

def model_modifier(m, bottleneck_ind):
    new_input = Input(shape = (m.layers[0].output.shape[1],))
    x = new_input
    for i in range(1, bottleneck_ind+1):
        x = m.layers[i](x)
    new_model = Model(new_input, x)

    new_model.layers[-1].activation = tf.keras.activations.linear

    return new_model

def loss(output):
    return (output[0][0], output[1][0], output[2][0])

def make_X(flux_train, ticid_train, ticid=[89305963, 147200394, 350716053]):
    X = []
    for i in ticid:
        ind = np.nonzero(ticid_train == float(i))[0][0]
        X.append(flux_train[ind])
    return np.array(X)


def get_activations(model, x_test, input_rms = False, rms_test = False,
                    ind=None):
    '''Returns intermediate activations.'''
    from keras.models import Model
    layer_outputs = [layer.output for layer in model.layers][1:]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    if input_rms:
        activations = activation_model.predict([x_test, rms_test])
    else:
        if type(ind) == type(None):
            activations = activation_model.predict(x_test)   
        else:
            activations = activation_model.predict(x_test[ind].reshape(1,-1))
    return activations

def get_bottleneck(model, x_test, p, save=False, output_dir='', vae=False):
    if vae:
        bottleneck_layer = vae_encoder.predict(x_test, p)

    else:
        bottleneck_layer = model.get_layer('bottleneck').output

    activation_model = Model(inputs=model.input,
                             outputs=bottleneck_layer)
    bottleneck = activation_model.predict(x_test)        
    
    if save:
        np.save(output_dir+'bottleneck_train.npy', bottleneck)
    else:
        return bottleneck

def compile_model(model, params):

    if params['optimizer'] == 'adam':
        # opt = optimizers.adam(lr = params['lr'], 
        #                       decay=params['lr']/params['epochs'])
        opt = optimizers.Adam(lr = params['lr'], 
                              decay=params['lr']/params['epochs'])        
    elif params['optimizer'] == 'adadelta':
        # opt = optimizers.adadelta(lr = params['lr'])
        opt = optimizers.Adadelta(lr = params['lr'])
        
    model.compile(optimizer=opt, loss=params['loss'])


def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same',
                    kernel_initializer='random_normal', activation='relu',
                    kernel_regularizer='l2', activity_regularizer='l2',
                    bias_regularizer='l2'):
    """Conv1DTranpose has not been implemented in Keras, so I convert the 1D
    tensor to a 2D tensor with an image width of 1, then apply
    Conv2Dtranspose.
    
    This code is inspired by this:
    https://stackoverflow.com/questions/44061208/how-to-implement-the-
    conv1dtranspose-in-keras
        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
    """
    
    # >> save shape of input tensor
    dim = input_tensor.get_shape().as_list()[1]

    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1),
                        strides=(strides, 1), padding=padding,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        activity_regularizer=activity_regularizer)(x)
    
    # >> explicitly set the shape, because TensorFlow hates me >:(  
    x = Lambda(lambda x: tf.ensure_shape(x, (None, strides*dim, 1, filters)))(x)
    
     
    # x.set_shape((None, strides*dim, 1, filters))
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    # x = x[:,:,0] # >> convert to 1D    
    
    x = Activation(activation)(x)
    return x

def swish(x, beta=1):
    '''https://www.bignerdranch.com/blog/implementing-swish-activation-function
    -in-keras/'''
    from keras.backend import sigmoid
    return (x*sigmoid(beta*x))

def mean_cubic_loss(y_true, y_pred): 
    # >> now compute MSE
    loss = K.mean(K.pow(K.abs(y_true - y_pred), 3))
    return loss

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Supervised learning algorithms (may not be up to date) ::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def cnn(x_train, y_train, x_test, y_test, params, num_classes=4):
    # -- encoding -------------------------------------------------------------
    encoded = cae_encoder(x_train, params)
    
    # -- supervised mode: softmax ---------------------------------------------
    x = Dense(int(num_classes),
          activation='softmax')(encoded.output)
    model = Model(encoded.input, x)
    model.summary()
        
    # -- compile model --------------------------------------------------------
    compile_model(model, params)

    # -- train model ----------------------------------------------------------
    history = model.fit(x_train, y_train, epochs=params['epochs'],
                        batch_size=params['batch_size'], shuffle=True,
                        validation_data=(x_test,y_test))
    
    return history, model

def cnn_mock(x_train, y_train, x_test, y_test, params, num_classes = 2):

    # -- encoding -------------------------------------------------------------
    encoded = cae_encoder(x_train, params)
    
    # -- supervised mode: softmax ---------------------------------------------
    x = Dense(int(num_classes),
          activation='softmax')(encoded.output)
    model = Model(encoded.input, x)
    model.summary()
        
    # -- compile model --------------------------------------------------------
    compile_model(model, params)

    # -- train model ----------------------------------------------------------
    history = model.fit(x_train, y_train, epochs=params['epochs'],
                        batch_size=params['batch_size'], shuffle=True,
                        validation_data=(x_test,y_test))
    
    return history, model

def mlp(x_train, y_train, x_test, y_test, params, resize=True):
    '''a simple classifier using fully-connected layers'''

    num_classes = np.shape(y_train)[1]
    input_dim = np.shape(x_train)[1]
    if resize:
        input_img = Input(shape = (input_dim,1))
        x = Flatten()(input_img)
    else:
        input_img = Input(shape = (input_dim,))
        x = input_img
    for i in range(len(params['hidden_units'])):
        x = Dense(params['hidden_units'][i],activation=params['activation'])(x)
    x = Dense(num_classes, activation='softmax')(x)
        
    model = Model(input_img, x)
    model.summary()
    compile_model(model, params, mlp=True)

    history = model.fit(x_train, y_train, epochs=params['epochs'],
                            batch_size=params['batch_size'], shuffle=True,
                            validation_data=(x_test, y_test))
        
    return history, model

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# def load_DAE_bottleneck(savepath, ticid):
#     print("Loading DAE-learned features...")
#     fnames = [f for f in os.listdir(savepath) if '_bottleneck_train.npy' in f]
    
#     with fits.open(savepath+'bottleneck_train.fits') as hdul:
#         bottleneck_train = hdul[0].data
#         ticid_train = hdul[1].data
        
#     fname = savepath+'bottleneck_test.fits'
#     if os.path.exists(fname):
#         with fits.open(fname) as hdul:
#             bottleneck_test = hdul[0].data
#             ticid_test = hdul[1].data
#         bottleneck_train = np.concatenate([bottleneck_train,
#                                            bottleneck_test], axis=0)
#         ticid_train = np.concatenate([ticid_train, ticid_test])

#     sorted_inds = np.argsort(ticid)
#     # >> intersect1d returns sorted arrays, so
#     # >> ticid == ticid[sorted_inds][np.argsort(sorted_inds)]
#     new_inds = np.argsort(sorted_inds)
#     _, comm1, comm2 = np.intersect1d(ticid, ticid_train, return_indices=True)

#     ticid = ticid_train[comm2]
#     bottleneck_train = bottleneck_train[comm2]
#     # !!
#     sectors = np.ones(np.shape(ticid)).astype('int')
    
#     return bottleneck_train, ticid, sectors

# def load_reconstructions(output_dir, ticid):
#     filo = fits.open(output_dir + 'x_predict_train.fits')
#     rcon = filo[0].data
#     ticid_filo = filo[1].data

#     sorted_inds = np.argsort(ticid)
#     new_inds = np.argsort(sorted_inds)
#     _, comm1, comm2 = np.intersect1d(ticid, ticid_filo, return_indices=True)
#     rcon = rcon[comm2][new_inds]

#     return rcon


            
