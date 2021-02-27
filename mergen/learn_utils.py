# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:50:41 2021

@author: Lindsey Gordon, Emma Chickles
learn_utils.py

To Do List: 
    - Better function documentation
    - Clean up usability of param searches
    - Kmeans param scan?
"""
import sklearn
from init import *
####### SIMPLEST WRAPPERS #####

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
    
def run_GMM(features, n_components = 2):
    from sklearn.mixture import GaussianMixture
    GMM = GaussianMixture(n_components=n_components, random_state=0).fit(features)
    return GMM
    
def run_LOF(features, n_neighbors = 20, p = 2, metric = 'minkowski', contamination = 0.1,
            algorithm = 'auto'):
    from sklearn.neighbors import LocalOutlierFactor
    
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, p=p, metric=metric,
                             contamination=contamination, algorithm=algorithm)
    clf.fit_predict(features)
    negative_factor = clf.negative_outlier_factor_
    
    lof = -1 * negative_factor
    return lof
    
    
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
                                acc = pf.plot_confusion_matrix(ticid, db.labels_,
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
                                pf.quick_plot_classification(time, flux,
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
                                    pf.plot_pca(bottleneck, db.labels_,
                                                output_dir=output_dir,
                                                prefix=prefix)
                                
                                if tsne:
                                    print('Plot t-SNE...')
                                    pf.plot_tsne(bottleneck, db.labels_,
                                                 output_dir=output_dir,
                                                 prefix=prefix)
                                # if tsne_clustering:
                                    
                                    
                            plt.close('all')
                            param_num +=1
    print("Plot paramscan metrics...")
    pf.plot_paramscan_metrics(output_dir+'dbscan-', parameter_sets, 
                              silhouette_scores, db_scores, ch_scores)
    #print(len(parameter_sets), len(num_classes), len(num_noisy), num_noisy)

    pf.plot_paramscan_classes(output_dir+'dbscan-', parameter_sets, 
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

def quick_hdbscan_param_search(features, min_samples=[2,3,4,5,6,7,8,15,50],
                               min_cluster_size=[2,3,5,15,50,100],
                               metric=['all'], p0=[1,2,3,4], output_dir='./'):
    
    import hdbscan
    with open(output_dir + 'hdbscan_param_search.txt', 'a') as f:
        f.write('{} {} {} {} {} {} {}\n'.format("min_cluster_size", "min_samples",
                                       "metric", "p", 'num_classes', 
                                       'num_noise', 'other_classes'))    
    if metric[0] == 'all':
        metric = list(hdbscan.dist_metrics.METRIC_MAPPING.keys())
        metric.remove('seuclidean')
        metric.remove('mahalanobis')
        metric.remove('wminkowski')
        metric.remove('haversine')
        metric.remove('cosine')
        metric.remove('arccos')
        metric.remove('pyfunc')        
        
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
                    classes, counts = np.unique(clusterer.labels_, return_counts=True)
                    
                    with open(output_dir + 'hdbscan_param_search.txt', 'a') as f:
                        f.write('{} {} {} {} {} {} {} {}\n'.format(min_cluster_size[i],
                                                       min_samples[k],
                                                       metric[j], p[n],
                                                       len(np.unique(classes))-1, 
                                                       counts[0], classes, counts))

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
                        hdu.writeto(output_dir + 'HDBSCAN_res'+str(param_num)+'.fits')
                    
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
                            acc = pf.plot_confusion_matrix(ticid, labels,
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
                        pf.quick_plot_classification(time, flux,ticid,target_info, 
                                                     features, labels,path=output_dir,
                                                     prefix=prefix,
                                                     title=title,
                                                     database_dir=database_dir,
                                                     single_file=single_file)
                    
                        pf.plot_cross_identifications(time, flux, ticid,
                                                      target_info, features,
                                                      labels, path=output_dir,
                                                      database_dir=database_dir,
                                                      data_dir=data_dir)
                        pf.plot_confusion_matrix(ticid, labels,
                                                  database_dir=database_dir,
                                                  single_file=single_file,
                                                  output_dir=output_dir,
                                                  prefix=prefix+'merge', merge_classes=True,
                                                  labels=[], parents=parents) 
                    
                        if pca:
                            print('Plot PCA...')
                            pf.plot_pca(features, labels,
                                        output_dir=output_dir,
                                        prefix=prefix)
                                    
                        if tsne:
                            print('Plot t-SNE...')
                            pf.plot_tsne(features,labels,
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
    
            
