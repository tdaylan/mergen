"""
Created on Thu Oct  8 20:27:13 2020

mergen.py
@authors: Lindsey Gordon (@lgordon), Emma Chickles (@emmachickles), Tansu Daylan
          (@tdaylan)

The "Mergen" pipeline performs unsupervised classification and outlier detection
using TESS light curves. To do this, we use convolutional autoencoders and 
feature engineering to produce low-dimensional representations of light curves.
Note that we're undergoing some major restructuring, so documentation is
currently spotty!

The Mergen methods are organized into five main sections:
1) Initialization
2) Data and Preprocessing
3) Feature Generation
4) Clustering and Outlier Analysis
5) Loading Mergen Products

To Do List:
* Set up pipeline function (kind of a run-all thing)
* Include example script?
* Include option to process multiple sectors
"""

from .__init__ import *
from . import data_utils    as dt
from . import catalog_utils as ct
from . import learn_utils   as lt
from . import plot_utils    as pt
from . import feature_utils as ft

class mergen(object):
    """ Main mergen class. Initialize this to work with everything else
    conveniently. """

    # ==========================================================================
    # == Initialization ========================================================
    # ==========================================================================

    def __init__(self, datapath, savepath, datatype, sector, mdumpcsv=None,
                 filelabel=None, runiter=False, numiter=1, numclstr=100,
                 parampath=None):
        """Creates mergen object from which most common routines can easily be
        run
        Parameters:
            * datapath: string, directory data is stored in
            * savepath: string, directory produces will be saved
            * datatype: string, indicates type of data being worked with.
                        options are: 
                        "SPOC", "FFI-Lygos", "FFI-QLP", "FFI-eleanor"
            * mdumpcsv : string, path to csv file containing TESS momentum dumps
            * filelabel : string, if you want to have all plots/files/folders
                          labelled specially        
            * sector  : int, TESS Observational Sector number (1,2,...,26,...)
            * runiter : bool, if True, runs iterative CAE scheme
            * numiter : int, number of iterations in the iterative CAE scheme
            * numclstr : int, number of clusters assumed by the GMM clustering
                         algorithm
            * parampath : string, path to txt file containing autoencoder
                        parameters
        """
        self.sector   = sector
        self.numclstr = numclstr

        self.datapath = datapath
        self.savepath = savepath
        self.datatype = datatype # >> SPOC or FFI
        self.ensbpath = self.savepath+'Ensemble-Sector_'+str(self.sector)+'/'

        if mdumpcsv is not None:
            self.mdumpcsv = mdumpcsv
        else:
            self.mdumpcsv = datapath + 'Table_of_momentum_dumps.csv'

        if filelabel is not None:
            self.filelabel = filelabel
        else:
            self.filelabel = "mergen"

        if parampath is not None:
            self.parampath = parampath
        else:
            self.parampath = savepath + 'caehyperparams.txt'

        # >> iterative scheme
        self.runiter = runiter
        self.numiter = numiter
        
        self.initiate_folder()

    def initiate_folder(self):
        """Make all the big folders"""
        print("Setting up CAE folder")
        self.CAEpath = self.ensbpath + "CAE/"
        dt.create_dir(self.CAEpath)

        print("Setting up DAE folder")
        self.DAEpath = self.ensbpath + "DAE/"
        dt.create_dir(self.DAEpath)j

        print("Setting up ENF folder")
        self.ENFpath = self.ensbpath + "ENF/"
        dt.create_dir(self.ENFpath)

    def run_all(self):
        return

    # ==========================================================================
    # == Data and Preprocessing ================================================
    # ==========================================================================

    def load_lightcurves_local(self):
        """Load in data saved in metafiles on datapath"""
        #check for self.datatype to determine loading scheme. 
        #figure out consistent stuff for FFI original locations
        if self.datatype == "FFI-Lygos":
            self.time, self.flux, self.errors, self.objid = \
            dt.load_all_lygos(self.datapath)
        elif self.datatype == "SPOC":
            self.flux, self.time, self.objid, self.target_info = \
            dt.load_data_from_metafiles(self.datapath, self.sector)
        
    def download_lightcurves(self):
        """Downloads and process light SPOC light curves, if not already
        downloaded."""
        dt.bulk_download_helper(datapath, sector=self.sector)
        dt.data_access_sector_by_bulk(datapath, sector=self.sector)

    def data_clean(self): 
        self.flux = dt.normalize(self.flux)

    def preprocess_dae(self):
        """Preprocessing for deep autoencoder. Returns:
            * pgram : LS periodograms"""
        self.pgram, self.flux, self.objid, self.target_info, self.time=\
        lt.DAE_preprocessing(self.flux, self.time, self.parampath, self.objid,
                             self.target_info, data_dir=self.datapath,
                             output_dir=self.ensbpath)
    def preprocess_cae(self):
        self.pflux, self.pflux_test = \
        lt.autoencoder_preprocessing(self.flux, self.time, self.parampath,
                                     ticid=self.objid, data_dir=self.datapath,
                                     output_dir=self.ensbpath)
    
    # ==========================================================================
    # == Feature Generation ====================================================
    # ==========================================================================

    def generate_engineered(self, version = 0, save = True):
        """ Run engineered feature creation"""
        self.feats = ft.create_save_featvec_homogenous_time(self.ENFpath,
                                                            self.time, 
                                                            self.flux,
                                                            self.filelabel,
                                                            version=version,
                                                            save=save)

    def generate_dae_features(self):
        """Trains deep autoencoder to extract representative features from
        periodograms."""
        self.model, self.hist, self.feats, self.rcon = \
        lt.deep_autoencoder(self.pgram, self.pgram, parampath=self.parampath,
                            ticid_train=self.objid, output_dir=self.ensbpathpath)

    def generate_cae_features(self):
        """Train convolutional autoencoder to extract representative
        features from lightcurves.
        Returns: 
            * model : Keras Model object
            * hist : Keras history dictionary
            * feats : CAE-derived features
            * rcon : reconstructions of the input light curves"""        
        self.model, self.hist, self.feats, self.feats_test, self.rcon_test, \
        self.recon = lt.conv_autoencoder(xtrain=self.pflux, ytrain=self.pflux, 
                                         params=self.parampath,
                                         output_dir=self.ensbpath,
                                         ticid_train=self.objid)
        return

    # ==========================================================================
    # == Clustering and Outlier Analysis =======================================
    # ==========================================================================

    def generate_clusters(self):
        """Run clustering algorithm on feature space.
        Returns:
            * clstr : array of cluster numbers, shape=(len(objid),)"""
        print('Performing clustering analysis in feature space...')
        self.clstr = lt.run_gmm(self.objid, self.feats, numclstr=self.numclstr,
                                savepath=self.ensbpath, runiter=self.runiter,
                                numiter=self.numiter)

    def generate_tsne(self):
        """Reduces dimensionality of feature space for visualization."""
        self.tsne = lt.load_tsne(self.ensbpath)

    def generate_predicted_otypes(self):
        """Predicts object types using known classifications in SIMBAD, ASAS-SN,
        and GCVS.
        Returns:
            * potd : predicted object type dictionary
            * potype : array of predicted object types, shape=(len(objid),)"""
        self.potd, self.potype = lt.label_clusters(self.ensbpath, self.sector,
                                                   self.objid, self.clstr,
                                                   self.totype, self.numtot,
                                                   self.totd)

    def produce_clustering_visualizations(self):
        '''Produces t-SNEs, confusion matrices, distribution plots, ensemble
        summary pie charts.'''
        pt.produce_clustering_visualizations(self.feats, self.numpot, self.tsne,
                                             self.ensbpath, self.potd,
                                             self.totd)


    def generate_novelty_scores(self):
        """
        Returns:
            * nvlty : novelty scores, shape=(len(objid),)"""
        print("Generating novelty scores...")
        self.nvlty = pt.generate_novelty_scores(self.feats, self.objid,
                                                self.ensbpath)

    def produce_novelty_visualizations(self):
        print("Producing novelty visualizations...")
        pt.produce_novelty_visualizations(self.nvlty, self.ensbpath, self.time,
                                          self.flux, self.objid)

    # ==========================================================================
    # == Loading Mergen Products ===============================================
    # ==========================================================================

    def load_features(self, typeFeatures):
        """ Load in feature metafiles stored in the datapath"""
        if typeFeatures == "ENF":
            self.feats = dt.load_ENF_feature_metafile(self.ENFpath)
        elif typeFeatures == "CAE": 
            print("Loading CAE-learned features...")
            self.feats = lt.load_bottleneck_from_fits(self.ensbpath, self.objid,
                                                      self.runiter, self.numiter)
        elif typeFeatures == "DAE": 
            print("Loading DAE-learned features...")
            self.feats = lt.load_DAE_bottleneck(self.ensbpath, self.objid)

    def load_gmm_clusters(self):
        """ clstr : array of cluster numbers, shape=(len(objid),)"""
        print('Loading GMM clustering results...')
        self.clstr = lt.load_gmm_from_txt(self.ensbpath, self.objid,
                                           self.runiter, self.numiter, 
                                           self.numclstr)

    def load_true_otypes(self):
        """ totype : true object types"""
        print('Loading ground truth object types...')
        self.totype = dt.load_otype_true_from_datadir(self.datapath,
                                                      self.sector,
                                                      self.objid)

    def numerize_true_otypes(self):
        """
        * unqtot : unique true object types
        * totd   : true object type dictionary, e.g. {0: 'RR-Lyrae', 1:...}
        * numtot : numerized true object types, shape=(len(objid),)
        """
        unqtot = np.unique(self.totype)
        self.totd = {i: unqtot[i] for i in range(len(unqtot))}
        self.numtot = np.array([np.nonzero(unqtot == ot)[0][0] for \
                                ot in self.totype])

    def load_pred_otypes(self):
        """ potype : predicted object types"""
        print('Loading predicted object types...')
        self.potype = dt.load_otype_pred_from_txt(self.ensbpath, self.sector,
                                                  self.objid)

    def numerize_pred_otypes(self):
        """
        * unqpot : unique predicted object types
        * potd   : predicted object type dictionary, e.g. {0: 'RR-Lyrae', 1:...}
        * numpot : numerized predicted object types, shape=(len(objid),)
        """
        unqpot = np.unique(self.potype)
        self.potd = {i: unqpot[i] for i in range(len(unqpot))}
        self.numpot = np.array([np.nonzero(unqpot == ot)[0][0] for \
                                ot in self.potype])

