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

    def __init__(self, datapath, savepath, datatype, sector,
                 ENF=True, CAE=True, DAE=True, 
                 mdumpcsv=None, filelabel=None, runiter=False, numiter=1,
                 numclstr=100, parampath=None):
        """Creates mergen object from which most common routines can easily be
        run
        Parameters:
            * datapath: string, directory data is stored in
            * savepath: string, directory produces will be saved
            * datatype: string, indicates type of data being worked with.
                        options are: 
                        "SPOC", "FFI-Lygos", "FFI-QLP", "FFI-eleanor"
            * sector : int, TESS Observation Sector number
            
            * ENF, CAE, DAE : booleans, feature generation methods
              * ENF : engineered features
              * CAE : features extracted from time-series data using a 
                      convolutional autoencoder
              * DAE : features extracted from LS-periodograms using a deep
                      autoencoder
            
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
        
        self.ENF = ENF
        self.CAE = CAE
        self.DAE = DAE
        
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
        """Create directories for each of the desired feature generation
        methods."""
        if self.CAE:
            self.CAEpath = self.ensbpath + "CAE/"
            dt.create_dir(self.CAEpath)

        if self.DAE:
            self.DAEpath = self.ensbpath + "DAE/"
            dt.create_dir(self.DAEpath)

        if self.ENF:
            self.ENFpath = self.ensbpath + "ENF/"
            dt.create_dir(self.ENFpath)

    def run(self):
        # >> load data
        self.load_lightcurves_local()
        
        # >> preprocessing and feature generation
        if self.DAE:
            self.data_preprocess("DAE")
            self.generate_dae_features()
            self.run_feature_analysis("DAE")
            self.run_vis("DAE")
        if self.CAE:
            self.data_preprocess("CAE")
            self.generate_cae_features()   
            self.run_feature_analysis("CAE")
            self.run_vis("CAE")
        if self.ENF:
            self.data_preprocess("ENF")
            self.generate_engineered()
            self.run_feature_analysis("ENF")
            self.run_vis("ENF")
            
    def run_pretrained(self):
        """Skips feature extraction and loads saved extracted features."""
        self.load_lightcurves_local()
        
        if self.DAE:
            self.load("DAE")
            self.run_vis("DAE")
        if self.CAE:
            self.load("CAE")
            self.run_vis("CAE")
        if self.ENF:
            self.load("ENF")
            self.run_vis("ENF")       

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
        dt.bulk_download_helper(self.datapath, sector=self.sector)
        dt.data_access_sector_by_bulk(self.datapath, sector=self.sector)

    def preprocess_data(self, featgen):
        if featgen == "ENF":
            self.flux = dt.normalize(self.flux)
        if featgen == "CAE":
            self.pflux, self.pflux_test = \
            lt.autoencoder_preprocessing(self.flux, self.time, self.parampath,
                                         ticid=self.objid,
                                         data_dir=self.datapath,
                                         output_dir=self.CAEpath)
        if featgen == "DAE":
            self.pgram, self.flux, self.objid, self.target_info, self.freq, self.time=\
            lt.DAE_preprocessing(self.flux, self.time, self.parampath,
                                 self.objid, self.target_info,
                                 data_dir=self.datapath, sector=self.sector,
                                 output_dir=self.DAEpath)

    
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
                            ticid_train=self.objid, output_dir=self.DAEpath)

    def generate_cae_features(self):
        """Train convolutional autoencoder to extract representative
        features from lightcurves.
        Returns: 
            * model : Keras Model object
            * hist : Keras history dictionary
            * feats : CAE-derived features
            * rcon : reconstructions of the input light curves"""        
        self.model, self.hist, self.feats, self.feats_test, self.rcon_test, \
        self.rcon = lt.conv_autoencoder(xtrain=self.pflux, ytrain=self.pflux, 
                                        params=self.parampath,
                                        output_dir=self.CAEpath,
                                        ticid_train=self.objid)
        return

    def produce_ae_visualizations(self,featgen):
        if featgen == "DAE":
            pt.produce_ae_visualizations(self.freq, self.pgram, self.rcon,
                                         self.DAEpath, self.objid, self.target_info,
                                         psd=True)

        if featgen == "CAE":
            pt.produce_ae_visualizations(self.time, self.pflux, self.rcon,
                                         self.CAEpath, self.objid, self.target_info,
                                         psd=False)

    # ==========================================================================
    # == Clustering and Outlier Analysis =======================================
    # ==========================================================================
    
    # >> featgen : feature generation method, e.g. 'CAE', 'DAE', 'ENF'
    
    def generate_clusters(self, featgen):
        """Run clustering algorithm on feature space.
        Returns:
            * clstr : array of cluster numbers, shape=(len(objid),)"""
        print('Performing clustering analysis in feature space...')
        self.clstr = lt.run_gmm(self.objid, self.feats, numclstr=self.numclstr,
                                savepath=self.ensbpath+featgen+'/',
                                runiter=self.runiter, numiter=self.numiter)

    def generate_tsne(self, featgen):
        """Reduces dimensionality of feature space for visualization."""
        self.tsne = lt.run_tsne(self.feats, savepath=self.ensbpath+featgen+'/')

    def generate_predicted_otypes(self, featgen):
        """Predicts object types using known classifications in SIMBAD, ASAS-SN,
        and GCVS.
        Returns:
            * potd : predicted object type dictionary
            * potype : array of predicted object types, shape=(len(objid),)"""
        self.potd, self.potype = \
            lt.label_clusters(self.ensbpath+featgen+'/', self.sector,
                              self.objid, self.clstr, self.totype, self.numtot,
                              self.totd)

    def produce_clustering_visualizations(self, featgen):
        '''Produces t-SNEs, confusion matrices, distribution plots, ensemble
        summary pie charts.'''
        pt.produce_clustering_visualizations(self.feats, self.numtot,
                                             self.numpot, self.tsne,
                                             self.ensbpath+featgen+'/',
                                             self.totd, self.potd)


    def generate_novelty_scores(self, featgen):
        """Returns: * nvlty : novelty scores, shape=(len(objid),)"""
        self.nvlty = pt.generate_novelty_scores(self.feats, self.objid,
                                                self.ensbpath+featgen+'/')

    def generate_rcon(self, featgen):
        self.rcon = lt.load_reconstructions(self.ensbpath+featgen+'/', self.objid)

    def produce_novelty_visualizations(self, featgen):
        pt.produce_novelty_visualizations(self.nvlty, self.ensbpath+featgen+'/',
                                          self.time, self.flux, self.objid)
        
    def run_feature_analysis(self, featgen):
        self.load_true_otypes()
        self.numerize_true_otypes()
        self.generate_clusters(featgen)
        self.generate_predicted_otypes(featgen)
        self.numerize_pred_otypes()
        
        self.generate_tsne(featgen)
        self.generate_novelty_scores(featgen)
        
    def run_vis(self, featgen):
        self.produce_clustering_visualizations(featgen)
        self.produce_novelty_visualizations(featgen)

    # ==========================================================================
    # == Loading Mergen Products ===============================================
    # ==========================================================================

    def load_features(self, featgen):
        """ Load in feature metafiles stored in the datapath"""
        if featgen == "ENF":
            self.feats = dt.load_ENF_feature_metafile(self.ENFpath)
        elif featgen == "CAE": 
            self.feats = \
                lt.load_bottleneck_from_fits(self.CAEpath, self.objid,
                                             self.runiter, self.numiter)
        elif featgen == "DAE": 
            self.feats = lt.load_DAE_bottleneck(self.DAEpath, self.objid)

    def load_gmm_clusters(self, featgen):
        """ clstr : array of cluster numbers, shape=(len(objid),)"""
        self.clstr = \
            lt.load_gmm_from_txt(self.ensbpath+featgen+'/', self.objid,
                                 self.runiter, self.numiter, self.numclstr)

    def load_reconstructions(self, featgen):
        self.rcon = lt.load_reconstructions(self.ensbpath+featgen+'/', self.objid)

    def load_nvlty(self, featgen):
        self.nvlty = pt.load_novelty_scores(self.ensbpath+featgen+'/',
                                            self.objid)

    def load_true_otypes(self):
        """ totype : true object types"""
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

    def load_pred_otypes(self, featgen):
        """ potype : predicted object types"""
        self.potype = dt.load_otype_pred_from_txt(self.ensbpath+featgen+'/',
                                                  self.sector, self.objid)

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

    def load_tsne(self, featgen):
        self.tsne = lt.load_tsne_from_fits(self.ensbpath+featgen+'/')
            
    def load(self, featgen):
        self.load_features(featgen)
        self.load_reconstructions(featgen)
        
        self.load_true_otypes()
        self.numerize_true_otypes()

        self.load_nvlty(featgen)
        self.load_gmm_clusters(featgen)

        self.load_pred_otypes(featgen)
        self.numerize_pred_otypes()
        
        self.load_tsne(featgen)
