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
* Fill in remaining functions
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
                 aeparam=None):
        """Creates mergen object from which most common routines can easily be
        run
        Parameters:
            * datapath: string, where any data are being stored
            * savepath: string, where the subfolders should be saved into
            * datatype: string, indicates type of data being worked with.
                        options are: 
                        "SPOC", "FFI-Lygos", "FFI-QLP", "FFI-eleanor"
            * mdumpcsv: string, path to csv file containing TESS momentum dumps
            * filelabel: string, if you want to have all plots/files/folders
                         labelled specially        
            * sector  : int, TESS Observational Sector number (1,2,...,26,...)
            * runiter : bool, if True, runs iterative CAE scheme
            * numiter : int, number of iterations in the iterative CAE scheme
            * numclstr : int, number of clusters assumed by the GMM clustering
                         algorithm
            * aeparam:  string, path to txt file containing autoencoder
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

        if aeparam is not None:
            self.aeparam = aeparam
        else:
            self.aeparam = savepath + 'caehyperparams.txt'

        # >> iterative scheme
        self.runiter = runiter
        self.numiter = numiter
        
        self.initiate_folder()

    def initiate_folder(self):
        """Make all the big folders"""
        print("Setting up CAE folder")
        # self.CAEpath = self.savepath + "CAE/"
        self.CAEpath = self.ensbpath + "CAE/"
        try:
            os.makedirs(self.CAEpath)
        except OSError:
            print ("Directory %s already exists" % self.CAEpath)
            
        print("Setting up ENF folder")
        # self.ENFpath = self.savepath + "ENF/"
        self.ENFpath = self.ensbpath + "ENF/"
        try:
            os.makedirs(self.ENFpath)
        except OSError:
            print ("Directory %s already exists" % self.ENFpath)
        return

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
            self.times, self.intensities, self.errors, self.identifiers = \
            dt.load_all_lygos(self.datapath)
        elif self.datatype == "SPOC":
            self.intensities, self.times, self.ticid, self.target_info = \
            dt.load_data_from_metafiles(self.datapath, self.sector)
        
    def download_lightcurves(self):
        '''Downloads and process light SPOC light curves, if not already
        downloaded.'''
        dt.bulk_download_helper(datapath, sector=self.sector)
        dt.data_access_sector_by_bulk(datapath, sector=self.sector)

    def data_clean(self):
        """ Cleans data up - just BASE cleanup of normalizing."""
        self.intensities = dt.normalize(self.intensities)
        return
    
    # ==========================================================================
    # == Feature Generation ====================================================
    # ==========================================================================

    def generate_engineered(self, version = 0, save = True):
        """ Run engineered feature creation"""
        self.feats = ft.create_save_featvec_homogenous_time(self.ENFpath,
                                                            self.times, 
                                                            self.intensities,
                                                            self.filelabel,
                                                            version=version,
                                                            save=save)

    def generate_cae(self):
        '''Train convolutional autoencoder to extract representative
        features from lightcurves.'''
        # TODO
        return

    def generate_cvae(self):
        '''Train convolutional variational autoencoder to extract representative
        features from lightcurves.'''

        return

    # ==========================================================================
    # == Clustering and Outlier Analysis =======================================
    # ==========================================================================

    def generate_clusters(self):
        print('Performing clustering analysis in feature space...')
        self.clstr = lt.run_gmm(self.ticid, self.feats, numclstr=self.numclstr,
                                savepath=self.ensbpath, runiter=self.runiter,
                                numiter=self.numiter)

    def generate_predicted_otypes(self):
        self.potd, self.potype = lt.label_clusters(self.ensbpath, self.sector,
                                                   self.ticid, self.clstr,
                                                   self.totype, self.numtot,
                                                   self.totd)

    def produce_clustering_visualizations(self):
        '''Produces t-SNEs, confusion matrices, distribution plots, ensemble
        summary pie charts.'''
        return

    def generate_outlier_scores(self):
        return

    def produce_outlier_visualizations(self):
        return

    # ==========================================================================
    # == Loading Mergen Products ===============================================
    # ==========================================================================

    def load_existing_features(self, typeFeatures):
        """ Load in feature metafiles stored in the datapath"""
        if typeFeatures == "ENF":
            self.feats = dt.load_ENF_feature_metafile(self.ENFpath)
        elif typeFeatures == "CAE":
            ### EMMA FILL THIS IN
            k = 6
        return

    def load_learned_features(self):
        print('Loading CAE-learned features...')
        self.feats = lt.load_bottleneck_from_fits(self.ensbpath,
                                                     self.ticid,
                                                     self.runiter,
                                                     self.numiter)

    def load_gmm_clusters(self):
        print('Loading GMM clustering results...')
        self.clstr = lt.load_gmm_from_txt(self.ensbpath, self.ticid,
                                           self.runiter, self.numiter, 
                                           self.numclstr)

    def load_true_otypes(self):
        print('Loading ground truth object types...')
        self.totype = dt.load_otype_true_from_datadir(self.datapath,
                                                      self.sector,
                                                      self.ticid)

    def numerize_true_otypes(self):
        unqtot = np.unique(self.totype)
        self.totd = {i: unqtot[i] for i in range(len(unqtot))}
        self.numtot = np.array([np.nonzero(unqtot == ot)[0][0] for \
                                ot in self.totype])

    def load_pred_otypes(self):
        print('Loading redicted object types...')
        self.potype = dt.load_otype_pred_from_txt(self.ensbpath, self.sector,
                                                  self.ticid)

    def numerize_pred_otypes(self):
        unqpot = np.unique(self.potype)
        self.potd = {i: unqpot[i] for i in range(len(unqpot))}
        self.numpot = np.array([np.nonzero(unqpot == ot)[0][0] for \
                                ot in self.potype])

