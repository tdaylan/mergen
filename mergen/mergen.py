"""
Created on Thu Oct  8 20:27:13 2020

mergen.py
@author: LG and EC

To Do List:
    - fill in remaining functions
    - Set up pipeline function (kind of a run-all thing)
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
    def __init__(self, datapath, savepath, datatype, mdumpcsv=None,
                 filelabel=None, sector=1, runiter=False, numiter=1,
                 numclstr=100, aeparam=None):
        """Creates mergen object from which most common routines can easily be
        run
        Parameters:
            * datapath: string, where any data are being stored
            * savepath: string, where the subfolders should be saved into
            * datatype: string, indicates type of data being worked with.
                        options are: 
                        "SPOC", "FFI-Lygos", "FFI-QLP", "FFI-eleanor"
            * mdumpcsv: string, path to csv containing TESS momentum dumps
                        (local)
            * filelabel: string, if you want to have all plots/files/folders
                         labelled specially        
        """
        self.sector   = sector
        self.numclstr = numclstr
        
        self.datapath = datapath
        self.savepath = savepath
        self.datatype = datatype #SPOC or FFI
        self.ensbpath = self.savepath+'Ensemble-Sector_'+str(self.sector)+'/'

        if mdumpcsv is not None: # >> CSV file containing TESS momentum dumps
            self.mdumpcsv = mdumpcsv
        else:
            self.mdumpcsv = datapath + 'Table_of_momentum_dumps.csv'

        if filelabel is not None:
            self.filelabel = filelabel
        else:
            self.filelabel = "mergen"

        if aeparam is not None: # >> TXT file containing autoencoder parameters
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
        self.CAEpath = self.savepath + "CAE/"
        try:
            os.makedirs(self.CAEpath)
        except OSError:
            print ("Directory %s already exists" % self.CAEpath)
            
        print("Setting up ENF folder")
        self.ENFpath = self.savepath + "ENF/"
        try:
            os.makedirs(self.ENFpath)
        except OSError:
            print ("Directory %s already exists" % self.ENFpath)
        return

    def load_lightcurves_local(self):
        """Load in data saved in metafiles on datapath"""
        #check for self.datatype to determine loading scheme. 
        #figure out consistent stuff for FFI original locations
        if self.datatype == "FFI-Lygos":
            self.times, self.intensities, self.errors, self.identifiers = \
            dt.load_all_lygos(self.datapath)
        elif self.datatype == "SPOC":
            self.times, self.intensities, self.ticid, self.target_info = \
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

    def load_existing_features(self, typeFeatures):
        """ Load in feature metafiles stored in the datapath"""
        if typeFeatures == "ENF":
            self.feats = dt.load_ENF_feature_metafile(self.ENFpath)
        elif typeFeatures == "CAE":
            ### EMMA FILL THIS IN
            k = 6
        return
    
    def generate_engineered(self, version = 0, save = True):
        """ Run engineered feature creation"""
        self.feats = ft.create_save_featvec_homogenous_time(self.ENFpath,
                                                            self.times, 
                                                            self.intensities,
                                                            self.filelabel,
                                                            version=version,
                                                            save=save)
        return

    def generate_cae(self):
        '''Train convolutional autoencoder to extract representative
        features from lightcurves.'''
        # TODO
        return

    def generate_vcae(self):
        '''Train variational convolutional autoencoder to extract representative
        features from lightcurves.'''

        return

    def generate_clusters(self):
        print('Performing clustering analysis in feature space...')
        self.clstr = lt.run_gmm(self.ticid, self.feats, numclstr=self.numclstr,
                                savepath=self.ensbpath, runiter=self.runiter,
                                numiter=self.numiter)
        return

    def load_learned_features(self):
        print('Loading CAE-learned features...')
        self.feats = lt.load_bottleneck_from_fits(self.ensbpath,
                                                     self.ticid,
                                                     self.runiter,
                                                     self.numiter)
        return

    def load_gmm_clusters(self):
        print('Loading GMM clustering results...')
        self.clstr = lt.load_gmm_from_txt(self.ensbpath, self.ticid,
                                           self.runiter, self.numiter, 
                                           self.numclstr)
        return

    def load_vtypes(self):
        print('Loading classifications...')
        self.vtype = lt.load_vtype_from_txt(self.ensbpath, self.sector,
                                            self.ticid)
        return

    def numerize_vtypes(self):
        self.uniqvtype = np.unique(self.vtype)
        self.numvtype = [np.nonzero(self.uniqvtype == vt)[0][0] for \
                         vt in self.vtype]
        return

    def run_all(self):
        return

