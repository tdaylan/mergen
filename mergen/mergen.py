"""
Created on Thu Oct  8 20:27:13 2020

mergen.py
@author: LG and EC

To Do List:
    - fill in remaining functions
    - Set up pipeline function (kind of a run-all thing)
"""

from .__init__ import *
from . import data_utils as dt
from . import catalog_utils as ct
from . import learn_utils as lt
from . import plot_utils as pt
from . import feature_utils as ft

#mg = mergen()
#mg.folder_initiate()
#mg.intensities

class mergen(object):
    """ Main mergen class. Initialize this to work with everything else
    conveniently. """
    def __init__(self, datapath, savepath, datatype, mdumpcsv, filelabel=None,
                 sector=1):
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
        
        self.datapath = datapath
        self.savepath = savepath
        self.datatype = datatype #SPOC or FFI
        # self.mdumpcsv = datapath + 'Table_of_momentum_dumps.csv'
        self.mdumpcsv = mdumpcsv
        self.sector   = sector
        if filelabel is not None:
            self.filelabel = filelabel
        else:
            self.filelabel = "mergen"
        
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
            self.features = dt.load_ENF_feature_metafile(self.ENFpath)
        elif typeFeatures == "CAE":
            ### EMMA FILL THIS IN
            k = 6
        return
    
    def generate_engineered(self, version = 0, save = True):
        """ Run engineered feature creation"""
        self.features = ft.create_save_featvec_homogenous_time(self.ENFpath,
                                                               self.times, 
                                                               self.intensities,
                                                               self.filelabel,
                                                               version=version,
                                                               save=save)
        return

    def generate_CAE(self):
        """Run CAE feature creation"""
        
        #EMMA FILL THIS IN
        
        return

    def run_all(self):
        return

