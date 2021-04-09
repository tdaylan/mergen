# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:14:55 2021

@author: Emma Chickles, Lindsey Gordon
catalog_utils.py

To Do List:
    Emma:
        -what is this is mergen and what of this is specific to your science stuff

"""
from __init__ import *
def get_tess_features(ticid):
    '''Query catalog data https://arxiv.org/pdf/1905.10694.pdf'''
    

    target = 'TIC '+str(int(ticid))
    catalog_data = Catalogs.query_object(target, radius=0.02, catalog='TIC')
    Teff = catalog_data[0]["Teff"]

    rad = catalog_data[0]["rad"]
    mass = catalog_data[0]["mass"]
    GAIAmag = catalog_data[0]["GAIAmag"]
    d = catalog_data[0]["d"]
    # Bmag = catalog_data[0]["Bmag"]
    # Vmag = catalog_data[0]["Vmag"]
    objType = catalog_data[0]["objType"]
    Tmag = catalog_data[0]["Tmag"]
    # lum = catalog_data[0]["lum"]

    return target, Teff, rad, mass, GAIAmag, d, objType, Tmag

def get_tess_feature_txt(ticid_list, out='./tess_features_sectorX.txt'):
    '''Queries 'TESS features' (i.e. Teff, rad, mass, GAIAmag, d) for each
    TICID and saves to text file.
    
    Can get ticid_list with:
    with open('all_targets_S019_v1.txt', 'r') as f:
        lines = f.readlines()
    ticid_list = []
    for line in lines[6:]:
        ticid_list.append(int(line.split()[0]))
    '''
    
    # !! 
    # TESS_features = []        
    for i in range(len(ticid_list)):
        print(i)
        try:
            features = get_tess_features(ticid_list[i])
            # TESS_features.append(features)
            with open(out, 'a') as f:
                f.write(' '.join(map(str, features)) + '\n')
        except:
            with open('./failed_get_tess_features.txt', 'a') as f:
                f.write(str(ticid_list[i])+'\n')


    
def build_simbad_database(out='./simbad_database.txt'):
    '''Object type follows format in:
    http://vizier.u-strasbg.fr/cgi-bin/OType?$1
    Can see other Simbad fields with Simbad.list_votable_fields()
    http://simbad.u-strasbg.fr/Pages/guide/sim-fscript.htx
    TODO  change votable field to otypes'''
    
    # -- querying object type -------------------------------------------------
    customSimbad = Simbad()
    # customSimbad.get_votable_fields()
    customSimbad.add_votable_fields('otype')
    
    # -- querying TICID for each object ---------------------------------------
    # >> first get all the TESS objects in the Simbad database
    res = customSimbad.query_catalog('tic')
    objects = list(res['MAIN_ID'])

    # >> now loop through all of the objects
    for i in range(len(objects)):
        # >> decode bytes object to convert to string
        obj = objects[i].decode('utf-8')
        bibcode = res['COO_BIBCODE'][i].decode('utf-8')
        otype = res['OTYPE'][i].decode('utf-8')
        
        #print(obj + ' ' + otype)
        
        # >> now query TICID
        obs_table = Observations.query_criteria(obs_collection='TESS',
                                                dataproduct_type='timeseries',
                                                objectname=obj)
        
        ticids = obs_table['target_name']
        for ticid in ticids:
            with open(out, 'a') as f:
                f.write(ticid + ',' + obj + ',' + otype + ',' + bibcode + '\n')
 

               
def get_simbad_classifications(ticid_list,
                               simbad_database_txt='./simbad_database.txt'):
    '''Query Simbad classification and bibcode from .txt file (output from
    build_simbad_database).
    Returns a list where simbad_info[i] = [ticid, main_id, obj type, bibcode]
    Object type follows format in:
    http://vizier.u-strasbg.fr/cgi-bin/OType?$1
    '''
    ticid_simbad = []
    main_id_list = []
    otype_list = []
    bibcode_list = []
    with open(simbad_database_txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            ticid, main_id, otype, bibcode = line[:-2].split(',')
            ticid_simbad.append(int(ticid)) 
            main_id_list.append(main_id)
            otype_list.append(otype)
            bibcode_list.append(bibcode)
    intersection, comm1, comm2 = np.intersect1d(ticid_list, ticid_simbad,
                                                return_indices=True)
    simbad_info = []
    for i in comm2:
        simbad_info.append([ticid_simbad[i], main_id_list[i], otype_list[i],
                            bibcode_list[i]])
    return simbad_info

def query_associated_catalogs(ticid):
    res=Catalogs.query_object('TIC ' + str(int(ticid)), radius=0.02,
                              catalog='TIC')[0]
    for i in ['HIP', 'TYC', 'UCAC', 'TWOMASS', 'ALLWISE', 'GAIA', 'KIC', 'APASS']:
        print(i + ' ' + str(res[i]) + '\n')

def query_simbad_classifications(ticid_list, output_dir='./', suffix=''):
    '''Call like this:
    query_simbad_classifications([453370125.0, 356473029])
    '''
    import time
    
    customSimbad = Simbad()
    customSimbad.add_votable_fields('otypes')
    # customSimbad.add_votable_fields('biblio')
    
    ticid_simbad = []
    otypes_simbad = []
    main_id_simbad = []
    bibcode_simbad = []
    
    with open(output_dir + 'all_simbad_classifications'+suffix+'.txt', 'a') as f:
        f.write('')    
    
    with open(output_dir + 'all_simbad_classifications'+suffix+'.txt', 'r') as f:
        lines = f.readlines()
        ticid_already_classified = []
        for line in lines:
            ticid_already_classified.append(float(line.split(',')[0]))
    

    for tic in ticid_list:
        
        res=None
        
        while res is None:
            try:
                if tic in ticid_already_classified:
                    print('Skipping TIC')
                    
                else:
                    print('get coords for TIC' + str(int(tic)))
                    
                    # >> get coordinates
                    target = 'TIC ' + str(int(tic))
                    catalog_data = Catalogs.query_object(target, radius=0.02,
                                                         catalog='TIC')[0]
                    # time.sleep(6)
            
                    
                    # -- get object type from Simbad --------------------------------------
                    
                    # >> first just try querying the TICID
                    res = customSimbad.query_object(target)
                    # time.sleep(6)
                    
                    # >> if no luck with that, try checking other IDs
                    if type(res) == type(None):
                        if type(catalog_data['TYC']) != np.ma.core.MaskedConstant:
                            target_new = 'TYC ' + str(catalog_data['TYC'])
                            res = customSimbad.query_object(target_new)
                            # time.sleep(6)
                            
                    if type(res) == type(None):
                        if type(catalog_data['HIP']) != np.ma.core.MaskedConstant:
                            target_new = 'HIP ' + str(catalog_data['HIP'])
                            res = customSimbad.query_object(target_new)
                            # time.sleep(6)
            
                    # # >> UCAC not added to Simbad yet
                    # if type(res) == type(None):
                    #     if type(catalog_data['UCAC']) != np.ma.core.MaskedConstant:
                    #         target_new = 'UCAC ' + str(catalog_data['UCAC'])
                    #         res = customSimbad.query_object(target_new)
                            
                    if type(res) == type(None):
                        if type(catalog_data['TWOMASS']) != np.ma.core.MaskedConstant:
                            target_new = '2MASS ' + str(catalog_data['TWOMASS'])
                            res = customSimbad.query_object(target_new)     
                            # time.sleep(6)
            
                    if type(res) == type(None):
                        if type(catalog_data['SDSS']) != np.ma.core.MaskedConstant:
                            target_new = 'SDSS ' + str(catalog_data['SDSS'])
                            res = customSimbad.query_object(target_new) 
                            # time.sleep(6)
            
                    if type(res) == type(None):
                        if type(catalog_data['ALLWISE']) != np.ma.core.MaskedConstant:
                            target_new = 'ALLWISE ' + str(catalog_data['ALLWISE'])
                            res = customSimbad.query_object(target_new)
                            # time.sleep(6)
                            
                    if type(res) == type(None):
                        if type(catalog_data['GAIA']) != np.ma.core.MaskedConstant:
                            target_new = 'Gaia ' + str(catalog_data['GAIA'])
                            res = customSimbad.query_object(target_new)      
                            # time.sleep(6)
                            
                    if type(res) == type(None):
                        if type(catalog_data['APASS']) != np.ma.core.MaskedConstant:
                            target_new = 'APASS ' + str(catalog_data['APASS'])
                            res = customSimbad.query_object(target_new)        
                            # time.sleep(6)
                            
                    if type(res) == type(None):
                        if type(catalog_data['KIC']) != np.ma.core.MaskedConstant:
                            target_new = 'KIC ' + str(catalog_data['KIC'])
                            res = customSimbad.query_object(target_new)    
                            # time.sleep(6)
                    
                    # # >> if still nothing, query with coordinates
                    # if type(res) == type(None):
                    #     ra = catalog_data['ra']
                    #     dec = catalog_data['dec']            
                    #     coords = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg))
                    #     res = customSimbad.query_region(coords, radius='0d0m2s')         
                    #     time.sleep(6)
                    
                    if type(res) == type(None):
                        print('failed :(')
                        res=0
                        with open(output_dir + 'all_simbad_classifications'+suffix+'.txt', 'a') as f:
                            f.write('{},{},{}\n'.format(tic, '', ''))              
                        ticid_simbad.append(tic)
                        otypes_simbad.append('none')
                        main_id_simbad.append('none')                
                    else:
                        otypes = res['OTYPES'][0].decode('utf-8')
                        main_id = res['MAIN_ID'].data[0].decode('utf-8')
                        ticid_simbad.append(tic)
                        otypes_simbad.append(otypes)
                        main_id_simbad.append(main_id)
                        
                        with open(output_dir + 'all_simbad_classifications'+suffix+'.txt', 'a') as f:
                            f.write('{},{},{}\n'.format(tic, otypes, main_id))
                            
                    # time.sleep(6)
            except:
                pass
                print('connection failed! Trying again now')
                    
                    
            
    return ticid_simbad, otypes_simbad, main_id_simbad
        


def query_vizier(ticid_list=None, out='./SectorX_GCVS.txt', catalog='gcvs',
                 dat_dir = '/Users/studentadmin/Dropbox/TESS_UROP/data/',
                 sector=20):
    '''http://www.sai.msu.su/gcvs/gcvs/vartype.htm'''
    
    # Vizier.ROW_LIMIT=-1
    # catalog_list=Vizier.find_catalogs('B/gcvs')
    # catalogs = Vizier.get_catalogs(catalog_list.keys())    
    # catalogs=catalogs[0]
    
    if type(ticid_list) == type(None):
        flux, x, ticid_list, target_info = \
            load_data_from_metafiles(dat_dir, sector, DEBUG=False,
                                     nan_mask_check=False)        
    
    ticid_viz = []
    otypes_viz = []
    main_id_viz = []
    ticid_already_classified = []
    
    # >> make sure output file exists
    with open(out, 'a') as f:
        f.write('')    
    
    with open(out, 'r') as f:
        lines = f.readlines()
        ticid_already_classified = []
        for line in lines:
            ticid_already_classified.append(float(line.split(',')[0]))
            
    
    for tic in ticid_list:
        if tic  in ticid_already_classified:
            print('Skipping '+str(tic))
        else:
            try:
                print('Running '+str(tic))
                target = 'TIC ' + str(int(tic))
                print('Query Catalogs')
                catalog_data = Catalogs.query_object(target, radius=0.02,
                                                     catalog='TIC')[0]
                ra = catalog_data['ra']
                dec = catalog_data['dec']            
                # coords = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg)) 
                # ra = coords.ra.deg
                # dec = coords
                v = Vizier(columns=['VarType', 'VarName'])
                print('Query Vizier')
                res = v.query_region(coord.SkyCoord(ra=ra, dec=dec,
                                                         unit=(u.deg, u.deg),
                                                         frame='icrs'),
                                          radius=0.003*u.deg, catalog=catalog)
                if len(res) > 0:
                    otype = res[0]['VarType'][0]
                    main_id = res[0]['VarName'][0]
                    ticid_viz.append(tic)
                    otypes_viz.append(otype)
                    main_id_viz.append(main_id)
                    # with open(out, 'a') as f:
                    #     f.write('{},{},{}\n'.format(tic, otype, main_id))              
                else:
                    otype = ''
                    main_id = ''
                    
                with open(out, 'a') as f:
                    f.write('{},{},{}\n'.format(tic, otype, main_id))    
            except:
                print('Connection failed! Trying again now')
                
    return ticid_viz, otypes_viz, main_id_viz

def get_otype_dict(data_dir='/Users/studentadmin/Dropbox/TESS_UROP/data/'):
    '''Return a dictionary of descriptions'''
    # d = {'a2': 'Variable Star of alpha2 CVn type',
    #      'ACYG': 'Variables of the Alpha Cygni type',
    #      'IR': 'Infra-Red source',
    #      'UV': 'UV-emission source',
    #      'X': 'X-ray source',
    #      'gB': 'gamma-ray Burst',
    #      'AR': 'Detached systems of the AR Lacertae type',
    #      'EB': 'Eclipsing binary',
    #      'Al': 'Eclipsing binary of Algol type',
    #      'bL': 'Eclipsing binary of beta Lyr type',
    #      'WU': 'Eclipsing binary of W UMa type',
    #      'EP': 'Star showing eclipses by its planet',
    #      'SB': 'Spectroscopic binary',
    #      'EI': 'Ellipsoidal variable Star',
    #      'CV': 'Cataclysmic Variable Star',
    #      'SNR': 'SuperNova Remnant',
    #      'Be': 'Be star',
    #      'Fl': 'Flare star',
    #      'V': 'Variable star',
    #      'HV': 'High-velocity star',
    #      'PM': }
    # d = {'ACYG': 'Variables of the Alpha Cygni type',
    #      'AR': 'Detached systems of the AR Lacertae type',
    #      'D': 'Detached systems',
    #      'DM': 'Detached main-sequence systems',
    #      'DW': 'Detached systems with a subgiant',
    #      'K': 'Contact systems',
    #      'KE': 'Contact systems of early (O-A) spectral type',
    #      'KW': 'Contact systems of the W UMa type',
    #      'SD': 'Semidetached systems',
    #      'GS': 'Systems with one or both giant and supergiant components',
    #      'RS': 'RS Canum Venaticorum-type systems',
    #      'CST': 'Nonvariable stars',
    #      'XPRM': 'X-ray systems consisting of a late-type dwarf (dK-dM) and a pulsar',
    #      'FKCOM': 'FK Comae Berenices-type variables',
    #      'GCAS': 'Eruptive irregular variables of the Gamma Cas type',
    #      'IA': 'Poorly studied irregular variables of early (O-A) spectral type'}
    
    d = {}
    
    with open(data_dir + 'otypes_gcvs.txt', 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        if len(line.split(' '*3)) > 1:
            otype = line.split(' '*3)[0]
            explanation = line.split(' '*3)[1].split('.')[0]
            d[otype] = explanation
    
    with open(data_dir + 'otypes_simbad.txt', 'r') as f:
        lines= f.readlines()
        
    for line in lines:
        if len(line.split('\t')) >= 3:
            otype = line.split('\t')[-2].split()[0]
            if len(otype) > 0:
                if otype[-1] == '*' and otype != '**':
                    otype = otype[:-1]
            explanation = ' '.join(line.split('\t')[-1].split())
            d[otype] = explanation
        
    return d

def get_parents_only(class_info, parents=['EB'],
                     parent_dict = {'EB': ['Al', 'bL', 'WU', 'EP', 'SB', 'SD'],
                                    'ACV': ['ACVO'],
                                    'D': ['DM', 'DS', 'DW'],
                                    'K': ['KE', 'KW'],
                                    'Ir': ['Or', 'RI', 'IA', 'IB', 'INA', 'INB']}):
    '''Finds all the objects with same parent and combines them into the same
    class
    '''
    classes = []
    new_class_info = []
    for i in range(len(class_info)):
        otype_list = class_info[i][1].split('|')
        new_otype_list=[]
        for otype in otype_list:
            for parent in parents:
                if otype in parent_dict[parent]:
                    new_otype = parent
                else:
                    new_otype = otype
                new_otype_list.append(new_otype)
                
        new_otype_list = np.unique(new_otype_list)
        new_class_info.append([class_info[i][0], '|'.join(new_otype_list),
                               class_info[i][2]])
            
    
    return np.array(new_class_info)

def correct_vizier_to_simbad(in_f='./SectorX_GCVS.txt',
                             out_f='./SectorX_GCVS_revised.txt',
                             uncertainty_flags=[':', '?', '*']):
    '''Make sure object types are the same'''
    with open(in_f, 'r') as f:
        lines = f.readlines()
        
    renamed = {'E': 'EB', 'EA': 'Al', 'EB': 'bL', 'EW': 'WU', 'ACV': 'a2',
               'ACVO': 'a2', 'BCEP': 'bC', 'BE':'Be', 'DCEP': 'cC',
               'DSCT': 'dS', 'DSCTC': 'dS', 'ELL': 'El', 'GDOR': 'gD',
               'I': 'Ir', 'IN': 'Or', 'IS': 'RI'}
        
    for line in lines:
        tic, otype, main = line.split(',')
        otype = otype.replace('+', '|')
        otype_list = otype.split('|')
        otype_list_new = []
        
        for o in otype_list:
            
            if len(o) > 0:
                # >> remove uncertainty_flags
                if o[-1] in uncertainty_flags:
                    o = o[:-1]
                    
                # >> remove (B)
                if '(' in o:
                    o = o[:o.index('(')]
                    
                if o in list(renamed.keys()):
                    o = renamed[o]
                # # >> rename object types to Simbad notation
                # if o == 'E':
                #     o = 'EB'
                # elif o == 'EA':
                #     o = 'Al'
                # elif o == 'EB':
                #     o = 'bL'
                # elif o == 'EW':
                #     o = 'WU'
                # elif o == 'ACV' or o == 'ACVO':
                #     o = 'a2'
                # elif o == 'BCEP':
                #     o = 'bC'
                # elif o == 'BE':
                #     o = 'Be'
                # elif o == ''
                
            otype_list_new.append(o)
                
                
        otype = '|'.join(otype_list_new)
        
        
        with open(out_f, 'a') as f:
            f.write(','.join([tic, otype, main]))

def quick_simbad(ticidasstring):
    """ only returns if it has a tyc id"""
    catalogdata = Catalogs.query_object(ticidasstring, radius=0.02, catalog="TIC")[0]
    try: 
        tyc = "TYC " + catalogdata["TYC"]
        customSimbad = Simbad()
        customSimbad.add_votable_fields("otypes")
        res = customSimbad.query_object(tyc)
        objecttype = res['OTYPES'][0].decode('utf-8')
    except: 
        objecttype = "there is no TYC for this object"
    return objecttype

def get_true_classifications(ticid_list,
                             database_dir='./databases/',
                             single_file=False,
                             useless_classes = ['*', 'IR', 'UV', 'X', 'PM',
                                                '?', ':'],
                             uncertainty_flags = ['*', ':', '?']):
    '''Query classifications and bibcode from *_database.txt file.
    Returns a list where class_info[i] = [ticid, obj type, bibcode]
    Object type follows format in:
    http://vizier.u-strasbg.fr/cgi-bin/OType?$1
    '''
    ticid_classified = []
    class_info = []
    
    # >> find all text files in directory
    if single_file:
        fnames = ['']
    else:
        fnames = fm.filter(os.listdir(database_dir), '*.txt')
    
    for fname in fnames:
        # >> read text file
        with open(database_dir + fname, 'r') as f:
            lines = f.readlines()
            for line in lines:
                ticid, otype, bibcode = line[:-1].split(',')
                

                
                # >> remove any repeats and any empty classes and sort
                otype_list = otype.split('|')
                # >> remove any candidate indicators
                for i in range(len(otype_list)):
                    if otype_list[i] != '**' and len(otype_list[i])>0:
                        if otype_list[i][-1] in uncertainty_flags:
                            otype_list[i] = otype_list[i][:-1]
                otype_list = np.unique(otype_list)
                # >> remove useless classes
                for u_c in useless_classes + ['']:
                    if u_c in otype_list:
                        otype_list =np.delete(otype_list,
                                              np.nonzero(otype_list == u_c))
                otype_list.sort()
                otype = '|'.join(otype_list)
                
                # >> only get classifications for ticid_list, avoid repeats
                # >> and only include objects with interesting lables
                ticid = float(ticid)
                if ticid in ticid_list and len(otype) > 0:
                    if ticid in ticid_classified:
                        ind = np.nonzero(np.array(ticid_classified) == ticid)[0][0]
                        new_class_info = class_info[ind][1] + '|' + otype
                        new_class_info = new_class_info.split('|')
                        new_class_info = '|'.join(np.unique(new_class_info))
                        class_info[ind][1] = new_class_info
                    else:
                        ticid_classified.append(ticid)
                        class_info.append([int(ticid), otype, bibcode])
                    
    # >> check for any repeats
    return np.array(class_info)
