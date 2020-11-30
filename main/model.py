# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# 
# 2020-05-26 - modellibrary.py
# Keras novelty detection in TESS dataset
# / Emma Chickles
# 
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 

from sklearn.manifold import TSNE
import os
import pdb
import matplotlib.pyplot as plt
import numpy as np
import plotting_functions as pf
import data_functions as df
from astropy.io import fits
from astropy.timeseries import LombScargle
import random
from sklearn.cluster import KMeans    

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
# from tensorflow.keras.utils.generic_utils import get_custom_objects

# import keras.backend as K
# import tensorflow as tf
# import keras
# from keras.models import Model
# from keras.layers import *
# from keras import optimizers
# from keras import metrics
# from keras.utils.generic_utils import get_custom_objects


def sector_mask_diag(sectors=[1,2,3,17,18,19,20], data_dir='./',
                      output_dir='./', custom_masks=None):
    
    num_sectors = len(sectors)
    all_flux = []
    all_ticid = []
    all_target_info = []
    all_x = []
    if type(custom_masks) == type(None):
        custom_masks = [[]]*num_sectors
    for i in range(num_sectors):
        flux, x, ticid, target_info = \
            df.load_data_from_metafiles(data_dir, sectors[i],
                                        nan_mask_check=True,
                                        custom_mask=custom_masks[i])       
        all_flux.append(flux)
        all_ticid.append(ticid)
        all_target_info.append(target_info)
        all_x.append(x)
        
    fig, ax  = plt.subplots(num_sectors)
    for i in range(num_sectors):
        ax[i].plot(all_x[i], all_flux[i][0], '.k', ms=2)
        pf.ticid_label(ax[i], all_ticid[i][0], all_target_info[i][0],
                       title=True)
        ax[i].set_title('Sector '+str(sectors[i])+'\n'+ax[i].get_title(),
                        fontsize='small')
    
    fig.tight_layout()
    fig.savefig(output_dir+'sector_masks.png')
    
def merge_sector_diag(data_dir, sectors=list(range(1, 29)), output_dir='./',
                      ncols=3):
    
    num_sectors = len(sectors)
    fig, ax = plt.subplots(num_sectors, ncols,
                           figsize=(5*ncols, 1.43*num_sectors))
    for i in range(num_sectors):
        sectorfile=np.loadtxt(data_dir+'Sector'+str(sectors[i])+\
                              '/all_targets_S'+'%03d'%sectors[i]+'_v1.txt')
        for j in range(ncols):
            if ncols == 1:
                a = ax[i]
            else:
                a = ax[i,j]
            ticid = sectorfile[j][0]
            time, flux, ticid = df.get_lc_file_and_data(output_dir, ticid)
            a.plot(time, flux, '.k', ms=2)
            a.set_title('Sector '+str(sectors[i]), fontsize='small')
        
    fig.tight_layout()
    fig.savefig(output_dir + 'sector_lightcurves.png')
    

def autoencoder_preprocessing(flux, time, p, ticid=None, target_info=None,
                              sector=1, mock_data=False,
                              validation_targets=[219107776],
                              DAE=False, features=False,
                              norm_type='standardization', input_rms=True,
                              input_psd = True, load_psd=False, n_pgram=1000,
                              train_test_ratio=0.9, data_dir='./',
                              split=False, output_dir='./',
                              use_tess_features=True,
                              use_tls_features=True,
                              use_rms=True, flux_plot=None,
                              concat_ext_feats=False):
    '''Preprocesses output from df.load_data_from_metafiles
    Shuffles array.
    Parameters:
        * flux : array of light curves, shape=(num light curves, num points)
        * ticid : list of TICIDs, shape=(num light curves)
        * target_info : [sector, cam, ccd, data_type, cadence] for each light
                        curve, shape=(num light curves, 5)
        * validation_targets : list of TICIDs to move from the training set to
                               testing set                        
        * DAE : preprocessing for deep autoencoder. if True, the following is
          required:
          * features : feature vector, shape=(num light curves, num features)
          If DAE is False, then performs preprocessing for convolutional
          autoencoder, and uses the folllowing parameters:
          * norm_type : either standardization, median_normalization,
                        minmax_normalization, none
          * input_rms : calculate RMS before normalizing
    '''
    
    # >> shuffle array
    print('Shuffling data...')
    inds = np.arange(len(flux))
    random.Random(4).shuffle(inds)
    flux = flux[inds]
    ticid = ticid[inds]
    target_info = np.array(target_info)[inds]
    if DAE:
        features = features[inds]
        
    # >> move specified targest to testing set
    if len(validation_targets) > 0:
        for t in validation_targets:
            target_ind = np.nonzero( ticid == t )
            if np.shape(target_ind)[1] == 0:
                print('!! TIC '+str(t)+' is not in data set')
            else:
                print('Moving '+str(t)+' to test set...')
                target_ind = target_ind[0][0]
            flux = np.insert(flux, -1, flux[target_ind], axis=0)
            flux = np.delete(flux, target_ind, axis=0)
            ticid = np.insert(ticid, -1, ticid[target_ind])
            ticid = np.delete(ticid, target_ind)
            target_info = np.insert(target_info, -1, target_info[target_ind],
                                    axis=0)
            target_info = np.delete(target_info, target_ind, axis=0) 
            if DAE:
                features = np.insert(features, -1, features[target_ind],
                                     axis=0)
                features = np.delete(features, target_ind, axis=0)
                
    # >> partition data and normalize
    if DAE:
        print('Partitioning data...')
        x_train, x_test, y_train, y_test, flux_train, flux_test,\
        ticid_train, ticid_test, target_info_train, target_info_test, time = \
            split_data_features(flux, features, time, ticid, target_info,
                                False, p, train_test_ratio=train_test_ratio)
        print('Standardizing feature vector...')
        x_train = df.standardize(x_train, ax=0)
        x_test = df.standardize(x_test, ax=0)
        
        return x_train, x_test, y_train, y_test, flux_train, flux_test, \
            ticid_train, ticid_test, target_info_train, target_info_test, time
    else:  
        if input_rms:
            print('Calculating RMS..')
            rms = df.rms(flux)
        else: rms_train, rms_test = False, False
        
        if input_psd:
            # >> get frequency array
            f, tmp = LombScargle(time, flux[0]).autopower()
            f = np.linspace(np.min(f), np.max(f), n_pgram)
                            
            if not load_psd:
                print('Calculating PSD..')
                
                psd = []
                for i in range(len(flux)):
                    print(str(i) + '/' + str(len(flux)))
                    # f, tmp = LombScargle(time, flux[i]).autopower()
                    tmp = LombScargle(time, flux[i]).power(f)
                    psd.append(tmp)
                psd = np.array(psd)
                
                # >> truncate
                if not p['fully_conv']:
                    new_length=int(np.shape(psd)[1] / \
                                   (2**(np.max(p['num_conv_layers'])/2)))*\
                                int((2**(np.max(p['num_conv_layers'])/2)))
                else:
                    new_length=int(np.shape(psd)[1] / \
                                   (2**(np.max(p['num_conv_layers']+1)/2)))*\
                                int((2**(np.max(p['num_conv_layers']+1)/2)))                    
                psd = np.delete(psd,np.arange(new_length,np.shape(psd)[1]),1)
                f = f[:new_length]

            else:
                # >> load data
                with fits.open(output_dir + 'psd_train.fits') as hdul:
                    psd_train = hdul[1].data            
                with fits.open(output_dir + 'psd_test.fits') as hdul:
                    psd_test = hdul[1].data    
            
        # flux_plot = df.normalize(flux) # >> divide by medina
        if norm_type == 'standardization':
            print('Standardizing fluxes...')
            flux = df.standardize(flux)
    
        elif norm_type == 'median_normalization':
            print('Normalizing fluxes (dividing by median)...')
            flux = df.normalize(flux)
            
        elif norm_type == 'minmax_normalization':
            print('Normalizing fluxes (changing minimum and range)...')
            mins = np.min(flux, axis = 1, keepdims=True)
            flux = flux - mins
            maxs = np.max(flux, axis=1, keepdims=True)
            flux = flux / maxs
            
        else:
            print('Light curves are not normalized!')
            
        print('Partitioning data...')
        # x_train, x_test, y_train, y_test, ticid_train, ticid_test,\
        # target_info_train, target_info_test, flux_train, flux_test, time = \
        #     split_data(flux, flux_plot, ticid, target_info, time, p,
        #                train_test_ratio=train_test_ratio,
        #                supervised=False) 
        x_train, x_test, y_train, y_test, ticid_train, ticid_test,\
        target_info_train, target_info_test, time = \
            split_data(flux, ticid, target_info, time, p,
                       train_test_ratio=train_test_ratio,
                       supervised=False)             
            
        if input_rms:
            rms_train = rms[:np.shape(x_train)[0]]
            rms_test = rms[-1 * np.shape(x_test)[0]:]
            
            
        if input_psd:
            if not load_psd:
                psd_train = psd[:np.shape(x_train)[0]]
                psd_test = psd[-1 * np.shape(x_test)[0]:]
                
                # >> save the PSDs to a fits file
                hdr = fits.Header()
                hdu = fits.PrimaryHDU(psd_train, header=hdr)
                hdu.writeto(output_dir + 'psd_train.fits')
                fits.append(output_dir + 'psd_train.fits', ticid_train)
                hdr = fits.Header()
                hdu = fits.PrimaryHDU(psd_test, header=hdr)
                hdu.writeto(output_dir + 'psd_test.fits')
                fits.append(output_dir + 'psd_test.fits', ticid_test)
                
            x_train = [x_train, psd_train]
            x_test = [x_test, psd_test]
            time = [time, f]
            fig, ax = plt.subplots(4, 2)
            for i in range(4):
                ax[i, 0].plot(time[0], x_train[0][i], '.k', markersize=2)
                ax[i, 1].plot(time[1], x_train[1][i])
                ax[i, 0].set_xlabel('Time [BJD - 2457000]')
                ax[i, 0].set_ylabel('Relative flux')
                ax[i, 1].set_xlabel('Frequency (Hz)')
                ax[i, 1].set_ylabel('PSD')
                ax[i, 1].set_xscale('log')
                ax[i, 1].set_yscale('log')
            fig.savefig(output_dir + 'x_train_PSD.png')
            
        if split:
            orbit_gap = np.argmax(np.diff(time))
            # >> split x_train and x_test at orbit gap
            x_train = np.split(x_train, [orbit_gap], axis=1)
            x_test = np.split(x_test, [orbit_gap], axis=1)
            
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
            x_train = [flux_train, external_features_train]
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
            x_test = [flux_test, external_features_test]
                        
        # return x_train, x_test, y_train, y_test, ticid_train, ticid_test, \
        #     target_info_train, target_info_test, rms_train, rms_test, \
        #     flux_train, flux_test, time
            
        return x_train, x_test, y_train, y_test, ticid_train, ticid_test, \
            target_info_train, target_info_test, rms_train, rms_test, time            

def bottleneck_preprocessing(sector, flux, ticid, target_info,
                             rms=None,
                             output_dir='./SectorX/',
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
        * sector : sector number, given as int
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
        with fits.open(bottleneck_dir + 'bottleneck_train.fits') as hdul:
            bottleneck_train = hdul[0].data        
        with fits.open(bottleneck_dir + 'bottleneck_test.fits') as hdul:
            bottleneck_test = hdul[0].data
        learned_feature_vector = np.concatenate([bottleneck_train,
                                                 bottleneck_test], axis=0)
        features.append(learned_feature_vector)
        
    if use_tls_features:
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
        tess_features = np.loadtxt(data_dir + 'Sector'+str(sector)+\
                                   '/tess_features_sector'+str(sector)+'.txt',
                                   delimiter=' ', usecols=[1,2,3,4,5,6,8])
        # >> take out any light curves with nans
        inds = np.nonzero(np.prod(~np.isnan(tess_features), axis=1))
        tess_features = tess_features[inds]
        
        
        intersection, comm1, comm2 = np.intersect1d(tess_features[:,0], ticid,
                                                    return_indices=True)
        # >> take intersection, and get rid of TICID column
        tess_features = tess_features[:,1:][comm1]
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
        features = df.standardize(features, ax=0)
            
    return features, flux, ticid, target_info

def run_model(x_train, y_train, x_test, y_test, p, supervised=False,
              mock_data=False):
    '''Runs autoencoder or CNN.'''
    if not supervised:
        history, model = conv_autoencoder(x_train, x_train, x_test,
                                             x_test, p)
    if supervised:
        if mock_data:
            history, model = cnn_mock(x_train, y_train, x_test, y_test, p)
        else:
            history, model = cnn(x_train, y_train, x_test, y_test, p)
        
    x_predict = model.predict(x_test)
    return history, model, x_predict
    
def param_summary(history, x_test, x_predict, p, output_dir, param_set_num,
                  title, supervised=False, y_test=False):
    from sklearn.metrics import confusion_matrix
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
            mse = np.average((x_predict - x_test)**2)
            f.write('mse '+ str(mse) + '\n')
        f.write('\n')
    
        
def model_summary_txt(output_dir, model):
    with open(output_dir + 'model_summary.txt', 'a') as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

# :: autoencoder ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def pretrain(p, output_dir, input_dim=18688, input_psd=True, input_rms=False,
             dataset_size=10000, f_mean=2., truncate=True, reshape=False,
             hyperparam_opt=False):
    
    # >> make some mock data
    x, x_train, x_test = \
        get_high_freq_mock_data(p=p, dataset_size=dataset_size,
                                input_dim=input_dim, f_mean=f_mean,
                                truncate=truncate, reshape=reshape,
                                hyperparam_opt=hyperparam_opt)    


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

def conv_autoencoder(x_train, y_train, x_test, y_test, params, 
                     val=True, split=False, input_features=False,
                     features=None, input_psd=False,
                     model_init=None, save_model=True, save_bottleneck=True,
                     predict=True, output_dir='./',
                     input_rms=False, rms_train=None, rms_test=None,
                     ticid_train=None, ticid_test=None,
                     train=True, weights_path='./best_model.hdf5',
                     concat_ext_feats=False):
    
    # -- making swish activation function -------------------------------------
    # get_custom_objects().update({'swish': Activation(swish)})
    
    # -- encoding -------------------------------------------------------------
    params['concat_ext_feats']=concat_ext_feats
    if split:
        encoded = encoder_split(x_train, params) # >> shared weights        
        
    elif input_psd:
        concat_ext_feats = True
        # encoded = encoder_split_diff_weights(x_train, params)
        encoded = encoder(x_train, params)   
        
    else:
        encoded = encoder(x_train, params)

    # -- decoding -------------------------------------------------------------
    if split:
        decoded = decoder_split(x_train, encoded.output, params)
    # elif input_psd:
    #     decoded = decoder_split_diff_weights(x_train, encoded.output, params)

    elif params['cvae']:
        # z_mean, z_log_var, z = encoded.output
        decoded = decoder(x_train, encoded.output, params)
        
    else:
        decoded = decoder(x_train, encoded.output, params)
        
        
    model = Model(encoded.input, decoded)
    # model = decoder(x_train, encoded, params)
        
    print(model.summary())
    
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
        
    
    # -- compile model --------------------------------------------------------
    print('Compiling model...')
    compile_model(model, params)

    # -- train model ----------------------------------------------------------
    if train:
        print('Training model...')
        # tf.keras.backend.clear_session()
        tensorboard_callback = keras.callbacks.TensorBoard(histogram_freq=0)
        
        checkpoint = keras.callbacks.ModelCheckpoint(output_dir+"model.hdf5",
                                                     monitor='loss', verbose=1,
                                                     save_best_only=True, mode='auto',
                                                     save_freq='epoch')
        
        if val:
            history = model.fit(x_train, x_train, epochs=params['epochs'],
                                batch_size=params['batch_size'], shuffle=True,
                                validation_data=(x_test, x_test),
                                callbacks=[checkpoint, tensorboard_callback])
        else:
            history = model.fit(x_train, x_train, epochs=params['epochs'],
                        batch_size=params['batch_size'], shuffle=True,
                        callbacks=[checkpoint, tensorboard_callback])
            
        if save_model:
            model.save(output_dir + 'model')      
            
    else:
        print('Loading weights...')
        model.load_weights(weights_path)
        history=None
        
    # -------------------------------------------------------------------------
        
    if save_bottleneck:
        bottleneck_train = \
            get_bottleneck(model, x_train, params, save=True, ticid=ticid_train,
                           out=output_dir+'bottleneck_train.fits')
        bottleneck = get_bottleneck(model, x_test, params, save=True,
                                    ticid=ticid_test,
                                    out=output_dir+'bottleneck_test.fits')    
        
    if predict:
        x_predict = model.predict(x_test)      
        hdr = fits.Header()
        if concat_ext_feats:
            hdu = fits.PrimaryHDU(x_predict[0], header=hdr)
        else:
            hdu = fits.PrimaryHDU(x_predict, header=hdr)
        hdu.writeto(output_dir + 'x_predict.fits')
        fits.append(output_dir + 'x_predict.fits', ticid_test)
        
        if train:
            if concat_ext_feats:
                param_summary(history, x_test[0], x_predict[0], params, output_dir, 
                              0,'')
            else:
                param_summary(history, x_test, x_predict, params, output_dir, 0,'')            
        model_summary_txt(output_dir, model)               
    
        return history, model, x_predict
    
    return history, model

def swish(x, beta=1):
    '''https://www.bignerdranch.com/blog/implementing-swish-activation-function
    -in-keras/'''
    from keras.backend import sigmoid
    return (x*sigmoid(beta*x))

# def cluster_layer(bottleneck, params):
#     from keras.layers import Lambda
#     from keras.engine.topology import Layer, InputSpec
#     input_spec = InputSpec(dtype=K.floatx(),
#                            shape=(None, params['latent_dim']))
#     q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
#     q **= (self.alpha + 1.0) / 2.0
#     q = K.transpose(K.transpose(q) / K.sum(q, axis=1))    

def DCEC(x_train, y_train, x_test, y_test, params, n_clusters=10):
    '''Inspired by
    https://github.com/XifengGuo/DCEC/blob/master/ConvAE.py
    We should change the initialization from Kmeans to DBSCAN, although we 
    can't specify the number of clusters which sucks.'''
    # >> initialize cluster centers using k-means
    kmeans = KMeans(n_clusters=n_clusters)
    encoded, feature_maps = encoder(x_train, params)
    bottleneck = encoded.output
    y_pred = kmeans.fit_predict(bottleneck)
    
def max_pool_layer(x, params):
    ksize = [1, params['pool_size'], 1, 1]
    strides = [1, params['pool_strides'], 1, 1]
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(x)
    # >> get pooling indices 
    _, argmax = tf.nn.max_pool_with_argmax(x, ksize, strides, 'SAME')
    # >> now pool
    x = Lambda(lambda x: tf.nn.max_pool_with_argmax(x, ksize, strides, 'SAME')[0])(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)     
    return x, argmax

def unpool_with_with_argmax(x, ind, params):
    """
    Modified From:
    https://github.com/sangeet259/tensorflow_unpooling/blob/master/unpool.py
    Currently buggy [2020-09-28]
    
      To unpool the tensor after  max_pool_with_argmax.
      Argumnets:
          pooled:    the max pooled output tensor
          ind:       argmax indices , the second output of max_pool_with_argmax
          ksize:     ksize should be the same as what you have used to pool
      Returns:
          unpooled:      the tensor after unpooling
      Some points to keep in mind ::
          1. In tensorflow the indices in argmax are flattened, so that a maximum value at position [b, y, x, c] becomes flattened index ((b * height + y) * width + x) * channels + c
          2. Due to point 1, use broadcasting to appropriately place the values at their right locations ! 
    """
    ksize = [1, params['pool_size'], 1, 1]
    strides = [1, params['pool_strides'], 1, 1]  
    pooled = Lambda(lambda x: K.expand_dims(x, axis=2))(x)
    
    # Get the the shape of the tensor in th form of a list
    input_shape = pooled.get_shape().as_list()
    # Determine the output shape
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
    # Ceshape into one giant tensor for better workability
    pooled_ = tf.reshape(pooled, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]])
    # The indices in argmax are flattened, so that a maximum value at position [b, y, x, c] becomes flattened index ((b * height + y) * width + x) * channels + c
    # Create a single unit extended cuboid of length bath_size populating it with continous natural number from zero to batch_size
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
    b = tf.ones_like(ind) * batch_range
    b_ = tf.reshape(b, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
    ind_ = tf.reshape(ind, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
    ind_ = tf.concat([b_, ind_],1)
    ref = tf.Variable(tf.zeros([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]))
    # Update the sparse matrix with the pooled values , it is a batch wise operation
    unpooled_ = tf.scatter_nd_update(ref, ind_, pooled_)
    # Reshape the vector to get the final result 
    unpooled = tf.reshape(unpooled_, [output_shape[0], output_shape[1], output_shape[2], output_shape[3]])
    
    x = Lambda(lambda x: K.squeeze(x, axis=2))(unpooled) 
    return unpooled

def unpool_layer(x, params):
    ksize = params['pool_size']
    strides = params['pool_strides']

def PSNR(y_true, y_pred):
    '''https://stackoverflow.com/questions/55844618/how-to-calculate-psnr-metric-in-keras'''
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1))))

def ssim_loss(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def custom_loss_tsne(y_true, y_pred):
    '''https://github.com/kylemcdonald/Parametric-t-SNE/blob/master/Parametric%20t-SNE%20%28Keras%29.ipynb'''
    import keras.backend as K  
    
    # >> compute TSNE
    P_input = compute_joint_probabilities(y_true)
    P_output = compute_joint_probabilities(y_pred)
    
    # >> now compute MSE
    loss = K.mean(K.square(P_output - P_input))
    
    return loss


def Hbeta(D, beta):
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(np.multiply(D, P)) / sumP
    P = P / sumP
    return H, P

def x2p(X, u=15, tol=1e-4, print_iter=500, max_tries=50, verbose=0):
    # Initialize some variables
    # n = X.shape[0]                     # number of instances
    n = 14394
    P = np.zeros((n, n))               # empty probability matrix
    beta = np.ones(n)                  # empty precision vector
    logU = np.log(u)                   # log of perplexity (= entropy)
    
    # Compute pairwise distances
    if verbose > 0: print('Computing pairwise distances...')
    sum_X = np.sum(np.square(X), axis=1)
    # note: translating sum_X' from matlab to numpy means using reshape to add a dimension
    D = sum_X + sum_X[:,None] + -2 * X.dot(X.T)

    # Run over all datapoints
    if verbose > 0: print('Computing P-values...')
    for i in range(n):
        
        if verbose > 1 and print_iter and i % print_iter == 0:
            print('Computed P-values {} of {} datapoints...'.format(i, n))
        
        # Set minimum and maximum values for precision
        betamin = float('-inf')
        betamax = float('+inf')
        
        # Compute the Gaussian kernel and entropy for the current precision
        indices = np.concatenate((np.arange(0, i), np.arange(i + 1, n)))
        Di = D[i, indices]
        H, thisP = Hbeta(Di, beta[i])
        
        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while abs(Hdiff) > tol and tries < max_tries:
            
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i]
                if np.isinf(betamax):
                    beta[i] *= 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i]
                if np.isinf(betamin):
                    beta[i] /= 2
                else:
                    beta[i] = (beta[i] + betamin) / 2
            
            # Recompute the values
            H, thisP = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1
        
        # Set the final row of P
        P[i, indices] = thisP
        
    if verbose > 0: 
        print('Mean value of sigma: {}'.format(np.mean(np.sqrt(1 / beta))))
        print('Minimum value of sigma: {}'.format(np.min(np.sqrt(1 / beta))))
        print('Maximum value of sigma: {}'.format(np.max(np.sqrt(1 / beta))))
    
    return P, beta

def compute_joint_probabilities(samples, batch_size=5000, d=2, perplexity=30, tol=1e-5, verbose=0):
    v = d - 1
    
    # Initialize some variables
    # n = samples.shape[0] # !! hard code
    n = 14394
    batch_size = min(batch_size, n)
    
    # Precompute joint probabilities for all batches
    if verbose > 0: print('Precomputing P-values...')
    # batch_count = int(n / batch_size)
    batch_count = n // batch_size
    P = np.zeros((batch_count, batch_size, batch_size))
    for i, start in enumerate(range(0, n - batch_size + 1, batch_size)):   
        curX = samples[start:start+batch_size]                   # select batch
        P[i], beta = x2p(curX, perplexity, tol, verbose=verbose) # compute affinities using fixed perplexity
        P[i][np.isnan(P[i])] = 0                                 # make sure we don't have NaN's
        P[i] = (P[i] + P[i].T) # / 2                             # make symmetric
        P[i] = P[i] / P[i].sum()                                 # obtain estimation of joint probabilities
        P[i] = np.maximum(P[i], np.finfo(P[i].dtype).eps)

    return P

def custom_loss_function(y_true, y_pred):
    '''Might be useful for testing:
        t=tf.convert_to_tensor(t, 'float32')
        t=tf.where(tf.is_nan(t), tf.zeros_like(t), t)
    '''
    import keras.backend as K
    from scipy import signal
    import tensorflow as tf
    
    # sess = tf.Session()
    # with sess.as_default():
        
    # >> calculating the mean squared error (L2 loss)
    l2_loss = K.mean(K.square(y_pred - y_true))
    
    # >> calculating Fourier Transform
    # >> note frequencies = [0, 1/(dt*n), ..., n/(2*dt*n)]
    yf_true = tf.signal.rfft(y_true)
    yf_true = tf.math.real(yf_true) # >> only keep real values
    yf_pred = tf.signal.rfft(y_pred)
    yf_pred = tf.math.real(yf_pred)
    
    # >> integrating over high frequencies (from f=0.3 to f=10 days^-1)
    # dt = 0.001388916488394898 # >> sampling interval = (sampling rate)^-1
    # n = y_true.get_shape().as_list()[0]
    # low = 0.3
    # high = 10.
    # low_ind = int(low*dt*n)
    # high_ind = int(high*dt*n)
    
    # dt = tf.constant(0.001388916488394898) # >> sampling interval
    # n = tf.constant(y_true.get_shape().as_list()[0]) # >> length of light curve
    # n = tf.cast(n, 'float32')
    # # n = tf.shape(y_true)
    # low = tf.constant(0.3)
    # high = tf.constant(10.)
    # low_ind = tf.math.round(low*dt*n)
    # low_ind = tf.cast(low_ind, 'int32')
    # high_ind = tf.math.round(high*dt*n)
    # high_ind = tf.cast(high_ind, 'int32')
    # size = high_ind - low_ind
    
    # >> normalize
    # yf_true = yf_true / n
    # yf_pred = yf_pred / n
    
    # power_true = tf.math.reduce_sum(yf_true[low_ind:high_ind])
    # power_pred = tf.math.reduce_sum(yf_pred[low_ind:high_ind])
    
    power_true = tf.math.reduce_sum(yf_true)
    power_pred = tf.math.reduce_sum(yf_pred)    
    
    # >> penalty for loss of high frequency information
    scale_factor = 1./15000
    penalty = tf.math.abs(scale_factor*power_true - scale_factor*power_pred)
    
    return l2_loss + penalty

def test_loss_function(y_true, y_pred, output_dir='./',
                       dt=0.001388916488394898):
    import tensorflow as tf
    import keras.backend as K
    
    N = len(y_true)
    
    # >> plot light curves
    fig, ax = plt.subplots(2)
    ax[0].plot(y_true, '.k', markersize=2)
    ax[0].plot(y_pred, '.', markersize=2)    
    ax[0].set_xlabel('Time [BJD - 2457000]')
    ax[0].set_ylabel('Relative flux')
    
    # >> convert to tensors
    y_true=tf.convert_to_tensor(y_true, 'float32')
    y_pred=tf.convert_to_tensor(y_pred, 'float32')
    
    # >> compute l2_loss
    l2_loss = K.mean(K.square(y_pred - y_true))
    
    # >> compute custom loss
    custom_loss = custom_loss_function(y_true, y_pred)
    
    # >> report loss
    ax[0].set_title('MSE: ' + str(K.get_value(l2_loss)) + ', custom loss: '+\
                    str(K.get_value(custom_loss)))
    
    # >> compute FFT
    yf_true = tf.signal.rfft(y_true)
    yf_true = tf.math.real(yf_true) # >> only keep real values
    yf_pred = tf.signal.rfft(y_pred)
    yf_pred = tf.math.real(yf_pred)    
    f = np.fft.fftfreq(N, dt)
    
    # >> plot FFT
    ax[1].plot(f[:N//2], K.get_value(yf_true)[:N//2], 'k')
    ax[1].plot(f[:N//2], K.get_value(yf_pred)[:N//2])
    ax[1].set_xscale('log')
    ax[1].set_xlabel('Frequency [days^-1]')
    ax[1].set_ylabel('Power')
    
    fig.savefig(output_dir + 'custom_loss.png')
    
    return fig, ax

def custom_loss_function1(y_true, y_pred):
    '''Fails when compiling. Also an error pops when running K.get_value(loss)'''
    import keras.backend as K
    from scipy import signal
    import tensorflow as tf
    from sklearn.mixture import GaussianMixture
    
    # sess = tf.Session()
    # with sess.as_default():
        
    # >> calculating the mean squared error (L2 loss)
    l2_loss = K.mean(K.square(y_pred - y_true))
    
    # >> applying high pass filter (cutoff = 1.)
    order=5
    T=0.001388916488394898
    fs=1/T
    nyq = 0.5*fs
    cutoff = 1./0.5*fs 
    b, a = signal.butter(order, cutoff, btype='high')        
    y_filt = signal.filtfilt(b, a, y_pred)    
    
    # >> calculate RMS of filtered light curve
    rms = K.sqrt(K.mean(K.square(y_filt)))
    rms = tf.cast(rms, 'float32')
    
    debug=False
    if debug:
        plt.figure()
        plt.plot()
    
    return l2_loss + rms
    
    
def iterative_cae(x_train, y_train, x_test, y_test, x, p, ticid_train, 
                  ticid_test, target_info_train, target_info_test, num_split=2,
                  output_dir='./', split=False, input_psd=False, 
                  database_dir='./', data_dir='./', train_psd_only=True,
                  momentum_dump_csv='./Table_of_momentum_dumps.csv', sectors=[],
                  concat_ext_feats=False):
    
    # >> load model
    model = keras.models.load_model(output_dir+'model')
    
    # >> get training set of highest reconstruction error
    x_train_predict = model.predict(x_train)
    x_test_predict = model.predict(x_test)
    if concat_ext_feats or input_psd: 
        err_train = (x_train[0] - x_train_predict[0])**2
        err_test = (x_test[0] - x_test_predict[0])**2
    else:      
        err_train = (x_train - x_train_predict)**2
        err_test = (x_test - x_test_predict)**2
    err_train = np.mean(err_train, axis=1)
    err_train = err_train.reshape(np.shape(err_train)[0])
    ranked_train = np.argsort(err_train)
    err_test = np.mean(err_test, axis=1)
    err_test = err_test.reshape(np.shape(err_test)[0])
    ranked_test = np.argsort(err_test)    
    del x_train_predict
    del x_test_predict
    del err_train
    del err_test
    
    # >> re-order arrays
    ticid_train = ticid_train[ranked_train]
    info_train = target_info_train[ranked_train]
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
    x_train = np.split(x_train, [int(train_len/num_split)])
    x_test = np.split(x_test, [int(test_len/num_split)])
    if concat_ext_feats or input_psd:
        x_train_feat = np.split(x_train_feat, [int(train_len/num_split)])
        x_train[0] = np.concatenate([x_train[0], x_train_feat[0]], axis=0)
        x_train[1] = np.concatenate([x_train[1], x_train_feat[1]], axis=0)        
        x_test_feat = np.split(x_test_feat, [int(test_len/num_split)])
        x_test[0] = np.concatenate([x_test[0], x_test_feat[0]], axis=0)
        x_test[1] = np.concatenate([x_test[1], x_test_feat[1]], axis=0)
    ticid_train = np.split(ticid_train, [int(train_len/num_split)])
    info_train = np.split(info_train, [int(train_len/num_split)])
    ticid_test = np.split(ticid_test, [int(test_len/num_split)])
    info_test = np.split(info_test, [int(test_len/num_split)])    
    
    if train_psd_only:
        x_train = x_train_feat
        x_test = x_test_feat
        x = x[1]
        input_psd=False
    
    pdb.set_trace()
    
    history_list = []
    model_list = []
    for i in range(num_split):
        # model_init = output_dir+'model'
        model_init = None
        history, model, x_predict = \
            conv_autoencoder(x_train[i], x_train[i], x_test[i], x_test[i], p,
                             val=False, split=split, save_model=True,
                             predict=True, save_bottleneck=True,
                             output_dir=output_dir+'iter'+str(i)+'-',
                             model_init=model_init)
        history_list.append(history)
        model_list.append(model)
        
        features = bottleneck_train
        pf.diagnostic_plots(history_new, model_new, p, output_dir+'iter'+str(i)+'-',
                            x, x_train[i], x_test[i],
                            x_predict, mock_data=False,
                            addend=0., prefix=str(i),
                            target_info_test=info_test[i],
                            target_info_train=info_train[i],
                            ticid_train=ticid_train[i],
                            ticid_test=ticid_test[i], percentage=False,
                            input_features=False,
                            input_rms=False, 
                            input_psd=input_psd, n_tot=40,
                            plot_epoch = True,
                            plot_in_out = True,
                            plot_in_bottle_out=False,
                            plot_latent_test = True,
                            plot_latent_train = True,
                            plot_kernel=False,
                            plot_intermed_act=False,
                            make_movie = False,
                            plot_lof_test=False,
                            plot_lof_train=False,
                            plot_lof_all=False,
                            plot_reconstruction_error_test=True,
                            plot_reconstruction_error_all=False,
                            load_bottleneck=True)  
         
        if concat_ext_feats:
            
            
            x_predict = model_new.predict(x_train)
            pf.diagnostic_plots(history_new, model_new, p, output_dir+'iter'+str(i)+'-', x,
                                x_train,
                                x_train, x_predict, mock_data=False,
                                addend=0.,
                                target_info_test=info_test[i],
                                target_info_train=info_train[i],
                                ticid_train=ticid_train[i],
                                ticid_test=ticid_test[i], percentage=False,
                                input_features=False,
                                input_rms=False, 
                                input_psd=True, n_tot=40,
                                plot_epoch = True,
                                plot_in_out = True,
                                plot_in_bottle_out=False,
                                plot_latent_test = False,
                                plot_latent_train = False,
                                plot_kernel=False,
                                plot_intermed_act=False,
                                make_movie = False,
                                plot_lof_test=False,
                                plot_lof_train=False,
                                plot_lof_all=False,
                                plot_reconstruction_error_test=True,
                                plot_reconstruction_error_all=False,
                                load_bottleneck=True)      

        features, flux_feat, ticid_feat, info_feat = \
            ml.bottleneck_preprocessing(sectors,
                                        np.concatenate([x_train[i], x_test[i]], axis=0),
                                        np.concatenate([ticid_train[i], ticid_test][i]),
                                        np.concatenate([target_info_train,
                                                        target_info_test]),
                                        rms=rms,
                                        data_dir=data_dir,
                                        bottleneck_dir=output_dir,
                                        output_dir=output_dir,
                                        use_learned_features=True,
                                        use_tess_features=use_tess_features,
                                        use_engineered_features=False,
                                        use_tls_features=use_tls_features,
                                        use_rms=use_rms, norm=True,
                                        cams=cams, ccds=ccds, log=True)  

        pf.plot_lof(x, x_train[i], ticid_train[i], features, 20,
                    output_dir+'iter'+str(i)+'-',
                    n_tot=100, target_info=info_train[i], prefix=str(i),
                    database_dir=database_dir, debug=True, addend=0.,
                    single_file=False, log=True, n_pgram=1000,
                    plot_psd=True, momentum_dump_csv=momentum_dump_csv)       
        
        best_param_set = [3, 3, 'canberra', None]
        print('Run HDBSCAN')
        _, _, acc = df.hdbscan_param_search(features, x, new_x_train[i], new_ticid_train[i],
                                      new_info_train[i], output_dir=output_dir+str(i),
                                      p0=[best_param_set[3]], single_file=False,
                                      database_dir=database_dir,
                                      metric=[best_param_set[2]],
                                      min_cluster_size=[best_param_set[0]],
                                      min_samples=[best_param_set[1]],
                                      DEBUG=True, pca=True, tsne=True,
                                      data_dir=data_dir, save=True)   
        
        gmm = GaussianMixture(n_components=200)
        labels = gmm.fit_predict(features)
        acc = pf.plot_confusion_matrix(new_ticid_train[i], labels,
                                       database_dir=database_dir,
                                       single_file=False,
                                       output_dir=output_dir+str(i),
                                       prefix='gmm-')          
        pf.quick_plot_classification(x, new_x_train[i], new_ticid_train[i],
                                     new_info_train[i], 
                                     features, labels,path=output_dir+str(i),
                                     prefix='gmm-',
                                     database_dir=database_dir)
        pf.plot_cross_identifications(x, new_x_train[i], new_ticid_train[i],
                                      new_info_train[i], features,
                                      labels, path=output_dir,
                                      database_dir=database_dir,
                                      data_dir=data_dir, prefix=str(i)+'gmm-')        
    return history_list, model_list

def cnn(x_train, y_train, x_test, y_test, params, num_classes=4):
    from keras.models import Model
    from keras.layers import Dense

    # -- encoding -------------------------------------------------------------
    encoded = encoder(x_train, params)
    
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
    from keras.models import Model
    from keras.layers import Dense

    # -- encoding -------------------------------------------------------------
    encoded = encoder(x_train, params)
    
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
    '''a simple classifier based on a fully-connected layer
    Deprecated 071720'''

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

def deep_autoencoder(x_train, y_train, x_test, y_test, params, resize = False,
                       batch_norm=True):
    '''No convolutional layers!'''
    from keras.models import Model
    from keras.layers import Input, Dense, Flatten, Reshape, BatchNormalization

    num_classes = np.shape(y_train)[1]
    input_dim = np.shape(x_train)[1]
    
    # params['hidden_units'] = list(range(params['max_dim'],
    #                                     params['latent_dim'],
    #                                     -params['step']))
    hidden_units = list(range(params['max_dim'],
                              params['latent_dim'],
                              -params['step']))    
    # params['hidden_units'] = list(range(16, params['latent_dim'],
    #                                     -params['step']))
    if hidden_units[-1] != params['latent_dim']:
        hidden_units.append(params['latent_dim'])
    
    if resize:
        input_img = Input(shape = (input_dim,1))
        x = Flatten()(input_img)
    else:
        input_img = Input(shape = (input_dim,))
        x = input_img
    for i in range(len(hidden_units)):
        x = Dense(hidden_units[i], activation=params['activation'],
                  kernel_initializer=params['initializer'])(x)
        if batch_norm: x = BatchNormalization()(x)
        
    x = Dense(params['latent_dim'], activation=params['activation'],
              kernel_initializer=params['initializer'])(x)
    for i in np.arange(len(hidden_units)-1, -1, -1):
        if batch_norm: x = BatchNormalization()(x)        
        x = Dense(hidden_units[i], activation=params['activation'],
                  kernel_initializer=params['initializer'])(x)

    if batch_norm: x = BatchNormalization()(x)    
    x = Dense(input_dim, activation=params['last_activation'],
              kernel_initializer=params['initializer'])(x)
    if resize:
        x = Reshape((input_dim, 1))(x)
        
    model = Model(input_img, x)
    model.summary()
    compile_model(model, params)

    history = model.fit(x_train, x_train, epochs=params['epochs'],
                        batch_size=params['batch_size'], shuffle=True,
                        validation_data=(x_test, x_test))
        
    return history, model

def encoder_DAE(x_train, params):
    from keras.models import Model
    from keras.layers import Input, Dense, Flatten, BatchNormalization
    
    input_dim = np.shape(x_train)[1]

def compile_model(model, params):

    if params['optimizer'] == 'adam':
        # opt = optimizers.adam(lr = params['lr'], 
        #                       decay=params['lr']/params['epochs'])
        opt = optimizers.Adam(lr = params['lr'], 
                              decay=params['lr']/params['epochs'])        
    elif params['optimizer'] == 'adadelta':
        opt = optimizers.adadelta(lr = params['lr'])
        
    model.compile(optimizer=opt, loss=params['losses'])

def sampling(args):
    """
    Adapted from https://github.com/piyush-kgp/VAE-MNIST-Keras
    Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1] # Returns the shape of tensor or variable as a tuple of int or None entries.
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# def encoder(x_train, params, reshape=False):
#     '''x_train is an array with shape (num light curves, num data points, 1).
#     params is a dictionary with keys:
#         * kernel_size : 3, 5
#         * latent_dim : dimension of bottleneck/latent space
#         * strides : 1
#         * epochs
#         * dropout
#         * num_filters : 8, 16, 32, 64...
#         * num_conv_layers : number of convolutional layers in entire
#           autoencoder (number of conv layers in encoder is num_conv_layers/2)
#         * num_consecutive : number of consecutive convolutional layers (can
#           currently only handle 1 or 2)
#         * batch_size : 128
#         * activation : 'elu'
#         * last_activation : 'linear'        
#         * optimizer : 'adam'
#         * losses : 'mean_squared_error', 'custom'
#         * lr : learning rate (e.g. 0.01)
#         * initializer: 'random_normal', 'random_uniform', ...
#     '''
    
#     if params['concat_ext_feats']:
#         input_dim = np.shape(x_train[0])[1]
#         input_dim1 = np.shape(x_train[1])[1]
#         input_img1 = Input(shape = (input_dim1,))
#         x1 = input_img1
#     else:
#         input_dim = np.shape(x_train)[1]
#     num_iter = int(params['num_conv_layers']/2)
    
#     if type(params['num_filters']) == np.int:
#         params['num_filters'] = list(np.repeat(params['num_filters'], num_iter))
#     if type(params['num_consecutive']) == np.int:
#         params['num_consecutive'] = list(np.repeat(params['num_consecutive'], num_iter))
    
#     encoder = Sequential()
    
#     encoder.add(Input(shape = (input_dim,)))
#     encoder.add(Reshape((input_dim, 1)))
    
#     for i in range(num_iter):
        
#         for j in range(params['num_consecutive'][i]):
#             encoder.add(Conv1D(params['num_filters'][i], int(params['kernel_size']),
#                     padding='same',
#                     kernel_initializer=params['initializer'],
#                     strides=params['strides'],
#                     kernel_regularizer=params['kernel_regularizer'],
#                     bias_regularizer=params['bias_regularizer'],
#                     activity_regularizer=params['activity_regularizer']))             

            
#             if params['batch_norm']:
#                 encoder.add(BatchNormalization())
            
#             encoder.add(Activation(params['activation']))
            
#         encoder.add(MaxPooling1D(params['pool_size'], padding='same'))
#         encoder.add(Dropout(params['dropout']))
        
#     encoder.add(Flatten())

#     encoder.add(Dense(params['latent_dim'], activation=params['activation'],
#                     kernel_initializer=params['initializer'],
#                     kernel_regularizer=params['kernel_regularizer'],
#                     bias_regularizer=params['bias_regularizer'],
#                     activity_regularizer=params['activity_regularizer']))

#     return encoder

def encoder(x_train, params, reshape=False):
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
    
    if params['concat_ext_feats']:
        input_dim = np.shape(x_train[0])[1]
        input_dim1 = np.shape(x_train[1])[1]
        input_img1 = Input(shape = (input_dim1,))
        x1 = input_img1
    else:
        input_dim = np.shape(x_train)[1]
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

    if params['fully_conv']:
        encoded = Conv1D(1, int(params['kernel_size']),
                activation=params['activation'], padding='same',
                kernel_initializer=params['initializer'],
                strides=params['strides'],
                kernel_regularizer=params['kernel_regularizer'],
                bias_regularizer=params['bias_regularizer'],
                activity_regularizer=params['activity_regularizer'])(x)
        
    elif params['concat_ext_feats']:
        for i in range(len(params['units'])):
            x1 = Dense(params['units'][i], activation=params['activation'],
                        kernel_initializer=params['initializer'],
                        kernel_regularizer=params['kernel_regularizer'],
                        bias_regularizer=params['bias_regularizer'],
                        activity_regularizer=params['activity_regularizer'])(x1)
            
        x = Flatten()(x)      
        x = Dense(params['latent_dim'], activation=params['activation'],
                  kernel_initializer=params['initializer'],
                  kernel_regularizer=params['kernel_regularizer'],
                  bias_regularizer=params['bias_regularizer'],
                  activity_regularizer=params['activity_regularizer'])(x)
        x = concatenate([x, x1])
        encoded = Dense(params['latent_dim']+params['units'][-1],
                  activation=params['activation'],
                  kernel_initializer=params['initializer'],
                  kernel_regularizer=params['kernel_regularizer'],
                  bias_regularizer=params['bias_regularizer'],
                  activity_regularizer=params['activity_regularizer'])(x)     
        
        
    elif params['cvae']:
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
        z = Lambda(sampling, output_shape=(params['latent_dim'],))([z_mean, z_log_var])         
        
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
                        activity_regularizer=params['activity_regularizer'])(x)
    
    if params['cvae']:
        encoder = Model(input_img, [z_mean, z_log_var, z])
        
    else:
        if params['concat_ext_feats']:
            encoder = Model([input_img, input_img1], encoded)
        else:
            encoder = Model(input_img, encoded)

    return encoder




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

# def skip_connection(feature_map_decoder, feature_map_encoder):
    #     connect_ind=len(feature_maps)-1-i*params['num_consecutive']-j
    #     x = Add()([x, feature_maps[connect_ind]])
    
    # return x

# def decoder(x_train, model, params):
#     import tensorflow as tf
    
#     if params['concat_ext_feats']:
#         input_dim = np.shape(x_train[0])[1]
#         input_dim1 = np.shape(x_train[1])[1]
#     else:
#         input_dim = np.shape(x_train)[1]
        
#     num_iter = int(params['num_conv_layers']/2)
#     reduction_factor = params['pool_size'] * params['strides']**params['num_consecutive'][0] 
#     tot_reduction_factor = reduction_factor**num_iter
    
#     if type(params['num_filters']) == np.int:
#         params['num_filters'] = list(np.repeat(params['num_filters'], num_iter))    
#     if type(params['num_consecutive']) == np.int:
#         params['num_consecutive'] = list(np.repeat(params['num_consecutive'], num_iter))

#     model.add(Dense(int(input_dim*params['num_filters'][-1]/tot_reduction_factor),
#                   kernel_initializer=params['initializer'],
#                   kernel_regularizer=params['kernel_regularizer'],
#                   bias_regularizer=params['bias_regularizer'],
#                   activity_regularizer=params['activity_regularizer']))
#     model.add(Reshape((int(input_dim/tot_reduction_factor),
#                          params['num_filters'][-1])))


#     for i in range(num_iter):
#         if params['dropout'] > 0:
#             model.add(Dropout(params['dropout']))
            
#         model.add(UpSampling1D(params['pool_size']))
        
        
#         for j in range(params['num_consecutive'][-1*i - 1]):
            
#             # >> last layer
#             if i == num_iter-1 and j == params['num_consecutive'][-1*i - 1]-1 \
#                 and not params['fully_conv']:
                
#                 if params['strides'] == 1: # >> faster than Conv1Dtranspose
#                     model.add(Conv1D(1, int(params['kernel_size']),
#                                       padding='same', strides=params['strides'],
#                                       kernel_initializer=params['initializer'],
#                                       kernel_regularizer=params['kernel_regularizer'],
#                                       bias_regularizer=params['bias_regularizer'],
#                                       activity_regularizer=params['activity_regularizer']))
                    
                
#                 # else: # !!
#                 #     decoder.add(Conv1DTranspose(x, 1, int(params['kernel_size']),
#                 #                padding='same',
#                 #                strides=params['strides'],
#                 #                kernel_initializer=params['initializer'],
#                 #                kernel_regularizer=params['kernel_regularizer'],
#                 #                bias_regularizer=params['bias_regularizer'],
#                 #           activity_regularizer=params['activity_regularizer']))
                    
#                 model.add(BatchNormalization())
#                 model.add(Activation(params['last_activation']))
#                 model.add(Reshape((input_dim,)))
                    
#                 if params['concat_ext_feats']:
#                     for i in range(len(params['units'])):
#                         x1 = Dense(params['units'][-1*i-1],
#                                    activation=params['activation'],
#                                    kernel_initializer=params['initializer'],
#                                    kernel_regularizer=params['kernel_regularizer'],
#                                    bias_regularizer=params['bias_regularizer'],
#                                    activity_regularizer=params['activity_regularizer'])(x1)
                        
#                     x1 = Dense(input_dim1,
#                                activation=params['activation'],
#                                kernel_initializer=params['initializer'],
#                                kernel_regularizer=params['kernel_regularizer'],
#                                bias_regularizer=params['bias_regularizer'],
#                                activity_regularizer=params['activity_regularizer'])(x1)                        
#                     decoded = [decoded, x1]
                    
#             else:
                
#                 if params['strides'] == 1:
#                     model.add(Conv1D(params['num_filters'][-1*i - 1],
#                                 int(params['kernel_size']),padding='same',
#                                 strides=params['strides'],
#                                 kernel_initializer=params['initializer'],
#                                 kernel_regularizer=params['kernel_regularizer'],
#                                 bias_regularizer=params['bias_regularizer'],
#                                 activity_regularizer=params['activity_regularizer']))  
#                 # else:
#                 #     x = Conv1DTranspose(x, params['num_filters'][-1*i - 1],
#                 #                int(params['kernel_size']), padding='same',
#                 #                strides=params['strides'],
#                 #                kernel_initializer=params['initializer'],
#                 #                kernel_regularizer=params['kernel_regularizer'],
#                 #                bias_regularizer=params['bias_regularizer'],
#                 #                activity_regularizer=params['activity_regularizer'])
#                 model.add(BatchNormalization())
#                 model.add(Activation(params['activation']))
        
#     return model


def decoder(x_train, bottleneck, params):
    import tensorflow as tf
    
    if params['concat_ext_feats']:
        input_dim = np.shape(x_train[0])[1]
        input_dim1 = np.shape(x_train[1])[1]
    else:
        input_dim = np.shape(x_train)[1]
        
    num_iter = int(params['num_conv_layers']/2)
    reduction_factor = params['pool_size'] * params['strides']**params['num_consecutive'][0] 
    tot_reduction_factor = reduction_factor**num_iter
    
    if type(params['num_filters']) == np.int:
        params['num_filters'] = list(np.repeat(params['num_filters'], num_iter))    
    if type(params['num_consecutive']) == np.int:
        params['num_consecutive'] = list(np.repeat(params['num_consecutive'], num_iter))
        

    if params['fully_conv']:
        x = bottleneck    
    else:
        if params['cvae']:
            z_mean, z_log_var, x = bottleneck
        elif params['concat_ext_feats']:
            x = bottleneck
            x1 = Dense(params['units'][-1],
                      kernel_initializer=params['initializer'],
                      kernel_regularizer=params['kernel_regularizer'],
                      bias_regularizer=params['bias_regularizer'],
                      activity_regularizer=params['activity_regularizer'])(bottleneck)            
        else:
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
                    
                if params['concat_ext_feats']:
                    for i in range(len(params['units'])):
                        x1 = Dense(params['units'][-1*i-1],
                                    activation=params['activation'],
                                    kernel_initializer=params['initializer'],
                                    kernel_regularizer=params['kernel_regularizer'],
                                    bias_regularizer=params['bias_regularizer'],
                                    activity_regularizer=params['activity_regularizer'])(x1)
                        
                    x1 = Dense(input_dim1,
                                activation=params['activation'],
                                kernel_initializer=params['initializer'],
                                kernel_regularizer=params['kernel_regularizer'],
                                bias_regularizer=params['bias_regularizer'],
                                activity_regularizer=params['activity_regularizer'])(x1)                        
                    decoded = [decoded, x1]
                    
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

    if params['fully_conv']:
        decoded = Conv1DTranspose(x, 1, int(params['kernel_size']),
                    activation=params['activation'], padding='same',
                    strides=params['strides'],
                    kernel_initializer=params['initializer'],
                    kernel_regularizer=params['kernel_regularizer'],
                    bias_regularizer=params['bias_regularizer'],
              activity_regularizer=params['activity_regularizer'])       
        decoded = Reshape((input_dim,))(decoded)          
    return decoded

def decgen(x_train, params):
    import tensorflow as tf
    
    if params['concat_ext_feats']:
        input_dim = np.shape(x_train[0])[1]
        input_dim1 = np.shape(x_train[1])[1]
    else:
        input_dim = np.shape(x_train)[1]
        
    num_iter = int(params['num_conv_layers']/2)
      
    input_img = Input(shape=(params['latent_dim'],))

    reduction_factor = params['pool_size'] * params['strides']
    tot_reduction_factor = reduction_factor**num_iter
    x = Dense(int(input_dim*params['num_filters'][-1]/tot_reduction_factor),
              kernel_initializer=params['initializer'],
              kernel_regularizer=params['kernel_regularizer'],
              bias_regularizer=params['bias_regularizer'],
              activity_regularizer=params['activity_regularizer'])(input_img)
    x = Reshape((int(input_dim/tot_reduction_factor),
                  params['num_filters'][-1]))(x)


    for i in range(num_iter):
        if params['dropout'] > 0:
            x = Dropout(params['dropout'])(x)
            
        if params['pool_size'] > 1:
            x = UpSampling1D(params['pool_size'])(x)            
        
        for j in range(params['num_consecutive'][-1*i - 1]):
            x = BatchNormalization()(x)
            if i == num_iter-1 and j == params['num_consecutive'][-1*i - 1]-1 \
                and not params['fully_conv']:
                
                if params['strides'] == 1: # >> faster than Conv1Dtranspose
                    x = Conv1D(1, int(params['kernel_size']),
                                      activation=params['last_activation'],
                                      padding='same', strides=params['strides'],
                                      kernel_initializer=params['initializer'],
                                      kernel_regularizer=params['kernel_regularizer'],
                                      bias_regularizer=params['bias_regularizer'],
                                      activity_regularizer=params['activity_regularizer'])(x)  
                
                else:
                    x = Conv1DTranspose(x, 1, int(params['kernel_size']),
                               activation=params['activation'], padding='same',
                               strides=1,
                               kernel_initializer=params['initializer'],
                               kernel_regularizer=params['kernel_regularizer'],
                               bias_regularizer=params['bias_regularizer'],
                          activity_regularizer=params['activity_regularizer'])
                    
                
                x = Reshape((input_dim,))(x)
                    
                if params['concat_ext_feats']:
                    for i in range(len(params['units'])):
                        x1 = Dense(params['units'][-1*i-1],
                                   activation=params['activation'],
                                   kernel_initializer=params['initializer'],
                                   kernel_regularizer=params['kernel_regularizer'],
                                   bias_regularizer=params['bias_regularizer'],
                                   activity_regularizer=params['activity_regularizer'])(x1)
                        
                    x1 = Dense(input_dim1,
                               activation=params['activation'],
                               kernel_initializer=params['initializer'],
                               kernel_regularizer=params['kernel_regularizer'],
                               bias_regularizer=params['bias_regularizer'],
                               activity_regularizer=params['activity_regularizer'])(x1)                        
                    x = [decoded, x1]
                    
            else:
                if j == 0:
                    stride = params['strides']
                else:
                    stride = 1
                
                if params['strides'] == 1:
                    x = Conv1D(params['num_filters'][-1*i - 1],
                                int(params['kernel_size']),
                                activation=params['activation'], padding='same',
                                strides=params['strides'],
                                kernel_initializer=params['initializer'],
                                kernel_regularizer=params['kernel_regularizer'],
                                bias_regularizer=params['bias_regularizer'],
                                activity_regularizer=params['activity_regularizer'])(x)   
                else:
                    x = Conv1DTranspose(x, params['num_filters'][-1*i - 1],
                               int(params['kernel_size']),
                               activation=params['activation'], padding='same',
                               strides=stride,
                               kernel_initializer=params['initializer'],
                               kernel_regularizer=params['kernel_regularizer'],
                               bias_regularizer=params['bias_regularizer'],
                               activity_regularizer=params['activity_regularizer'])
                
                if params['encoder_decoder_skip']:
                    connect_ind=len(feature_maps)-1-i*params['num_consecutive'][-1*i - 1]-j
                    x = Add()([x, feature_maps[connect_ind]])
                    x = Activation(params['activation'])(x)
                              

    if params['fully_conv']:
        x = Conv1DTranspose(x, 1, int(params['kernel_size']),
                   activation=params['activation'], padding='same',
                   strides=params['strides'],
                   kernel_initializer=params['initializer'],
                   kernel_regularizer=params['kernel_regularizer'],
                   bias_regularizer=params['bias_regularizer'],
              activity_regularizer=params['activity_regularizer'])       
        x = Reshape((input_dim,))(x)        

    # x = Reshape((input_dim,1))(x)  
    decoded = Model(input_img, x)
    return decoded



def discriminator(x_train, params, reshape=False):
    input_dim = np.shape(x_train)[-1]
    input_img = Input(shape=(input_dim,))
    x = Reshape((input_dim, 1))(input_img)
    x = Conv1D(8, kernel_size=int(params['kernel_size']),
               activation=params['activation'],
               strides=params['strides'], padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Conv1D(16, kernel_size=int(params['kernel_size']),
               activation=params['activation'],
               strides=params['strides'], padding='same')(x)    
    x = BatchNormalization()(x)

    x = Conv1D(32, kernel_size=int(params['kernel_size']),
               activation=params['activation'],
               strides=params['strides'], padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Conv1D(32, kernel_size=int(params['kernel_size']),
               activation=params['activation'],
               strides=params['strides'], padding='same')(x)    

    x = BatchNormalization()(x)

    dec = Flatten()(x)
    dec = Dense(1, activation='sigmoid')(x)

    return Model([input_img], [dec, x])


def encoder_split(x, params):
    from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten
    from keras.layers import Dense, concatenate, BatchNormalization
    from keras.models import Model

    input_imgs = [Input(shape=(np.shape(a)[1], 1)) for a in x]
    num_iter = int((params['num_conv_layers'])/2)    

    for i in range(num_iter):

        if i == 0:
            conv_1 = Conv1D(params['num_filters'], int(params['kernel_size']),
                            activation=params['activation'], padding='same',
                            kernel_initializer=params['initializer'])            
            x = [conv_1(a) for a in input_imgs]
            if params['num_consecutive']==2:
                conv_2 = Conv1D(params['num_filters'],
                                int(params['kernel_size']),
                                activation=params['activation'],
                                padding='same',
                                kernel_initializer=params['initializer'])                
                x = [conv_2(a) for a in x]
        else:
            conv_3 = Conv1D(params['num_filters'], int(params['kernel_size']),
                            activation=params['activation'], padding='same',
                            kernel_initializer=params['initializer'])                 
            x = [conv_3(a) for a in x]

            if params['num_consecutive']==2:
                conv_4 = Conv1D(params['num_filters'],
                                int(params['kernel_size']),
                                activation=params['activation'],
                                padding='same',
                                kernel_initializer=params['initializer'])                 
                x = [conv_4(a) for a in x]
                  
            
        batch_norm = BatchNormalization()
        x = [batch_norm(a) for a in x]
            
        maxpool_1 = MaxPooling1D(2, padding='same')
        x = [maxpool_1(a) for a in x]
        
        dropout_1 = Dropout(params['dropout'])
        x = [dropout_1(a) for a in x]

    x = [Flatten()(a) for a in x]
    
    x = [Dense(int(params['latent_dim']/2),
               activation=params['activation'])(a) for a in x]
    
    encoded = concatenate(x)
    encoder = Model(inputs=input_imgs, outputs=encoded)
    return encoder

def encoder_split_diff_weights(x, params):
    from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten
    from keras.layers import Dense, concatenate, BatchNormalization
    from keras.models import Model

    input_imgs = [Input(shape=(np.shape(a)[1], 1)) for a in x]
    num_iter = int((params['num_conv_layers'])/2)    

    for i in range(num_iter):

        if i == 0:
            conv_1_0 = Conv1D(params['num_filters'],
                              int(params['kernel_size']),
                              activation=params['activation'], padding='same',
                              kernel_initializer=params['initializer'])     
            conv_1_1 = Conv1D(params['num_filters'],
                              int(params['kernel_size']),
                              activation=params['activation'], padding='same',
                              kernel_initializer=params['initializer'])                        
            x = [conv_1_0(input_imgs[0]), conv_1_1(input_imgs[1])]
            if params['num_consecutive']==2:
                conv_2_0 = Conv1D(params['num_filters'],
                                int(params['kernel_size']),
                                activation=params['activation'],
                                padding='same',
                                kernel_initializer=params['initializer'])    
                conv_2_1 = Conv1D(params['num_filters'],
                                int(params['kernel_size']),
                                activation=params['activation'],
                                padding='same',
                                kernel_initializer=params['initializer'])                  
                x = [conv_2_0(x[0]), conv_2_1(x[1])]
        else:
            conv_3_0 = Conv1D(params['num_filters'],
                              int(params['kernel_size']),
                              activation=params['activation'], padding='same',
                              kernel_initializer=params['initializer'])           
            conv_3_1 = Conv1D(params['num_filters'],
                              int(params['kernel_size']),
                              activation=params['activation'], padding='same',
                              kernel_initializer=params['initializer'])             
            x = [conv_3_0(x[0]), conv_3_1(x[1])]

            if params['num_consecutive']==2:
                conv_4_0 = Conv1D(params['num_filters'],
                                int(params['kernel_size']),
                                activation=params['activation'],
                                padding='same',
                                kernel_initializer=params['initializer'])                 
                conv_4_1 = Conv1D(params['num_filters'],
                                int(params['kernel_size']),
                                activation=params['activation'],
                                padding='same',
                                kernel_initializer=params['initializer'])                   
                x = [conv_4_0(x[0]), conv_4_1(x[1])]
                  
            
        batch_norm = BatchNormalization()
        x = [batch_norm(a) for a in x]
            
        maxpool_1 = MaxPooling1D(2, padding='same')
        x = [maxpool_1(a) for a in x]
        
        dropout_1 = Dropout(params['dropout'])
        x = [dropout_1(a) for a in x]

    x = [Flatten()(a) for a in x]
    
    dense_0 = Dense(int(params['latent_dim']/2),
                    activation=params['activation'])
    dense_1 = Dense(int(params['latent_dim']/2),
                    activation=params['activation'])    
    x = [dense_0(x[0]), dense_1(x[1])]
    
    encoded = concatenate(x)
    encoder = Model(inputs=input_imgs, outputs=encoded)
    return encoder

def decoder_split(x, bottleneck, params):
    from keras.layers import Dense, Reshape, Conv1D, UpSampling1D, Dropout
    from keras.layers import Lambda, concatenate, BatchNormalization
    from keras import backend as K

    num_iter = int((params['num_conv_layers'])/2)
    
    input_dim_1 = np.shape(x[0])[1]
    dense_1 = Dense(int(input_dim_1*params['num_filters']/2**num_iter),
                    kernel_initializer=params['initializer'])
    
    input_dim_2 = np.shape(x[1])[1]
    # dense_2 = Dense(int(input_dim_2*params['num_filters']/2**num_iter),
    #                 kernel_initializer=params['initializer'])
    dense_2 = Dense(int(input_dim_2/(2**(num_iter)))*params['num_filters'],
                    kernel_initializer=params['initializer'])
    
    x = [dense_1(bottleneck), dense_2(bottleneck)]
    
    reshape_1 = Reshape((int(input_dim_1/(2**(num_iter))),
                         params['num_filters']))
    reshape_2 = Reshape((int(input_dim_2/(2**(num_iter))),
                         params['num_filters']))    
    x = [reshape_1(x[0]), reshape_2(x[1])]
    for i in range(num_iter):
        dropout_1 = Dropout(params['dropout'])
        x = [dropout_1(a) for a in x]
        
        upsampling_1 = UpSampling1D(2)
        x = [upsampling_1(a) for a in x]
        
        batch_norm = BatchNormalization()
        x = [batch_norm(a) for a in x]
        
        
        if i == num_iter-1:
            if params['num_consecutive'] == 2:
                conv_4 = Conv1D(params['num_filters'],
                                int(params['kernel_size']),
                                activation=params['activation'], padding='same',
                                kernel_initializer=params['initializer'])                   
                x = [conv_4(a) for a in x]
            conv_1 = Conv1D(1, int(params['kernel_size']),
                            activation=params['last_activation'],
                            padding='same',
                            kernel_initializer=params['initializer'])
            decoded = [conv_1(a) for a in x]
            # decoded = concatenate(x)
        else:
            if params['num_consecutive'] == 2:
                conv_2 = Conv1D(params['num_filters'], int(params['kernel_size']),
                activation=params['activation'], padding='same',
                kernel_initializer=params['initializer'])   
                x = [conv_2(a) for a in x]
            conv_3 = Conv1D(params['num_filters'], int(params['kernel_size']),
                            activation=params['activation'], padding='same',
                            kernel_initializer=params['initializer'])                   
            x = [conv_3(a) for a in x]
    return decoded

def decoder_split_diff_weights(x, bottleneck, params):
    from keras.layers import Dense, Reshape, Conv1D, UpSampling1D, Dropout
    from keras.layers import Lambda, concatenate, BatchNormalization
    from keras import backend as K

    num_iter = int((params['num_conv_layers'])/2)
    
    input_dim_1 = np.shape(x[0])[1]
    dense_1 = Dense(int(input_dim_1*params['num_filters']/2**num_iter),
                    kernel_initializer=params['initializer'])
    
    input_dim_2 = np.shape(x[1])[1]
    # dense_2 = Dense(int(input_dim_2*params['num_filters']/2**num_iter),
    #                 kernel_initializer=params['initializer'])
    dense_2 = Dense(int(input_dim_2/(2**(num_iter)))*params['num_filters'],
                    kernel_initializer=params['initializer'])
    
    x = [dense_1(bottleneck), dense_2(bottleneck)]
    
    reshape_1 = Reshape((int(input_dim_1/(2**(num_iter))),
                         params['num_filters']))
    reshape_2 = Reshape((int(input_dim_2/(2**(num_iter))),
                         params['num_filters']))    
    x = [reshape_1(x[0]), reshape_2(x[1])]
    for i in range(num_iter):
        dropout_1 = Dropout(params['dropout'])
        x = [dropout_1(a) for a in x]
        
        upsampling_1 = UpSampling1D(2)
        x = [upsampling_1(a) for a in x]
        
        batch_norm = BatchNormalization()
        x = [batch_norm(a) for a in x]
        
        
        if i == num_iter-1:
            if params['num_consecutive'] == 2:
                conv_4_0 = Conv1D(params['num_filters'],
                                int(params['kernel_size']),
                                activation=params['activation'], padding='same',
                                kernel_initializer=params['initializer'])  
                conv_4_1 = Conv1D(params['num_filters'],
                                int(params['kernel_size']),
                                activation=params['activation'], padding='same',
                                kernel_initializer=params['initializer'])                  
                x = [conv_4_0(x[0]), conv_4_1(x[1])]
            conv_1_0 = Conv1D(1, int(params['kernel_size']),
                            activation=params['last_activation'],
                            padding='same',
                            kernel_initializer=params['initializer'])
            conv_1_1 = Conv1D(1, int(params['kernel_size']),
                            activation=params['last_activation'],
                            padding='same',
                            kernel_initializer=params['initializer'])            
            decoded = [conv_1_0(x[0]), conv_1_1(x[1])]
        else:
            if params['num_consecutive'] == 2:
                conv_2_0 = Conv1D(params['num_filters'],
                                  int(params['kernel_size']),
                                  activation=params['activation'],
                                  padding='same',
                                  kernel_initializer=params['initializer'])   
                conv_2_1 = Conv1D(params['num_filters'],
                                  int(params['kernel_size']),
                                  activation=params['activation'],
                                  padding='same',
                                  kernel_initializer=params['initializer'])                  
                x = [conv_2_0(x[0]), conv_2_1(x[1])]
            conv_3_0 = Conv1D(params['num_filters'], int(params['kernel_size']),
                            activation=params['activation'], padding='same',
                            kernel_initializer=params['initializer'])      
            conv_3_1 = Conv1D(params['num_filters'], int(params['kernel_size']),
                            activation=params['activation'], padding='same',
                            kernel_initializer=params['initializer'])              
            x = [conv_3_0(x[0]), conv_3_1(x[1])]
    return decoded

def create_mlp(input_dim):
    '''Build multi-layer perceptron neural network model for numerical data
    (rms)'''
    from keras.models import Model
    from keras.layers import Dense, Input
    input_img = Input(shape = (input_dim,))
    x = Dense(8, activation='relu')(input_img)
    x = Dense(4, activation='relu')(x)
    x = Dense(1, activation='linear')(x)
    
    model = Model(input_img, x)
    return model
    
# :: preprocessing data :::::::::::::::::::::::::::::::::::::::::::::::::::::::

def split_data_features(flux, features, time, ticid, target_info, classes, p,
                        train_test_ratio = 0.9,
                        cutoff=16336, supervised=False, interpolate=False,
                        resize_arr=False, truncate=False):

    # >> truncate (must be a multiple of 2**num_conv_layers)
    if truncate:
        new_length = int(np.shape(flux)[1] / \
                     (2**(np.max(p['num_conv_layers'])/2)))*\
                     int((2**(np.max(p['num_conv_layers'])/2)))
        flux=np.delete(flux,np.arange(new_length,np.shape(flux)[1]),1)
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
        y_test = np.array(y_test)
        x_train = np.copy(features[train_inds])
        x_test = np.copy(features[test_inds])
        flux_train = np.copy(flux[train_inds])
        flux_test = np.copy(flux[test_inds])
        ticid_train = np.copy(ticid[train_inds])
        ticid_test = np.copy(ticid[test_inds])
        target_info_train = np.copy(target_info[train_inds])
        target_info_test = np.copy(target_info[test_inds])
    else:
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

def split_data(flux, ticid, target_info, time, p,
               train_test_ratio = 0.9,
               cutoff=16336,
               supervised=False, classes=False, interpolate=False,
               resize_arr=False, truncate=True):
    '''need to update, might not work'''
        
    if truncate:
        # >> dim reduced each iteration
        reduction_factor = np.max(p['pool_size'])* np.max(p['strides'])**np.max(p['num_consecutive'] )
        # reduction_factor = np.max(p['pool_size'])* np.max(p['strides'])
        
        num_iter = np.max(p['num_conv_layers'])/2
        tot_reduction_factor = reduction_factor**num_iter
        if p['fully_conv']:
            # >> 1 more conv layer
            tot_reduction_factor = tot_reduction_factor * np.max(p['strides'])
        new_length = int(np.shape(flux)[1] / \
                     tot_reduction_factor)*\
                     int(tot_reduction_factor)
        flux=np.delete(flux,np.arange(new_length,np.shape(flux)[1]),1)
        # flux_plot=np.delete(flux_plot, np.arange(new_length,np.shape(flux_plot)[1]),1)
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
        x_train = flux[:split_ind]
        x_test = flux[split_ind:]
        ticid_train = ticid[:split_ind]
        ticid_test = ticid[split_ind:]
        target_info_train = target_info[:split_ind]
        target_info_test = target_info[split_ind:]    
        y_test, y_train = [None, None]
        # flux_train = flux_plot[:split_ind]
        # flux_test = flux_plot[split_ind:]
        
    if resize_arr:
        x_train =  np.resize(x_train, (np.shape(x_train)[0],
                                       np.shape(x_train)[1], 1))
        x_test =  np.resize(x_test, (np.shape(x_test)[0],
                                       np.shape(x_test)[1], 1))
    return x_train, x_test, y_train, y_test, ticid_train, ticid_test, \
        target_info_train, target_info_test, time
    

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

def get_bottleneck(model, x_test, p, DAE=True, save=False, ticid=None,
                   out=None):
    if p['fully_conv']:
        inds = np.nonzero(['conv1d' in x.name for x in model.layers])[0]
        bottleneck_ind = inds[-1]        
    else:
        inds = np.nonzero(['dense' in x.name for x in model.layers])[0]
        
        # >> bottleneck layer is the first Dense layer
        if DAE:
            bottleneck_ind = inds[int(len(inds)/2)-1]
        else:
            bottleneck_ind = inds[0]
                        
    activation_model = Model(inputs=model.input,
                             outputs=model.layers[bottleneck_ind].output)
    bottleneck = activation_model.predict(x_test)    
    
    if p['fully_conv']:
        bottleneck = np.squeeze(bottleneck, axis=-1)
    
    bottleneck = df.standardize(bottleneck, ax=0)
    
    if save:
        hdr = fits.Header()
        hdu = fits.PrimaryHDU(bottleneck, header=hdr)
        hdu.writeto(out)    
        fits.append(out, ticid)          
    
    return bottleneck

# :: mock data ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

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
    flux = df.standardize(flux)
    
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

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
# def get_target_list(sector_num, output_dir='./'):
#     '''Get TICID from sector_num (given as int)'''
#     from astroquery.mast import Observations
#     from astroquery.mast import Tesscut
#     obs_table = Observations.query_criteria(obs_collection='TESS',
#                                             dataproduct_type='TIMESERIES',
#                                             sequence_number=sector_num)
    
#     print(obs_table)
#     target_list = np.copy(obs_table['target_name'])
    
#     cam_list = []
#     ccd_list = []
#     for target in target_list:
#         obj_name = 'TIC ' + target
#         try:
#             obj_table = Tesscut.get_sectors(obj_name)
#             ind = np.nonzero(obj_table['sector']==sector_num)
#             cam_list.append(obj_table['camera'][ind][0])
#             ccd_list.append(obj_table['ccd'][ind][0])
            
#             with open(output_dir+'tess-s00'+str(sector_num)+'.txt', 'a') as f:
#                 f.write(obj_name + ' {} {} {}\n'.format(obj_table['sector'][ind][0],
#                                                         obj_table['camera'][ind][0],
#                                                         obj_table['ccd'][ind][0]))
#         except:
#             print('failed! '+target)
#             with open(output_dir+'tess-s00'+str(sector_num)+'skip.txt', 'a') as f:
#                 f.write(obj_name+'\n')
                        
#     # >> also save .txt files for each camera and ccd
#     cam_list = np.array(cam_list)
#     ccd_list = np.array(ccd_list)
#     print(np.unique(cam_list))
#     print(np.unique(ccd_list))
#     for cam in range(4):
#         for ccd in range(4):
#             inds = np.nonzero( (cam_list==cam) * (cam_list==ccd) )[0]
#             with open(output_dir+'tess-s00'+str(sector_num)+'-'+str(cam)+'-'+\
#                       str(ccd)+'.txt', 'a') as f:
#                 for i in inds:
#                     f.write(target_list[i]+'\n')
        
#     return target_list
    
        
# # :: pull files with astroquery :::::::::::::::::::::::::::::::::::::::::::::::
# # adapted from pipeline.py

# def get_lc(ticid, out='./', DEBUG_INTERP=False, download_fits=True,
#            prefix=''):
#     '''input a ticid, returns light curve'''
#     from astroquery.mast import Observations
#     from astropy.io import fits
#     import fnmatch
    
#     # >> download fits file
#     targ = 'TIC ' + str(int(ticid))
    
#     if download_fits:
#         try: 
#             obs_table = Observations.query_object(targ, radius=".02 deg")
            
#             # >> find all data products for ticid
#             data_products_by_obs = Observations.get_product_list(obs_table[0:2])
            
#             filter_products = \
#                 Observations.filter_products(data_products_by_obs,
#                                              dataproduct_type = 'timeseries',
#                                              description = 'Light curves',
#                                              extension='fits')
                
#             # >> download fits file
#             manifest = \
#                 Observations.download_products(filter_products,
#                                                download_dir = out)
#         except (ConnectionError, OSError, TimeoutError):
#             print(targ + "could not be accessed due to an error")
        
#     # >> find fits file
#     fnames_all = os.listdir(out)
#     fname = fnmatch.filter(fnames_all, '*'+str(int(ticid))+'*fits*')[0]
    
#     # >> read fits file
#     f = fits.open(out+fname)
#     time = f[1].data['TIME']
#     flux = f[1].data['PDCSAP_FLUX']
#     return time, flux
    

# def get_fits_files(mypath, target_list):
#     '''target_list from tess.txt generated in get_target_list'''
#     from astroquery.mast import Observations
#     for ticid in target_list:
#         targ = 'TIC ' + str(int(ticid))
#         try: 
#             obs_table = Observations.query_object(targ, radius=".02 deg")
#             data_products_by_obs = Observations.get_product_list(obs_table[0:2])
            
#             filter_products = \
#                 Observations.filter_products(data_products_by_obs,
#                                              dataproduct_type = 'timeseries',
#                                              description = 'Light curves',
#                                              extension='fits')
#             manifest = \
#                 Observations.download_products(filter_products,
#                                                download_dir = mypath)
#         except (ConnectionError, OSError, TimeoutError):
#             print(targ + "could not be accessed due to an error")  

# def get_bottleneck_from_activations(model, activations, p, input_features=False, 
#                    features=False, input_rms=False, rms=False):
#     '''[deprecated 070320]
#     Get bottleneck layer, with shape (num light curves, latent dimension)
#     Parameters:
#         * model : Keras Model()
#         * activations : from get_activations()
#         * p : parameter set, with p['latent_dim'] = dimension of latent space
#         * input_features : bool
#         * features : array of features to concatenate with bottleneck, must be
#                      given if input_features=True
#         * rms : list of RMS must be given if input_rms=True
#     '''

#     # >> first find all Dense layers
#     inds = np.nonzero(['dense' in x.name for x in model.layers])[0]
    
#     # >> now check which Dense layers has number of units = latent_dim
#     for ind in inds:
#         ind = ind - 1 # >> len(activations) = len(model.layers) - 1, since
#                       #    activations doesn't incluthe Input layer
#         num_units = np.shape(activations[ind])[1]
#         if num_units == p['latent_dim']:
#             bottleneck_ind = ind
    
#     bottleneck = activations[bottleneck_ind]
    
#     if input_features: # >> concatenate features to bottleneck
#         bottleneck = np.concatenate([bottleneck, input_features], axis=1)
#     if input_rms:
#         bottleneck = np.concatenate([bottleneck,
#                                       np.reshape(rms, (np.shape(rms)[0],1))],
#                                     axis=1)
        
#     return bottleneck     


# def encoder_external_features(x, params):
#     from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten
#     from keras.layers import Dense, concatenate, BatchNormalization
#     from keras.models import Model

#     input_imgs = [Input(shape=(np.shape(x[0])[1], 1)),
#                   Input(shape=(np.shape(x[1])[0]))]
#     num_iter = int((params['num_conv_layers'])/2)    

#     for i in range(num_iter):

#         if i == 0:
#             x[0] = Conv1D(params['num_filters'],
#                               int(params['kernel_size']),
#                               activation=params['activation'], padding='same',
#                               kernel_initializer=params['initializer'])(x[0])
#             if params['num_consecutive']==2:
#                 x[0] = Conv1D(params['num_filters'],
#                                 int(params['kernel_size']),
#                                 activation=params['activation'],
#                                 padding='same',
#                                 kernel_initializer=params['initializer'])(x[0])
#         else:
#             x[0] = Conv1D(params['num_filters'],
#                               int(params['kernel_size']),
#                               activation=params['activation'], padding='same',
#                               kernel_initializer=params['initializer'])(x[0])

#             if params['num_consecutive']==2:
#                 x[0] = Conv1D(params['num_filters'],
#                                 int(params['kernel_size']),
#                                 activation=params['activation'],
#                                 padding='same',
#                                 kernel_initializer=params['initializer'])(x[0])
                  
            
#         x[0] = BatchNormalization()(x[0])
            
#         maxpool_1 = MaxPooling1D(2, padding='same')
#         x = [maxpool_1(a) for a in x]
        
#         dropout_1 = Dropout(params['dropout'])
#         x = [dropout_1(a) for a in x]

#     x = [Flatten()(a) for a in x]
    
#     dense_0 = Dense(int(params['latent_dim']/2),
#                     activation=params['activation'])
#     dense_1 = Dense(int(params['latent_dim']/2),
#                     activation=params['activation'])    
#     x = [dense_0(x[0]), dense_1(x[1])]
    
#     encoded = concatenate(x)
#     encoder = Model(inputs=input_imgs, outputs=encoded)
#     return encoder     
    

# def run_VAEGAN(x_train, p, kernel=5, columns=18688,
#                rows=1, channel=3):
#     from keras import optimizers
#     from VAEGAN import decgen
    
#     # noise = np.random.normal(0, 1, (p['batch_size'], 256))
#     # optimizers
#     if p['optimizer'] == 'adam':
#         opt = optimizers.adam(lr = p['lr'],  decay=p['lr']/p['epochs'])    
#     # SGDop = SGD(lr=0.0003)
#     # ADAMop = Adam(lr=0.0002)
    
#     G = decgen(p['kernel_size'], p['num_filters'], rows, columns, channel)
#     G.compile(optimizer=p['optimizer'], loss='mse')
#     G.summary()
    
#     # >> training model
#     history = G.fit()
    
#     # G.load_weights('generator.h5')
#     # image = G.predict(noise)
#     # image = np.uint8(image * 127.5 +127.5)
#     # plt.imshow(image[0]), plt.show()    
            
        # if i != 0 and params['encoder_skip'] and not params['full_feed_forward_highway']:
        #     f_x = feature_maps[-1 * params['num_consecutive'][i] - 1]
        #     # >> projection layer
        #     f_x = Conv1D(params['num_filters'][i-1], int(params['kernel_size']),
        #             activation=params['activation'], padding='same',
        #             kernel_initializer=params['initializer'],
        #             strides=params['strides'],
        #             kernel_regularizer=params['kernel_regularizer'],
        #             bias_regularizer=params['bias_regularizer'],
        #             activity_regularizer=params['activity_regularizer'])(f_x)
        #     # >> add feature maps
        #     x = Add()([x, f_x])
        #     x = Activation(params['activation'])(x)    
        # if i != 0 and params['full_feed_forward_highway']:
        #     for k in range(i):
        #         f_x = feature_maps[(-1 - k) * params['num_consecutive'][i] - 1]
        #         # >> projection layer
        #         f_x = Conv1D(params['num_filters'][i-1], int(params['kernel_size']),
        #                 activation=params['activation'], padding='same',
        #                 kernel_initializer=params['initializer'],
        #                 strides= params['strides']**(1+k),
        #                 kernel_regularizer=params['kernel_regularizer'],
        #                 bias_regularizer=params['bias_regularizer'],
        #                 activity_regularizer=params['activity_regularizer'])(f_x)
        #         # >> add feature maps
        #         x = Add()([x, f_x])                
        #     x = Activation(params['activation'])(x) 


            # if j == 0:
            #     stride = params['strides']
            # else:
            #     stride = 1

            # if params['share_pool_inds']:
            #     x, argmax = max_pool_layer(x, params)
            #     pool_masks.append(argmax)
            # else:

        # if params['pool_size'] != 1:
        #     if params['share_pool_inds']:
        #         argmax=pool_masks[-1*i-1]
        #         x = unpool_with_with_argmax(x, argmax, params)
        #     else:

        # if i != num_iter - 1:
        #     params['activity_regularizer'] = None
        # else: 
        #     params['activity_regularizer'] = 'l2'
