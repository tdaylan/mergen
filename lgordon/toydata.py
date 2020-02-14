# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 23:21:54 2020

@author: conta
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense



#constant things
sigma = 5
mu = 250
#set up toydata thing
toydata = []
x = np.arange(0,500,1)
#produce data and store it in toydata list
for i in range(0,500):
    flatdata = np.repeat(1,500) + (np.random.normal(mu,sigma,500)/250)
    toydata.append(flatdata)
    #
    y = ( 2.*np.pi*sigma**2. )**-.5 * np.exp( -.5 * (x-mu)**2. / sigma**2. ) * 2
    bumpdata = y + 1 + (np.random.normal(mu, sigma, 500)/250)
    toydata.append(bumpdata)

#plot two random datasets from within toydata to be sure it worked correctly.
plt.plot(toydata[356])
plt.plot(toydata[357])
