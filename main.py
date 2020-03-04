from __future__ import print_function

import pandas as pd
import numpy as np
import keras
import re
import copy
import random

import sys
sys.path.append('../')

from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import fashion_mnist
from keras.datasets import imdb
from keras.datasets import reuters
from keras.datasets import boston_housing

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.initializers import *

import HyperTube
from HyperTube import *
import HyperTube_BO
import HyperTube_GA
import HyperTube_PR
import Models_CNN 
from Models_CNN import *

# MNIST and CIFAR-10

# MNIST
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
dataset = "MNIST"

# CIFAR10
# (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
# dataset = "CIFAR10"

# Fashion FMNIST
# (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
# dataset = "FMNIST"

# Fashion CIFAR100
# (X_train, Y_train), (X_test, Y_test) = cifar100.load_data()

num_classes = 10
L_train = X_train.shape[0]
X_train_0, Y_train_0, X_test_0, Y_test_0 = X_train, Y_train, X_test, Y_test

HP_0 = {"model_code":"CNN", "seed":10, "num_classes": 10, "epochs":50, "batch_size": 100, "val_frac": 0.50,  
      "reg1":0, "reg2":0.02, "loss": "categorical_crossentropy", "lr":0.001, "rho":0.9, "momentum": 0.9, "beta_1":0.9, 
      "beta_2":0.999, "decay":0.0, "schedule_decay":0.004, "model_type":"LeNet_5", "optimizer":"Adam", "metrics":"categorical_accuracy",
        "k_size_1":5, "k_size_2":5, "p_size":2, "drop1":0.25, "drop2":0.5, "early_stopping": False, "inherit": False}

HHP_0 = {"level": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], "n_init": 10, "update_method":None, "update_rate":0.5} # n=5000 # m=8

HPga={}
HPga["drop1"] = np.linspace(0,0.6,128)
HPga["drop2"] = np.linspace(0,0.6,128)
HPga["lr"] = np.exp(np.linspace(np.log(0.0001), np.log(0.01), 256))
HPcv_0 = {'drop1':np.linspace(0,0.6,128), 'drop2':np.linspace(0,0.6,128), "lr":np.exp(np.linspace(np.log(0.0001), np.log(0.01), 256))} 
val_frac = HP_0['val_frac']

for i in range(10):
    
    ind = np.random.choice(L_train,10000,replace = False)
    ind_train = ind.tolist()

    X_train = X_train_0[ind_train,:,:]
    Y_train = Y_train_0[ind_train]

    L_test = X_test.shape[0]
    # ind1 = np.random.choice(L_test,1000,replace = False)
    ind1 = np.random.choice(L_test,5000,replace = False)
    ind_test = ind1.tolist()

    X_test = X_test_0[ind_test,:,:]
    Y_test = Y_test_0[ind_test]
    
    img_rows = X_train.shape[1]
    img_cols = X_train.shape[2]

    try:
        channels = X_train.shape[3]
    except:
        channels = 1

    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], channels, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channels)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    Y_test = keras.utils.to_categorical(Y_test, num_classes)
    
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    HP_0["input_shape"] = input_shape
    HP_0["seed"] = random.randint(0,1000)
    HP_0["epochs"] = 50
    HHP_0["n_init"] = 100
    
    train_X, train_Y, val_X, val_Y = val_split_CNN(X_train, Y_train, frac = 0.5)
    test_X, test_Y = X_test, Y_test
    
    print('train_X shape:', train_X.shape)
    print('train_Y shape:', train_Y.shape)
    print('val_X shape:', val_X.shape)
    print('val_Y shape:', val_Y.shape)
    print('test_X shape:', test_X.shape)
    print('test_Y shape:', test_Y.shape)
    
    HP_random = model_sample_init_hp(HP_0, HHP_0, HPcv_0)
    
    HP = copy.deepcopy(HP_0)
    HHP = copy.deepcopy(HHP_0)
    HPcv = copy.deepcopy(HPcv_0)
    HP["epochs"] = 50
    HHP["n_init"] = 100
    HHP["setting"] = "D"
    fn2 = 'CNN_HP_HyperTube_FMNIST_D.csv'
    model_list, eva_val_min_list, eva_test_min_list, meva_val_list, meva_test_list =  model_online(train_X, train_Y, val_X, val_Y, test_X, test_Y, HP, HHP, HPcv, fn2, other_utils=None)  
    
    HP = copy.deepcopy(HP_0)
    HHP = copy.deepcopy(HHP_0)
    HPcv = copy.deepcopy(HPcv_0)
    HP["epochs"] = 50
    HHP["n_init"] = 100
    HHP["setting"] = "EDR"
    fn2 = 'CNN_HP_HyperTube_FMNIST_EDR.csv'
    model_list, eva_val_min_list, eva_test_min_list, meva_val_list, meva_test_list, hp_list_min, eva_val_final, eva_test_final =  model_online(train_X, train_Y, val_X, val_Y, test_X, test_Y, HP, HHP, HPcv, fn2, other_utils=None)  
    
    HP = copy.deepcopy(HP_0)
    HHP = copy.deepcopy(HHP_0)
    HPcv = copy.deepcopy(HPcv_0)
    HP["epochs"] = 50
    HHP["n_init"] = 100
    HHP["setting"] = "EDR_1"
    fn2 = 'CNN_HP_HyperTube_FMNIST_EDR_1.csv'
    model_list, eva_val_min_list, eva_test_min_list, meva_val_list, meva_test_list =  model_online(train_X, train_Y, val_X, val_Y, test_X, test_Y, HP, HHP, HPcv, fn2, other_utils=None)  

    HP = copy.deepcopy(HP_0)
    HHP = copy.deepcopy(HHP_0)
    HPcv = copy.deepcopy(HPcv_0)
    HP["epochs"] = 200
    HHP["n_init"] = 25
    HHP["setting"] = "P"
    fn2 = 'CNN_HP_HyperTube_CIFAR10_P_2.csv'
    model_list, eva, hp_list_min, eva_min, meva = model_online(train_X, train_Y, val_X, val_Y, HP, HHP, HPcv, fn2, other_utils=None)  
    
    HP = copy.deepcopy(HP_0)
    HHP = copy.deepcopy(HHP_0)
    HPcv = copy.deepcopy(HPcv_0)
    HP["epochs"] = 250
    HHP["n_init"] = 20
    HHP["setting"] = "P"
    fn2 = 'CNN_HP_HyperTube_CIFAR10_P_3.csv'
    model_list, eva, hp_list_min, eva_min, meva = model_online(train_X, train_Y, val_X, val_Y, HP, HHP, HPcv, fn2, other_utils=None)
    
for i in range(10):
    HP = copy.deepcopy(HP_0)
    HHP = copy.deepcopy(HHP_0)
    HPcv = copy.deepcopy(HPcv_0)
    HHP["update_method"] = "GA"
    HHP["adjusted"] = False
    HHP["update_rate"] = 0.5
    HHP["HPga"] = HPga
    HHP["mutation_prob"] = 0.2
    HHP["prob_cross"] = 0.2
    HHP["cross_mutation_prob"] = 0
    HHP["setting"] = "D"
    fn2 = 'CNN_HP_HyperTube_'+dataset+ '_' + HHP["setting"] + '_' + HHP["update_method"] + '_1.csv'
    model_list, eva_val_min_list, eva_test_min_list, meva_val_list, meva_test_list, min_history_full, ave_history_full =  model_online_1(train_X, train_Y, val_X, val_Y, test_X, test_Y, HP, HHP, HPcv, HP_random, fn2, other_utils=None)
    
    HP = copy.deepcopy(HP_0)
    HHP = copy.deepcopy(HHP_0)
    HPcv = copy.deepcopy(HPcv_0)
    HHP["update_method"] = None
    HHP["adjusted"] = False
    HHP["setting"] = "D"
    fn2 = 'CNN_HP_HyperTube_'+dataset+ '_' + HHP["setting"] + '_1.csv'
    model_list, eva_val_min_list, eva_test_min_list, meva_val_list, meva_test_list, min_history_full, ave_history_full =  model_online_1(train_X, train_Y, val_X, val_Y, test_X, test_Y, HP, HHP, HPcv, HP_random, fn2, other_utils=None)  
    








