#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 1.0-alpha Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.signal
from scipy.stats import iqr
import matplotlib.pyplot as plt

from vame.util.auxiliary import read_config


'''
create test and train data - normalize all the y and z separately
'''

def traindata_fixed(cfg, files, testfraction, num_features, savgol_filter, check_parameter):
    X_train = []
    pos = []
    pos_temp = 0
    pos.append(0)
    
   
        
    for file in files:
        print("z- scoring with x and y separately %s" %file)
        path_to_file = os.path.join(cfg['project_path'],"data", file, file+'-PE-seq.npy')
        data = np.load(path_to_file)
        #normalize the data first time x and y axis separately
        # X_mean_x = np.mean((data.T)[:,::2], axis = None)
        # X_mean_y = np.mean((data.T)[:,1::2], axis = None)

        # X_std_x= np.std((data.T)[:,::2], axis = None)
        # X_std_y = np.std((data.T)[:,1::2], axis = None)
       

        X_z = data.T.copy()
       
        
        # X_z[:,::2] = (X_z[:,::2] - X_mean_x) / X_std_x #normalize the data for each feature
        # X_z[:,1::2] = (X_z[:,1::2] - X_mean_y) / X_std_y

        # #normalize second time all together
        # X_mean = np.mean((X_z), axis = None)
        # X_std= np.std((X_z), axis = None)
        # X_z = (X_z - X_mean) / X_std

        

        
        X_len = len(data.T)
        pos_temp += X_len
        pos.append(pos_temp)
        X_train.append(X_z)
    
    X = np.concatenate(X_train, axis=0).T

    X_med = X
        
    num_frames = len(X_med.T)
    test = int(num_frames*testfraction)
    
    z_test =X_med[:,:test]
    z_train = X_med[:,test:]
    

    #save numpy arrays the the test/train info:
    np.save(os.path.join(cfg['project_path'],"data", "train",'train_seq.npy'), z_train)
    np.save(os.path.join(cfg['project_path'],"data", "train", 'test_seq.npy'), z_test)

    #save for each file separately   
    for i, file in enumerate(files):
            np.save(os.path.join(cfg['project_path'],"data", file, file+'-PE-seq-clean.npy'), X_med[:,pos[i]:pos[i+1]])
        
    print('Lenght of train data: %d' %len(z_train.T))
    print('Lenght of test data: %d' %len(z_test.T))


def create_trainset(config, check_parameter=False):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    legacy = cfg['legacy'] #it is false
    fixed = cfg['egocentric_data']
    
    if not os.path.exists(os.path.join(cfg['project_path'],'data','train',"")):
        os.mkdir(os.path.join(cfg['project_path'],'data','train',""))

    files = []
    if cfg['all_data'] == 'No':
        for file in cfg['video_sets']:
            use_file = input("Do you want to train on " + file + "? yes/no: ")
            if use_file == 'yes':
                files.append(file)
            if use_file == 'no':
                continue
    else:
        for file in cfg['video_sets']:
            files.append(file)

    print("Creating training dataset...")
    
    print("Creating trainset from the vame.csv_to_numpy() output ")
    traindata_fixed(cfg, files, cfg['test_fraction'], cfg['num_features'], cfg['savgol_filter'], check_parameter)
    
    