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
from vame.util.auxiliary import read_config

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0] 

def interp_1d(v):
    nans, idx = nan_helper(v)
    if np.all(nans):
        return v  # nothing to interpolate
    v[nans] = np.interp(idx(nans), idx(~nans), v[~nans])
    return v

def interpol(arr):
    """
    Interpolates NaNs in the x and y columns of a (T, 3) pose array.
    Only column 0 and 1 are interpolated (x and y).
    """
    arr = arr.copy()  # safer
    arr[:, 0] = interp_1d(arr[:, 0])
    arr[:, 1] = interp_1d(arr[:, 1])
    return arr

#######################################


def csv_to_numpy(config):
    """
    This is a function to convert your pose-estimation.csv file to a numpy array.

    Note that this code is only useful for data which is a priori egocentric, i.e. head-fixed
    or otherwise restrained animals.

    example use:
    vame.csv_to_npy('pathto/your/config/yaml', 'path/toYourFolderwithCSV/')
    """
    config_file = Path(config).resolve()
    cfg = read_config(config_file)

    path_to_file = cfg['project_path']
    filename = cfg['video_sets']
    confidence = cfg['pose_confidence']
    

    for file in filename:
        print(file)
        # Read in your .csv file, skip the first two rows and create a numpy array
        data = pd.read_csv(os.path.join(path_to_file,"videos","pose_estimation",file+'.csv'), skiprows = 1, header=None)
        data_mat = pd.DataFrame.to_numpy(data)
        data_mat = data_mat[:,1:]

        pose_list = []
        n_bodyparts = data_mat.shape[1] // 3
        
        # split into [x, y, p] per bodypart
        for i in range(n_bodyparts):
            pose_list.append(data_mat[:, i*3:(i+1)*3])
         
        # find low confidence and set them to NaN
        for p in pose_list:
          low_conf = p[:, 2] <= confidence
          p[low_conf, 0:2] = np.nan
      
        # interpolate NaNs (in-place by reassigning)
        for k in range(len(pose_list)):
            pose_list[k] = interpol(pose_list[k])
            
        # concatenate and drop confidence columns
        positions = np.concatenate(pose_list, axis=1)        # shape (T, 3*n_bodyparts)
        final_positions = positions.reshape(-1, n_bodyparts, 3)[:, :, :2].reshape(-1, 2*n_bodyparts)

        # save the final_positions array with np.save()
        np.save(os.path.join(path_to_file,'data',file,file+"-PE-seq.npy"), final_positions.T)
        print("conversion from DeepLabCut csv to numpy complete...") #each row is a feature without the score(21 features and ~23000 data)

    print("Your data is now in right format and you can call vame.create_trainset()")
