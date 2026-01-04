#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 0.1 Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os


class SEQUENCE_DATASET(Dataset):
    def __init__(self, path_to_file, data, train, temporal_window, normalize=True):
        self.temporal_window = temporal_window
        
        file_path = os.path.join(path_to_file, data)
        self.X = np.load(file_path)
        
        if self.X.shape[0] > self.X.shape[1]:
            self.X = self.X.T
            
        self.num_frames = self.X.shape[1]
        
        # Pre-normalize data
        self.X = self.X.astype(np.float32)

        if normalize:
            mean_path = os.path.join(path_to_file, 'seq_mean.npy')
            std_path = os.path.join(path_to_file, 'seq_std.npy')
            
            if train and not os.path.exists(mean_path):
                print("Compute mean and std for temporal dataset.")
                self.mean = np.mean(self.X)
                self.std = np.std(self.X)
                np.save(mean_path, self.mean)
                np.save(std_path, self.std)
            else:
                self.mean = np.load(mean_path)
                self.std = np.load(std_path)
            
            self.X = (self.X - self.mean) / self.std

        if train:
            print('Initialize train data. Frames %d' % self.num_frames)
        else:
            print('Initialize test data. Frames %d' % self.num_frames)
        
    def __len__(self):
        return self.num_frames - self.temporal_window

    def __getitem__(self, index):
        start = index
        end = start + self.temporal_window
        sequence = self.X[:, start:end]
        return torch.from_numpy(sequence)
    
    
    
    
    
