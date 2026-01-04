#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 0.1 Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r"D:\SilviaData\ScriptOnGithub\VAME")

from vame.util.auxiliary import *
from vame.util.csv_to_npy_original import csv_to_numpy
from vame.util.create_file_for_vame import create_file_for_vame
from vame.util.auxiliary import read_config