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

sys.path.insert(0, r"D:\SilviaData\ScriptOnGithub\Detection_Behaviour_Motifs")

from vame.initialize_project import init_new_project
from vame.model import create_trainset
from vame.model import train_model
from vame.model import evaluate_model
from vame.analysis import pose_segmentation
from vame.analysis import motif_videos
from vame.analysis import community
from vame.analysis import community_videos
from vame.analysis import visualization
from vame.analysis import generative_model
from vame.analysis import gif
from vame.util.csv_to_npy import csv_to_numpy
from vame.util.align_egocentrical import egocentric_alignment
from vame.util import auxiliary
from vame.util.auxiliary import update_config
from vame.util.create_file_for_vame import create_file_for_vame
from vame.util.auxiliary import read_config
from vame.analysis import cluster_latent_space_silvia
from vame.analysis import umap_visualization_silvia
from vame.analysis import community_analysis_silvia
from vame.util.create_file_for_vame import create_file_for_vame
#from vame.etoghram import etoghram
from vame.util.auxiliary import read_config
from vame.analysis.plot_losses import plot_losses
from vame.Gui.plot_feature_reconstruction import plot_feature_reconstruction
from vame.Auxiliary_functions.add_user_motifs import add_user_motifs
from vame.Gui.motif_marker_app import MotifMarkerApp

