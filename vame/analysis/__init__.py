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

from vame.analysis.pose_segmentation import pose_segmentation
from vame.analysis.videowriter import motif_videos, community_videos
from vame.analysis.community_analysis import community
from vame.analysis.umap_visualization import visualization
from vame.analysis.generative_functions import generative_model
from vame.analysis.gif_creator import gif
from vame.analysis.cluster_latent_space_silvia import cluster_latent_space_silvia
from vame.analysis.umap_visualization_silvia import  umap_visualization_silvia
from vame.analysis.community_analysis_silvia import  community_analysis_silvia
from vame.analysis.tree_hierarchy import graph_to_tree, draw_tree, traverse_tree_cutline



