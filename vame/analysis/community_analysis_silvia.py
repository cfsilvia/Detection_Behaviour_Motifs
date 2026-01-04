import os
import umap
import scipy
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from vame.util.auxiliary import read_config
from vame.analysis.tree_hierarchy import graph_to_tree, draw_tree, traverse_tree_cutline


class community_analysis_silvia:
    def __init__(self,config):
       #read config file
       config_file = Path(config).resolve()
       cfg = read_config(config_file)
       #read all the necesary parameters
       self.model_name = cfg['model_name']
       self.n_cluster = cfg['n_cluster']
       self.parameterization = cfg['parameterization']
       self.project_path = cfg['project_path']
       self.file_exp = cfg['video_sets']
       file_1 = "latent_vector_" + self.file_exp[0] + ".npy"
       self.file_latent_vector = os.path.join(self.project_path,"results", self.file_exp[0], self.model_name,self.parameterization+'-'+str(self.n_cluster), file_1 )
       self.save_data = os.path.join(cfg['project_path'],"results",self.file_exp[0],self.model_name,self.parameterization+'-'+str(self.n_cluster),"")
       self.file_labels = os.path.join(self.save_data,str(self.n_cluster)+'_km_label_'+ self.file_exp[0] + '.npy')


    def  __call__(self, cut_tree = None):
       self.labels = np.load(self.file_labels) #only for one file
       transition_matrices = self.compute_transition_matrices()
       communities_all, trees = self.create_community_bag( transition_matrices, cut_tree)
       community_labels_all = self.get_community_labels( communities_all) 
       a=1
    ##############################################
    '''
    '''
    def compute_transition_matrices(self):
       adj, trans, mat = self. get_adjacency_matrix()
       transition_matrices = trans
       return transition_matrices
    '''
    '''
    def get_adjacency_matrix(self):
       temp_matrix = np.zeros((self.n_cluster,self.n_cluster), dtype=np.float64)
       adjacency_matrix = np.zeros((self.n_cluster,self.n_cluster), dtype=np.float64)
       cntMat = np.zeros((self.n_cluster))
       self.labels = np.squeeze(self.labels, axis = 0)
       steps = len(self.labels)

       for i in range(self.n_cluster):
          for k in range(steps-1): #go through the frames with labels
             idx = self.labels[k]
             if idx == i:
                idx2 = self.labels[k+1] #consecutive label -one if it jumps to this cluster
                if idx == idx2: #if it is the same cluster
                   continue
                else:
                   cntMat[idx2] =cntMat[idx2] + 1 
          temp_matrix[i] = cntMat
          cntMat = np.zeros((self.n_cluster))

       for k in range(steps-1):
         idx = self.labels[k]
         idx2 = self.labels[k+1]
         if idx == idx2:
            continue
         adjacency_matrix[idx,idx2] = 1 # one if the jump is from one to the other
         adjacency_matrix[idx2,idx] = 1  

       transition_matrix = self. get_transition_matrix(temp_matrix)

       return adjacency_matrix, transition_matrix, temp_matrix
    
    '''
    '''
    def get_transition_matrix(self, adjacency_matrix, threshold = 0.0):
       row_sum=adjacency_matrix.sum(axis=1)
       transition_matrix = adjacency_matrix/row_sum[:,np.newaxis]
       transition_matrix[transition_matrix <= threshold] = 0
       if np.any(np.isnan(transition_matrix)):
            transition_matrix=np.nan_to_num(transition_matrix)
            
       return transition_matrix
    

   
    '''
    '''
    def create_community_bag(self, transition_matrices, cut_tree):
       trees = []
       communities_all = []
       _, usage = np.unique(self.labels, return_counts=True)
       T = graph_to_tree(usage, transition_matrices, self.n_cluster, merge_sel=1) 
       trees.append(T)
       if cut_tree != None:
            community_bag =  traverse_tree_cutline(T,cutline=cut_tree)
            communities_all.append(community_bag)
            draw_tree(T)
       else:
            draw_tree(T)
            plt.pause(0.5)
            flag_1 = 'no'
            while flag_1 == 'no':
                cutline = int(input("Where do you want to cut the Tree? 0/1/2/3/..."))
                community_bag =  traverse_tree_cutline(T,cutline=cutline)
                print(community_bag)
                flag_2 = input('\nAre all motifs in the list? (yes/no/restart)')
                if flag_2 == 'no':
                    while flag_2 == 'no':
                        add = input('Extend list or add in the end? (ext/end)')
                        if add == "ext":
                            motif_idx = int(input('Which motif number? '))
                            list_idx = int(input('At which position in the list? (pythonic indexing starts at 0) '))
                            community_bag[list_idx].append(motif_idx)
                        if add == "end":
                            motif_idx = int(input('Which motif number? '))
                            community_bag.append([motif_idx])
                        print(community_bag)
                        flag_2 = input('\nAre all motifs in the list? (yes/no/restart)')
                if flag_2 == "restart":
                    continue
                if flag_2 == 'yes':
                    communities_all.append(community_bag)
                    flag_1 = 'yes'
            
       return communities_all, trees



    '''
    '''
    def get_community_labels(self, communities_all):
    # transform parameterized latent vector into communities
     community_labels_all = []
     num_comm = len(communities_all)  
     community_labels = np.zeros_like(self.labels)
     for i in range(num_comm):
            clust = np.array(communities_all[i])
            for j in range(len(clust)):
                find_clust = np.where(self.labels == clust[j])[0]
                community_labels[find_clust] = i
        
            community_labels = np.int64(scipy.signal.medfilt(community_labels, 7))  
            community_labels_all.append(community_labels)

     return community_labels_all