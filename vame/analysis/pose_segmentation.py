#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Silvia adaptation get latent space and segment it 
"""

import os
import tqdm
import torch
import pickle
import numpy as np
from pathlib import Path


from hmmlearn import hmm
from sklearn.cluster import KMeans

from vame.util.auxiliary import read_config
from vame.model.rnn_model import RNN_VAE

import matplotlib.pyplot as plt


###########################################
class  pose_segmentation:
    def __init__(self,config):
        config_file = Path(config).resolve()
        self.cfg = read_config(config_file)
        self.model_name = self.cfg['model_name']
        self.n_cluster = self.cfg['n_cluster']
        self.parameterization = self.cfg['parameterization']
        self.files = self.cfg['video_sets']
        
        print('Pose segmentation for VAME model: %s \n' %self.model_name)

    def __call__(self):
      # create folders if they didn't exist
        for folders in self.cfg['video_sets']:
            if not os.path.exists(os.path.join(self.cfg['project_path'],"results",folders,self.model_name,"")):
                os.mkdir(os.path.join(self.cfg['project_path'],"results",folders,self.model_name,""))
            
      ###
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            print("Using CUDA")
            print('GPU active:',torch.cuda.is_available())
            print('GPU used:',torch.cuda.get_device_name(0))
        else:
            print("CUDA is not working! Attempting to use the CPU...")
            torch.device("cpu")
        ###
        model = self. load_model()
        ###
        latent_vector_files = self.embedd_latent_vectors(model)
        #####
        #clusterization
        print("For all files the same parameterization of latent vectors is applied for %d cluster" %self.n_cluster)
        label, clust_center = self.same_parameterization(latent_vector_files)
        #reoder for further use and get information of label......
        labels, cluster_centers, motif_usages = self.reoder_from_cluster_model(latent_vector_files,label, clust_center)
        #now are plotting all the motif_usages and save plot in descending order
        self.plot_usage(motif_usages )
        #now is savings

        self.saving_latent_space(latent_vector_files, labels, motif_usages)
         
         

###########################################
    '''
    load a model
    '''
    def load_model(self):
         # load Model
        ZDIMS = self.cfg['zdims']
        FUTURE_DECODER = self.cfg['prediction_decoder']
        TEMPORAL_WINDOW = self.cfg['time_window']*2
        FUTURE_STEPS = self.cfg['prediction_steps']
        NUM_FEATURES = self.cfg['num_features']
        hidden_size_layer_1 = self.cfg['hidden_size_layer_1']
        hidden_size_layer_2 = self.cfg['hidden_size_layer_2']
        hidden_size_rec = self.cfg['hidden_size_rec']
        hidden_size_pred = self.cfg['hidden_size_pred']
        dropout_encoder = self.cfg['dropout_encoder']
        dropout_rec = self.cfg['dropout_rec']
        dropout_pred = self.cfg['dropout_pred']
        softplus = self.cfg['softplus']

        model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1, 
                                hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder, 
                                dropout_rec, dropout_pred, softplus).cuda()
        
        model.load_state_dict(torch.load(os.path.join(self.cfg['project_path'],'model','best_model',self.model_name+'_'+self.cfg['Project']+'.pkl')))
        model.eval()

        return model
    
    '''
    create the latent space
       '''
    def embedd_latent_vectors(self,model):
        project_path = self.cfg['project_path']
        temp_win = self.cfg['time_window']
        num_features = self.cfg['num_features']
        latent_vector_files = [] 

        for file in self.files:
            print('Embedding of latent vector for file %s' %file)
            #data has a shape(num features, num_Frames)
            data = np.load(os.path.join(project_path,'data',file,file+'-PE-seq-clean.npy'))
            latent_vector_list = []

            with torch.no_grad():
                 for i in tqdm.tqdm(range(data.shape[1] - temp_win)): #progress bar
                     data_sample_np = data[:,i:temp_win+i].T
                     data_sample_np = np.reshape(data_sample_np, (1, temp_win, num_features))
                     h_n = model.encoder(torch.from_numpy(data_sample_np).type('torch.FloatTensor').cuda())
                     mu, _, _ = model.lmbda(h_n)
                     latent_vector_list.append(mu.cpu().data.numpy())
            latent_vector = np.concatenate(latent_vector_list, axis=0)
            latent_vector_files.append(latent_vector) 
        return latent_vector_files        
    
    '''
    clusterization of  latent space
    save all the hidden markov model
    '''
    def same_parameterization(self, latent_vector_files):
        random_state = self.cfg['random_state_kmeans']
        n_init = self.cfg['n_init_kmeans']
        clust_center = np.array([])
        label = np.array([])
        
        latent_vector_cat = np.concatenate(latent_vector_files, axis=0)

        if self.parameterization == "kmeans":
            print("Using kmeans as parameterization!")
            kmeans = KMeans(init='k-means++', n_clusters = self.n_cluster, random_state=42, n_init=20).fit(latent_vector_cat)
            kmeans = KMeans(init='k-means++', n_clusters=self.n_cluster, random_state=random_state, n_init=n_init).fit(latent_vector_cat)
            clust_center = kmeans.cluster_centers_
            label = kmeans.predict(latent_vector_cat)
            label = kmeans.labels_
        elif self.parameterization == "hmm":
            print("Using a HMM as parameterization!")
            hmm_model = hmm.GaussianHMM(n_components = self.n_cluster, covariance_type="full", n_iter=100)
            hmm_model.fit(latent_vector_cat)
            label = hmm_model.predict(latent_vector_cat)
            save_data = os.path.join(self.cfg['project_path'], "results", "")
            with open(save_data+"hmm_trained.pkl", "wb") as file: pickle.dump(hmm_model, file)
  
        return label, clust_center
    
    '''
    arrange labels to be used later
    '''
    def reoder_from_cluster_model(self, latent_vector_files, label, clust_center):
        labels = []
        cluster_centers = []
        motif_usages = []

        idx = 0
        for i, file in enumerate(self.files):
              file_len = latent_vector_files[i].shape[0]
              labels.append(label[idx:idx+file_len])
              if self.parameterization == "kmeans":
                 cluster_centers.append(clust_center)

              motif_usage = self.get_motif_usage(label[idx:idx+file_len])
              motif_usages.append(motif_usage)
              idx += file_len
        return labels, cluster_centers, motif_usages
    
    '''
    get motifs usages- fill with zeros places without clusters
    '''

    def get_motif_usage(self, label):
        motif_usage_all = np.unique(label, return_counts=True)
        motif_usage = np.zeros(self.n_cluster, dtype=int)
        motif_usage[motif_usage_all[0]] = motif_usage_all[1]
       
        return motif_usage

    
    
    '''
    saving latent space for each experiment
    '''
    def saving_latent_space(self,latent_vectors, labels, motif_usages, cluster_center = None):
        for idx, file in enumerate(self.files):
                print(os.path.join(self.cfg['project_path'],"results",file,"",self.model_name,self.parameterization+'-'+str(self.n_cluster),""))
                if not os.path.exists(os.path.join(self.cfg['project_path'],"results",file,self.model_name,self.parameterization+'-'+str(self.n_cluster),"")):                    
                    try:
                        os.mkdir(os.path.join(self.cfg['project_path'],"results",file,"",self.model_name,self.parameterization+'-'+str(self.n_cluster),""))
                    except OSError as error:
                        print(error)   
                    
                save_data = os.path.join(self.cfg['project_path'],"results",file,self.model_name,self.parameterization+'-'+str(self.n_cluster),"")
                np.save(os.path.join(save_data,str(self.n_cluster)+'_km_label_'+file), labels[idx])
                if self.parameterization=="kmeans":
                    np.save(os.path.join(save_data,'cluster_center_'+file), cluster_center[idx])
                np.save(os.path.join(save_data,'latent_vector_'+file), latent_vectors[idx])
                np.save(os.path.join(save_data,'motif_usage_'+file), motif_usages[idx])
    
        
        print("You succesfully extracted motifs with VAME! From here, you can proceed running vame.motif_videos() ")

    '''
    find number motifs and plot
    complete with zero missing values for last motifs
    '''
    def plot_usage(self, motif_usage ):
        output_path = os.path.join(self.cfg['project_path'],"results",self.parameterization+'-'+str(self.n_cluster) +"_motif_usage.pdf")
        
        sum_vec = np.sum(motif_usage, axis=0)
        total_usage = sum_vec.sum()
        motif_ids = np.arange(self.n_cluster)
        order = np.argsort(sum_vec)[::-1]
        sum_sorted = sum_vec[order]
        sum_sorted_percent = (sum_sorted/total_usage)*100
        motif_sorted = motif_ids[order]

        x = np.arange(self.n_cluster)

        fig, ax = plt.subplots()
        ax.plot(x, sum_sorted_percent, marker="o")
        ax.axhline(1, linestyle = '--', linewidth = 1, color ='red')
        plt.xlabel("Original motifs")
        plt.ylabel("Motifs usage (%)")
        # set x-ticks to original motif numbers
        ax.set_xticks(x)
        ax.set_xticklabels(motif_sorted, rotation=90)

        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(x)
        ax_top.set_xticklabels(x + 1)  # ranks start at 1
        ax_top.set_xlabel("Motif rank")

        plt.tight_layout()
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.show()







