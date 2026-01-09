import os
import umap
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap, BoundaryNorm

from vame.util.auxiliary import read_config

LABEL_COLORS_28 = {
0:  "#1f77b4",  # blue
    1:  "#ff7f0e",  # orange
    2:  "#2ca02c",  # green
    3:  "#d62728",  # red
    4:  "#9467bd",  # purple
    5:  "#8c564b",  # brown
    6:  "#e377c2",  # pink
    7:  "#7f7f7f",  # gray
    8:  "#bcbd22",  # olive
    9:  "#17becf",  # cyan

    10: "#393b79",  # dark blue
    11: "#637939",  # dark green
    12: "#8c6d31",  # mustard
    13: "#843c39",  # dark red
    14: "#7b4173",  # dark purple

    15: "#3182bd",  # steel blue
    16: "#e6550d",  # burnt orange
    17: "#31a354",  # medium green
    18: "#756bb1",  # lavender
    19: "#636363",  # dark gray

    20: "#6baed6",  # light blue
    21: "#fd8d3c",  # light orange
    22: "#74c476",  # light green
    23: "#9e9ac8",  # light purple
    24: "#969696",  # light gray

    25: "#bcbddc",  # pale lavender
    26: "#fdae6b",  # pale orange
    27: "#a1d99b",  # pale green

    # --- additional distinct colors ---
    28: "#1b9e77",  # teal green
    29: "#d95f02",  # dark orange
    30: "#7570b3",  # muted violet
    31: "#e7298a",  # magenta
    32: "#66a61e",  # lime green
    33: "#e6ab02",  # golden yellow
    34: "#a6761d",  # ochre
    35: "#666666",  # neutral dark gray
    36: "#1f9bcf",  # bright cyan-blue
    37: "#b15928",  # chestnut
    38: "#f781bf",  # light magenta
    39: "#4daf4a",  # vivid green
}





class umap_visualization_silvia:
    def __init__(self,config,add_manual_data,exp_file = None ):
       #read config file
       config_file = Path(config).resolve()
       cfg = read_config(config_file)
       #read all the necesary parameters
       self.model_name = cfg['model_name']
       self.n_cluster = cfg['n_cluster']
       self.parameterization = cfg['parameterization']
       self.project_path = cfg['project_path']
       self.file_exp = cfg['video_sets']
       self.add_manual_data = add_manual_data
       if (exp_file != None) and (exp_file != 'all'):
         idx = (cfg['video_sets']).index(exp_file)

         file_1 = "latent_vector_" + self.file_exp[idx] + ".npy"
         self.file_latent_vector = os.path.join(self.project_path,"results", self.file_exp[idx], self.model_name,self.parameterization+'-'+str(self.n_cluster), file_1 )
         self.save_data = os.path.join(cfg['project_path'],"results",self.file_exp[idx],self.model_name,self.parameterization+'-'+str(self.n_cluster),"")
         self.file_labels = os.path.join(self.save_data,str(self.n_cluster)+'_km_label_'+ self.file_exp[idx] + '.npy')
         self.umap_without_labels = os.path.join(self.save_data,str(self.n_cluster)+'_umap_without_label_number_'+ self.file_exp[idx] + '.pdf')
         self.umap_without_labels_manual = os.path.join(self.save_data,str(self.n_cluster)+'_umap_without_label_number_manuallabels_'+ self.file_exp[idx] + '.pdf')
         self.umap_with_labels = os.path.join(self.save_data,str(self.n_cluster)+'_umap_with_label_number_'+ self.file_exp[idx] + '.pdf')
         self.number_file = idx
       else:
           self.files = self.join_all_files(cfg['video_sets'])
       
       self.value = exp_file
       #parameters for umap
       self.num_points = cfg['num_points']
       self.min_dist=cfg['min_dist']
       self.n_neighbors=cfg['n_neighbors']
       self.random_state = cfg['random_state']

       if self.add_manual_data:
            self.output_data = os.path.join(cfg['project_path'], 'videos', 'pose_estimation','labels_manually_' + cfg['video_sets'][0] + '.npy')
            self.label_data = np.load(self.output_data)

    def  __call__(self, label = None):
        match self.value:
            case 'all':
                latent_vectors = []
                length_files = []
                for f in self.files:
                    latent_vector = np.load(f)
                    latent_vectors.append(latent_vector)
                    length_files.append(latent_vector.shape[0])

                self.latent_vector = np.concatenate(latent_vectors, axis=0)
                self.num_points = self.latent_vector.shape[0]
                print("Embedding %d data points.." %self.num_points)
                embed = self.create_embedding()
                print("Visualizing %d data points.. " %self.num_points)
                self.umap_label_vis_all(embed,length_files)
        

            case _: #one file   
               self.latent_vector = np.load(self.file_latent_vector)
               self.num_points = self.latent_vector.shape[0]
               # if self.num_points > self.latent_vector.shape[0]:
               #         self.num_points = self.latent_vector.shape[0]
               print("Embedding %d data points.." %self.num_points)
               embed = self.create_embedding()
               print("Visualizing %d data points.. " %self.num_points)
               if label == None:                    
                    self.umap_vis(embed)
               elif label == "motif":
                    motif_label = np.load(self.file_labels)

                    self.umap_label_vis(embed,motif_label)
        
                
                   

    #################################################################################    

    '''
    input: latent space
    output: embedding vectors
    '''
    def create_embedding(self):
        
        # print("Compute embedding for file %s" % self.file_exp[self.number_file])
         reducer = umap.UMAP(n_components=2, min_dist = self.min_dist, n_neighbors = self.n_neighbors, 
                    random_state=self.random_state) 
         embed = reducer.fit_transform(self.latent_vector[:self.num_points,:])

         return embed
    
    '''
    input: embed vectors with number of points to plot
    output: plot without labels
    '''
    def umap_vis(self, embed):
         fig, ax = plt.subplots(figsize=(6,6))
         ax.scatter(embed[:self.num_points,0], embed[:self.num_points,1], s = 10, alpha = 0.6, c = 'blue', edgecolors = 'none')
         ax.set_aspect('equal', 'datalim')
         ax.set_xlabel("Embedding Dimension 1")
         ax.set_ylabel("Embedding Dimension 2")
         ax.set_title("2D Embedding Scatter Plot")

         ax.grid(False)

         fig.savefig(self.umap_without_labels, format = "pdf")

         plt.show()

    '''
    input: embed vectors with number of points to plot
    output: plot with labels
    '''
    def umap_label_vis(self,embed,label):
         fig, ax = plt.subplots(figsize=(7, 7))

         # Generate 28 distinct colors using tab20 + tab20b
         fixed_colors = [LABEL_COLORS_28[i] for i in range(self.n_cluster)]

        # Create a discrete colormap
         cmap = ListedColormap(fixed_colors)

        #ensures that cluster i always maps to color i
         norm = BoundaryNorm(boundaries=np.arange(-0.5, self.n_cluster + 0.5, 1),ncolors= self.n_cluster)


         sc = ax.scatter(embed[:self.num_points, 0], embed[:self.num_points, 1], c=label[:self.num_points], cmap = cmap, norm=norm, s=10, alpha=0.7, edgecolors='none')
         
         if self.add_manual_data:
             # Overlay manual motif labels
             motif_labels = self.label_data[:, 2]  # Assuming the third column contains motif labels
             motif_indices = np.where(motif_labels > 0)[0]
             ax.scatter(embed[motif_indices, 0], embed[motif_indices, 1], c='black', marker = 'x', s=30, linewidths=0.5,alpha = 0.6, label='Manual Motifs')

         # Add colorbar with cluster ticks
         cbar = plt.colorbar(sc, ax=ax, boundaries=np.arange(-0.5, self.n_cluster + 0.5, 1))
         cbar.set_ticks(np.arange(self.n_cluster))
         cbar.set_label("Cluster ID")

         ax.set_aspect('equal', 'datalim')
         ax.set_xlabel("Embedding Dimension 1")
         ax.set_ylabel("Embedding Dimension 2")
         ax.set_title("2D Embedding Scatter Plot")

         ax.grid(False)
         if self.add_manual_data:
            fig.savefig(self.umap_without_labels_manual, format = "pdf")
         else:
            fig.savefig(self.umap_without_labels , format = "pdf")


         for cluster_id in range(self.n_cluster):
             # Get points belonging to this cluster
            cluster_points = embed[label == cluster_id,:]
            if len(cluster_points) > 0:
              # Compute cluster centroid
              centroid = cluster_points.mean(axis=0)
              # Place text at centroid
              ax.text(centroid[0], centroid[1], str(cluster_id),
            fontsize=9, fontweight='bold',
            ha='center', va='center',
            color='black',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
        )

         fig.savefig(self.umap_with_labels, format = "pdf")

         plt.show()
        

    '''
     join all the files
    '''
    def join_all_files(self,cfg):
        files = []
        for f in cfg:
         file_1 = "latent_vector_" + f + ".npy"
         self.file_latent_vector = os.path.join(self.project_path,"results", f, self.model_name,self.parameterization+'-'+str(self.n_cluster), file_1 )
         files.append(self.file_latent_vector)
        return files
    
    '''
     map of all files , 
    '''
    def umap_label_vis_all(self,embed,length_files):
        fig, ax = plt.subplots(figsize=(7, 7))

        colors = ["salmon", "seagreen", "royalblue"]
        
        l_initial = 0
        for count, l in enumerate(length_files):
           ax.scatter(embed[l_initial:l_initial + l, 0],embed[l_initial:l_initial + l, 1], color=colors[count],label=self.file_exp[count], s=10, alpha=0.7, edgecolors='none')
           l_initial += l

        ax.set_xlabel("Embedding Dimension 1")
        ax.set_ylabel("Embedding Dimension 2")
        ax.set_title("2D Embedding Scatter Plot")
        
        ax.legend(title="Files", loc="best", markerscale=2,frameon=False)

        ax.grid(False)

        plt.show()