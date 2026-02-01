#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA-based segmentation of latent space with HMM clustering.
Takes pre-computed latent vectors and performs dimensionality reduction via PCA,
followed by HMM-based segmentation on the reduced space.
"""

import os
import pickle
import numpy as np
from pathlib import Path

from hmmlearn import hmm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from vame.util.auxiliary import read_config


class pose_segmentation_pca:
    """
    PCA-based analysis of latent space with HMM clustering.
    Takes pre-computed latent vectors and performs dimensionality reduction via PCA,
    followed by HMM-based segmentation on the reduced space.
    """
    def __init__(self, config, exp_file = None):
   
        config_file = Path(config).resolve()
        self.cfg = read_config(config_file)
        self.model_name = self.cfg['model_name']
        self.parameterization = self.cfg['parameterization']
        self.n_cluster = self.cfg['n_cluster']
        self.project_path = self.cfg['project_path']
        self.file_exp = self.cfg['video_sets']
        self.n_pca_components = None #can be None then choice to get a square matrix
        if (exp_file != None) and (exp_file != 'all'):
         #idx = (cfg['video_sets']).index(exp_file)
         idx =0    

         file_1 = "latent_vector_" + self.file_exp[idx] + ".npy"
         self.files = os.path.join(self.project_path,"results", self.file_exp[idx], self.model_name,'self.parameterization'+'-'+str(self.n_cluster), file_1 )
         self.save_data = os.path.join(self.cfg['project_path'],"results",self.file_exp[idx],self.model_name,'_PCA_'+'-'+str(self.n_cluster),"")
         
         # Create directory if it doesn't exist
         if not os.path.exists(self.save_data):
             try:
                 os.makedirs(self.save_data, exist_ok=True)
             except OSError as error:
                 print(error)
         
        else:
           self.files = self.join_all_files(self.cfg['video_sets'])
        print(f'PCA Segmentation for VAME model: {self.model_name}')
        print(f'Using {self.n_pca_components} PCA components\n')

    def __call__(self):
        """
        Execute full PCA + HMM analysis pipeline
        """
        # Create output folders
        # for folders in self.files:
        #     # if not os.path.exists(os.path.join(self.cfg['project_path'], "results", folders, self.model_name, "")):
        #     #     os.makedirs(os.path.join(self.cfg['project_path'], "results", folders, self.model_name, ""), exist_ok=True)
        
        # Perform PCA with 90% variance threshold
        pca, latent_pca_files, variance_ratio = self.pca_latent_space(
            variance_threshold=0.90,
            save_results=False  # Don't save here, save after HMM
        )
        
        # Save PCA results
        self.save_pca_results(pca, latent_pca_files, variance_ratio)
        
        # Apply HMM on PCA space
        print("\nApplying HMM on PCA-transformed latent space...")
        hmm_model = self.hmm_on_pca(latent_pca_files)
        
        # Get HMM predictions
        latent_pca_cat = latent_pca_files
        labels = hmm_model.predict(latent_pca_cat)
        
        # Reorganize labels by file
        labels_by_file, motif_usages = self.reorder_labels_from_hmm(latent_pca_files, labels)
        
        # Plot results
        self.plot_usage(motif_usages)
      #  self.plot_pca_analysis(pca, labels_by_file)
        
        # Save results
        self.save_pca_hmm_results(latent_pca_files, labels_by_file, motif_usages, hmm_model)
        
        print("PCA+HMM analysis completed successfully!")
        return pca, latent_pca_files, hmm_model, labels_by_file, motif_usages
    
    """
        Perform PCA on latent space vectors
        
        Parameters:
        -----------
        latent_vector_files : list of np.ndarray
            List of latent space vectors for each file
        n_components : int, optional
            Number of PCA components. If None, uses components that explain variance_threshold
        variance_threshold : float
            Target cumulative explained variance (default: 0.90 for 90%)
        save_results : bool
            Whether to save PCA results to disk
        
        Returns:
        --------
        pca : sklearn.decomposition.PCA
            Fitted PCA model
        latent_pca_files : list of np.ndarray
            Transformed latent vectors in PCA space for each file
        explained_variance_ratio : np.ndarray
            Explained variance ratio for each component
        """

    def pca_latent_space(self,  variance_threshold=0.90, save_results=True):
        #read numpy files
        # Load and concatenate all latent vectors from files
        latent_vector_files = []
        
        # Handle both single file and multiple files
        if isinstance(self.files, str):
            # Single file case
            latent_vector_files.append(np.load(self.files))
        else:
            # Multiple files case
            for file_path in self.files:
                latent_vector_files.append(np.load(file_path))
        
        latent_vector_cat = np.concatenate(latent_vector_files, axis=0)
        
        # Set default n_components if not provided
        if self.n_pca_components is None:
            # Fit with max components first to find components for target variance
            n_components_max = min(latent_vector_cat.shape[0], latent_vector_cat.shape[1])
            pca_temp = PCA(n_components=n_components_max, random_state=self.cfg.get('random_state_kmeans', 42))
            pca_temp.fit(latent_vector_cat)
            
            # Find number of components for target variance
            cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum_var >= variance_threshold) + 1
            print(f"Components needed for {variance_threshold*100:.0f}% variance: {n_components}")
            
            # Update instance variable
            self.n_pca_components = n_components
        
        # Fit PCA with determined number of components
        print(f"Performing PCA with {n_components} components on latent space...")
        pca = PCA(n_components=n_components, random_state=self.cfg.get('random_state_kmeans', 42))
        pca.fit(latent_vector_cat)
        
        # Transform each file's latent vectors
        # latent_pca_files = []
        # for i, file in enumerate(self.files):
        #     latent_pca = pca.transform(latent_vector_files[i])
        #     latent_pca_files.append(latent_pca)

        latent_pca_files = pca.transform(latent_vector_cat)
        
        actual_variance = np.sum(pca.explained_variance_ratio_)
        print(f"Total explained variance: {actual_variance:.4f} ({actual_variance*100:.2f}%)")
        print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
        
        return pca, latent_pca_files, pca.explained_variance_ratio_
    
    """
        Save PCA analysis results to disk
        
        Parameters:
        -----------
        pca : sklearn.decomposition.PCA
            Fitted PCA model
        latent_pca_files : list of np.ndarray
            PCA-transformed latent vectors for each file
        variance_ratio : np.ndarray
            Explained variance ratio for each component
        """
    def save_pca_results(self, pca, latent_pca_files, variance_ratio):
       
        output_path = os.path.join(self.cfg['project_path'], "results", "pca_hmm_analysis")
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path, exist_ok=True)
            except OSError as error:
                print(error)
        
        # Save PCA transformed vectors for each file
        for idx, file in enumerate(self.files):
            # Extract file name from path if it's a full path
            if isinstance(file, str):
                file_name = os.path.basename(file).replace('.npy', '')
            else:
                file_name = str(idx)
            np.save(os.path.join(output_path, f'pca_latent_vector_{file_name}'), latent_pca_files)
        
        # Save PCA model and variance info
        with open(os.path.join(output_path, 'pca_model.pkl'), 'wb') as f:
            pickle.dump(pca, f)
        
        np.save(os.path.join(output_path, 'explained_variance_ratio'), pca.explained_variance_ratio_)
        np.save(os.path.join(output_path, 'cumsum_variance'), np.cumsum(pca.explained_variance_ratio_))
        
        print(f"PCA results saved to {output_path}")
    """
        Train HMM on PCA-transformed latent space
        
        Parameters:
        -----------
        latent_pca_files : list of np.ndarray
            PCA-transformed latent vectors for each file
        n_iter : int
            Number of iterations for HMM fitting
        
        Returns:
        --------
        hmm_model : hmmlearn.hmm.GaussianHMM
            Fitted HMM model
        """
    def hmm_on_pca(self, latent_pca_files, n_iter=100):
        
        latent_pca_cat = latent_pca_files
        
        print(f"Training HMM with {self.n_cluster} states on PCA space...")
        hmm_model = hmm.GaussianHMM(n_components=self.n_cluster, covariance_type="full", n_iter=n_iter)
        hmm_model.fit(latent_pca_cat)
        
        print("HMM training completed!")
        return hmm_model
    """
        Reorganize HMM labels by file
        
        Parameters:
        -----------
        latent_pca_files : list of np.ndarray
            PCA-transformed latent vectors
        labels : np.ndarray
            HMM predictions on concatenated vectors
        
        Returns:
        --------
        labels_by_file : list of np.ndarray
            Labels reorganized by file
        motif_usages : list of np.ndarray
            Usage statistics for each motif per file
        """
    def reorder_labels_from_hmm(self, latent_pca_files, labels):
        
        labels_by_file = []
        motif_usages = []
        
        idx = 0
        for i, file in enumerate(self.files):
            file_len = latent_pca_files.shape[0]
            file_labels = labels[idx:idx + file_len]
            labels_by_file.append(file_labels)
            
            motif_usage = self.get_motif_usage(file_labels)
            motif_usages.append(motif_usage)
            idx += file_len
        
        return labels_by_file, motif_usages
    """
        Get motif usage counts and fill missing clusters with zeros
        
        Parameters:
        -----------
        label : np.ndarray
            Cluster labels
        
        Returns:
        --------
        motif_usage : np.ndarray
            Usage count for each motif
        """
    def get_motif_usage(self, label):
        
        motif_usage_all = np.unique(label, return_counts=True)
        motif_usage = np.zeros(self.n_cluster, dtype=int)
        motif_usage[motif_usage_all[0]] = motif_usage_all[1]
        return motif_usage
    """
        Plot motif usage across all files
        
        Parameters:
        -----------
        motif_usages : list of np.ndarray
            Motif usage for each file
        """
    def plot_usage(self, motif_usages):
       
        output_path = os.path.join(self.cfg['project_path'], "results", f"pca_hmm_{self.n_cluster}_motif_usage.pdf")
        
        sum_vec = np.sum(motif_usages, axis=0)
        total_usage = sum_vec.sum()
        motif_ids = np.arange(self.n_cluster)
        order = np.argsort(sum_vec)[::-1]
        sum_sorted = sum_vec[order]
        sum_sorted_percent = (sum_sorted / total_usage) * 100
        motif_sorted = motif_ids[order]
        
        x = np.arange(self.n_cluster)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, sum_sorted_percent, marker="o", linewidth=2, markersize=8)
        ax.axhline(1, linestyle='--', linewidth=1, color='red', label='1% threshold')
        plt.xlabel("Original motifs", fontsize=12)
        plt.ylabel("Motifs usage (%)", fontsize=12)
        plt.title("PCA+HMM Motif Usage Distribution", fontsize=14)
        
        # Set x-ticks to original motif numbers
        ax.set_xticks(x)
        ax.set_xticklabels(motif_sorted, rotation=90)
        
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(x)
        ax_top.set_xticklabels(x + 1)  # ranks start at 1
        ax_top.set_xlabel("Motif rank", fontsize=12)
        ax.legend()
        
        plt.tight_layout()
        fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Motif usage plot saved to {output_path}")
        plt.show()
    """
        Plot PCA analysis results including scree plot and 2D projection
        
        Parameters:
        -----------
        pca : sklearn.decomposition.PCA
            Fitted PCA model
        labels_by_file : list of np.ndarray
            HMM labels for each file
        """
    def plot_pca_analysis(self, pca, labels_by_file):
        
        output_dir = os.path.join(self.cfg['project_path'], "results", "pca_hmm_plots")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Plot 1: Scree plot with explained variance
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Cumulative explained variance
        axes[0].plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                     np.cumsum(pca.explained_variance_ratio_), 'bo-', linewidth=2, markersize=6)
        axes[0].axhline(0.95, linestyle='--', color='red', label='95% variance')
        axes[0].set_xlabel('Number of Components', fontsize=12)
        axes[0].set_ylabel('Cumulative Explained Variance Ratio', fontsize=12)
        axes[0].set_title('PCA Scree Plot - Cumulative Variance', fontsize=13)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Individual explained variance for top 10 components
        n_top = min(10, len(pca.explained_variance_ratio_))
        axes[1].bar(range(1, n_top + 1), pca.explained_variance_ratio_[:n_top], color='steelblue')
        axes[1].set_xlabel('Principal Component', fontsize=12)
        axes[1].set_ylabel('Explained Variance Ratio', fontsize=12)
        axes[1].set_title('Top 10 Components Explained Variance', fontsize=13)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        scree_path = os.path.join(output_dir, "pca_scree_plot.pdf")
        fig.savefig(scree_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"PCA scree plot saved to {scree_path}")
        plt.show()
        
        # Plot 2: 2D PCA projection with HMM labels
        if pca.n_components_ >= 2:
            # Load original latent vectors for 2D projection
            latent_vector_files = []
            if isinstance(self.files, str):
                latent_vector_files.append(np.load(self.files))
            else:
                for file_path in self.files:
                    latent_vector_files.append(np.load(file_path))
            latent_vector_cat = np.concatenate(latent_vector_files, axis=0)
            
            # Use first 2 components from existing PCA
            transformed_2d = pca.components_[:2].T @ (latent_vector_cat - pca.mean_).T
            transformed_2d = transformed_2d.T
            
            fig, ax = plt.subplots(figsize=(12, 10))
            labels_cat = np.concatenate(labels_by_file)
            
            # Create scatter plot colored by HMM state
            scatter = ax.scatter(transformed_2d[:, 0], transformed_2d[:, 1], 
                               c=labels_cat, cmap='tab20', alpha=0.6, s=30, edgecolors='none')
            
            cbar = plt.colorbar(scatter, ax=ax, label='HMM State')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
            ax.set_title('2D PCA Projection of Latent Space (colored by HMM state)', fontsize=13)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            projection_path = os.path.join(output_dir, "pca_2d_projection_hmm.pdf")
            fig.savefig(projection_path, format="pdf", bbox_inches="tight", dpi=300)
            print(f"2D PCA projection saved to {projection_path}")
            plt.show()
        
        # Plot 3: 3D projection if we have at least 3 components
        if pca.n_components_ >= 3:
            # Load original latent vectors for 3D projection
            latent_vector_files = []
            if isinstance(self.files, str):
                latent_vector_files.append(np.load(self.files))
            else:
                for file_path in self.files:
                    latent_vector_files.append(np.load(file_path))
            latent_vector_cat = np.concatenate(latent_vector_files, axis=0)
            
            # Use first 3 components from existing PCA
            transformed_3d = pca.components_[:3].T @ (latent_vector_cat - pca.mean_).T
            transformed_3d = transformed_3d.T
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            labels_cat = np.concatenate(labels_by_file)
            
            scatter = ax.scatter(transformed_3d[:, 0], transformed_3d[:, 1], transformed_3d[:, 2],
                               c=labels_cat, cmap='tab20', alpha=0.6, s=20, edgecolors='none')
            
            plt.colorbar(scatter, ax=ax, label='HMM State', shrink=0.5)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=10)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=10)
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})', fontsize=10)
            ax.set_title('3D PCA Projection of Latent Space (colored by HMM state)', fontsize=13)
            
            plt.tight_layout()
            projection_3d_path = os.path.join(output_dir, "pca_3d_projection_hmm.pdf")
            fig.savefig(projection_3d_path, format="pdf", bbox_inches="tight", dpi=300)
            print(f"3D PCA projection saved to {projection_3d_path}")
            plt.show()

    """
        Save all PCA+HMM analysis results to disk
        
        Parameters:
        -----------
        latent_pca_files : list of np.ndarray
            PCA-transformed latent vectors
        labels_by_file : list of np.ndarray
            HMM labels for each file
        motif_usages : list of np.ndarray
            Motif usage for each file
        hmm_model : hmmlearn.hmm.GaussianHMM
            Fitted HMM model
        """
    def save_pca_hmm_results(self, latent_pca_files, labels_by_file, motif_usages, hmm_model):
       
        for idx, file in enumerate(self.files):
            output_path = os.path.join(self.cfg['project_path'], "results", f"pca_{self.n_pca_components}_hmm_{self.n_cluster}", "")
            
            if not os.path.exists(output_path):
                try:
                    os.makedirs(output_path, exist_ok=True)
                except OSError as error:
                    print(error)
            
            # Save PCA transformed vectors
            np.save(os.path.join(output_path, f'pca_latent_vector_{Path(file).name}'), latent_pca_files)
            
            # Save HMM labels
            np.save(os.path.join(output_path, f'hmm_labels_{Path(file).name}'), labels_by_file)
            
            # Save motif usage
            np.save(os.path.join(output_path, f'motif_usage_{Path(file).name}'), motif_usages)
        
        # Save HMM model globally
        global_path = os.path.join(self.cfg['project_path'], "results", "pca_hmm_analysis")
        if not os.path.exists(global_path):
            os.makedirs(global_path, exist_ok=True)
        
        with open(os.path.join(global_path, 'hmm_model_pca.pkl'), 'wb') as f:
            pickle.dump(hmm_model, f)
        
        print(f"PCA+HMM results saved to individual file directories and {global_path}")


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
    