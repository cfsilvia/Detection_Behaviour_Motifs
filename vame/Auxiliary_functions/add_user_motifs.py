import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from vame.util.auxiliary import read_config


class add_user_motifs:
    def __init__(self, config):
         config_file = Path(config).resolve()
         self.config = read_config(config_file)
         self.data_path = os.path.join(self.config['project_path'], 'videos', 'pose_estimation', self.config['video_sets'][0] + '.csv')
         self.motifs_path = os.path.join(self.config['project_path'], 'videos', 'pose_estimation','labels_manually_' + self.config['video_sets'][0] + '.xlsx')
         self.output_data = os.path.join(self.config['project_path'], 'videos', 'pose_estimation','labels_manually_' + self.config['video_sets'][0] + '.npy')
         self.output_figure = os.path.join(self.config['project_path'], 'videos', 'pose_estimation','labels_manually_' + self.config['video_sets'][0] + '.pdf')

    def __call__(self,feature):
       #load data
         data = pd.read_csv(self.data_path)
         labels_motifs = pd.read_excel(self.motifs_path)
         df_out = self.create_label_data(labels_motifs, data, feature)
         #saved data
         np.save(self.output_data, df_out)
         print(f"Saved labels to {self.output_data}")
         #create figure
         self.plot_regions(df_out, feature)

    def create_label_data(self, labels_motifs, data, feature):
         data['label1'] = 0
         frame_col_csv = data.columns[0]          # first column = frame number

         for _, row in labels_motifs.iterrows():
                start_frame = row['start']
                end_frame = row['end']
                motif_label = row['label']
                data.loc[(data[frame_col_csv] >= start_frame) & (data[frame_col_csv] <= end_frame), 'label1'] = motif_label

         df_out = data[[frame_col_csv, feature, "label1"]].to_numpy()

         return df_out

    def plot_regions(self, df_out, feature):
         
         data_plot = df_out[:, 1]
         frames = np.arange(len(data_plot))
         labels = df_out[:, 2]

         plt.figure(figsize=(15, 5))
         plt.plot(frames, data_plot, color='black', linewidth=0.4, label=feature)
         
         ymin, ymax = np.min(data_plot), np.max(data_plot)
         plt.fill_between(frames, ymin, ymax, where=labels==1, color='pink', alpha=0.7, label='Motif')
         
         plt.xlabel('Frames')
         plt.ylabel('Feature value')
         plt.title(f'Feature {feature} with motifs')
         plt.legend()
         plt.savefig(self.output_figure)
         plt.show()
         print(f"Saved figure to {self.output_figure}")

                
               
            
