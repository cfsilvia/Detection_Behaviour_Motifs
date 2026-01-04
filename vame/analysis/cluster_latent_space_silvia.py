import os
import tqdm
import torch
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import umap


from hmmlearn import hmm
from sklearn.cluster import KMeans

from vame.util.auxiliary import read_config
from vame.model.rnn_model import RNN_VAE

class cluster_latent_space_silvia:
    def __init__(self,config,exp_file = None):
       #read config file
       config_file = Path(config).resolve()
       cfg = read_config(config_file)
       self.model_name = cfg['model_name']
       self.n_cluster = cfg['n_cluster']
       self.parameterization = cfg['parameterization']
       self.project_path = cfg['project_path']
       self.file_exp = cfg['video_sets']
       if exp_file != None:
         idx = (cfg['video_sets']).index(exp_file)


       file_1 = "latent_vector_" + self.file_exp[idx] + ".npy"
       self.file_latent_vector = os.path.join(self.project_path,"results", self.file_exp[idx], self.model_name,self.parameterization+'-'+str(self.n_cluster), file_1 )
       self.save_data = os.path.join(cfg['project_path'],"results",self.file_exp[idx],self.model_name,self.parameterization+'-'+str(self.n_cluster),"")
       self.file_labels = os.path.join(self.save_data,str(self.n_cluster)+'_km_label_'+ self.file_exp[idx] + '.npy')
       self.video_file = os.path.join(self.project_path, "videos", self.file_exp[idx] + '.avi') 
       self.cluster_start = cfg['time_window'] / 2
       self.data_file = os.path.join(self.project_path, "videos", "pose_estimation" ) 
       self.number_video = idx # number video to check

    def  __call__(self):
        print('Pose segmentation for VAME model: %s \n' %self.model_name)
        self.original_data = pd.read_csv(os.path.join(self.project_path,"videos","pose_estimation",self.file_exp[self.number_video] +'.csv'), skiprows = 1, header=None)
        self.original_frames = self.original_data.iloc[:,0]
        self.labels = np.load(self.file_labels)
        self.create_movies_with_same_motifs()
 ############################################################      

    

    '''
    create movies with the same motif
    '''
    def create_movies_with_same_motifs(self):
       #settings
       
       file_path_output = os.path.join(self.data_file,(self.file_exp[self.number_video] + "_information_after_clustering.xlsx"))
       csv_file = os.path.join(self.data_file, (self.file_exp[self.number_video] + ".csv"))
       df_csv = pd.read_csv(csv_file)

       df_new = pd.DataFrame(columns = [*df_csv.columns, "motif", "#event", "#frame"])
       #read labels
       motifs = np.unique(self.labels)
       
       for m in motifs:
             #read movie 
             capture = cv2.VideoCapture(self.video_file)
             #vid information
             if capture.isOpened():
                width  = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = capture.get(cv2.CAP_PROP_FPS)
                print('fps:', fps)
                #extract frames for a given motifs
                frames_number = np.where(self.labels.squeeze() == m)[0]
                frames = self.original_frames[frames_number]
                
                self.write_movie(m,capture,frames, width, height, fps)
                #df_new = self.write_data(df_new,m,frames, df_csv)
               
                #df_new.to_excel(file_path_output, index = False)  
       #df_new.to_excel(file_path_output, index = False)     


    '''
    write a movie
    '''
    def write_movie(self, m, capture, frames_numbers,width, height, fps):
       #create mp4 writer
       
       output_path = os.path.join(self.project_path,"results", self.file_exp[self.number_video], 
                                  self.model_name,self.parameterization+'-'+str(self.n_cluster), 
                                  self.file_exp[self.number_video] + "_snapshot_" + str(m) + "_.mp4")

       
      #  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      #  writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
       writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (int(width), int(height)))
       #
       differences = np.diff(frames_numbers)
       number_serie = 0
       for count, f in enumerate(frames_numbers):
          capture.set(cv2.CAP_PROP_POS_FRAMES, self.cluster_start + f)
          ret, frame = capture.read()
          if ret:
            #writer.write(frame)
            #add text
            if (count > 0) and differences[count-1] > 1:
                number_serie += 1
                text = str(int(number_serie)) + ' New Event  ' + str(self.cluster_start + f ) + str(number_serie) 
            else:
                text = str(int(number_serie)) + ' Frame  ' + str(self.cluster_start + f )
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 4
            thickness = 8
            color = (0, 0, 255)  # Red text (BGR format)
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            frame_height, frame_width = frame.shape[:2]
            position = (
                (frame_width - text_width) // 2,
                (frame_height + text_height) // 2
                )
            cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
            writer.write(frame)

          else:
            print(f"Could not extract frame {f}")

       capture.release()
       writer.release()
       print(f"Created video: {output_path}")

    '''
     write data
    '''
    def write_data(self,df_new,m,frames_numbers, df_csv):
       differences = np.diff(frames_numbers)
       number_serie = 0
       for count, f in enumerate(frames_numbers):
            if (count > 0) and differences[count-1] > 1:
                number_serie += 1
            df_aux = df_csv.iloc[[f]]
            df_aux['motif'] = m
            df_aux['#event'] = number_serie
            df_aux['#frame'] = f

            df_new = pd.concat([df_new, df_aux], ignore_index=True)
           
       return df_new

    
   