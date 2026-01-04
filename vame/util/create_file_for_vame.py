import pandas as pd
import numpy as np
import os
from pathlib import Path
from vame.util.auxiliary import read_config

class create_file_for_vame:
    def __init__(self,config, sheetname, upper_tube, lower_tube, minimum_x,
                maximum_x, width_mole, height_mole):
        config_file = Path(config).resolve()
        cfg = read_config(config_file)
        self.path_files = os.path.join(cfg['project_path'], 'videos', 'pose_estimation')
        self._filenames = cfg['video_sets']
        self._upper_tube_all = upper_tube
        self._lower_tube_all = lower_tube
        self._sheetname = sheetname
        self._minimum_x_all = minimum_x
        self._maximum_x_all =  maximum_x
        self._width_mole_all = width_mole
        self._height_mole_all = height_mole
    '''
    for each file do preprocessing
    '''
    def __call__(self):
         
        for count, f in enumerate(self._filenames):
            self._upper_tube = self._upper_tube_all[count]
            self._lower_tube = self._lower_tube_all[count]
            self._minimum_x = self._minimum_x_all[count]
            self._maximum_x = self._maximum_x_all[count]
            self._width_mole = self._width_mole_all[count]
            self._height_mole = self._height_mole_all[count]

            self.preprocess(f)

        
        
    ######################################################
    '''
    do preprocessing
    '''
    def preprocess(self, f):
        df_original = self.read_excel(f)
        df = df_original.copy()
        if df is not None:
           #all the rows with the animal at the exit of the cage

           #remove nan non detected
           updated_df = self.remove_non_detected_animal(df)
           #remove columns nothing detected through all the frames
           updated_df = self.remove_all_zero_values(updated_df)
           #replace 0 nodetection with nan
           updated_df = self.fill_no_detection(updated_df)
           #remove outside the tube
           updated_df = self.remove_data_outside_tube(updated_df)
           #interpolate before removing middle
           updated_df = self.interpolate_nan_data(updated_df) 
           
           relative_df = updated_df.copy()
           absolute_df = updated_df.copy()
           
           #get relative features normalized by mole size
           relative_df = self.remove_middle(relative_df)
           relative_df = self.remove_columns_with_middle(relative_df)
           #add suffix _rel
           relative_df.columns = [relative_df.columns[0]] + [col + '_rel' for col in relative_df.columns[1:]]

           #get absolute features 
           absolute_df = self.normalize_absolute(absolute_df)
           #add _abs suffix
           absolute_df.columns = [absolute_df.columns[0]] + [col + '_abs' for col in absolute_df.columns[1:]]



           #concatenate the 2 files
           updated_df_total = pd.merge(absolute_df, relative_df, on = "original_frames")
           
           self._filename = os.path.join(self.path_files,(f + '.csv'))
           self.save_as_csv(updated_df_total)
           print("The csv file was saved")
        else:
            updated_df = None
                 





    def read_excel(self,f):
        file = os.path.join(self.path_files,(f + '.xlsx'))
        try:
           df = pd.read_excel(file , sheet_name=self._sheetname) 
           return df 
        except Exception as e:
         print(f"An error occurred while reading the Excel file: {e}")
         return None
    '''
    Fill no detection in which some part of the blind mole was not detected and is zero , replacing by nan for doing interpolation
    ''' 
    def fill_no_detection(self,df):   
        updated_df = df.copy()
        for col in updated_df.columns:
            if '_x' in col.lower() or '_y' in col.lower():
                updated_df[col] = updated_df[col].replace(0, np.nan)
        return updated_df
    
    '''
    Remove the y which are outside the tube
    '''
    def remove_data_outside_tube(self,df):
        updated_df = df.copy()
        for col in updated_df.columns:
            if  '_y' in col.lower():
                # Replace values greater than the threshold with NaN
                updated_df[col] = df[col].mask(df[col] > self._lower_tube, np.nan)
                updated_df[col] = df[col].mask(df[col] < self._upper_tube, np.nan)
        return updated_df
    '''
    Remove the middle and divide by size ofthe blind mole. I will get between -1 to 1 for either x or y
    '''
    def  remove_middle(self,df): 
        df_updated = df.copy()
        for col in  df.columns:
            if '_x' in col.lower():
                df_updated[col] = (df[col] - df['BMR_Middle_x'])/self._width_mole
            elif '_y' in col.lower():
                df_updated[col] = (df['BMR_Middle_y'] -df[col])/self._height_mole
        return df_updated
    
    '''
    Normalize absolute coordinates
    do in a way to go from -1 to 1 in x positive and 1 to -1 in the -y direction
    '''
    def normalize_absolute(self, df):
        df_updated = df.copy()
        avg_x = (self._maximum_x + self._minimum_x)/2
        delta_x = (self._maximum_x - self._minimum_x)/2
        avg_y = (self._upper_tube + self._lower_tube)/2
        delta_y = (-self._upper_tube + self._lower_tube)/2

        for col in  df.columns:
            if '_x' in col.lower():
                        df_updated[col] = (df[col] - avg_x)/delta_x
            elif '_y' in col.lower():
                        df_updated[col] = (avg_y - df[col])/delta_y 
        return df_updated




    '''
    Remove the columns related with the middle
    '''
    def remove_columns_with_middle(self, df): 
        df_cleaned = df.loc[:,~df.columns.str.contains("BMR_Middle")]
        return df_cleaned  
     
    '''
    interpolate nan data for each columns
    '''   
    def interpolate_nan_data(self, df):
        df_interpolate = df.copy()
        for col in df.columns:  
            df_interpolate[col] = df[col].interpolate().fillna(method="bfill").fillna(method="ffill").values  
        return df_interpolate    
    
    '''
    save as csv
    '''
    def save_as_csv(self, updated_df):
        updated_df.to_csv(self._filename, index=False,quoting=0)
        
    '''
    remove when the animal is not detected. all the rows with Nan
    '''    
    def remove_non_detected_animal(self,df):
        cols_to_check = df.columns[1:] #first columns are frames
        df_without_nan = df[~df[cols_to_check].isna().all(axis = 1)].reset_index(drop=True)
        df_without_nan = df_without_nan.rename(columns={df_without_nan.columns[0]: "original_frames"})
        return df_without_nan
    
    '''
    remove all the columns which are zero including the score
    '''
    def remove_all_zero_values(self, df):
        zero_cols = df.columns[(df == 0).all()].tolist()
        #get base names-split from the right
        base_names = list({name.rsplit('_', 1)[0] for name in zero_cols})
        
        df_new = df.drop(columns = [col for col in df.columns if any(col.startswith(base) for base in base_names)])

        return df_new