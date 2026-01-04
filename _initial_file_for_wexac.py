import sys

sys.path.insert(0, r"D:\Silvia\ScriptOnGithub\Detection_Behaviour_Motifs")
import vame



def main_menu(data):
    choice = data['choice']
    match choice:  
        case '1':
            # Initialize your project
            config = vame.init_new_project(project=data['project'], videos=data['videos'], working_directory=data['working_directory'], videotype='.mp4')

        case '2': #create file 
            obj_1 = vame.create_file_for_vame( user_data['config'], data['sheetname'], data['upper_tube'], data['lower_tube'], 
                                              user_data['minimum_x'], user_data['maximum_x'], user_data['width_mole'], user_data['height_mole'] )

            obj_1()
        case '3':
        #     # you can use the following to convert your .csv to a .npy array, ready to train vame on it
              vame.csv_to_numpy(data['config'])
        case '4':
        #      # # create the training set for the VAME model
              vame.create_trainset(data['config'], check_parameter=False)
        case '5':
        #      # # Train VAME:
              vame.train_model(data['config'])
        case '6':
              # # Evaluate model
              vame.evaluate_model(data['config'])
        case '7':
        #      # # Segment motifs/pose
              obj = vame.pose_segmentation(data['config'])
              obj()

        case '8':
        #     #do umap over the latent space either to each experiment or all together
             obj = vame.umap_visualization_silvia(data['config'],"BMR10_with_landmarks_left")
             obj("motif") #argument could be: blank nothing or "motif"

        case '9':
        #     #find motifs on the movies
            obj = vame.cluster_latent_space_silvia(data['config'],"BMR10_with_landmarks_left")
            obj() #commands to add : "cluster", "usage_motifs", "find_motifs_on_the_movies"
        # case '8':#not used
        #     #Create motif videos to get insights about the fine grained poses
        #     vame.motif_videos(data['config'], videoType='.mp4')
        # case '9':#notused
        #     #Create etograms of the motifs
        #     input_file = user_data['labels_file']
        #     fps = user_data['fps']  # adjust to your recording's frame rate
        #     output_csv_path = user_data['file_etogrhams']
        #     offset = 30/2  # depends on your window size (usually half of it)
        #     vame.etoghram(input_file, output_csv_path,fps, offset)
       
        
        # case '12':
        #     # # OPTIONAL: Create behavioural hierarchies via community detection
        #     obj = vame.community_analysis_silvia(data['config'])  
        #     obj(2)   #cut_tree = 2
        case '13':
        ##plot kloss
          obj = vame.plot_losses(data['config']) 
          obj()    
            

        case _:
             return "Invalid option"

 

if __name__ == "__main__":
    
    user_data = {}
    user_data['choice'] = '9'  
    user_data['working_directory'] = '/home/labs/kimchi/cfsilvia/Blind_mole/Motifs_Detection/' 
    user_data['project']='BMR-VAME-Project'
    user_data['videos'] = ['/home/labs/kimchi/cfsilvia/Blind_mole/Motifs_Detection//original_data/BMR10_with_landmarks_left.xlsx']  # it is inside the working directory in original_data and the video in original_videos

   # user_data['original_data'] = 'D:/Silvia/Data/Data_for_vame/BMR10/VAME/BMR10_with_landmarks_left.xlsx' #'U:/Users/Ruthi/2025/BMR10/VAME/BMR10_with_landmarks_left.xlsx'
    user_data['sheetname'] = 'BMR'
    user_data['upper_tube'] = [1269] #for bmr10
    user_data['lower_tube'] = [1454]
    user_data['minimum_x'] = [983]
    user_data['maximum_x'] = [2211]
    user_data['width_mole'] = [579]
    user_data['height_mole'] = [164]
   
    user_data['config'] = '/home/labs/kimchi/cfsilvia/Blind_mole/Motifs_Detection/' + 'BMR-VAME-Project-Jan1-2026' + '/config.yaml' #'U:/Users/Ruthi/2025/BMR10/VAME/' + 'BMR10-VAME-Project-Jul24-2025' + '/config.yaml'
    
    #user_data['labels_file'] = r"U:\Users\Ruthi\2025\BMR10\VAME\BMR10-VAME-Project-Nov19-2025\results\BMR10_with_landmarks_left\VAME\hmm-10\10_km_label_BMR10_with_landmarks_left.npy" #r"U:\Users\Ruthi\2025\BMR10\VAME\BMR10-VAME-Project-Jul24-2025\results\BMR10_with_landmarks_left\VAME\hmm-10\10_km_label_BMR10_with_landmarks_left.npy"
    user_data['fps'] = 24.00
    #user_data['file_etogrhams'] = r"U:\Users\Ruthi\2025\BMR10\VAME\BMR10-VAME-Project-Nov19-2025\results\BMR10_with_landmarks_left\VAME\hmm-10\ethogram_aligned.csv" #r"U:\Users\Ruthi\2025\BMR10\VAME\BMR10-VAME-Project-Jul24-2025\results\BMR10_with_landmarks_left\VAME\hmm-10\ethogram_aligned.csv"
    
    # user_data['directory_results'] = r"D:\Silvia\Data\Data_for_vame\BMR10\VAME\BMR10-VAME-Project-Nov25-2025\results\BMR10_with_landmarks_left\VAME\hmm-10\"
    # user_data['number_clusters'] = 10
    # user_data['name_data'] = "BMR10_with_landmarks_left"

    main_menu(user_data)