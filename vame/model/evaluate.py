#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 0.1 Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import torch.utils.data as Data

from vame.util.auxiliary import read_config
from vame.model.rnn_vae import RNN_VAE
from vame.model.dataloader import SEQUENCE_DATASET

def plot_reconstruction(filepath, test_loader, seq_len_half, model, model_name,
                        FUTURE_DECODER, FUTURE_STEPS, suffix=None, device=torch.device('cpu')):
    #x = test_loader.__iter__().next()
    dataiter = iter(test_loader)
    x = next(dataiter)
    x = x.permute(0,2,1).float().to(device)
    data = x[:,:seq_len_half,:]
    data_fut = x[:,seq_len_half:seq_len_half+FUTURE_STEPS,:]
    if FUTURE_DECODER:
        x_tilde, future, latent, mu, logvar = model(data)

        fut_orig = data_fut.cpu()
        fut_orig = fut_orig.data.numpy()
        fut = future.cpu()
        fut = fut.detach().numpy()

    else:
        x_tilde, latent, mu, logvar = model(data)

    data_orig = data.cpu()
    data_orig = data_orig.data.numpy()
    data_tilde = x_tilde.cpu()
    data_tilde = data_tilde.detach().numpy()

    if FUTURE_DECODER:
        fig, axs = plt.subplots(2, 5)
        fig.suptitle('Reconstruction [top] and future prediction [bottom] of input sequence')
        for i in range(5):
            axs[0,i].plot(data_orig[i,:,49], color='k', label='Sequence Data')
            axs[0,i].plot(data_tilde[i,:,49], color='r', linestyle='dashed', label='Sequence Reconstruction')

            axs[1,i].plot(fut_orig[i,:,49], color='k')
            axs[1,i].plot(fut[i,:,49], color='r', linestyle='dashed')
        axs[0,0].set(xlabel='time steps', ylabel='reconstruction')
        axs[1,0].set(xlabel='time steps', ylabel='predction')
        fig.savefig(os.path.join(filepath,"evaluate",'Future_Reconstruction.png'))

    else:
        fig, ax1 = plt.subplots(1, 5)
        for i in range(5):
            fig.suptitle('Reconstruction of input sequence')
            ax1[i].plot(data_orig[i,:,49], color='k', label='Sequence Data')
            ax1[i].plot(data_tilde[i,:,49], color='r', linestyle='dashed', label='Sequence Reconstruction')
        fig.set_tight_layout(True)
        if not suffix:
            fig.savefig(os.path.join(filepath,'evaluate','Reconstruction_'+model_name+'.png'), bbox_inches='tight')
        elif suffix:
            fig.savefig(os.path.join(filepath,'evaluate','Reconstruction_'+model_name+'_'+suffix+'.png'), bbox_inches='tight')


def plot_loss(cfg, filepath, model_name):
    basepath = os.path.join(cfg['project_path'],"model","model_losses")
    train_loss = np.load(os.path.join(basepath,'train_losses_'+model_name+'.npy'))
    test_loss = np.load(os.path.join(basepath,'test_losses_'+model_name+'.npy'))
    mse_loss_train = np.load(os.path.join(basepath,'mse_train_losses_'+model_name+'.npy'))
    mse_loss_test = np.load(os.path.join(basepath,'mse_test_losses_'+model_name+'.npy'))
#    km_loss = np.load(os.path.join(basepath,'kmeans_losses_'+model_name+'.npy'), allow_pickle=True)
    km_losses = np.load(os.path.join(basepath,'kmeans_losses_'+model_name+'.npy'))
    kl_loss = np.load(os.path.join(basepath,'kl_losses_'+model_name+'.npy'))
    fut_loss = np.load(os.path.join(basepath,'fut_losses_'+model_name+'.npy'))

#    km_losses = []
#    for i in range(len(km_loss)):
#        km = km_loss[i].cpu().detach().numpy()
#        km_losses.append(km)

    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle('Losses of our Model')
    ax1.set(xlabel='Epochs', ylabel='loss [log-scale]')
    ax1.set_yscale("log")
    ax1.plot(train_loss, label='Train-Loss')
    ax1.plot(test_loss, label='Test-Loss')
    ax1.plot(mse_loss_train, label='MSE-Train-Loss')
    ax1.plot(mse_loss_test, label='MSE-Test-Loss')
    ax1.plot(km_losses, label='KMeans-Loss')
    ax1.plot(kl_loss, label='KL-Loss')
    ax1.plot(fut_loss, label='Prediction-Loss')
    ax1.legend()
    #fig.savefig(filepath+'evaluate/'+'MSE-and-KL-Loss'+model_name+'.png')
    fig.savefig(os.path.join(filepath,"evaluate",'MSE-and-KL-Loss'+model_name+'.png'))

def calculate_correlation(cfg, X_norm, full_recon, model_name, suffix=None, columns_names=None):
    print("Calculating correlation and RMSE between original and reconstructed data...")
    correlations = []
    rmses = []
    num_features = X_norm.shape[0]
    
    for i in range(num_features):
        # Check for constant arrays to avoid RuntimeWarning/NaNs
        if np.std(X_norm[i]) == 0 or np.std(full_recon[i]) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(X_norm[i], full_recon[i])[0, 1]
        correlations.append(corr)

        rmse = np.sqrt(np.mean((X_norm[i] - full_recon[i])**2))
        rmses.append(rmse)
        
    df = pd.DataFrame({'Feature_Index': range(num_features), 'Correlation': correlations, 'RMSE': rmses, 'Feature_Name': columns_names if columns_names else None})
    
    name = f"Correlation_{model_name}" + (f"_{suffix}" if suffix else "")
    save_path = os.path.join(cfg['project_path'], "model", "evaluate", name + ".xlsx")
    
    df.to_excel(save_path, index=False)
    print(f"Saved correlation and RMSE results to {save_path}")

#reconstruct full sequence 
'''
the idea to model in chunks to 30 frames and then to reconstruct the full sequence 
'''
def reconstruct_full_sequence(cfg, model, model_name, device, suffix=None):
    print("Reconstructing full sequence...")
    path_to_file = os.path.join(cfg['project_path'], "data", "train")
    data_path = os.path.join(path_to_file, 'train_seq.npy')
    
    if not os.path.exists(data_path):
        print(f"File {data_path} not found.")
        return

    X = np.load(data_path)
    if X.shape[0] > X.shape[1]:
        X = X.T
        
    mean_path = os.path.join(path_to_file, 'seq_mean.npy')
    std_path = os.path.join(path_to_file, 'seq_std.npy')
    normalize = cfg.get('normalize_data', True)
    if normalize and os.path.exists(mean_path) and os.path.exists(std_path):
        mean = np.load(mean_path)
        std = np.load(std_path)
        X_norm = (X - mean) / std
    else:
        X_norm = X

    seq_len = cfg['time_window']
    num_features = X_norm.shape[0]
    num_frames = X_norm.shape[1]
    
    remainder = num_frames % seq_len
    pad_width = seq_len - remainder if remainder != 0 else 0
    #fill with zeros the end 
    X_padded = np.pad(X_norm, ((0,0), (0, pad_width)), mode='constant') if pad_width > 0 else X_norm
        
    X_T = X_padded.T
    num_chunks = X_T.shape[0] // seq_len #number of full sequences of length seq_len
    chunks = X_T.reshape(num_chunks, seq_len, num_features)
    chunks_tensor = torch.from_numpy(chunks).float().to(device)
    
    reconstructions = []
    batch_size = 128
    
    model.eval()
    with torch.no_grad():
        for i in range(0, num_chunks, batch_size):
            batch = chunks_tensor[i:i+batch_size]
            recon = model(batch)[0]
            reconstructions.append(recon.cpu().numpy())
            
    full_recon = np.concatenate(reconstructions, axis=0)
    full_recon = full_recon.reshape(-1, num_features).T
    if pad_width > 0:
        full_recon = full_recon[:, :-pad_width]
        
    name = f"Full_Reconstruction_{model_name}" + (f"_{suffix}" if suffix else "")
    np.save(os.path.join(cfg['project_path'], "model", "evaluate", name + ".npy"), full_recon)
    print(f"Saved full reconstruction to {name}.npy")
    #add names of features
    columns_names = extract_landmarks_names(cfg)
    #save correlation between original and reconstructed data
    calculate_correlation(cfg, X_norm, full_recon, model_name, suffix, columns_names)
    # Plot example feature reconstruction
    

    feature_idx = 49
    if feature_idx < num_features:
        plt.figure(figsize=(20, 5))
        plt.plot(X_norm[feature_idx, :], 'k', label='Original', alpha=0.5, linewidth=0.5)
        plt.plot(full_recon[feature_idx, :], 'r--', label='Reconstruction', alpha=0.5, linewidth=0.5)
        plt.title(f'Full Reconstruction Feature {feature_idx}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(cfg['project_path'], "model", "evaluate", name + ".tif"), dpi=300)
        plt.close()

'''
extract features names
'''
def extract_landmarks_names(cfg):
        fname = cfg['video_sets']
        header = pd.read_csv(os.path.join(cfg['project_path'],"videos","pose_estimation",fname[0] +'.csv'), nrows=0)
        columns_names = header.columns.tolist()[1:]  # Skip the first column which is usually an index or frame number
        #remove the columns which its name contains the name score
        columns_names = [name for name in columns_names if 'score' not in name.lower()]
        return columns_names

'''
evaluation of partial temporal model
'''

def eval_temporal(cfg, use_gpu, model_name, snapshot=None, suffix=None):
    SEED = 19
    ZDIMS = cfg['zdims']
    FUTURE_DECODER = cfg['prediction_decoder']
    TEMPORAL_WINDOW = cfg['time_window']*2
    FUTURE_STEPS = cfg['prediction_steps']
    NUM_FEATURES = cfg['num_features']
    TEST_BATCH_SIZE = 64
    hidden_size_layer_1 = cfg['hidden_size_layer_1']
    hidden_size_layer_2 = cfg['hidden_size_layer_2']
    hidden_size_rec = cfg['hidden_size_rec']
    hidden_size_pred = cfg['hidden_size_pred']
    dropout_encoder = cfg['dropout_encoder']
    dropout_rec = cfg['dropout_rec']
    dropout_pred = cfg['dropout_pred']
    softplus = cfg['softplus']
    normalize = cfg.get('normalize_data', True)

    filepath = os.path.join(cfg['project_path'],"model")

    device = torch.device("cuda" if use_gpu else "cpu")
    seq_len_half = int(TEMPORAL_WINDOW/2)

    if use_gpu:
        torch.cuda.manual_seed(SEED)

    model = RNN_VAE(TEMPORAL_WINDOW,ZDIMS,NUM_FEATURES,FUTURE_DECODER,FUTURE_STEPS, hidden_size_layer_1,
                    hidden_size_layer_2, hidden_size_rec, hidden_size_pred, dropout_encoder,
                    dropout_rec, dropout_pred, softplus).to(device)
    
    load_path = snapshot if snapshot else os.path.join(cfg['project_path'],"model","best_model",model_name+'_'+cfg['Project']+'.pkl')
    map_location = None if use_gpu else device
    model.load_state_dict(torch.load(load_path, map_location=map_location))
    
    model.eval() #toggle evaluation mode
    #change to train data instead of test data
    testset = SEQUENCE_DATASET(os.path.join(cfg['project_path'],"data", "train",""), data='train_seq.npy', train=False, temporal_window=TEMPORAL_WINDOW, normalize=normalize)
    test_loader = Data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True)

    plot_reconstruction(filepath, test_loader, seq_len_half, model, model_name, FUTURE_DECODER, FUTURE_STEPS, suffix=suffix, device=device)
    plot_loss(cfg, filepath, model_name)
    reconstruct_full_sequence(cfg, model, model_name, device, suffix)


def evaluate_model(config, use_snapshots=False):
    """
    Evaluation of testset.
        
    Parameters
    ----------
    config : str
        Path to config file.
    model_name : str
        name of model (same as in config.yaml)
    use_snapshots : bool
        Whether to plot for all snapshots or only the best model.
    """
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']

    if not os.path.exists(os.path.join(cfg['project_path'],"model","evaluate")):
        os.mkdir(os.path.join(cfg['project_path'],"model","evaluate"))
    evaluate_path = os.path.join(cfg['project_path'], "model", "evaluate")
    os.makedirs(evaluate_path, exist_ok=True)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")
        print('GPU active:',torch.cuda.is_available())
        print('GPU used:',torch.cuda.get_device_name(0))
    else:
        torch.device("cpu")
        print("CUDA is not working, or a GPU is not found; using CPU!")

    print("\n\nEvaluation of %s model. \n" %model_name)   
    if not use_snapshots:
        eval_temporal(cfg, use_gpu, model_name)
    elif use_snapshots:
        snapshots=os.listdir(os.path.join(cfg['project_path'],'model','best_model','snapshots'))
        for snap in snapshots:
            fullpath = os.path.join(cfg['project_path'],"model","best_model","snapshots",snap)
            epoch=snap.split('_')[-1]
            eval_temporal(cfg, use_gpu, model_name, snapshot=fullpath, suffix='snapshot'+str(epoch))

    print(f"You can find the results of the evaluation in '{evaluate_path}' \n"
          "OPTIONS:\n"
          "- vame.pose_segmentation() to identify behavioral motifs.\n"
          "- re-run the model for further fine tuning. Check again with vame.evaluate_model()")
