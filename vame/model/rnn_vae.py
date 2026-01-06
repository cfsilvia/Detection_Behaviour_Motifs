#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 0.1 Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import torch
from torch import nn
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import os
import numpy as np
from pathlib import Path

from vame.util.auxiliary import read_config
from vame.model.dataloader import SEQUENCE_DATASET
from vame.model.rnn_model import RNN_VAE, RNN_VAE_LEGACY

# make sure torch uses cuda for GPU computing
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")
    print('GPU active:',torch.cuda.is_available())
    print('GPU used:',torch.cuda.get_device_name(0))
else:
    torch.device("cpu")

def reconstruction_loss(x, x_tilde, reduction):
    mse_loss = nn.MSELoss(reduction=reduction)
    rec_loss = mse_loss(x_tilde,x)
    return rec_loss

def future_reconstruction_loss(x, x_tilde, reduction):
    mse_loss = nn.MSELoss(reduction=reduction)
    rec_loss = mse_loss(x_tilde,x)
    return rec_loss

def cluster_loss(H, kloss, lmbda, batch_size):
    gram_matrix = (H.T @ H) / batch_size
    _ ,sv_2, _ = torch.svd(gram_matrix)
    sv = torch.sqrt(sv_2[:kloss])
    loss = torch.sum(sv)
    return lmbda*loss


def kullback_leibler_loss(mu, logvar):
    # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # I'm using torch.mean() here as the sum() version depends on the size of the latent vector
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


def kl_annealing(epoch, kl_start, annealtime, function):
    """
        Annealing of Kullback-Leibler loss to let the model learn first
        the reconstruction of the data before the KL loss term gets introduced.
    """
    if epoch > kl_start:
        if function == 'linear':
            new_weight = min(1, (epoch-kl_start)/(annealtime))

        elif function == 'sigmoid':
            new_weight = float(1/(1+np.exp(-0.9*(epoch-annealtime))))
        else:
            raise NotImplementedError('currently only "linear" and "sigmoid" are implemented')

        return new_weight

    else:
        new_weight = 0
        return new_weight


def gaussian(ins, is_training, seq_len, std_n=0.8):
    if is_training:
        emp_std = ins.std(1)*std_n
        emp_std = emp_std.unsqueeze(2).repeat(1, 1, seq_len)
        emp_std = emp_std.permute(0,2,1)
        noise = Variable(ins.data.new(ins.size()).normal_(0, 1))
        return ins + (noise*emp_std)
    return ins

'''
Training function for VAME model
'''
def train(train_loader, epoch, model, optimizer, anneal_function, BETA, kl_start,
          annealtime, seq_len, future_decoder, future_steps, scheduler, mse_red, 
          mse_pred, kloss, klmbda, bsize, noise):
    model.train()
    
    # Initialize accumulators
    total_train_loss = 0.0
    total_mse_loss = 0.0
    total_kl_loss = 0.0
    total_kmeans_loss = 0.0
    total_fut_loss = 0.0
    
    seq_len_half = int(seq_len / 2)
    device = torch.device("cuda" if use_gpu else "cpu")

    for batch_idx, data_item in enumerate(train_loader):
        # Prepare data: [batch, seq_len, features] -> [batch, features, seq_len]
        data_item = data_item.permute(0, 2, 1).float().to(device)

        data = data_item[:, :seq_len_half, :]
        fut = data_item[:, seq_len_half:seq_len_half+future_steps, :]

        if noise:
            data_input = gaussian(data, True, seq_len_half)
        else:
            data_input = data

        # Forward pass
        if future_decoder:
            data_tilde, future, latent, mu, logvar = model(data_input)
            fut_rec_loss = future_reconstruction_loss(fut, future, mse_pred)
        else:
            data_tilde, latent, mu, logvar = model(data_input)
            fut_rec_loss = torch.tensor(0.0, device=device)

        # Calculate losses
        rec_loss = reconstruction_loss(data, data_tilde, mse_red)
        kl_loss = kullback_leibler_loss(mu, logvar)
        kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
        
        kl_weight = kl_annealing(epoch, kl_start, annealtime, anneal_function)
        
        loss = rec_loss + (BETA * kl_weight * kl_loss) + (kl_weight * kmeans_loss)
        if future_decoder:
            loss += fut_rec_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_train_loss += loss.item()
        total_mse_loss += rec_loss.item()
        total_kl_loss += kl_loss.item()
        total_kmeans_loss += kmeans_loss.item()
        if future_decoder:
            total_fut_loss += fut_rec_loss.item()

    # Calculate averages
    num_batches = len(train_loader)
    avg_train_loss = total_train_loss / num_batches
    avg_mse_loss = total_mse_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    avg_kmeans_loss = total_kmeans_loss / num_batches
    avg_fut_loss = total_fut_loss / num_batches

    # Step scheduler
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(avg_train_loss)
    else:
        scheduler.step()

    # Logging
    weighted_kl = BETA * kl_weight * avg_kl_loss
    weighted_kmeans = kl_weight * avg_kmeans_loss

    if future_decoder:
        print('Train loss: {:.3f}, MSE-Loss: {:.3f}, MSE-Future-Loss {:.3f}, '
              'KL-Loss: {:.3f}, Kmeans-Loss: {:.3f}, weight: {:.2f}'.format(
              avg_train_loss, avg_mse_loss, avg_fut_loss, weighted_kl, weighted_kmeans, kl_weight))
    else:
        print('Train loss: {:.3f}, MSE-Loss: {:.3f}, KL-Loss: {:.3f}, '
              'Kmeans-Loss: {:.3f}, weight: {:.2f}'.format(
              avg_train_loss, avg_mse_loss, weighted_kl, weighted_kmeans, kl_weight))

    return kl_weight, avg_train_loss, weighted_kmeans, avg_kl_loss, avg_mse_loss, avg_fut_loss

'''
test function for VAME model
'''
def test(test_loader, epoch, model, optimizer, BETA, kl_weight, seq_len, mse_red, kloss, klmbda, future_decoder, bsize):
    model.eval() # toggle model to inference mode
    
    # Initialize accumulators
    total_test_loss = 0.0
    total_mse_loss = 0.0
    total_kl_loss = 0.0
    total_kmeans_loss = 0.0
    
    seq_len_half = int(seq_len / 2)
    device = torch.device("cuda" if use_gpu else "cpu")

    with torch.no_grad():
        for data_item in test_loader:
            # Prepare data: [batch, seq_len, features] -> [batch, features, seq_len]
            data_item = data_item.permute(0, 2, 1).float().to(device)
            data = data_item[:, :seq_len_half, :]

            # Forward pass
            if future_decoder:
                recon_images, _, latent, mu, logvar = model(data)
            else:
                recon_images, latent, mu, logvar = model(data)

            # Calculate losses
            rec_loss = reconstruction_loss(data, recon_images, mse_red)
            kl_loss = kullback_leibler_loss(mu, logvar)
            kmeans_loss = cluster_loss(latent.T, kloss, klmbda, bsize)
            
            loss = rec_loss + (BETA * kl_weight * kl_loss) + (kl_weight * kmeans_loss)

            # Accumulate metrics
            total_test_loss += loss.item()
            total_mse_loss += rec_loss.item()
            total_kl_loss += kl_loss.item()
            total_kmeans_loss += kmeans_loss.item()

    # Calculate averages
    num_batches = len(test_loader)
    if num_batches > 0:
        avg_test_loss = total_test_loss / num_batches
        avg_mse_loss = total_mse_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        avg_kmeans_loss = total_kmeans_loss / num_batches
    else:
        avg_test_loss = avg_mse_loss = avg_kl_loss = avg_kmeans_loss = 0.0

    weighted_kl = BETA * kl_weight * avg_kl_loss
    weighted_kmeans = kl_weight * avg_kmeans_loss

    print('Test loss: {:.3f}, MSE-Loss: {:.3f}, KL-Loss: {:.3f}, Kmeans-Loss: {:.3f}'.format(
          avg_test_loss, avg_mse_loss, weighted_kl, weighted_kmeans))

    return avg_mse_loss, avg_test_loss, weighted_kmeans

###########################################################
#-----------------------------Auxiliarty functions of main function --------------------------------------------------------
def _setup_directories(cfg):
    model_path = os.path.join(cfg['project_path'], 'model')
    best_model_path = os.path.join(model_path, 'best_model')
    snapshots_path = os.path.join(best_model_path, 'snapshots')
    losses_path = os.path.join(model_path, 'model_losses')

    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path, exist_ok=True)
        os.makedirs(snapshots_path, exist_ok=True)
        os.makedirs(losses_path, exist_ok=True)

def _get_device():
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")
        print('GPU active:', torch.cuda.is_available())
        print('GPU used: ', torch.cuda.get_device_name(0))
        return torch.device("cuda")
    else:
        print("warning, a GPU was not found... proceeding with CPU (slow!) \n")
        return torch.device("cpu")

def _init_model(cfg, device):
    legacy = cfg['legacy']
    temporal_window = cfg['time_window'] * 2
    zdims = cfg['zdims']
    fixed = cfg['egocentric_data']
    num_features = cfg['num_features']
    # if not fixed:
    #     num_features -= 2
    
    future_decoder = cfg['prediction_decoder']
    future_steps = cfg['prediction_steps']
    
    # RNN params
    hidden_size_layer_1 = cfg['hidden_size_layer_1']
    hidden_size_layer_2 = cfg['hidden_size_layer_2']
    hidden_size_rec = cfg['hidden_size_rec']
    hidden_size_pred = cfg['hidden_size_pred']
    dropout_encoder = cfg['dropout_encoder']
    dropout_rec = cfg['dropout_rec']
    dropout_pred = cfg['dropout_pred']
    softplus = cfg['softplus']

    RNN = RNN_VAE_LEGACY if legacy else RNN_VAE
    
    model = RNN(temporal_window, zdims, num_features, future_decoder, future_steps, 
                hidden_size_layer_1, hidden_size_layer_2, hidden_size_rec, 
                hidden_size_pred, dropout_encoder, dropout_rec, dropout_pred, softplus)
    
    return model.to(device)

def _load_pretrained_weights(model, cfg):
    if cfg['pretrained_weights']:
        pretrained_model = cfg['pretrained_model']
        project_path = cfg['project_path']
        project_name = cfg['Project']
        
        path1 = os.path.join(project_path, 'model', 'best_model', f"{pretrained_model}_{project_name}.pkl")
        
        if os.path.exists(path1):
             print(f"Loading pretrained weights from model: {path1}\n")
             model.load_state_dict(torch.load(path1))
             return 0, 1 # KL_START, ANNEALTIME
        elif os.path.exists(pretrained_model):
             print(f"Loading pretrained weights from {pretrained_model}\n")
             model.load_state_dict(torch.load(pretrained_model))
             return 0, 1
        else:
             print("Could not load pretrained model. Check file path in config.yaml.")
    
    return cfg['kl_start'], cfg['annealtime']

def _get_dataloaders(cfg, temporal_window):
    train_batch_size = cfg['batch_size']
    test_batch_size = int(cfg['batch_size']/4)
    normalize = cfg.get('normalize_data', True)
    
    trainset = SEQUENCE_DATASET(os.path.join(cfg['project_path'],"data", "train",""), data='train_seq.npy', train=True, temporal_window=temporal_window, normalize=normalize)
    testset = SEQUENCE_DATASET(os.path.join(cfg['project_path'],"data", "train",""), data='test_seq.npy', train=False, temporal_window=temporal_window, normalize=normalize)

    train_loader = Data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, drop_last=True)
    test_loader = Data.DataLoader(testset, batch_size=test_batch_size, shuffle=True, drop_last=True)
    
    return train_loader, test_loader, train_batch_size, test_batch_size

def _save_logs(project_path, model_name, logs, current_test_loss):
    base = os.path.join(project_path, 'model', 'model_losses')
    np.save(os.path.join(base, f'train_losses_{model_name}'), logs['train_losses'])
    np.save(os.path.join(base, f'test_losses_{model_name}'), logs['test_losses'])
    np.save(os.path.join(base, f'kmeans_losses_{model_name}'), logs['kmeans_losses'])
    np.save(os.path.join(base, f'kl_losses_{model_name}'), logs['kl_losses'])
    np.save(os.path.join(base, f'weight_values_{model_name}'), logs['weight_values'])
    np.save(os.path.join(base, f'mse_train_losses_{model_name}'), logs['mse_losses'])
    np.save(os.path.join(base, f'mse_test_losses_{model_name}'), current_test_loss)
    np.save(os.path.join(base, f'fut_losses_{model_name}'), logs['fut_losses'])

#--------------------------------------------Main Function -----------------------------------------------------
'''
input: config file 
output: trained model
'''
def train_model(config):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    
    model_name = cfg['model_name']
    print("Train Variational Autoencoder - model name: %s \n" % model_name)
    
    _setup_directories(cfg)
    device = _get_device()
    
    # For reproducibility between run and run
    SEED = 19
    torch.manual_seed(SEED)
    if device.type == 'cuda':
        torch.cuda.manual_seed(SEED)

    temporal_window = cfg['time_window'] * 2
    
    # Initialize Model
    model = _init_model(cfg, device)
    
    # Load Pretrained Weights if wants a pretrained model
    kl_start, annealtime = _load_pretrained_weights(model, cfg)
    
    # Data Loaders
    train_loader, test_loader, train_batch_size, test_batch_size = _get_dataloaders(cfg, temporal_window)
    
    # Optimizer & Scheduler
    learning_rate = cfg['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    
    scheduler_step_size = cfg['scheduler_step_size']
    if cfg['scheduler']:
        print('Scheduler step size: %d, Scheduler gamma: %.2f\n' %(scheduler_step_size, cfg['scheduler_gamma']))
        # Thanks to @alexcwsmith for the optimized scheduler contribution
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=cfg['scheduler_gamma'], patience=cfg['scheduler_step_size'], threshold=1e-3, threshold_mode='rel')
        #scheduler = ReduceLROnPlateau(optimizer, 'min', factor=cfg['scheduler_gamma'], patience=cfg['scheduler_step_size'], threshold=1e-3, threshold_mode='rel', verbose=True)
    else:
        scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=1, last_epoch=-1)
        
    # Training Loop Variables
    epochs = cfg['max_epochs']
    beta = cfg['beta']
    snapshot_interval = cfg['model_snapshot']
    
    # Loss params
    mse_rec_red = cfg['mse_reconstruction_reduction']
    mse_pred_red = cfg['mse_prediction_reduction']
    kmeans_loss = cfg['kmeans_loss']
    kmeans_lambda = cfg['kmeans_lambda']
    anneal_function = cfg['anneal_function']
    noise = cfg['noise']
    
    future_decoder = cfg['prediction_decoder']
    future_steps = cfg['prediction_steps']
    
    print(f'Latent Dimensions: {cfg["zdims"]}, Time window: {cfg["time_window"]}, Batch Size: {train_batch_size}, Beta: {beta}, lr: {learning_rate:.4f}\n')
    
    # Logging
    logs = {
        'train_losses': [], 'test_losses': [], 'kmeans_losses': [], 'kl_losses': [],
        'weight_values': [], 'mse_losses': [], 'fut_losses': []
    }
    
    best_loss = 999999
    convergence = 0
    
    print("Start training... ")
    for epoch in range(1, epochs):
        print("Epoch: %d" % epoch)
        
        # Train
        weight, train_loss, km_loss, kl_loss, mse_loss, fut_loss = train(
            train_loader, epoch, model, optimizer, anneal_function, beta, kl_start,
            annealtime, temporal_window, future_decoder, future_steps, scheduler,
            mse_rec_red, mse_pred_red, kmeans_loss, kmeans_lambda, train_batch_size, noise
        )
        
        # Test
        current_loss, test_loss, _ = test(
            test_loader, epoch, model, optimizer, beta, weight, temporal_window,
            mse_rec_red, kmeans_loss, kmeans_lambda, future_decoder, test_batch_size
        )
        
        # Update logs
        logs['train_losses'].append(train_loss)
        logs['test_losses'].append(test_loss)
        logs['kmeans_losses'].append(km_loss)
        logs['kl_losses'].append(kl_loss)
        logs['weight_values'].append(weight)
        logs['mse_losses'].append(mse_loss)
        logs['fut_losses'].append(fut_loss)
        
        # Save Best Model
        if weight > 0.99 and current_loss <= best_loss:
            best_loss = current_loss
            print("Saving model!")
            torch.save(model.state_dict(), os.path.join(cfg['project_path'], "model", "best_model", f"{model_name}_{cfg['Project']}.pkl"))
            convergence = 0
        else:
            convergence += 1
            
        # Save Snapshot
        if epoch % snapshot_interval == 0:
            print("Saving model snapshot!\n")
            torch.save(model.state_dict(), os.path.join(cfg['project_path'], 'model', 'best_model', 'snapshots', f"{model_name}_{cfg['Project']}_epoch_{epoch}.pkl"))
            
        # Check Convergence
        if convergence > cfg['model_convergence']:
            print('Finished training...')
            print('Model converged. Please check your model with vame.evaluate_model(). \n'
                  'You can also re-run vame.trainmodel() to further improve your model. \n'
                  'Make sure to set _pretrained_weights_ in your config.yaml to "true" \n'
                  'and plug your current model name into _pretrained_model_. \n'
                  'Hint: Set "model_convergence" in your config.yaml to a higher value. \n'
                  '\n'
                  'Next: \n'
                  'Use vame.pose_segmentation() to identify behavioral motifs in your dataset!')
            break
        
        # Save Logs to disk
        _save_logs(cfg['project_path'], model_name, logs, current_loss)
        
        print("\n")

    if convergence < cfg['model_convergence']:
        print('Finished training...')
        print('Model seems to have not reached convergence. You may want to check your model \n'
              'with vame.evaluate_model(). If your satisfied you can continue. \n'
              'Use vame.pose_segmentation() to identify behavioral motifs! \n'
              'OPTIONAL: You can re-run vame.train_model() to improve performance.')
