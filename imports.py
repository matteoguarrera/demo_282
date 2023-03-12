root_folder = ""
import os
import sys
# import inspect
sys.path.append(root_folder)
# from collections import Counter
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import random
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import IPython
from ipywidgets import interactive, widgets, Layout
from IPython.display import display, HTML

from demo_282.RNN import RecurrentClassifier 

####################################################################################
# Visualizations 
# Could be an idea inserting a slider to move t_warmup
def plot_norms(out_GT, out_rnn, out_kf, t_warmup = 0):

    differences_kf = (out_GT - out_kf).squeeze()[:,t_warmup:]
    differences_rnn = (out_GT - out_rnn).squeeze()[:,t_warmup:]

    # norms
    norms_kf = np.array([np.linalg.norm(differences_kf,ord=1,axis=1), 
                        np.linalg.norm(differences_kf,ord=2,axis=1), 
                        np.linalg.norm(differences_kf,ord=np.inf,axis=1)]).T

    norms_rnn = np.array([np.linalg.norm(differences_rnn,ord=1,axis=1), 
                          np.linalg.norm(differences_rnn,ord=2,axis=1), 
                          np.linalg.norm(differences_rnn,ord=np.inf,axis=1)]).T

    fig, ax = plt.subplots(1,4, figsize = (20,4))
    ax[0].plot(differences_kf[5], label='kalman') # Select and arbitrary run
    ax[0].plot(differences_rnn[5], label='rnn') # Select and arbitrary run
    ax[0].legend()
    for i in range(3):
      ax[i+1].hist(norms_kf[:,i], bins='auto', label='kalman', alpha = 0.5)
      ax[i+1].hist(norms_rnn[:,i], bins='auto', label='rnn', alpha = 0.5)
      ax[i+1].legend()


    ax[0].title.set_text('[Sample run] Actual Differences')
    ax[1].title.set_text('1 norm')
    ax[2].title.set_text('2 norm')
    ax[3].title.set_text('inf norm')



####################################################################################
# Train

def train(hyperparam_dict, seed, TRAIN_DATA, TEST_DATA):
  
  fig, ax = plt.subplots(1,1)
  batch_size = hyperparam_dict['batch_size']
  device = hyperparam_dict['device']
  criterion = hyperparam_dict['criterion']

  model = RecurrentClassifier(num_layers = 1, hidden_size = hyperparam_dict['hidden_size'], 
                                 model_type = hyperparam_dict['model_type'])
  
  list_to_device = lambda th_obj: [tensor.to(device) for tensor in th_obj]
  
  optimizer = optim.Adam(model.parameters(), 
                        lr=hyperparam_dict['lr'], 
                        weight_decay=hyperparam_dict['weight_decay'])    # use adam optimizer

  model.to(device)
  model.train()
  train_losses = []
  valid_losses = []
  for epoch in range(hyperparam_dict['epochs']):
      random.seed(seed + epoch)
      np.random.seed(seed + epoch)
      torch.manual_seed(seed + epoch)
      n_correct, n_total = 0, 0
      progress_bar = tqdm(range(0, len(TRAIN_DATA)//batch_size))
      for i in progress_bar: 

          batch_X = TRAIN_DATA[batch_size*i:batch_size*(i+1), :, 0].unsqueeze(-1)
          batch_Y = TRAIN_DATA[batch_size*i:batch_size*(i+1), :, 1].unsqueeze(-1)

          logits = model(batch_X)
          loss = criterion(logits, batch_Y)
          train_losses.append(loss.item())

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          # wandb.log({"train loss": loss})

          if (i + 1) % 10 == 0:
              progress_bar.set_description(f"Epoch: {epoch:2d} Loss: {np.mean(train_losses[-10:]):.3f}")
        
      ax.plot(train_losses);
      ax.set_xlabel('iterations')
      ax.set_ylabel('loss')
      ax.grid()
      
  with torch.no_grad():
      progress_bar = tqdm(range(0, len(TEST_DATA)//batch_size))    
      for i in progress_bar:
          batch_X = TEST_DATA[batch_size*i:batch_size*(i+1), :, 0].unsqueeze(-1)
          batch_Y = TEST_DATA[batch_size*i:batch_size*(i+1), :, 1].unsqueeze(-1)

          logits = model(batch_X)
          loss = criterion(logits, batch_Y)
          valid_losses.append(loss.item())

          if (i + 1) % 10 == 0:
              progress_bar.set_description(f"[{i}]")

#       ax2.plot(valid_losses)
  # wandb.log({"eval accuracy": np.mean(valid_losses)})

      
  print(f"Eval Accuracy: {np.mean(valid_losses)}")
  
  # Mark the run as finished
  # wandb.finish()
  
  return train_losses, valid_losses, model
