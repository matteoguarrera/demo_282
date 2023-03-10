root_folder = ""
import os
import sys
import inspect
sys.path.append(root_folder)
from collections import Counter
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import random
import numpy as np
import json

import matplotlib.pyplot as plt
# from utils import validate_to_array

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import IPython
from ipywidgets import interactive, widgets, Layout
from IPython.display import display, HTML



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
