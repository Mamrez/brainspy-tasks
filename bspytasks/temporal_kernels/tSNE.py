from time import time

import numpy as np
import pandas as pd
import torch
import os
import librosa
import sklearn

# For plotting
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import plotly.graph_objects as go

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

#For standardising the dat
from sklearn.preprocessing import StandardScaler

#PCA
from sklearn.manifold import TSNE

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import DataLoader, Dataset, random_split
from itertools import chain
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from main_with_dnpu_preprocess_ti46 import post_dnpu_down_sample

if __name__ == '__main__':

    # loading raw_data
    empty = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_in_house_arsenic/empty/"
    dataset, label = [], []
    train = True

    # Loading dnpu pre-processed data
    projection_idx = 26

    dnpu_preprocessed_data_dir    = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti46/in_elec_4/train_data.npy"
    dnpu_preprocessed_label_dir= "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti46/in_elec_4/train_labels.npy"

    data_dnpu_preprocessed = post_dnpu_down_sample(
        np.load(
            dnpu_preprocessed_data_dir
        )
    )
    label_dnpu_preprocessed = np.load(
        dnpu_preprocessed_label_dir
    )

    x_dnpu = np.zeros((len(data_dnpu_preprocessed), 500))
    y_dnpu = np.zeros((len(data_dnpu_preprocessed)))

    for i in range(len(x_dnpu)):
        x_dnpu[i] = data_dnpu_preprocessed[i][projection_idx]
        y_dnpu[i] = label_dnpu_preprocessed[i]

    # mean std normalizing
    x_dnpu = StandardScaler().fit_transform(x_dnpu)

    # Choose here between raw or dnpu pre-processed data
    tsne_1 = TSNE(n_components=2,verbose=0, perplexity=5, n_iter=1000).fit_transform(x_dnpu)
    # tsne_2 = TSNE(n_components=2,verbose=0, perplexity=10, n_iter=1000).fit_transform(x_dnpu)
    # tsne_3 = TSNE(n_components=2,verbose=0, perplexity=30, n_iter=1000).fit_transform(x_dnpu)
    # tsne_4 = TSNE(n_components=2,verbose=0, perplexity=50, n_iter=1000).fit_transform(x_dnpu)


    # fig, axes = plt.subplots(2, 2)

    # axes[0, 0].scatter(tsne_1[:, 0], tsne_1[:, 1], s= 80, c=y_dnpu, cmap='Spectral')
    # axes[0, 1].scatter(tsne_2[:, 0], tsne_2[:, 1], s= 80, c=y_dnpu, cmap='Spectral')
    # axes[1, 0].scatter(tsne_3[:, 0], tsne_3[:, 1], s= 80, c=y_dnpu, cmap='Spectral')
    # axes[1, 1].scatter(tsne_4[:, 0], tsne_4[:, 1], s= 80, c=y_dnpu, cmap='Spectral')

    # plt.gca().set_aspect('equal', 'datalim')
    # plt.colorbar(boundaries=np.arange(11)-0.5, ax=axes[0]).set_ticks(np.arange(10))
    
    # plt.show()
    # print("")


    plt.scatter(tsne_1[:, 0], tsne_1[:, 1], s= 80, c=y_dnpu, cmap='Spectral')
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries = np.arange(11)-0.5).set_ticks(np.arange(10))

    plt.show()
