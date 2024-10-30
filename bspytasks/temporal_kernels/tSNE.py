from time import time

import numpy as np
import pandas as pd
import torch
import os
import librosa
import sklearn

# For plotting
# from matplotlib import offsetbox
import matplotlib.pyplot as plt
# import matplotlib.patheffects as PathEffects
import seaborn as sns
# import plotly.graph_objects as go

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

#For standardising the dat
from sklearn.preprocessing import StandardScaler

from main_with_dnpu_preprocess_ti46 import load_audio_dataset

#PCA
from sklearn.manifold import TSNE
import umap

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import DataLoader, Dataset, random_split
from itertools import chain
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from main_with_dnpu_preprocess_ti46 import post_dnpu_down_sample

def plot_raw_dataset_tsne():
    empty = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_in_house_arsenic/empty/"

    dataset, label = load_audio_dataset(
        (empty, "C:/Users/Mohamadreza/Documents/ti_spoken_digits/female_speaker"),
        min_max_scale= True,
        low_pass_filter= False
    )

    np_dataset = np.zeros((2075, 12500))
    np_label = np.zeros((2075))
    for i in range(len(dataset)):
        np_dataset[i][0:len(dataset[i])] = dataset[i]
        np_label[i] = label[i]

    # tsne_1 = TSNE(n_components=2,verbose=0, perplexity=50, n_iter=5000).fit_transform(np_dataset)
    # plt.figure()
    # plt.scatter(tsne_1[:, 0], tsne_1[:, 1], s= 80, c=np_label, cmap='Spectral')
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.colorbar(boundaries = np.arange(11)-0.5).set_ticks(np.arange(10))
        
    trans = umap.UMAP().fit(np_dataset)

    plt.figure()
    plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s= 80, c=np_label, cmap='Spectral')
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries = np.arange(11)-0.5).set_ticks(np.arange(10))

    plt.show()

    
def plot_with_umap():

    dnpu_preprocessed_data_dir  = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_digits/boron_8_electrode_35nm_etched/data.npy"
    dnpu_preprocessed_label_dir = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_digits/boron_8_electrode_35nm_etched/labels.npy"

    n_samples = 500
    data_dnpu_preprocessed = post_dnpu_down_sample(
        np.load(
            dnpu_preprocessed_data_dir
        ),
        n_samples = n_samples
    )
    label_dnpu_preprocessed = np.load(
        dnpu_preprocessed_label_dir
    )

    for projection_idx in range(0, 63):
        x_dnpu = np.zeros((len(data_dnpu_preprocessed), n_samples))
        y_dnpu = np.zeros((len(data_dnpu_preprocessed)))
        for i in range(len(x_dnpu)):
            x_dnpu[i] = data_dnpu_preprocessed[i][projection_idx]
            y_dnpu[i] = label_dnpu_preprocessed[i]

        # mean std normalizing
        x_dnpu = StandardScaler().fit_transform(x_dnpu)

        trans = umap.UMAP(n_neighbors=10, min_dist=0.1).fit(x_dnpu)

        plt.figure()
        plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s= 80, c=y_dnpu, cmap='Spectral')
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(boundaries = np.arange(11)-0.5).set_ticks(np.arange(10))

    plt.show()

    print("")





if __name__ == '__main__':

    # plot_with_umap()

    plot_raw_dataset_tsne()

    # loading raw_data
    empty = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_in_house_arsenic/empty/"
    dataset, label = [], []
    train = True

    # Loading dnpu pre-processed data
    projection_idx = 31

    dnpu_preprocessed_data_dir  = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_digits/boron_8_electrode_35nm_etched/data.npy"
    dnpu_preprocessed_label_dir = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_digits/boron_8_electrode_35nm_etched/labels.npy"

    n_samples = 1250
    data_dnpu_preprocessed = post_dnpu_down_sample(
        np.load(
            dnpu_preprocessed_data_dir
        ),
        n_samples = n_samples
    )
    label_dnpu_preprocessed = np.load(
        dnpu_preprocessed_label_dir
    )

    data_dnpu_preprocessed = data_dnpu_preprocessed[0:250][:][:]

    for projection_idx in range(0, 63):
        x_dnpu = np.zeros((len(data_dnpu_preprocessed), n_samples))
        y_dnpu = np.zeros((len(data_dnpu_preprocessed)))
        for i in range(len(x_dnpu)):
            x_dnpu[i] = data_dnpu_preprocessed[i][projection_idx]
            y_dnpu[i] = label_dnpu_preprocessed[i]

        # mean std normalizing
        x_dnpu = StandardScaler().fit_transform(x_dnpu)

        # Choose here between raw or dnpu pre-processed data
        # tsne_1 = TSNE(n_components=2,verbose=0, perplexity=5, n_iter=1000).fit_transform(x_dnpu)
        # tsne_1 = TSNE(n_components=2,verbose=0, perplexity=10, n_iter=1000).fit_transform(x_dnpu)
        # tsne_1 = TSNE(n_components=2,verbose=0, perplexity=30, n_iter=5000).fit_transform(x_dnpu)
        tsne_1 = TSNE(n_components=2,verbose=0, perplexity=50, n_iter=5000).fit_transform(x_dnpu)
        # tsne_1 = TSNE(n_components=2,verbose=0, perplexity=100, n_iter=6000).fit_transform(x_dnpu)


        # fig, axes = plt.subplots(2, 2)

        # axes[0, 0].scatter(tsne_1[:, 0], tsne_1[:, 1], s= 80, c=y_dnpu, cmap='Spectral')
        # axes[0, 1].scatter(tsne_2[:, 0], tsne_2[:, 1], s= 80, c=y_dnpu, cmap='Spectral')
        # axes[1, 0].scatter(tsne_3[:, 0], tsne_3[:, 1], s= 80, c=y_dnpu, cmap='Spectral')
        # axes[1, 1].scatter(tsne_4[:, 0], tsne_4[:, 1], s= 80, c=y_dnpu, cmap='Spectral')

        # plt.gca().set_aspect('equal', 'datalim')
        # plt.colorbar(boundaries=np.arange(11)-0.5, ax=axes[0]).set_ticks(np.arange(10))
        
        # plt.show()
        # print("")

        plt.figure()
        plt.scatter(tsne_1[:, 0], tsne_1[:, 1], s= 80, c=y_dnpu, cmap='Spectral')
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(boundaries = np.arange(11)-0.5).set_ticks(np.arange(10))

    plt.show()

    print("")
