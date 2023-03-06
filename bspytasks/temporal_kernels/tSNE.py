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
    # data_dir = ("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spoken_digit_task/spoken_mnist/spoken_mnist/train", empty)
    # for subdir, _, files in chain.from_iterable(
    #     os.walk(path) for path in data_dir
    # ):
    #     for file in files:
    #         tmp, _ = librosa.load(os.path.join(subdir, file), sr=None, dtype=np.float32)
    #         if train == True:
    #             if subdir[-5:] == "train":
    #                 dataset.append(tmp)
    #                 label.append(file[0])
    #         elif train == False:
    #             if subdir[-4:] == "test":
    #                 dataset.append(tmp)
    #                 label.append(file[0])
    # x_raw = np.zeros((len(dataset), 8000))
    # y_raw = np.zeros((len(dataset)))
    # for i in range(len(dataset)):
    #     x_raw[i][0:len(dataset[i])] = dataset[i]
    #     y_raw[i] = label[i]
    # x_raw = StandardScaler().fit_transform(x_raw)

    # Loading dnpu pre-processed data
    projection_idx = 22

    dnpu_preprocessed_data_dir    = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti46/in_elec_4/dnpu_output.npy"
    dnpu_preprocessed_label_dir= "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti46/in_elec_4/labels.npy"

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
    tsne = TSNE(random_state = 42, n_components=2,verbose=0, perplexity=35, n_iter=500).fit_transform(x_dnpu)
    plt.scatter(tsne[:, 0], tsne[:, 1], s= 80, c=y_dnpu, cmap='Spectral')
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title('Visualizing Spoken MNIST through t-SNE, DNPU pre-process', fontsize=22)
    plt.show()
    print("")

    # tsne2 = TSNE(random_state = 42, n_components=3,verbose=0, perplexity=40, n_iter=300).fit_transform(x_dnpu)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(tsne2[:, 0], tsne2[:, 1],tsne2[:,2], s= 45, c=y_dnpu, cmap='Spectral')
    # ax.set_xlim(-10,10)
    # ax.set_ylim(-10,10)
    # ax.set_zlim(-10,10)
    # plt.title('Visualizing Kannada MNIST through t-SNE in 3D', fontsize=22)
    # plt.show()