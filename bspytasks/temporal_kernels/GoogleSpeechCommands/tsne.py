from time import time

import numpy as np
import pandas as pd
import torch
import os
import librosa
import sklearn
import scipy

# For plotting
# from matplotlib import offsetbox
import matplotlib.pyplot as plt
# import matplotlib.patheffects as PathEffects
import seaborn as sns
# import plotly.graph_objects as go

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

#For standardising the dat
from sklearn.preprocessing import StandardScaler


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

def butter_lowpass(cutoff, order, fs):
    return scipy.signal.butter( N = order, 
                                Wn = cutoff, 
                                btype = 'low', 
                                analog = False,
                                fs= fs,
                                output = 'sos'
    )

def butter_lowpass_filter(data, cutoff, order, fs):
    sos = butter_lowpass(cutoff, order = order, fs=fs)
    return scipy.signal.sosfilt(sos, data)



if __name__ == '__main__':

    num_classes = 14

    # loading labels
    labels_np = np.load("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernels/GoogleSpeechCommands/dataset/SUBSET/numpy_audios/labels_np.npy", 
                        allow_pickle=True)[0 * 200 : num_classes * 200]

    # loading data with np.memmap
    folder_path = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernels/GoogleSpeechCommands/dataset/dnpu_measurements"

    dataset = np.zeros(((num_classes * 200, 64, 496)))
    
    i = 0
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith("6.npy") or filename.endswith("13.npy"):
            file_path = os.path.join(folder_path, filename)
            file = np.load(file_path, allow_pickle=True)
            # low pass filtering and downsampling
            # for j in range(0, file.shape[0]):
            #     for k in range(0, file.shape[1]):
            #         file[j][k] = butter_lowpass_filter(
            #             file[j][k],
            #             cutoff = 496//2,
            #             order = 4,
            #             fs = 8000
            #         )

            # memmap_array[i * 200 : (i + 3) * 200][:][:] = file[:,:,0:7936:16]
            dataset[i * 200 : (i + 7) * 200][:][:] = file[:,:,0:7936:16]
            i += 7

    # 50, 48, 42, 34, 32
    projection_idx = 62

    x_dnpu = StandardScaler().fit_transform(
        dataset[:,projection_idx,:]
    )

    tsne_1 = TSNE(
        n_components=2,
        verbose=0,
        perplexity=50,
        n_iter=2000,
        init='pca'
    ).fit_transform(
        x_dnpu
    )

    plt.figure()
    plt.scatter(
        tsne_1[:, 0], tsne_1[:, 1], s = 80, c = labels_np, cmap='Spectral'
    )
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries = np.arange(num_classes)-0.5).set_ticks(np.arange(num_classes))
    plt.show()

    # trans = umap.UMAP().fit(x_dnpu)

    # plt.figure()
    # plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s= 80, c=labels_np, cmap='Spectral')
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.colorbar(boundaries = np.arange(4)-0.5).set_ticks(np.arange(3))
    # plt.show()

    print()
