"""
Created on Thu Jan 10 2023
This file contains classes related to the ring data generation and loading. 
@author: github.com/lkriener/yin_yang_data_set (by Reza)
"""
import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

class YinYangDataset(Dataset):
    def __init__(self, r_small=0.1, r_big=0.5, size=1000, seed=42, transform=None):
        super(YinYangDataset, self).__init__()
        self.rng = np.random.RandomState(seed)
        self.transform = transform
        self.r_small = r_small
        self.r_big = r_big
        self.__vals = []
        self.__cs = []
        self.class_names = ['yin', 'yang' , 'dot']
        for i in range(size):
            # keep num of class instances balanced by using rejection sampling
            # choose class for this sample
            goal_class = self.rng.randint(3)
            x, y, c = self.get_sample(goal=goal_class)
            # add mirrod axis values
            # x_flipped = 1. - x
            # y_flipped = 1. - y
            # val = np.array([x, y, x_flipped, y_flipped])
            val = np.array([x, y])
            c = np.array([c])
            self.__vals.append(val)
            self.__cs.append(c)

    def get_sample(self, goal=None):
        # sample until goal is satisfied
        found_sample_yet = False
        while not found_sample_yet:
            # sample x,y coordinates
            x, y = self.rng.rand(2) * 2. * self.r_big
            # check if within yin-yang circle
            if np.sqrt((x - self.r_big)**2 + (y - self.r_big)**2) > self.r_big:
                continue
            # check if they have the same class as the goal for this sample
            c = self.which_class(x, y)
            if goal is None or c == goal:
                found_sample_yet = True
                break
        return x, y, c

    def which_class(self, x, y):
        # equations inspired by
        # https://link.springer.com/content/pdf/10.1007/11564126_19.pdf
        d_right = self.dist_to_right_dot(x, y)
        d_left = self.dist_to_left_dot(x, y)
        criterion1 = d_right <= self.r_small
        criterion2 = d_left > self.r_small and d_left <= 0.5 * self.r_big
        criterion3 = y > self.r_big and d_right > 0.5 * self.r_big
        is_yin = criterion1 or criterion2 or criterion3
        is_circles = d_right < self.r_small or d_left < self.r_small
        if is_circles:
            return 2
        return int(is_yin)

    def dist_to_right_dot(self, x, y):
        return np.sqrt((x - 1.5 * self.r_big)**2 + (y - self.r_big)**2)

    def dist_to_left_dot(self, x, y):
        return np.sqrt((x - 0.5 * self.r_big)**2 + (y - self.r_big)**2)

    def __getitem__(self, index):
        sample = (self.__vals[index].copy(), self.__cs[index])
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.__cs)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    dataset_train = YinYangDataset(size=10000, seed=42)
    dataset_validation = YinYangDataset(size=1000, seed=41)
    dataset_test = YinYangDataset(size=1000, seed=40)

    batchsize_train = 20
    batchsize_eval = len(dataset_test)

    train_loader = DataLoader(dataset_train, batch_size=batchsize_train, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset_validation, batch_size=batchsize_eval, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset_test, batch_size=batchsize_eval, shuffle=False, drop_last=True)

    fig, axes = plt.subplots(ncols=3, sharey=True, figsize=(15, 8))
    titles = ['Training set', 'Validation set', 'Test set']
    for i, loader in enumerate([train_loader, val_loader, test_loader]):
        axes[i].set_title(titles[i])
        axes[i].set_aspect('equal', adjustable='box')
        xs = []
        ys = []
        cs = []
        for batch, batch_labels in loader:
            for j, item in enumerate(batch):
                x1, y1 = item
                c = batch_labels[j]
                xs.append(x1)
                ys.append(y1)
                cs.append(c)
        xs = np.array(xs)
        ys = np.array(ys)
        cs = np.array(cs)
        axes[i].scatter(xs[cs == 0], ys[cs == 0], color='C0', edgecolor='k', alpha=0.7)
        axes[i].scatter(xs[cs == 1], ys[cs == 1], color='C1', edgecolor='k', alpha=0.7)
        axes[i].scatter(xs[cs == 2], ys[cs == 2], color='C2', edgecolor='k', alpha=0.7)
        axes[i].set_xlabel('x1')
        if i == 0:
            axes[i].set_ylabel('y1')
        
    plt.show()