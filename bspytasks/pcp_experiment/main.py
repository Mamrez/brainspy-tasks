import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import torchvision
import tqdm
import scipy
import librosa
import os
import sklearn
import pickle

from torchvision import transforms
from tqdm import tqdm
from itertools import chain
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SeparatedKernels(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 6, kernel_size = 5) -> None:
        super(SeparatedKernels, self).__init__()
        self.RKernel = nn.Conv2d(
            in_channels     = 1,
            out_channels    = out_channels,
            kernel_size     = kernel_size
        )
        self.GKernel = nn.Conv2d(
            in_channels     = 1,
            out_channels    = out_channels,
            kernel_size     = kernel_size
        )
        self.BKernel = nn.Conv2d(
            in_channels     = 1,
            out_channels    = out_channels,
            kernel_size     = kernel_size
        )
    
    def forward(self, x):
        # x -> (batch_size, 3, h, w)
        x1 = self.RKernel(x[:, 0, :, :][:, None, :,:])
        x2 = self.RKernel(x[:, 1, :, :][:, None, :,:])
        x3 = self.RKernel(x[:, 2, :, :][:, None, :,:])

        # return torch.sum(torch.stack((x1, x2, x3)), dim = 2)
        return x1[:, :] + x2[:, :] + x3[:, :]

class Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = SeparatedKernels(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_cifar(
        visualize = False,
        batch_size = 16
):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if visualize:
        # get some random training images
        dataiter = iter(trainloader)
        images, labels = next(dataiter)

        # show images
        imshow(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    
    return trainloader, testloader

def train(
    model,
    epochs
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr                  = 1e-3,
        weight_decay        = 5e-5
    )

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

if __name__ == "__main__":
    trainloader, test_loader = load_cifar(False, batch_size=16)
    model = Network()

    train(
        model  = model.to(device),
        epochs = 100,

    )

