import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys
import os
import librosa
import sklearn
import torchaudio

from torch.utils.data import DataLoader, Dataset, random_split

import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np
import scipy

from tqdm import tqdm
from itertools import chain
from torchvision import transforms

class M4(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        # self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        # self.bn1 = nn.BatchNorm1d(n_channel)
        # self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_input, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        # x = self.conv1(x.resize(x.size(0), 1, x.size(1)))
        # x = self.bn1(x)
        # x = self.pool1(x)
        x = self.conv2(x.resize(x.size(0), 1, x.size(1))[:, :, 0:7936:16])
        x = F.silu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.silu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.silu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x.resize(x.size(0), 1, x.size(1)))
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.tanh(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.tanh(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.tanh(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


class CNN_LSTM(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        # convolution -> batch_norm -> activation function -> pooling
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)

        # LSTM
        self.lstm = nn.LSTM(input_size = 30, hidden_size = 128, num_layers = 2, batch_first = True)

        self.fc1 = nn.Linear(32 * 128, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, n_output)

    def forward(self, x):
        x = self.conv1(x.resize(x.size(0), 1, x.size(1)))
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.tanh(self.bn2(x))
        x = self.pool2(x)

        x, _ = self.lstm(x)
        x = F.tanh(self.fc1(x.resize(x.size(0), x.size(1) * x.size(2))))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

def train(
    model, num_epochs, train_loader, test_loader, device, batch_size
):
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr          = 0.01,
        steps_per_epoch = int(len(train_loader)),
        epochs          = num_epochs,
        anneal_strategy = 'linear'
    )

    model.train()
    accuracies = [0]
    
    for epoch in range(num_epochs):
        current_loss = 0.
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, data in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

                inputs = data[0].to(device)
                targets = data[1].type(torch.LongTensor).to(device)
                
                outputs = torch.squeeze(model(inputs))
                loss = loss_fn(outputs, targets)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                scheduler.step()
                
                current_loss += loss.item()

                # if i % batch_size  == batch_size - 1:
                #     current_loss = 0.

                tepoch.set_postfix(loss=current_loss/(i+1), accuracy = accuracies[-1])

            if epoch >= 40:
                model.eval()
                with torch.no_grad():
                    correct, total = 0, 0
                    for i, data in enumerate(test_loader):
                        inputs = data[0].to(device)
                        targets = data[1].type(torch.LongTensor).to(device)
                        outputs = torch.squeeze(model(inputs))
                        _, predicted = torch.max(outputs, 1)
                        total += data[1].size(0)
                        correct += (predicted == targets).sum().item()

                accuracies.append(100*correct/total)
                model.train()
            

    print("-----------------------------------------")
    print("Best test accuracy: ", np.max(accuracies))

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = KWSModelRawAudio(
    #     30
    # )

    # model = CNN_LSTM(n_output=12)

    num_classes = 35
    model = M5(n_input = 1, n_output=num_classes, n_channel=64)
    # model = M4(n_input = 1, n_output=12, n_channel=64)

    # Load the previsouly saved ones
    dataset_np = np.load("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernels/GoogleSpeechCommands/dataset/SUBSET/numpy_audios/dataset_np.npy", allow_pickle=True)[:num_classes * 200]
    labels_np = np.load("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernels/GoogleSpeechCommands/dataset/SUBSET/numpy_audios/labels_np.npy", allow_pickle=True)[:num_classes * 200]

    # for j in range(0, dataset_np.shape[0]):
    #     dataset_np[j] = butter_lowpass_filter(
    #         dataset_np[j],
    #         cutoff = 1000,
    #         order = 5,
    #         fs = 8000
    #     )

    train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(
            dataset_np,
            labels_np, 
            test_size = 0.1, 
            train_size = 0.9,
            stratify=labels_np,
            random_state = 0
        )

    # test set
    torch_data_test = torch.empty(size=(len(test_data), 8000))
    torch_targets_test = torch.empty(size=(len(test_data),))
    for i in range(0, len(test_data)):
        torch_data_test[i] = torch.Tensor(test_data[i])
        torch_targets_test[i] = test_label[i]
    testset = torch.utils.data.TensorDataset(torch_data_test, torch_targets_test)
    
    # train set
    torch_data_train = torch.empty(size=(len(train_data), 8000))
    torch_targets_train = torch.empty(size=(len(train_data),))
    for i in range(0, len(train_data)):
        torch_data_train[i] = torch.Tensor(train_data[i])
        torch_targets_train[i] = train_label[i]
    trainset = torch.utils.data.TensorDataset(torch_data_train, torch_targets_train)

    train_loader = DataLoader(
        trainset,
        batch_size  = 32,
        shuffle     = True,
        drop_last   = True
    )

    test_loader = DataLoader(
        testset,
        batch_size  = 64,
        shuffle     = False,
        drop_last   = False
    )

    train(
        model           = model,
        num_epochs      = 100,
        train_loader    = train_loader,
        test_loader     = test_loader,
        device          = device,
        batch_size      = 16
    )
