import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sklearn
import scipy
from sklearn import model_selection

from torch.utils.data import DataLoader, Dataset, random_split

import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np

from tqdm import tqdm
from itertools import chain
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class M4(nn.Module):
    def __init__(self, n_output, n_channel):
        super().__init__()
        # self.bn0 = nn.BatchNorm1d(n_channel)
        # self.conv1 = nn.Conv1d(n_channel, n_channel, kernel_size=8, stride=1, padding=3)
        # self.bn1 = nn.BatchNorm1d(n_channel)
        # self.pool1 = nn.MaxPool1d(4)
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
        # x = self.bn0(x)
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

def train(
    model, num_epochs, train_loader, test_loader, device
):
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=1e-4
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

            if epoch >= 20:
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


if __name__ == '__main__':

    num_classes = 21
    model = M4(
        n_output    = num_classes,
        n_channel   = 64
    )

    # loading labels
    labels_np = np.load("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernels/GoogleSpeechCommands/dataset/SUBSET/numpy_audios/labels_np.npy", 
                        allow_pickle=True)[0 * 200 : num_classes * 200]

    # loading data with np.memmap
    folder_path = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernels/GoogleSpeechCommands/dataset/dnpu_measurements"
    # memmap_dataset_path = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernels/GoogleSpeechCommands/dataset/dnpu_measurements/memmap_dataset.npy"
    
    # 496 is a number after first software convolution
    # memmap_array = np.memmap(memmap_dataset_path, dtype='float32', mode='w+', shape=((num_classes * 200, 64, 496)))

    dataset = np.zeros(((num_classes * 200, 64, 496)))
    
    i = 0
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith("6.npy") or filename.endswith("13.npy") or filename.endswith("20.npy"):
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
    
    # memmap_array.flush()

    train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(
            dataset,
            labels_np, 
            test_size = 0.1, 
            train_size = 0.9,
            stratify=labels_np,
            random_state = 0
        )

    # data augmentation
    # train_label_aug = np.repeat(train_label, 2)
    # train_data_aug = np.repeat(train_data, 2, axis=0)
    # for i in range(len(train_data_aug)):
    #     if i%2 == 0:
    #         train_data_aug[i] += 0.35 * np.max(np.abs(train_data_aug[i])) * np.random.normal(size=((train_data_aug.shape[1], train_data_aug.shape[2])))
    # train_data = train_data_aug
    # train_label = train_label_aug

    # test set
    torch_data_test = torch.empty(size=(len(test_data), 64, 496))
    torch_targets_test = torch.empty(size=(len(test_data),))
    for i in range(0, len(test_data)):
        torch_data_test[i] = torch.Tensor(test_data[i])
        torch_targets_test[i] = test_label[i]
    testset = torch.utils.data.TensorDataset(torch_data_test, torch_targets_test)
    
    # train set
    torch_data_train = torch.empty(size=(len(train_data), 64, 496))
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
        num_epochs      = 500,
        train_loader    = train_loader,
        test_loader     = test_loader,
        device          = device
        )
