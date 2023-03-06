import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys
import os
import librosa
import tqdm
import scipy
import sklearn

from torch.utils.data import DataLoader, Dataset, random_split

import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np

from tqdm import tqdm
from itertools import chain
from torchvision import transforms
from brainspy.utils.manager import get_driver

_global_param = 32
_mean, _std = 0, 0

def post_dnpu_down_sample(
    input, # (500, 32, 12500)
    receptive_field = 4, # in milisecond
    sample_rate = 12500, # hz
):
    kernel_size = int(sample_rate * receptive_field * 0.001) # 50 points per window
    o_size = int(np.shape(input)[2] // kernel_size) # 250

    # correct this part
    output = np.zeros((np.shape(input)[0], np.shape(input)[1], 500))
    for i in range(len(input)):
        for j in range(len(input[i])):
            for k in range(0, 500):
                output[i][j][k] = input[i][j][k * 25]

    # for i in range(len(input)):
    #     for j in range(len(input[i])):
    #         for k in range(0, kernel_size):
    #             output[i][j][k] = input[i][j][k * o_size - 1]
    
    return output

class M4Compact(nn.Module):
    def __init__(self, input_ch, n_channels=32) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_ch)
        self.conv1 = nn.Conv1d(input_ch, int(0.5 * n_channels), kernel_size=3)
        self.bn2 = nn.BatchNorm1d(int(0.5 * n_channels))
        self.pool1 = nn.MaxPool1d(8)
        self.conv2 = nn.Conv1d(int(0.5 * n_channels), int(0.5 * n_channels), kernel_size=3)
        self.bn3   = nn.BatchNorm1d(int(0.5 * n_channels))
        self.pool2 = nn.MaxPool1d(8)
        self.fc1   = nn.Linear(int(0.5 * n_channels), 10)
    
    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn3(x)
        x = F.relu(x)

        # x = self.pool2(x)

        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

class ToTensor(object):
    def __call__(self, sample) -> object:
        audio_data, audio_label = sample['audio_data'], sample['audio_label']
        return {
            'audio_data'        : torch.tensor(audio_data, dtype=torch.float),
            'audio_label'       : torch.tensor(np.asarray(audio_label, dtype=np.float32), dtype=torch.float)
        }

class Normalize(object):
    def __call__(self, sample) -> object:
        audio_data, audio_label = sample['audio_data'], sample['audio_label']

        for i in range(len(audio_data)):
            audio_data[i] = (audio_data[i] - _mean[i] / _std[i])

        return {
            'audio_data' : audio_data,
            'audio_label': audio_label
        }

# calculates mean and std
def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data in enumerate(dataloader):
        # Mean over batch, audio signal, but not over the channels
        audio_data = data[1]['audio_data']
        channels_sum += torch.mean(audio_data, dim=[0, 2])
        channels_squared_sum += torch.mean(audio_data**2, dim=[0, 2])
        num_batches += 1
    
    mean_temp = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std_temp = (channels_squared_sum / num_batches - mean_temp ** 2) ** 0.5

    return mean_temp, std_temp

# loads from .npy files
class DNPUAudioDataset(Dataset):
    def __init__(
        self,
        data_dir,
        label_dir,
        transform,
        projections_to_remove = [],
    ) -> None:
    
        self.transform = transform

        self.dataset_tmp = np.load(data_dir)
        self.label = np.load(label_dir)

        self.dataset_tmp = post_dnpu_down_sample(self.dataset_tmp, 4, 12500)

        if projections_to_remove:
            # new dataset size -> 450, 32-rem, 496
            self.dataset = np.zeros((np.shape(self.dataset_tmp)[0], _global_param - len(projections_to_remove), np.shape(self.dataset_tmp)[2]))
            # loop over new number of projections
            cnt = 0
            for i in range(0, _global_param):
                if not(i in projections_to_remove):
                    self.dataset[:, cnt, :] = self.dataset_tmp[:, i, :]
                    cnt += 1
        else:
            self.dataset = self.dataset_tmp
        
        assert len(self.dataset) == len(self.label), "Error in loading data!"
    
    def __len__(self) -> None:
        return len(self.dataset)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        
        data = self.dataset[index]
        label = self.label[index]

        sample = {
            'audio_data': data,
            'audio_label': label
        }

        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
def train(
    model, num_epochs, weight_decay, train_loader, test_loader, device, batch_size
):
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        weight_decay    = weight_decay
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr          = 0.005,
        steps_per_epoch = int(len(train_loader)),
        epochs          = num_epochs,
        anneal_strategy = 'cos',
        cycle_momentum  = True
    )

    model.train()
    accuracies = [0]
    
    for epoch in range(num_epochs):
        with tqdm(train_loader, unit="batch") as tepoch:
            # print("Starting epoch: ", epoch + 1)
            current_loss = 0.
            i = 0
            # for i, (data, target) in enumerate(train_loader):
            for data in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                # i = data[0]


                inputs = data[1]['audio_data'].to(device)
                targets = data[1]['audio_label'].type(torch.LongTensor).to(device)
                
                outputs = torch.squeeze(model(inputs))
                loss = loss_fn(outputs, targets)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                scheduler.step()
                
                current_loss += loss.item()

                if i % batch_size  == batch_size - 1:
                #     # print("Loss after mini-batch ", i+1, ": ", current_loss/batch_size)
                #     print(f"Train Epoch: {epoch} [{i * len(data)}/{len(train_loader.dataset)} ({100. * i / len(train_loader):.0f}%)]/tLoss: {loss.item():.6f}")
                    current_loss = 0.
                i += 1

                tepoch.set_postfix(loss=current_loss, accuracy = np.max(accuracies))

            if epoch >= 40:
                model.eval()
                correct, total = 0, 0
                for data in enumerate(test_loader):
                    inputs = data[1]['audio_data'].to(device)
                    targets = data[1]['audio_label'].type(torch.LongTensor).to(device)
                    outputs = torch.squeeze(model(inputs))
                    _, predicted = torch.max(outputs, 1)
                    total += data[1]['audio_label'].size(0)
                    correct += (predicted == targets).sum().item()

                # print("Test accuracy: ", 100 * correct / total)
                accuracies.append(100*correct/total)
                model.train()
            
            if epoch >= 80:
                with torch.no_grad():
                    print("")
            

    print("-----------------------------------------")
    print("Best test accuracy: ", np.max(accuracies))


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    empty = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_in_house_arsenic/empty/"
    projections_to_remove = [] # [3, 6, 11, 15]

    # Bellow is a bit complex type fo coding. I concatenate the whole training dataset to calculate mean and std (global variables)
    # Then, I use these variables to normalize both training AND test dataset.
    dataset_for_mean =  DNPUAudioDataset(
                            data_dir    = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti46/in_elec_4/dnpu_output.npy",
                            label_dir= "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti46/in_elec_4/labels.npy",
                            transform   = ToTensor(),
                            projections_to_remove= projections_to_remove
    )
                                    
    _mean, _std = get_mean_and_std(
        DataLoader(
            dataset_for_mean,
            batch_size= len(dataset_for_mean),
            shuffle= False,
            drop_last= False   
        )
    )

    transform = transforms.Compose([
        ToTensor(),
        Normalize()
    ])

    batch_size = 32

    dataset = DNPUAudioDataset(
        data_dir    = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti46/in_elec_4/dnpu_output.npy",
        label_dir= "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti46/in_elec_4/labels.npy",
        transform   = transform,
        projections_to_remove= projections_to_remove
    )

    trainset, testset = torch.utils.data.random_split(
        dataset,
        lengths = [450, 50]
    )


    train_loader = DataLoader(
        # torch.utils.data.ConcatDataset([train_set1, train_set2, train_set3, train_set4, train_set5]),
        trainset,
        batch_size  = batch_size,
        shuffle     = True,
        drop_last   = False
    )

    test_loader = DataLoader(
        # torch.utils.data.ConcatDataset([test_set1, test_set2, test_set3, test_set4, test_set5]),
        testset,
        batch_size  = 8,
        shuffle     = False,
        drop_last   = False
    )


    model = M4Compact(
        input_ch = 32 - len(projections_to_remove),
    )
    print("Number of learnable params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))


    train(
        model           = model,
        num_epochs      = 100,
        weight_decay    = 10e-5,
        train_loader    = train_loader,
        test_loader     = test_loader,
        device          = device,
        batch_size      = batch_size,
    )

