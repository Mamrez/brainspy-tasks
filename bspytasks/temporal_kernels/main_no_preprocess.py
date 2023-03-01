import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys
import os
import librosa

from torch.utils.data import DataLoader, Dataset, random_split

import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np

from tqdm import tqdm
from itertools import chain
from torchvision import transforms

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
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
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

class ToTensor(object):
    def __call__(self, sample) -> object:
        audio_data, audio_label = sample['audio_data'], sample['audio_label']
        return {
            'audio_data'        : torch.tensor(audio_data, dtype=torch.float),
            'audio_label'       : torch.tensor(np.asarray(audio_label, dtype=np.float32), dtype=torch.float)
        }

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    m = torch.nn.ConstantPad1d((0, 8000 - len(batch[0])), 0)
    batch[0] = m(batch[0])
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    batch = batch.reshape(batch.size(0), 1, batch.size(1))
    return batch #.permute(0, 2, 1)

def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    # for waveform, label in batch['audio_label']:
    #     tensors += [waveform]
    #     targets += [label]

    for i in range(len(batch)):
        waveform = batch[i]['audio_data']
        label = batch[i]['audio_label']
        tensors += [waveform]
        targets += [label]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

class AudioDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform,
        train = True,
    ) -> None:
    
        self.transform = transform

        self.dataset, self.label = [], []

        for subdir, _, files in chain.from_iterable(
            os.walk(path) for path in data_dir
        ):
            for file in files:
                tmp, _ = librosa.load(os.path.join(subdir, file), sr=None, dtype=np.float32)
                if train == True:
                    if subdir[-5:] == "train":
                        self.dataset.append(tmp)
                        self.label.append(file[0])
                elif train == False:
                    if subdir[-4:] == "test":
                        self.dataset.append(tmp)
                        self.label.append(file[0])

        assert len(self.dataset) == len(self.label), "Error in loading dataset!"
    
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
    model, num_epochs, train_loader, test_loader, device, batch_size
):
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr              = 0.001, 
        weight_decay    = 10e-5
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr          = 0.01,
        steps_per_epoch = int(len(train_loader)),
        epochs          = num_epochs,
        anneal_strategy = 'linear'
    )

    model.train()
    model.conv1.requires_grad_(False)
    for epoch in range(num_epochs):
        print("Starting epoch: ", epoch + 1)
        current_loss = 0.
        for i, (data, target) in enumerate(train_loader):
            inputs = data.to(device)
            targets = target.type(torch.LongTensor).to(device)
            
            outputs = torch.squeeze(model(inputs))
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            current_loss += loss.item()
            if i % batch_size  == batch_size - 1:
                print("Loss after mini-batch ", i+1, ": ", current_loss/batch_size)
                current_loss = 0.
    
    model.eval()
    correct, total = 0, 0
    for i, (data, target) in enumerate(test_loader):
        inputs = data.to(device)
        targets = target.type(torch.LongTensor).to(device)
        
        outputs = torch.squeeze(model(inputs))

        _, predicted = torch.max(outputs, 0)
        total += target.size(0)
        correct += (predicted == targets).sum().item()

    print("Test accuracy: ", 100 * correct / total)


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    empty = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_in_house_arsenic/empty/"

    transform = transforms.Compose([
            ToTensor()
    ])

    batch_size = 16

    model = M5(
        n_input     = 1,
        n_output    = 10,
        stride      = 16,
        n_channel   = 32
    )

    model.conv1.requires_grad_(False)
    print("Number of learnable params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))


    train_set = AudioDataset(
        data_dir    = ("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spoken_digit_task/spoken_mnist/spoken_mnist/train", empty),
        transform   = transform,
        train       = True   
    )
    train_loader = DataLoader(
        train_set,
        batch_size  = batch_size,
        shuffle     = True,
        collate_fn= collate_fn,
        drop_last   = True
    )

    test_set = AudioDataset(
        data_dir    = ("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spoken_digit_task/spoken_mnist/spoken_mnist/test", empty),
        transform   = transform,
        train       = False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size  = 1,
        shuffle     = False,
        collate_fn= collate_fn,
        drop_last   = False
    )

    train(
        model           = model,
        num_epochs      = 25,
        train_loader    = train_loader,
        test_loader     = test_loader,
        device          = device,
        batch_size      = batch_size
    )
