import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys
import os
import librosa
import sklearn

from torch.utils.data import DataLoader, Dataset, random_split

import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np

from tqdm import tqdm
from itertools import chain
from torchvision import transforms

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=26, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=125, stride=stride)
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

# def pad_sequence(batch):
#     # Make all tensor in a batch the same length by padding with zeros
#     batch = [item.t() for item in batch]
#     m = torch.nn.ConstantPad1d((0, 12500 - len(batch[0])), 0)
#     batch[0] = m(batch[0])
#     batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
#     batch = batch.reshape(batch.size(0), 1, batch.size(1))
#     return batch #.permute(0, 2, 1)

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(
        batch,
        batch_first     = True,
        padding_value   = 0.
    )
    return batch.reshape(batch.size(0), 1, batch.size(1))

def collate_fn(batch):
    tensors, targets = [], []
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
        self.train = train
        self.dataset, self.label = [], []

        self.max_length = 19000
        self.min_length = 0

        for subdir, _, files in chain.from_iterable(
            os.walk(path) for path in data_dir
        ):
            for file in files:
                tmp, _ = librosa.load(os.path.join(subdir, file), sr=12500, dtype=np.float32)

                # tmp = np.pad(
                #     tmp,
                #     pad_width = (0, self.max_length - len(tmp)),
                #     mode = 'constant',
                #     constant_values = 0.
                # )

                # # trimming -> not using the trimming leads to the highest accuracy
                tmp, _ = librosa.effects.trim(tmp, frame_length = 128, hop_length = 8, top_db=25)
                if len(tmp) < 12500:
                    tmp = np.pad(
                        tmp,
                        pad_width= (0, 12500 - len(tmp)),
                        mode = 'constant',
                        constant_values= 0.
                    )
                else:
                    tmp = tmp[0:12500]
                    print("Warning! cropping audio data...")

                # scaling
                scale = np.max(np.abs(tmp))
                tmp = tmp * (1/scale) * 1.

                self.dataset.append(tmp)
                self.label.append(file[1])
            
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(self.label)
        self.label = le.transform(self.label)

        assert len(self.dataset) == len(self.label), "Error in loading dataset!"

        self.balanced_train_data, self.balanced_test_data, self.balanced_train_label, self.balanced_test_label = sklearn.model_selection.train_test_split(
            self.dataset,
            self.label, 
            test_size = 0.1, 
            train_size = 0.9,
            stratify=self.label,
            random_state = 1
        )
    
    def __len__(self) -> None:
        if self.train:
            return len(self.balanced_train_data)
        else:
            return len(self.balanced_test_data)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        
        if self.train:
            data = self.balanced_train_data[index]
            label = self.balanced_train_label[index]
        else:
            data = self.balanced_test_data[index]
            label = self.balanced_test_label[index]

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

        _, predicted = torch.max(outputs, 1)
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
        n_output    = 26,
        stride      = 16,
        n_channel   = 32
    )

    # model.conv1.requires_grad_(False)
    print("Number of learnable params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    train_set = AudioDataset(
        data_dir    = ("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernels_ti_alpha/recordings", empty),
        transform   = transform,
        train       = True   
    )

    test_set = AudioDataset(
        data_dir    = ("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernels_ti_alpha/recordings", empty),
        transform   = transform,
        train       = False   
    )

    train_loader = DataLoader(
        train_set,
        batch_size  = batch_size,
        shuffle     = True,
        collate_fn= collate_fn,
        drop_last   = False
    )

    test_loader = DataLoader(
        test_set,
        batch_size  = batch_size,
        shuffle     = False,
        collate_fn  = collate_fn,
        drop_last   = False
    )

    train(
        model           = model,
        num_epochs      = 100,
        train_loader    = train_loader,
        test_loader     = test_loader,
        device          = device,
        batch_size      = batch_size
    )
