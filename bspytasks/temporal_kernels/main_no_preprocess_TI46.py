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
from torch.utils.tensorboard import SummaryWriter


import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np

from tqdm import tqdm
from itertools import chain
from torchvision import transforms
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class M1(nn.Module):
    def __init__(self, n_input=1, n_output=16):
        super().__init__()
        # self.bn1 = nn.BatchNorm1d(1)
        # self.conv1 = nn.Conv1d(n_input, n_output, kernel_size=3, stride=1)
        # self.pool1 = nn.MaxPool1d(8)
        self.fc1 = nn.Linear(1250, 10) 

    def forward(self, x):
        # x = self.bn1(x)
        # x = F.relu(self.conv1(x))
        # x = self.pool1(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

class M1_for_DNPU(nn.Module):
    def __init__(self, n_input, n_output=16):
        super().__init__()
        # self.bn1 = nn.BatchNorm1d(n_input)
        # self.conv1 = nn.Conv1d(n_input, n_output, kernel_size=3, stride=1)
        # self.pool1 = nn.MaxPool1d(8)
        self.fc1 = nn.Linear(1250*64, 10)

    def forward(self, x):
        # x = self.bn1(x)
        # x = F.relu(self.conv1(x))
        # x = self.pool1(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

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
    m = torch.nn.ConstantPad1d((0, 1250 - len(batch[0])), 0)
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
        waveform = batch[i]['audio_data'][::10]
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

        self.train = train
        self.transform = transform
        self.dataset, self.label = [], []
        self.balanced_train_data, self.balanced_test_data, self.balanced_train_label, self.balanced_test_label = [], [], [], []

        max_length = 0

        for subdir, _, files in chain.from_iterable(
            os.walk(path) for path in data_dir
        ):
            for file in files:
                tmp, _ = librosa.load(os.path.join(subdir, file), sr=None, dtype=np.float32)
                tmp, _ = librosa.effects.trim(tmp, frame_length=128, hop_length=1, top_db=12)
                self.dataset.append(tmp)
                self.label.append(file[1])
                if len(tmp) > max_length:
                    max_length = len(tmp)

        assert len(self.dataset) == len(self.label), "Error in loading dataset!"

        # Converting into arrays:
        self.dataset_np = np.zeros((len(self.dataset), max_length))
        self.label_np = np.zeros((len(self.label)))

        for i in range(0, len(self.dataset)):
            self.dataset_np[i][0 : len(self.dataset[i])] = self.dataset[i]
            self.label_np[i] = self.label[i]

        self.balanced_train_data, self.balanced_test_data, self.balanced_train_label, self.balanced_test_label = sklearn.model_selection.train_test_split(
            self.dataset_np,
            self.label_np, 
            test_size = 0.1, 
            train_size = 0.9,
            stratify=self.label_np,
            random_state = 0
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
    LOSS = []
    accuracies = [0]
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr              = 0.0005, 
        weight_decay    = 1e-5
    )
    for epoch in range(num_epochs):
        if epoch != 0:
            model.eval()
            with torch.no_grad():
                correct, total = 0, 0
                for i, (data, label) in enumerate(test_loader):
                    label = label.type(torch.LongTensor)
                    output = torch.squeeze(model(data))
                    _, predicted = torch.max(output, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                accuracies.append(100*correct/total)  
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            current_loss = 0.
            for i, (data, label) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                label = label.type(torch.LongTensor)
                output = torch.squeeze(model(data))
                loss = loss_fn(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # clamping DNPU control voltages
                # DNPUControlVoltageClamp(model, -0.30, 0.30)
                current_loss += loss.item()
                tepoch.set_postfix(
                    loss = current_loss / (i + 1),
                    accuracy = accuracies[-1]
                )
                LOSS.append(current_loss / (i + 1))      


def try_with_DNPU():
    """
       The code supports for single DNPU projection.
    """

    batch_size = 16
    sample_shape = (64, 1250)

    single_dnpu_projection_index = 13

    # - Create dataset
    np_data_test = np.load("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_digits/boron_8_electrode_35nm_etched/testset_numpy.npy", allow_pickle=True)
    np_data_train = np.load("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_digits/boron_8_electrode_35nm_etched/trainset_numpy.npy", allow_pickle=True)

    torch_data_test = torch.empty(size=(len(np_data_test), 64, sample_shape[1]))
    torch_targets_test = torch.empty(size=(len(np_data_test),))

    torch_data_train = torch.empty(size=(len(np_data_train), 64, sample_shape[1]))
    torch_targets_train = torch.empty(size=(len(np_data_train),))

    for sample_idx, sample in enumerate(np_data_test):
        torch_data_test[sample_idx] = sample["audio_data"] #[single_dnpu_projection_index-1:single_dnpu_projection_index,:]
        torch_targets_test[sample_idx] = sample["audio_label"].long()
    
    for sample_idx, sample in enumerate(np_data_train):
        torch_data_train[sample_idx] = sample["audio_data"] #[single_dnpu_projection_index-1:single_dnpu_projection_index,:]
        torch_targets_train[sample_idx] = sample["audio_label"].long()


    dataset_test = torch.utils.data.TensorDataset(torch_data_test, torch_targets_test)
    dataset_train = torch.utils.data.TensorDataset(torch_data_train, torch_targets_train)

    test_loader = DataLoader(
        dataset_test,
        batch_size=2,
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = M1_for_DNPU(
        n_input= 64,
        n_output= 8
    )

    print("Number of learnable params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))


    train(
        model           = model,
        num_epochs      = 100,
        train_loader    = train_loader,
        test_loader     = test_loader,
        device          = device,
        batch_size      = batch_size
    )

def plot_accuracies():
    linear_with_dnpu = np.load("acc_with_dnpu_linear.npy")
    linear_without_dnpu = np.load("acc_without_dnpu_linear.npy")
    cnn_with_dnpu = np.load("acc_with_dnpu_cnn.npy")
    cnn_without_dnpu = np.load("acc_without_dnpu_cnn.npy")

    plt.plot(np.arange(50), linear_without_dnpu, linewidth=3, label="linear_without_dnpu")
    plt.plot(np.arange(50), linear_with_dnpu, linewidth=3, label="linear_with_dnpu")
    plt.plot(np.arange(50), cnn_without_dnpu, linewidth=3, label="cnn_without_dnpu")
    plt.plot(np.arange(50), cnn_with_dnpu, linewidth=3, label="cnn_with_dnpu")

    plt.legend()
    plt.show()

    print()

if __name__ == '__main__':

    # plot_accuracies()

    empty = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_in_house_arsenic/empty/"

    transform = transforms.Compose([
            ToTensor()
    ])

    batch_size = 16

    # model = M1(
    #     n_input     = 1,
    #     n_output    = 16
    #     # stride      = 16,
    #     # n_channel   = 32
    # )

    # print("Number of learnable params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # train_set = AudioDataset(
    #     data_dir    = ("C:/Users/Mohamadreza/Documents/ti_spoken_digits/female_speaker", empty),
    #     transform   = transform,
    #     train       = True   
    # )

    # test_set = AudioDataset(
    #     data_dir    = ("C:/Users/Mohamadreza/Documents/ti_spoken_digits/female_speaker", empty),
    #     transform   = transform,
    #     train       = False   
    # )

    # # NOTE: collate_fn downsamples raw audio data to 1250 points 
    # train_loader = DataLoader(
    #     train_set,
    #     batch_size  = batch_size,
    #     shuffle     = True,
    #     collate_fn  = collate_fn,
    #     drop_last   = True
    # )

    # test_loader = DataLoader(
    #     test_set,
    #     batch_size  = 2,
    #     shuffle     = True,
    #     collate_fn  = collate_fn,
    #     drop_last   = True
    # )

    # train(
    #     model           = model,
    #     num_epochs      = 50,
    #     train_loader    = train_loader,
    #     test_loader     = test_loader,
    #     device          = device,
    #     batch_size      = batch_size
    # )

    try_with_DNPU()
