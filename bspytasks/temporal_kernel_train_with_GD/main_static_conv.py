import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch import multiprocessing as mp

import tqdm
import scipy
import librosa
import os
import sklearn

from tqdm import tqdm
from itertools import chain
from torch.utils.data import Dataset, DataLoader

from brainspy.processors.processor import Processor
from brainspy.processors.dnpu import DNPU

EMPTY = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_in_house_arsenic/empty/"
NUM_DNPUs = 4

# GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def butter_lowpass(cutoff, order, fs):
    return scipy.signal.butter( N = order, 
                                Wn = cutoff, 
                                btype = 'lowpass', 
                                analog=False,
                                fs= fs
    )

def butter_lowpass_filter(data, cutoff, order, fs):
    b, a = butter_lowpass(cutoff, order = order, fs=fs)
    y = scipy.signal.filtfilt(b = b, 
                            a = a, 
                            x = data
    )
    return y

class StaticDNPUConvolutionKernel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        training_data = torch.load(
            "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernel_train_with_GD/SurrogateModels/model_2/training_data_smg.pt",
            map_location = device
        )
        SMG = Processor(
            configs = {
                "processor_type" : "simulation",
                "electrode_effects": {
                    "amplification": [28.5]
                },
                "waveform":{
                    "slope_length" : 0,
                    "plateau_length": 1
                }
            },
            info = training_data["info"],
            model_state_dict = training_data['model_state_dict']
        ) 

        # instanciating a DNPU
        self.DNPU = DNPU(
            processor = SMG,
            data_input_indices = [[2, 3, 4]]
        )
    
    # implement 1-D convolution (k = 3)
    def forward(self, x):
        out = torch.vmap(
            func = self.DNPU,
            in_dims= 0,
        )(x.unfold(1, 3, 1))

        return out.squeeze()

class StaticDNPUConvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(StaticDNPUConvolutionLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dnpus = nn.ModuleList(
            [
                nn.ModuleList([StaticDNPUConvolutionKernel().to(device) for _ in range(self.in_channels)]) for _ in range(self.out_channels)
            ]
        )

    def forward(self, x):
        # Following section works for bspy env.
        # for loop implementation
        # x -> (batch_size = 6, in_channel = 4, 8780)
        out = torch.zeros((x.size(0), self.out_channels, x.size(2) - 2), device = device)
        for out_ch_idx in range(0, self.out_channels):
            outout_in_channel = torch.zeros((x.size(0), x.size(2) - 2), device = device)
            for in_ch_idx in range(0, self.in_channels):
                outout_in_channel += self.dnpus[out_ch_idx][in_ch_idx](x[:, in_ch_idx,:]).squeeze()
            out[:, out_ch_idx, :] = outout_in_channel
        
        return out

def load_audio_dataset(
    data_dir            = None,
    min_max_scale       = None,
    low_pass_filter     = None,
    same_size_audios    = None
):
    dataset, label = [], []
    max_length = 0

    for subdir, _, files in chain.from_iterable(
        os.walk(path) for path in data_dir
    ):
        for file in files:
            # Loading audio file;
            # First performing low pass filtering, and then trimming
            tmp, sr = librosa.load(os.path.join(subdir, file), sr=None, mono=True, dtype=np.float32)
            # Amplification, to be chosen in accordance to DNPU performance
            if min_max_scale == True:
                scale = np.max(np.abs(tmp))
                tmp = tmp * (1/scale)
            if low_pass_filter == True:
                tmp = butter_lowpass_filter(
                    tmp, 5000, 3, sr
                )
            # Removing silence
            tmp, _ = librosa.effects.trim(
                y               = tmp, 
                top_db          = 12,
                ref             = np.max,
                frame_length    = 128, 
                hop_length      = 4
            )
            
            if len(tmp) > max_length:
                max_length = len(tmp)
                if max_length % 10 != 0:
                    max_length += (10 - (max_length % 10))

            dataset.append(tmp)
            label.append(file[1])

    if same_size_audios == None:
        return dataset, label
    elif same_size_audios == "MAX":
        dataset_numpy = np.zeros((len(dataset), max_length))
        label_numpy = np.zeros((len(dataset)))
        for i in range(len(dataset)):
            dataset_numpy[i][0:len(dataset[i])] = dataset[i]
            label_numpy[i] = label[i]
        return dataset_numpy, label_numpy

class ToTensor(object):
    def __call__(self, data, label) -> object:
        return torch.tensor(data, dtype=torch.float), torch.tensor(np.asarray(label, dtype=np.float32), dtype=torch.float)

class AudioDataset(Dataset):
    def __init__(self, audios, labels, transforms) -> None:
        super(AudioDataset, self).__init__()
        self.transform = transforms
        self.audios = audios
        self.labels = labels
        assert len(self.audios) == len(self.labels), "Error in loading dataset!"
    
    def __len__(self):
        return len(self.audios)

    def __targets__(self):
        return self.labels

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        
        if self.transform:
            data, label = self.transform(self.audios[index], self.labels[index])
            return data, label
        else:
            return self.audios[index], self.labels[index]

class M3_dnpu_preprocessed(nn.Module):
    def __init__(self, n_input = 64, n_output=10, n_channel = 32):
        super().__init__()
        self.bn2 = nn.BatchNorm1d(n_input)
        self.conv2 = StaticDNPUConvolutionLayer(in_channels = n_input, out_channels = n_channel)
        # self.conv2 = nn.Conv1d(in_channels= n_input, out_channels=n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(8)
        # self.conv3 = StaticDNPUConvolutionLayer(in_channels = n_channel, out_channels = n_channel)
        self.conv3 = nn.Conv1d(in_channels= n_channel, out_channels=n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(n_channel, n_output)

    def forward(self, x, train = None, epoch_index = None):
        x = F.silu(self.bn2(x))

        x = self.conv2(x)
        x = F.silu(self.bn3(x))
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.silu(self.bn4(x))
        x = self.pool3(x)

        x = F.avg_pool1d(x, x.shape[-1])

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

def train(
        model,
        num_epochs,
        train_loader,
        test_loader,
        save = True,
):
    LOSS = []
    accuracies = [0]
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(   
        model.parameters(),   
        weight_decay    = 1e-5
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr          = 0.05,
        steps_per_epoch = int(len(train_loader)),
        epochs          = num_epochs,
        anneal_strategy = 'cos',
        cycle_momentum  = True
    )


    model = model.to(device)
    for epoch in range(num_epochs):
        if epoch != 0:
            model.eval()
            with torch.no_grad():
                correct, total = 0, 0
                for input in enumerate(test_loader):

                    data = input[1]['audio_data'].to(device)
                    label = input[1]['audio_label'].type(torch.LongTensor).to(device)
                    output = torch.squeeze(model(data.to(device)))

                    _, predicted = torch.max(output, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                accuracies.append(100*correct/total)  
        # saving the test set
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            current_loss = 0.
            i = 0
            for input in enumerate(tepoch):
                i += 1
                tepoch.set_description(f"Epoch {epoch}")

                data = input[1]['audio_data'].to(device)
                label = input[1]['audio_label'].type(torch.LongTensor).to(device)

                output = torch.squeeze(model(data.to(device)))
                loss = loss_fn(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                current_loss += loss.item()
                tepoch.set_postfix(
                    loss = current_loss / (i + 1),
                    accuracy = accuracies[-1]
                )
                LOSS.append(current_loss / (i + 1))
            if save:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_sate_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': LOSS[-1],
                        'accuracy': accuracies[-1]
                    }, 
                    "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernel_train_with_GD/SurrogateModels/model_2/training_results.pt"
                )
        scheduler.step()
    return model.state_dict()

class DNPUAudioDataset(Dataset):
    def __init__(
        self,
        transform,
    ) -> None:
        
        self.transform = transform
    
    def __len__(self) -> None:
        return len(self.dataset)

    def __targets__(self) -> None:
        return self.label

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

class ToTensor(object):
    def __call__(self, sample) -> object:
        audio_data, audio_label = sample['audio_data'], sample['audio_label']
        return {
            'audio_data'        : torch.tensor(audio_data, dtype=torch.float),
            'audio_label'       : torch.tensor(np.asarray(audio_label, dtype=np.float32), dtype=torch.float)
        }


def train_with_dnpu_preprocessed_data():
    batch_size = 16

    # - Create dataset
    # load dataset
    dataset = torch.load(
        "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_digits/boron_8_electrode_35nm_etched/dataset_1250.pt"
    )

    # stratified train eval split
    train_idx, test_idx = sklearn.model_selection.train_test_split(
        np.arange(dataset.__len__()),
        test_size = 0.1,
        random_state = 7,
        shuffle = True,
        stratify = dataset.__targets__()
    )

    # selecting train eval subsets
    trainset = torch.utils.data.Subset(
        dataset,
        train_idx
    )
    evalset = torch.utils.data.Subset(
        dataset,
        test_idx
    )

    # pytorch train dataloader
    train_loader = DataLoader(
        trainset,
        batch_size  = batch_size,
        shuffle     = True,
        drop_last   = True
    )

    # pytorch evaluation dataloader
    eval_loader = DataLoader(
        evalset,
        batch_size = 5,
        shuffle    = False,
        drop_last  = True
    )

    model = M3_dnpu_preprocessed().to(device)

    _ = train(
        model,
        100,
        train_loader = train_loader,
        test_loader = eval_loader,
        save = True
    )

if __name__ == "__main__":

    _ = train_with_dnpu_preprocessed_data()
