import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

import scipy
import librosa
import os
import sklearn

from itertools import chain
from torch.utils.data import Dataset, DataLoader

from brainspy.processors.processor import Processor
from brainspy.processors.dnpu import DNPU

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
            "SMG/training_data_smg.pt"        
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
                nn.ModuleList([StaticDNPUConvolutionKernel() for _ in range(self.in_channels)]) for _ in range(self.out_channels)
            ]
        )

    def forward(self, x):
        # Following section works for bspy env.
        # for loop implementation
        # x -> (batch_size = 6, in_channel = 4, 8780)
        out = torch.zeros((x.size(0), self.out_channels, x.size(2) - 2))
        for out_ch_idx in range(0, self.out_channels):
            outout_in_channel = torch.zeros((x.size(0), x.size(2) - 2))
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
    def __call__(self, sample) -> object:
        audio_data, audio_label = sample['audio_data'], sample['audio_label']
        return {
            'audio_data'        : torch.tensor(audio_data, dtype=torch.float),
            'audio_label'       : torch.tensor(np.asarray(audio_label, dtype=np.float32), dtype=torch.float)
        }
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

class M3_dnpu_preprocessed(nn.Module):
    def __init__(self, n_input = 64, n_output=10, n_channel = 32):
        super().__init__()
        self.bn2 = nn.BatchNorm1d(n_input)
        self.conv2 = StaticDNPUConvolutionLayer(in_channels = n_input, out_channels = n_channel)
        # self.conv2 = nn.Conv1d(in_channels= n_input, out_channels=n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(8)
        self.conv3 = StaticDNPUConvolutionLayer(in_channels = n_channel, out_channels = n_channel)
        # self.conv3 = nn.Conv1d(in_channels= n_channel, out_channels=n_channel, kernel_size=3)
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
        return F.log_softmax(x, dim=2).squeeze()

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        eval_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.eval_data = eval_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for data in self.train_data:
            source = data['audio_data'].to(self.gpu_id)
            targets = data['audio_label'].to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
        print("Checkpoint saved, running evaluation ...")
        with torch.no_grad():
            correct, total = 0, 0
            for input in enumerate(self.eval_data):
                data = input[1]['audio_data']
                label = input[1]['audio_label'].type(torch.LongTensor)
                output = torch.squeeze(self.model(data))
                _, predicted = torch.max(output, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            print("Evaluation accuracy:", 100 * correct/total)

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

def load_train_objs():
    # load the dataset
    dataset = torch.load(
        "dataset/dataset_1250.pt"
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
    train_set = torch.utils.data.Subset(
        dataset,
        train_idx
    )
    eval_set = torch.utils.data.Subset(
        dataset,
        test_idx
    )

    # load the model
    model = M3_dnpu_preprocessed()

    optimizer = torch.optim.AdamW(   
        model.parameters(),   
        weight_decay    = 1e-5
    )
    return train_set, eval_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size = batch_size,
        pin_memory = True,
        shuffle = False,
        sampler = DistributedSampler(dataset)
    )

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset, evalset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    eval_data = DataLoader(evalset, batch_size = 5, shuffle = False, drop_last = True)
    trainer = Trainer(model, train_data, eval_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":

    dataset, evalset, model, optimizer = load_train_objs()
    # train_data = prepare_dataloader(dataset, 8)
    # eval_data = DataLoader(evalset, batch_size = 5, shuffle = False, drop_last = True)
    # b_sz = len(next(iter(train_data))[0])


    # import argparse
    # parser = argparse.ArgumentParser(description='simple distributed training job')
    # parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    # parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    # parser.add_argument('--batch_size', default=8, type=int, help='Input batch size on each device (default: 32)')
    # args = parser.parse_args()

    # world_size = torch.cuda.device_count()
    # mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
    
    
    # mp.spawn(main, args=(world_size, 1, 100, 8), nprocs=world_size)

    # test
    



