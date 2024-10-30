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
import math

from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np

from tqdm import tqdm
from itertools import chain
from torchvision import transforms
from brainspy.utils.manager import get_driver
from copy import deepcopy
from datetime import datetime

def apply_noise(weight, device, std = 0.067):
    with torch.no_grad():
        noise = std * torch.max(torch.abs(weight)) * torch.randn_like(weight, device=weight.device)
        noisy_weight = weight.clone() + noise
    return noisy_weight

class noisy_Linear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(noisy_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            noisy_weight = apply_noise(self.weight, self.weight.device)
        else:
            noisy_weight = self.weight

        return F.linear(input, noisy_weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class noisy_Conv1d(torch.nn.modules.conv._ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        # we create new variables below to make mypy happy since kernel_size has
        # type Union[int, Tuple[int]] and kernel_size_ has type Tuple[int]
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        super(noisy_Conv1d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _single(0), groups, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            noisy_weight = apply_noise(self.weight, device=self.weight.device)
        else:
            noisy_weight = self.weight
        
        return self._conv_forward(input, noisy_weight, self.bias)


class M4(nn.Module):
    def __init__(self, input_ch, n_channels=32, hw_train = False) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_ch)
        self.pool1 = nn.MaxPool1d(4)

        if hw_train:
            self.conv1 = noisy_Conv1d(input_ch, n_channels, kernel_size= 3, device=device)
        else:
            self.conv1 = nn.Conv1d(input_ch, n_channels, kernel_size=3, stride = 1)

        self.bn2 = nn.BatchNorm1d(n_channels)
        self.pool2 = nn.MaxPool1d(4)

        if hw_train:
            self.conv2 = noisy_Conv1d(n_channels, 2 * n_channels, kernel_size=3, device=device)
        else:
            self.conv2 = nn.Conv1d(n_channels, 2 * n_channels, kernel_size=3)

        self.bn3   = nn.BatchNorm1d(2 * n_channels)

        self.pool3 = nn.MaxPool1d(4)

        if hw_train:
            self.conv3 = noisy_Conv1d(2 * n_channels, 2 * n_channels, kernel_size=3, device=device)
        else:
            self.conv3 = nn.Conv1d(2 * n_channels, 2 * n_channels, kernel_size=3)

        self.bn4 = nn.BatchNorm1d(2 * n_channels)
        self.pool4 = nn.MaxPool1d(4)

        if hw_train:
            self.fc1 = noisy_Linear(2 * n_channels, 26, device=device)
        else:
            self.fc1   = nn.Linear(2 * n_channels, 26)
    
    def forward(self, x):
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.conv1(x)
        x = F.relu(self.bn2(x))

        x = self.pool2(x)

        x = self.conv2(x)
        x = F.relu(self.bn3(x))

        x = self.pool3(x)

        x = self.conv3(x)
        x = F.relu(self.bn4(x))

        x = self.pool4(x)

        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

def weight_clamp(model, alpha = 2.5):
    model.eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            _std = torch.std(param)
            param.clamp_(-alpha * _std, alpha*_std)

def add_noise(model, cs):
    noisy_model = deepcopy(model)
    for m in noisy_model.modules():
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
        # if isinstance(m, (noisy_Conv1d, noisy_Linear)):
            w = m.weight.data
            w_abs_max = w.abs().max()
            # - Bring to unit
            w /= w_abs_max
            noise = torch.randn_like(w) * (cs[3] + cs[2]*w.abs() + cs[1]*w.abs()**2 + cs[0]*w.abs()**3)
            w += noise
            w *= w_abs_max
            m.weight.data = w 
    return noisy_model

def test(
    model,
    test_loader,
    device
):
        
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for data in enumerate(test_loader):
            inputs = data[1]['audio_data'].to(device)
            targets = data[1]['audio_label'].type(torch.LongTensor).to(device)
            outputs = torch.squeeze(model(inputs))
            _, predicted = torch.max(outputs, 1)
            total += data[1]['audio_label'].size(0)
            correct += (predicted == targets).sum().item() 
    
    # print("Test accuracy: ", 100 * correct / total)
    

    return 100 * correct / total

def train(
    model, num_epochs, weight_decay, train_loader, test_loader, device, batch_size
):
    
    # Tensorboard configs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("runs/ti_alpha_{}".format(timestamp))

    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr = 0.01,
        weight_decay    = weight_decay
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr          = 0.01,
        total_steps     = num_epochs,
        anneal_strategy = 'cos',
        cycle_momentum  = True
    )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, 
    #     T_max       = num_epochs,
    # )

    model.train()
    accuracies = []
    best_acc = 0.
    
    for epoch in range(num_epochs):
        
        accuracies.append(
            test(
                model,
                test_loader,
                device
            )
        )

        if accuracies[-1] > best_acc:
            torch.save(model.state_dict(), "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernels_ti_alpha/models/HWAware_model_a2_5.pt")
            best_acc = accuracies[-1]
        
        if best_acc >= 94:
            break

        with tqdm(train_loader, unit="batch") as tepoch:
            current_loss = 0.
            i = 0
            model.train()
            for data in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

                inputs = data[1]['audio_data'].to(device)
                targets = data[1]['audio_label'].type(torch.LongTensor).to(device)
                
                outputs = torch.squeeze(model(inputs))
                loss = loss_fn(outputs, targets)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                
                current_loss += loss.item()

                if i % batch_size  == batch_size - 1:
                    current_loss = 0.
                i += 1

                tepoch.set_postfix(loss=current_loss, top_acc = best_acc, curr_acc = accuracies[-1])

        scheduler.step()

        writer.add_scalar(
            "Accuracy",
            torch.tensor(accuracies[-1]),
            epoch
        )
    writer.close()
    print("-----------------------------------------")

def hw_train(
    model,
    num_epochs,
    train_loader,
    test_loader,
    device,
    batch_size,
    weight_decay
):
    print("Starting the hardware aware training...")

    # Tensorboard configs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("runs/hw_aware_trainer_{}".format(timestamp))

    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr              = 0.002,
        weight_decay    = weight_decay
    )

    # scheduler = torch.optim.lr_scheduler.CyclicLR(
    #     optimizer       = optimizer,
    #     base_lr         = 0.0005,
    #     max_lr          = 0.05,
    #     step_size_up    = 2000,
    #     mode            = 'triangular2',
    #     cycle_momentum  = False
    # )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr          = 0.01,
        total_steps     = num_epochs,
        anneal_strategy = 'cos',
        cycle_momentum  = True
    )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, 
    #     T_max       = num_epochs,
    # )

    accuracies = [0.]
    best_acc = 0.

    for epoch in range(num_epochs):
        accuracies.append(
            test(
                model,
                test_loader,
                device
            )
        )

        if accuracies[-1] > best_acc:
            if epoch != 0:
                torch.save(model.state_dict(), "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernels_ti_alpha/models/HWAware_model_a2_5_noisy.pt")
                best_acc = accuracies[-1]
        
        if best_acc >= 94:
            break
        
        with tqdm(train_loader, unit="batch") as tepoch:
            current_loss = 0.
            i = 0
            model.train()
            for data in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

                inputs = data[1]['audio_data'].to(device)
                
                targets = data[1]['audio_label'].type(torch.LongTensor).to(device)
                
                outputs = torch.squeeze(model(inputs))
                loss = loss_fn(outputs, targets)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                
                current_loss += loss.item()
                if i % batch_size  == batch_size - 1:
                    current_loss = 0.
                i += 1

                # Here we do the clamping
                weight_clamp(model, 2.5)

                tepoch.set_postfix(loss=current_loss, top_acc = best_acc, curr_acc = accuracies[-1])

        scheduler.step()

        writer.add_scalar(
            "Learning rate",
            # scheduler.get_last_lr()[-1],
            optimizer.param_groups[0]['lr'],
            epoch
        )
        writer.add_scalar(
            "Loss",
            current_loss,
            epoch
        )
        writer.add_scalar(
            "Accuracy",
            torch.tensor(accuracies[-1]),
            epoch
        )
        writer.add_histogram(
            "conv1.weight", 
            model.conv1.weight,
            epoch
        )
        writer.add_histogram(
            "conv2.weight", 
            model.conv2.weight,
            epoch
        )
        writer.add_histogram(
            "fc1.weight", 
            model.fc1.weight,
            epoch
        )
        writer.flush()

    print("-----------------------------------------")
    print("Final test: ")
    print("-----------------------------------------")
    writer.close()

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
        scaler = sklearn.preprocessing.MinMaxScaler((-1, 1))
        audio_data = scaler.fit_transform(audio_data)
        return {
            'audio_data'        : audio_data,
            'audio_label'       : audio_label,
        }

class balanced_dataset(Dataset):
    def __init__(self,
                transform, 
                train = True,
                loaded_channels = 32,
    ) -> None:
        self.train = train
        self.transform = transform
        self.data = np.load("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/in_elec_4_var/dataset.npy", allow_pickle=True)

        data_num = len(self.data)

        self.concat_org_dataset = np.zeros((data_num, loaded_channels, 1250))
        self.concat_org_label = np.zeros((data_num))

        self.balanced_train_data = np.zeros((int(0.9 * data_num), loaded_channels, 1250))
        self.balanced_train_label = np.zeros((int(0.9 * data_num)))

        self.balanced_test_data = np.zeros((int(0.1 * data_num), loaded_channels, 1250))
        self.balanced_test_label = np.zeros((int(0.1 * data_num)))

        for i in range(data_num):
            for j in range(loaded_channels):
                self.concat_org_dataset[i][j] = self.data.__getitem__(i)['audio_data'][j]
            self.concat_org_label[i] = self.data.__getitem__(i)['audio_label']

        # balanced splitting dataset into 450 and 50
        self.balanced_train_data, self.balanced_test_data, self.balanced_train_label, self.balanced_test_label = sklearn.model_selection.train_test_split(
            self.concat_org_dataset,
            self.concat_org_label, 
            test_size = 0.1, 
            train_size = 0.9,
            stratify=self.concat_org_label,
            random_state = 0
        )

        print("")

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
        

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 16

    train_loader = DataLoader(
        balanced_dataset(transforms.Compose([ToTensor()]), train=True),
        batch_size  = batch_size,
        shuffle     = True,
        drop_last   = False
    )

    test_loader = DataLoader(
        balanced_dataset(transforms.Compose([ToTensor()]), train=False),
        batch_size  = 64,
        shuffle     = False,
        drop_last   = False
    )

    model = M4(
        input_ch = 32,
        hw_train= False
    )
    model = model.to(device)

    print("Number of learnable params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Normal train, model is saved...
    train(
        model           = model,
        num_epochs      = 200,
        weight_decay    = 10e-4,
        train_loader    = train_loader,
        test_loader     = test_loader,
        device          = device,
        batch_size      = batch_size,
    )

    # # loading the model
    # model = M4(
    #     input_ch = 32,
    #     hw_train= True
    # )
    # model.load_state_dict(torch.load("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernels_ti_alpha/models/HWAware_model_a2_5.pt"))
    # model.to(device)

    # hw_train(
    #     model           = model,
    #     num_epochs      = 500,
    #     train_loader    = train_loader,
    #     test_loader     = test_loader,
    #     device          = device,
    #     batch_size      = batch_size,
    #     weight_decay    = 10e-3
    # )

    # # Evaluation
    # model = M4(
    #     input_ch = 32,
    #     hw_train= False
    # )
    # model.load_state_dict(torch.load("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernels_ti_alpha/models/HWAware_model_a2_5_noisy.pt"))
    # model.to(device)
    # model.eval()

    # # - Obtain the non-noisy accuracy
    # fp_acc = test(model, test_loader, device)

    # # - Obtain accuracy under noise influence
    # cs = [0.294462, -0.452322, 0.226837, 0.015175]
    # accs = []
    # for _ in range(250):
    #     noisy_model = add_noise(model, cs)
    #     accs.append(test(noisy_model, test_loader, device))
    #     print(f"FP acc. {fp_acc}% noisy acc {accs[-1]}%")
    
    # print("Average of noise accuracy: ", np.mean(accs))
    # print("Min/max of noise accuracy: ", np.min(accs), "/", np.max(accs))
    # print("Accuracy fluctuation: ", np.max(accs)-np.min(accs) ,"%")

    # _ = plt.hist(accs, bins='auto')
    # plt.title("Distribution of validation accuracy after clipping and noise injection")
    # plt.xlim(right=100)
    # plt.xlabel("Accuracy (100%)")
    # plt.ylabel("Distribution over 200 runs")
    # plt.show()
