import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


import matplotlib.pyplot as plt
import numpy as np

import tqdm
import scipy
import librosa
import os
import sklearn
import pickle
import wfdb

from torchvision import transforms
from tqdm import tqdm
from itertools import chain
from torch.utils.data import Dataset, DataLoader

from brainspy.utils.manager import get_driver
from brainspy.utils.io import load_configs

# import Lori's codes
from gd_inputs import input_perturbation
from gd import dI_dV

# global constants
SLOPE_LENGTH = 10
REST_LENGTH = 100
DRIVER = get_driver(
    load_configs(
        "configs/defaults/processors/hw_ECG.yaml"
    )["driver"]
)
EMPTY = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_in_house_arsenic/empty/"
GD_CONFIGS = load_configs("bspytasks/temporal_kernel_train_with_GD/gd_configs_ECG.yml")
DNPU_INPUT_ELECTRODE = GD_CONFIGS["inputs"][0]
DNPU_IDX = 0
NUM_DNPUs = 1

class ToTensor(object):
    def __call__(self, sample) -> object:
        data, label = sample['data'], sample['label']
        return {
            'data'        : torch.tensor(data, dtype=torch.float),
            'label'       : torch.tensor(np.asarray(label, dtype=np.float32), dtype=torch.float)
        }

class MITBIHDataset(Dataset):
    def __init__(self, transform, normalize = True) -> None:
        super().__init__()
        self.transform = transform

        self.data = np.zeros((2274, 180))
        self.labels = np.zeros((2274))

        record = wfdb.rdrecord("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/mit_arrhythmia/208")
        annotation = wfdb.rdann("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/mit_arrhythmia/208", 'atr')

        signals = record.p_signal[:,0]
        annotations = annotation.sample

        for i in range(2, 2273):
            temp = signals[annotations[i] - 90 : annotations[i] + 90]
            self.data[i - 2] = temp
            if normalize == True:
                self.data[i - 2] = (self.data[i - 2] - self.data[i - 2].min()) / (self.data[i - 2].max() - self.data[i - 2].min())
            if annotation.symbol[i] == "L" or annotation.symbol[i] == "R" or annotation.symbol[i] == "N": 
                # 0 for no problem
                self.labels[i - 2] = 0
            else:
                # 1 for detecting arrhythmia
                self.labels[i - 2] = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        sample= {
            "data": self.data[index],
            "label": self.labels[index]
        }
        if self.transform:
            sample = self.transform(sample)
        
        return sample

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


def DNPUControlVoltageClamp(model, min, max):
    model.eval()
    with torch.no_grad():
        for p in model.dnpu_layer.parameters():
            p.data.clamp_(min, max)

class CustomLossFunction(nn.Module):
    """
        Custom CrossEntropy loss function to implement the regularization.
        A penalty will applied to the loss value if the control voltages of
        the DNPU(s) goes out of the range.
    """
    def __init__(self, model, min, max) -> None:
        super().__init__()
        self.model = model
        self.min = min
        self.max = max
    
    def forward(self, outputs, targets):
        loss = F.cross_entropy(outputs, targets)

        penalty_value = 0.
        with torch.no_grad():
            for value in model.dnpu_layer.weight.detach().numpy():
                if value >= self.max or value <= self.min:
                    penalty_value += value ** 2
        
        return loss + penalty_value


def single_forward_measurement(
        # input -> batch_size * MAX_length
        input,
        weight

):
    outputs = np.zeros((len(input), np.shape(input)[1]//2), dtype=np.float32)
    ext_weights = np.insert(weight, DNPU_INPUT_ELECTRODE, 0.)

    for i in range(len(input)):
        input_to_dnpu = np.zeros(
            (7, len(input[i]) + (2 * SLOPE_LENGTH) + REST_LENGTH)
        )
        # Setting electrodes
        for m in range(7):
            if m != DNPU_INPUT_ELECTRODE:
                ramp_up = np.linspace(
                    0, ext_weights[m], SLOPE_LENGTH
                )
                plateau = np.linspace(
                    ext_weights[m], ext_weights[m], np.shape(input_to_dnpu)[1] - 2 * SLOPE_LENGTH
                )
                ramp_down = np.linspace(
                    ext_weights[m], 0, SLOPE_LENGTH
                )
                input_to_dnpu[m] = np.concatenate((
                    ramp_up, plateau, ramp_down
                ))
            else:
                # Setting audio input
                input_to_dnpu[m, SLOPE_LENGTH + REST_LENGTH : -SLOPE_LENGTH] = input[i]
        output = DRIVER.forward_numpy(input_to_dnpu.T)
        output = output[SLOPE_LENGTH + REST_LENGTH : -SLOPE_LENGTH, 0]
        output = output - np.mean(output)
        # output = butter_lowpass_filter(
        #     output, 5000, 4, 1568
        # )
        outputs[i] = output[::2]
    return outputs

def backward_measurement(
    # weight it DNPU control voltages
    weight
):
    # creating the "random" input
    NUM_RANDOM_EXP = 7
    inputs = np.linspace(-0.1,0.1, NUM_RANDOM_EXP)
    derivatives = np.zeros((NUM_RANDOM_EXP, 6))
    for i in range(NUM_RANDOM_EXP):
        # creating waveforms
        waveform_to_dnpu = input_perturbation(inputs[i], weight.detach().cpu().numpy(), GD_CONFIGS)

        # DNPU measurement for perturbations
        output = DRIVER.forward_numpy(waveform_to_dnpu.T)

        # mask
        output = output[GD_CONFIGS["ramping_points"] + GD_CONFIGS["waiting_points"] : -(GD_CONFIGS["ramping_points"] + GD_CONFIGS["waiting_points"]), 0]

        # multiply by refernce
        derivatives[i] = dI_dV(
                        output,
                        GD_CONFIGS
                    )
    # FIX for SIGN FLIP
    derivatives_avg = np.zeros((6))
    derivatives_T = derivatives.T

    for i in range(len(derivatives_avg)):
        signs = np.sign(derivatives_T[i])
        unique_signs, counts = np.unique(signs, return_counts=True)
        maj_sign = unique_signs[np.argmax(counts)]
        for j in range(NUM_RANDOM_EXP):
            if np.sign(derivatives_T[i][j]) != maj_sign:
                derivatives_T[i][j] *= -1.0
        derivatives_avg[i] = np.average(derivatives_T[i])

    return torch.from_numpy(derivatives_avg)

LIST_OF_CVs = [[] for i in range(NUM_DNPUs)]
LIST_OF_GRADs = [[] for i in range(NUM_DNPUs)]
LIST_OF_DERIVATIVES = [[] for i in range(NUM_DNPUs)]


class DNPUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        # input -> batch_size * audio_length
        # e.g., 32 * 12,500
        ctx.save_for_backward(weight)
        outputs = single_forward_measurement(
                    input  = input.detach().cpu().numpy(),
                    weight = weight.detach().cpu().numpy()
        )   
        return torch.tensor(outputs, requires_grad = True)

    @staticmethod
    def backward(ctx, grad_output):
        global DNPU_IDX
        weight = ctx.saved_tensors[0]
        if ctx.needs_input_grad[0]:
            print("Warning!!")
        if ctx.needs_input_grad[1]:
            derivatives = backward_measurement(
                # here input is the output of the device
                weight
            )
            # grad_output -> [batch_size, 878]
            # grad_output = torch.mean(grad_output, 1)
            signs = torch.sign(torch.mean(grad_output, 1))
            grad_output = torch.mean(torch.abs(grad_output), 1) * signs

            grad_weight = torch.reshape(grad_output, (grad_output.size(0),1)) * derivatives
        # Saving the training data
        
        LIST_OF_CVs[DNPU_IDX % NUM_DNPUs].append(
            weight.data.detach().cpu().numpy().copy()
        )
        LIST_OF_GRADs[DNPU_IDX % NUM_DNPUs].append(grad_weight.detach().cpu().numpy())
        LIST_OF_DERIVATIVES[DNPU_IDX % NUM_DNPUs].append(derivatives.detach().cpu().numpy())
        with open("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernel_train_with_GD/data_mnist/cvs.pkl", "wb") as fp:
            pickle.dump(LIST_OF_CVs, fp)
        with open("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernel_train_with_GD/data_mnist/grads.pkl", "wb") as fp:
            pickle.dump(LIST_OF_GRADs, fp)
        with open("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernel_train_with_GD/data_mnist/drvs.pkl", "wb") as fp:
            pickle.dump(LIST_OF_DERIVATIVES, fp)
        DNPU_IDX += 1
        
        return None, grad_weight

class DNPULayer(nn.Module):
    def __init__(self) -> None:
        super(DNPULayer, self).__init__()
        # defining control voltages (6) as learnable parameters for backward
        self.weight = nn.Parameter(
            0.35 * (torch.rand(6) - 0.5)
        )
    def forward(self, input):
        return DNPUFunction.apply(input, self.weight)
    def _return_params(self):
        return self.weight.detach().cpu().numpy()

class DNPUClassifier(nn.Module):
    def __init__(self) -> None:
        super(DNPUClassifier, self).__init__()
        self.dnpu_layer = DNPULayer()
        self.linear_layer = nn.Linear(90, 2)
    def forward(self, x):
        
        x = self.dnpu_layer(x)
        x = self.linear_layer(x)
        return F.log_softmax(x, dim=1)
    def _get_params(self):
        return self.dnpu_layer._return_params()
 
class LinearLayer(nn.Module):
    def __init__(self) -> None:
        super(LinearLayer, self).__init__()
        self.linear_layer = nn.Linear(90, 2)
    def forward(self, x):
        # x = x.reshape(16, 180)
        x = self.linear_layer(x)
        return F.log_softmax(x, dim=1)

def train(
        model,
        num_epochs,
        train_loader,
        test_loader,
        save = True,
        DNPU_train_enabled = True
):
    LOSS = []
    accuracies = [0]
    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = CustomLossFunction(model, min = -0.20, max = 0.20)
    optimizer = torch.optim.SGD(
        [
            # {"params": model.dnpu_layer.weight, "lr" : 5e-3},
            {"params": model.linear_layer.parameters(), "lr" : 5e-4},
        ],
        momentum    = 0.9,
    )
    for epoch in range(num_epochs):
        if epoch != 0:
            model.eval()
            with torch.no_grad():
                correct, total = 0, 0
                for i, data in enumerate(test_loader):
                    input = data['data']
                    label = data['label'].type(torch.LongTensor)
                    output = torch.squeeze(model(input))
                    _, predicted = torch.max(output, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                accuracies.append(100*correct/total)  
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            current_loss = 0.
            for i, data in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                input = data['data']
                label = data['label'].type(torch.LongTensor)
                output = torch.squeeze(model(input))
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
            if save:
                if DNPU_train_enabled:
                    np.savez("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernel_train_with_GD/data/training_data_dnpu_trained.npz", Loss=LOSS, Acc=accuracies)
                else:
                    np.savez("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernel_train_with_GD/data/training_data.npz", Loss=LOSS, Acc=accuracies)
                    

    return model.state_dict()

            
if __name__ == "__main__":

    batch_size = 10
    transform = transforms.Compose([
            ToTensor()
    ])
    dataset = MITBIHDataset(
        transform = transform,
        normalize = True
    )

    train_set, val_set = torch.utils.data.random_split(
        dataset, [0.5, 0.5]
    )

    train_loader = torch.utils.data.DataLoader(dataset = train_set,
                                                batch_size = batch_size,
                                                shuffle = True,
                                                drop_last = True
    )


    test_loader = torch.utils.data.DataLoader(dataset = val_set,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            drop_last = True
    )

    # model = LinearLayer()
    model = DNPUClassifier()

    _ = train (
        model,
        num_epochs      = 100,
        train_loader    = train_loader,
        test_loader     = test_loader,
        save            = True,
        DNPU_train_enabled = True
    )

    DRIVER.close_tasks()

    pass

