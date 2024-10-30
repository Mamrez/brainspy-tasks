import torch
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
SLOPE_LENGTH = 1000
REST_LENGTH = 2500
DRIVER = get_driver(
    load_configs(
        "configs/defaults/processors/hw.yaml"
    )["driver"]
)
EMPTY = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_in_house_arsenic/empty/"
GD_CONFIGS = load_configs("bspytasks/temporal_kernel_train_with_GD/gd_configs.yml")
DNPU_INPUT_ELECTRODE = GD_CONFIGS["inputs"][0]
DNPU_IDX = 0
NUM_DNPUs = 1

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
                tmp = tmp * (1/scale) * 0.75
            if low_pass_filter == True:
                tmp = butter_lowpass_filter(
                    tmp, 4000, 3, sr
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
            # CAREFUL!!!
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

def single_forward_measurement(
        # input -> batch_size * MAX_length
        input,
        weight

):
    if np.shape(input)[1] == 878:
        for i in range(len(input)):
            input[i] = input[i] * (1 / np.max(np.abs(input[i]))) * 0.75
        input = input.repeat(10, axis=1)

    outputs = np.zeros((len(input), np.shape(input)[1]//10), dtype=np.float32)
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
        output = butter_lowpass_filter(
            output, 5000, 4, 12500
        )
        outputs[i] = output[::10]
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
            # Here is a challenge to solve!
            # print("Warning!!")
            pass
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
        with open("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernel_train_with_GD/data/cvs.pkl", "wb") as fp:
            pickle.dump(LIST_OF_CVs, fp)
        with open("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernel_train_with_GD/data/grads.pkl", "wb") as fp:
            pickle.dump(LIST_OF_GRADs, fp)
        with open("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernel_train_with_GD/data/drvs.pkl", "wb") as fp:
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

class FeedForwardDNPU(nn.Module):
    def __init__(self) -> None:
        super(FeedForwardDNPU, self).__init__()
        # DNPU: 1: 8780 -> 878
        self.dnpu_1 = DNPULayer()
        self.layer_norm_1 = nn.LayerNorm(878)

        # LinerLayer: 1 -> 878 -> 878
        self.linear_1 = nn.Linear(878, 878)

        # DNPU: 2: 878 -> 878
        self.dnpu_2 = DNPULayer()
        self.layer_norm_2 = nn.LayerNorm(878)

        # LinearLayer: 2 -> 878 -> 10
        self.linear_2 = nn.Linear(878, 10)
    
    def forward(self, x):
        x = self.dnpu_1(x)
        x = self.layer_norm_1(x)
        x = self.linear_1(x)

        x = self.dnpu_2(x)
        x = self.layer_norm_2(x)
        x = self.linear_2(x)

        return F.log_softmax(x, dim = 1)

    def _get_params(self):
        pass

class DNPUClassifier(nn.Module):
    def __init__(self) -> None:
        super(DNPUClassifier, self).__init__()
        self.dnpu_layer = DNPULayer()
        # self.bn = nn.BatchNorm1d(878)
        self.ln = nn.LayerNorm(878)
        self.linear_layer = nn.Linear(878, 10)
    def forward(self, x):
        x = self.dnpu_layer(x)
        x = self.ln(x)
        x = self.linear_layer(x)
        return F.log_softmax(x, dim=1)
    def _get_params(self):
        return self.dnpu_layer._return_params()


class OnlyLinearLayer(nn.Module):
    def __init__(self) -> None:
        super(OnlyLinearLayer, self).__init__()
        self.ln = nn.LayerNorm(878)
        self.linear_layer = nn.Linear(878, 10)
    def forward(self, x):
        x = self.ln(x)
        x = self.linear_layer(x)
        return F.log_softmax(x, dim=1)

def layerwise_train(
        model,
        audios = None,
        labels = None,
        batch_size = 8
):
    # # Fetch the control voltages from DNPU Layer
    # initial_control_voltages = model._get_params()

    # # Measure (all) train and test sets
    # outputs = single_forward_measurement(
    #     input = audios,
    #     weight = initial_control_voltages
    # )

    # # Train linear layer with measured data (num_epochs // 2) and report accuracy
    # dataset = AudioDataset(
    #     audios      = outputs,
    #     labels      = labels,
    #     transforms  = ToTensor()
    # )

    # train_idx, test_idx = sklearn.model_selection.train_test_split(
    #     np.arange(dataset.__len__()),
    #     test_size       = .1,
    #     random_state    = 7,
    #     shuffle         = True,
    #     stratify        = dataset.__targets__()
    # )

    # # Subset dataset for train and val
    # trainset = torch.utils.data.Subset(dataset, train_idx)
    # testset = torch.utils.data.Subset(dataset, test_idx)

    # train_loader = DataLoader(
    #     trainset,
    #     batch_size  = batch_size, 
    #     shuffle     = True,
    #     drop_last   = False
    # )

    # test_loader = DataLoader(
    #     testset,
    #     batch_size  = batch_size,
    #     shuffle     = False,
    #     drop_last   = False
    # )

    # model = OnlyLinearLayer()

    # model_state_dict = train(
    #     model,
    #     num_epochs      = 100,
    #     train_loader    = train_loader,
    #     test_loader     = test_loader,
    #     save            = True, 
    #     DNPU_train_enabled = False
    # )


    # Activate DNPU layer training; Train bothe layers normally
    dataset = AudioDataset(
        audios      = audios,
        labels      = labels,
        transforms  = ToTensor()
    )

    train_idx, test_idx = sklearn.model_selection.train_test_split(
        np.arange(dataset.__len__()),
        test_size       = .1,
        random_state    = 7,
        shuffle         = True,
        stratify        = dataset.__targets__()
    )

    # Subset dataset for train and val
    trainset = torch.utils.data.Subset(dataset, train_idx)
    testset = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(
        trainset,
        batch_size  = batch_size,
        shuffle     = True,
        drop_last   = False
    )

    test_loader = DataLoader(
        testset,
        batch_size  = batch_size,
        shuffle     = False,
        drop_last   = False
    )
    model = FeedForwardDNPU()

    _ = train (
        model,
        num_epochs      = 100,
        train_loader    = train_loader,
        test_loader     = test_loader,
        save            = True,
        DNPU_train_enabled = True
    )

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
    # optimizer = torch.optim.SGD(
    #     [
    #         # {"params": model.dnpu_layer.weight, "lr" : 5e-3},
    #         {"params": model.linear_layer.parameters(), "lr" : 5e-4},
    #     ],
    #     momentum    = 0.9,
    # )
    optimizer = torch.optim.SGD(
        [
            {"params": model.linear_2.parameters(), "lr": 5e-4},
            {"params": model.dnpu_2.weight, "lr" : 5e-4}
        ],
        lr = 5e-4,
        momentum=0.9
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
            if save:
                if DNPU_train_enabled:
                    np.savez("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernel_train_with_GD/data/training_data_dnpu_trained.npz", Loss=LOSS, Acc=accuracies)
                else:
                    np.savez("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernel_train_with_GD/data/training_data.npz", Loss=LOSS, Acc=accuracies)
                    

    return model.state_dict()

            
if __name__ == "__main__":

    batch_size = 8
    audios, labels = load_audio_dataset(
        data_dir        = (EMPTY, "C:/Users/Mohamadreza/Documents/ti_spoken_digits/female_train"),
        min_max_scale   = True,
        low_pass_filter = True,
        # same_size_audios: can be "NONE" or an "MAX"
        # None -> keep every audio as what it is
        # "MAX" -> extend to maximum audio
        # if "MAX" is chosen, data is returned as numpy arrays, otherwise as list
        same_size_audios = "MAX"
    )

    # dataset = AudioDataset(
    #     audios      = audios,
    #     labels      = labels,
    #     transforms  = ToTensor()
    # )

    # train_idx, test_idx = sklearn.model_selection.train_test_split(
    #     np.arange(dataset.__len__()),
    #     test_size       = .1,
    #     random_state    = 7,
    #     shuffle         = True,
    #     stratify        = dataset.__targets__()
    # )

    # # Subset dataset for train and val
    # trainset = torch.utils.data.Subset(dataset, train_idx)
    # testset = torch.utils.data.Subset(dataset, test_idx)

    # train_loader = DataLoader(
    #     trainset,
    #     batch_size  = batch_size,
    #     shuffle     = True,
    #     drop_last   = False
    # )

    # test_loader = DataLoader(
    #     testset,
    #     batch_size  = batch_size,
    #     shuffle     = False,
    #     drop_last   = False
    # )
    
    # model = MultipleDNPUClassifier(
    #     num_dnpus   = NUM_DNPUs
    # )
    model = FeedForwardDNPU()

    # _ = train(
    #     model,
    #     100,
    #     1e-3,
    #     1e-5,
    #     train_loader,
    #     test_loader,
    #     batch_size,
    #     save = False,
    #     DNPU_train_enabled = True
    # )

    layerwise_train(
        model = model,
        audios = audios,
        labels = labels
    )

    DRIVER.close_tasks()

    pass

