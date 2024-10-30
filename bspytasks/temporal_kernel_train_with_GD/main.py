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

import datetime

# global constants
SLOPE_LENGTH = 1500
REST_LENGTH = 3000
DRIVER = get_driver(
    load_configs(
        "configs/defaults/processors/hw.yaml"
    )["driver"]
)
EMPTY = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_in_house_arsenic/empty/"
GD_CONFIGS = load_configs("bspytasks/temporal_kernel_train_with_GD/gd_configs.yml")
DNPU_INPUT_ELECTRODE = GD_CONFIGS["inputs"][0]
DNPU_IDX = 0
NUM_DNPUs = 4

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
            for dnpu_layer in model.dnpu_layer.children():
                for i in range(len(dnpu_layer)):
                    for value in dnpu_layer[i].weight.detach().cpu():
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
        weight,
        train = True,
        dnpu_conv_index = 0,
        epoch_index = 0
        
):
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
            output, 6125, 4, 12500
        )
        outputs[i] = output[::10]
    # here save for inference
    timestamp_string = str(datetime.datetime.now().strftime("%Y-%m-%d %H%M%S"))
    PATH = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernel_train_with_GD/data/meas_28/inference_data_dnpu_conv"+str(dnpu_conv_index)+"_epoch"+str(epoch_index)+timestamp_string+".npy"
    if train == False:
        np.save(
            PATH,
            outputs
        )
    return outputs

def backward_measurement(
    # weight it DNPU control voltages
    weight
):
    # creating the "random" input
    NUM_RANDOM_EXP = 5
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
    def forward(ctx, input, weight, train = True, dnpu_conv_index = 0, epoch_index = 0):
        # input -> batch_size * audio_length
        # e.g., 32 * 12,500
        ctx.save_for_backward(weight)
        outputs = single_forward_measurement(
                    input  = input.detach().cpu().numpy(),
                    weight = weight.detach().cpu().numpy(),
                    train = train,
                    dnpu_conv_index = dnpu_conv_index,
                    epoch_index = epoch_index
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
        with open("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernel_train_with_GD/data/meas_28/cvs.pkl", "wb") as fp:
            pickle.dump(LIST_OF_CVs, fp)
        with open("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernel_train_with_GD/data/meas_28/grads.pkl", "wb") as fp:
            pickle.dump(LIST_OF_GRADs, fp)
        with open("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernel_train_with_GD/data/meas_28/drvs.pkl", "wb") as fp:
            pickle.dump(LIST_OF_DERIVATIVES, fp)
        DNPU_IDX += 1
        
        return None, grad_weight, None, None, None
    
class DNPULayer(nn.Module):
    def __init__(self, dnpu_conv_index) -> None:
        super(DNPULayer, self).__init__()
        # defining control voltages (6) as learnable parameters for backward
        self.weight = nn.Parameter(
            0.35 * (torch.rand(6) - 0.5)
        )
        self.dnpu_conv_index = dnpu_conv_index
    def forward(self, input, train = True, epoch_index = 0):
        return DNPUFunction.apply(input, self.weight, train, self.dnpu_conv_index, epoch_index)
    def _return_params(self):
        return self.weight.detach().cpu().numpy()

class MultipleDNPUConvLayer(nn.Module):
    def __init__(self, num_dnpus) -> None:
        super(MultipleDNPUConvLayer, self).__init__()
        self.num_dnpus = num_dnpus
        self.dnpus = nn.ModuleList([DNPULayer(i) for i in range(self.num_dnpus)])

    def forward(self, x, train = True, epoch_index =0):
        out = torch.zeros((16, self.num_dnpus, 971))
        for i in range(self.num_dnpus):
            out[:, i, :] = self.dnpus[i](x, train = train, epoch_index = epoch_index)
        return out

class DNPUClassifierWithLinear(nn.Module):
    def __init__(self, num_dnpus = 4, n_output=10, ) -> None:
        super().__init__()
        self.num_dnpus = num_dnpus
        self.dnpu_layer = MultipleDNPUConvLayer(self.num_dnpus)
        self.bn = nn.BatchNorm1d(self.num_dnpus)
        self.flat = nn.Flatten()
        self.linear = nn.Linear(self.num_dnpus * 971, n_output)
    
    def forward(self, x, train = True, epoch_index =0):
        x = self.dnpu_layer(x, train = train, epoch_index = epoch_index)
        x = self.bn(x)
        x = self.linear(self.flat(x))

        return F.log_softmax(x, dim=1)


class M3(nn.Module):
    def __init__(self, n_input=1, n_output=10, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, 2, kernel_size = 80, stride=stride)
        self.bn2 = nn.BatchNorm1d(2)
        self.pool1 = nn.MaxPool1d(8)
        self.conv2 = nn.Conv1d(2, n_channel, kernel_size = 3)
        self.bn3 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(8)
        self.conv3 = nn.Conv1d(n_channel, n_channel, kernel_size = 3)
        self.bn4 = nn.BatchNorm1d(n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(n_channel, n_output)

    def forward(self, x, train = None, epoch_index = None):
        x = x.reshape(16, 1, 9710)

        x = self.conv1(x)
        x = F.silu(self.bn2(x))
        x = self.pool1(x)

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
    
class M5(nn.Module):
    def __init__(self, n_input=1, n_output=10, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, 2, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm1d(2)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(2, n_channel, kernel_size = 3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size = 3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size = 3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(3)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x, train = None, epoch_index = None):
        x = x.reshape(16, 1, 9710)
        x = self.conv1(x)
        x = F.silu(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.silu(self.bn2(x))
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.silu(self.bn3(x))
        x = self.pool3(x)

        x = self.conv4(x)
        x = F.silu(self.bn4(x))
        x = self.pool4(x)
        
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

class DNPUClassifier(nn.Module):
    def __init__(self, n_input = 2, n_output = 10, n_channel = 32):
        super().__init__()
        # self.dnpu_layer = DNPULayer()
        self.n_input = n_input
        self.dnpu_layer = MultipleDNPUConvLayer(num_dnpus=2)
        self.bn1 = nn.BatchNorm1d(n_input)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_input, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x, train = True, epoch_index = 0):
        x = self.dnpu_layer(x, train = train, epoch_index = epoch_index)
        x = x.reshape(16, self.n_input, 971)
        x = F.tanh(self.bn1(x))

        x = self.pool1(x)
        x = self.conv2(x)
        x = F.silu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.silu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.silu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


# def layerwise_train(
#         model,
#         audios = None,
#         labels = None,
#         batch_size = 16
# ):
#     # # Fetch the control voltages from DNPU Layer
#     initial_control_voltages = model._get_params()

#     # Measure (all) train and test sets
#     outputs = single_forward_measurement(
#         input = audios,
#         weight = initial_control_voltages
#     )

#     # Train linear layer with measured data (num_epochs // 2) and report accuracy
#     dataset = AudioDataset(
#         audios      = outputs,
#         labels      = labels,
#         transforms  = ToTensor()
#     )

#     train_idx, test_idx = sklearn.model_selection.train_test_split(
#         np.arange(dataset.__len__()),
#         test_size       = .1,
#         random_state    = 7,
#         shuffle         = True,
#         stratify        = dataset.__targets__()
#     )

#     # Subset dataset for train and val
#     trainset = torch.utils.data.Subset(dataset, train_idx)
#     testset = torch.utils.data.Subset(dataset, test_idx)

#     train_loader = DataLoader(
#         trainset,
#         batch_size  = batch_size, 
#         shuffle     = True,
#         drop_last   = True
#     )

#     test_loader = DataLoader(
#         testset,
#         batch_size  = batch_size,
#         shuffle     = True,
#         drop_last   = True
#     )

#     model = OnlyConvLayer()

#     model_state_dict = train(
#         model,
#         num_epochs      = 100,
#         train_loader    = train_loader,
#         test_loader     = test_loader,
#         save            = True
        #  )

#     print()


#     # # Activate DNPU layer training; Train both layers normally
#     # dataset = AudioDataset(
#     #     audios      = audios,
#     #     labels      = labels,
#     #     transforms  = ToTensor()
#     # )

#     # train_idx, test_idx = sklearn.model_selection.train_test_split(
#     #     np.arange(dataset.__len__()),
#     #     test_size       = .1,
#     #     random_state    = 7,
#     #     shuffle         = True,
#     #     stratify        = dataset.__targets__()
#     # )

#     # # Subset dataset for train and val
#     # trainset = torch.utils.data.Subset(dataset, train_idx)
#     # testset = torch.utils.data.Subset(dataset, test_idx)

#     # train_loader = DataLoader(
#     #     trainset,
#     #     batch_size  = batch_size,
#     #     shuffle     = True,
#     #     drop_last   = False
#     # )

#     # test_loader = DataLoader(
#     #     testset,
#     #     batch_size  = batch_size,
#     #     shuffle     = False,
#     #     drop_last   = False
#     # )
#     # model = DNPUClassifier()

#     # with torch.no_grad():
#     #     model.linear_layer.weight.copy_(model_state_dict['linear_layer.weight'])
#     #     model.linear_layer.bias.copy_(model_state_dict["linear_layer.bias"])
#     #     for i in range(0, len(model.dnpu_layers)):
#     #         model.dnpu_layer.weight.copy_(torch.tensor(initial_control_voltages[i]))

#     # # set the linear layer NOT to be trained
#     # with torch.no_grad():
#     #     model.linear_layer.requires_grad_(False)

#     _ = train (
#         model,
#         num_epochs      = 100,
#         train_loader    = train_loader,
#         test_loader     = test_loader,
#         save            = True,
#     )

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
    loss_fn = CustomLossFunction(model, min = -0.35, max = 0.35)
    # optimizer = torch.optim.SGD(
    #     [
    #         # {"params": model.dnpu_layer.weight, "lr" : 5e-2},
    #         {"params": model.conv1.parameters(), "lr" : 5e-4},
    #         {"params": model.bn1.parameters(), "lr" : 5e-4},
    #         {"params": model.conv2.parameters(), "lr" : 5e-4},
    #         {"params": model.bn2.parameters(), "lr" : 5e-4},
    #         {"params": model.fc1.parameters(), "lr" : 5e-4},
    #     ],
    #     lr = 5e-4,
    #     momentum    = 0.9,
    # )

    optimizer = torch.optim.AdamW(
        # [
        #     {"params": model.dnpu_layer.weight, "lr" : 1e-3},
        #     {"params": model.conv_1d.parameters(), "lr" : 1e-3},
        #     {"params": model.linear_layer.parameters(), "lr" : 1e-3},
        # ],     
        model.parameters(),   
        lr              = 0.0005, 
        weight_decay    = 1e-6
    )

    for epoch in range(num_epochs):
        if epoch != 0:
            model.eval()
            with torch.no_grad():
                correct, total = 0, 0
                for i, (data, label) in enumerate(test_loader):
                    label = label.type(torch.LongTensor)
                    output = torch.squeeze(model(data, train = False, epoch_index = epoch))
                    _, predicted = torch.max(output, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                accuracies.append(100*correct/total)  
        # saving the test set
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            current_loss = 0.
            for i, (data, label) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                label = label.type(torch.LongTensor)
                output = torch.squeeze(model(data, train=True, epoch_index = epoch))
                loss = loss_fn(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # clamping DNPU control voltages
                # DNPUControlVoltageClamp(model, -0.45, 0.45)
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
                        'loss': LOSS[i],
                        'accuracy': accuracies[-1]
                    }, 
                    "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernel_train_with_GD/data/meas_28/model.pt"
                )
                    

    return model.state_dict()

            
if __name__ == "__main__":

    batch_size = 16
    audios, labels = load_audio_dataset(
        data_dir        = (EMPTY, "C:/Users/Mohamadreza/Documents/ti_spoken_digits/female_speaker"),
        min_max_scale   = True,
        low_pass_filter = True,
        # same_size_audios: can be "NONE" or an "MAX"
        # None -> keep every audio as what it is
        # "MAX" -> extend to maximum audio
        # if "MAX" is chosen, data is returned as numpy arrays, otherwise as list
        same_size_audios = "MAX"
    )

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
        drop_last   = True
    )

    test_loader = DataLoader(
        testset,
        batch_size  = batch_size,
        shuffle     = False,
        drop_last   = True
    )
    
    # model = DNPUClassifier()

    model = DNPUClassifierWithLinear(num_dnpus = 4)
    # model = M3()

    _ = train(
        model = model,
        num_epochs = 100,
        train_loader = train_loader,
        test_loader = test_loader,
        save = False
    )

    # layerwise_train(
    #     model = model,
    #     audios = audios,
    #     labels = labels
    # )

    DRIVER.close_tasks()

    pass

