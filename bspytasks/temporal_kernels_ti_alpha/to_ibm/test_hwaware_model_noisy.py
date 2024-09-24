import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Conv1Wrapper(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size,1)
        )
    def forward(self, x):
        x = self.conv(x.unsqueeze(3))
        return x.squeeze(3)

def wrap_conv1d(model):
    for m in model.modules():
        if isinstance(m, (Conv1Wrapper, torch.nn.Conv2d)): continue
        for n,child in m.named_children():
            if isinstance(child, torch.nn.Conv1d):
                weight = child.weight.data
                bias = child.bias.data
                new_mod = Conv1Wrapper(in_channels=weight.shape[1], out_channels=weight.shape[0], kernel_size=weight.shape[2])
                new_mod.conv.weight.data = weight.unsqueeze(3)
                new_mod.conv.bias.data = bias
                m.__setattr__(n, new_mod)
    return model

class M4Compact(nn.Module):
    def __init__(self, input_ch, n_channels=32) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_ch)
        self.conv1 = nn.Conv1d(input_ch, 32, kernel_size=8)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(8)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3)
        self.bn3   = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(4)
        self.fc1   = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.bn1(x)

        x = self.conv1(x)
        x = self.bn2(x)
        x = F.tanh(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn3(x)
        x = F.tanh(x)

        x = self.pool2(x)

        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

def validate(model, test_loader):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = torch.squeeze(model(inputs))
            _, predicted = torch.max(outputs, 1)
            total += len(targets)
            correct += (predicted == targets).sum().item()
    return 100 * correct / total

def add_noise(model, cs):
    noisy_model = deepcopy(model)
    for m in noisy_model.modules():
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
            w = m.weight.data
            w_abs_max = w.abs().max()
            # - Bring to unit
            w /= w_abs_max
            noise = torch.randn_like(w) * (cs[3] + cs[2]*w.abs() + cs[1]*w.abs()**2 + cs[0]*w.abs()**3)
            w += noise
            w *= w_abs_max
            m.weight.data = w 
    return noisy_model



    def apply_noise(weight):
        std_dev = 0.067
        with no_grad():
            noise = std_dev * weight.abs().max() * randn_like(weight, device=weight.device)
            out_weight = weight.clone() + noise
        return out_weight
    
    def forward(self, input):
        weight = self.weight
        if self.training:
            noisy_weights = apply_noise(weight)
        else:
            noisy_weights = weight
        return self._conv_forward(input, noisy_weights)


if __name__ == '__main__':

    batch_size = 10
    sample_shape = (32, 500)
    
    # - Create dataset
    np_data = np.load("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernels/to_ibm/testset.npy", allow_pickle=True)
    torch_data = torch.empty(size=(len(np_data),sample_shape[0], sample_shape[1]))
    torch_targets = torch.empty(size=(len(np_data),))
    for sample_idx, sample in enumerate(np_data):
        torch_data[sample_idx] = sample["audio_data"]
        torch_targets[sample_idx] = sample["audio_label"].long()

    dataset = torch.utils.data.TensorDataset(torch_data, torch_targets)

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = M4Compact(input_ch=32)
    model.load_state_dict(torch.load("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernels/to_ibm/HWAware_model_a2_5_noisy_2.pt", map_location=device))
    model = model.to(device= device)
    model.eval()
    model = wrap_conv1d(model)

    # - Obtain the non-noisy accuracy
    fp_acc = validate(model, test_loader)

    # - Obtain accuracy under noise influence
    cs = [0.294462, -0.452322, 0.226837, 0.015175]

    accs = []
    for _ in range(500):
        noisy_model = add_noise(model, cs)
        accs.append(validate(noisy_model, test_loader))
        print(f"FP acc. {fp_acc}% noisy acc {accs[-1]}%")
    
    print("Accuracy mean: ", np.mean(accs))
    print("Accuracy standard deviation: ", np.std(accs))
    print("Accuracy variance: ", np.var(accs))
    print("Min/max of noise accuracy: ", np.min(accs), "/", np.max(accs))
    print("Accuracy fluctuation: ", np.max(accs) - np.min(accs) ,"%")
    

    # cnt, bins = np.histogram(accs, bins=20)
    # plt.stairs(cnt, bins)
    # plt.show()

    _ = plt.hist(accs, bins=10)
    plt.title("Distribution of validation accuracy after clipping and noise injection")
    plt.xlim(right=100)
    plt.xlabel("Accuracy (100%)")
    plt.ylabel("Distribution over 500 runs")
    plt.show()