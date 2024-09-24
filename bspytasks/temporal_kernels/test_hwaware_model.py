import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm


class M4Compact(nn.Module):
    def __init__(self, input_ch) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_ch)
        self.conv1 = nn.Conv1d(input_ch, 16, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(8)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=3)
        self.bn3   = nn.BatchNorm1d(16)
        self.pool2 = nn.MaxPool1d(8)
        self.fc1   = nn.Linear(16, 10)
    
    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

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

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = DataLoader(
        np.load("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti46/in_elec_4/testset.npy", allow_pickle=True),
        batch_size  = 2,
        shuffle     = False,
        drop_last   = False
    )

    model = M4Compact(
        input_ch = 32,
    )

    print("Number of learnable params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model.load_state_dict(
        torch.load(
            "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernels/models/HWAware_model_a4.pt",
            map_location = device
        )
    )

    model = model.to(
        device= device
    )

    model.eval()
    acc = test(
        model,
        test_loader,
        device
    )

    print("Test accuracy: ", acc)

    with torch.no_grad():
        for name, param in model.named_parameters():
            print("")
