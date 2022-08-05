from distutils.command.config import config
from pickletools import optimize
import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

from torchvision import transforms

import dataset_ti_alpha

from brainspy.utils.manager import get_driver

from sklearn import preprocessing

input_size = 1024
hidden_layer_size = 512
num_classes = 26


class LinearLayer(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_classes) -> None:
        super(LinearLayer, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_layer_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_layer_size, num_classes)
    
    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        return output

def train(
        model,
        num_epoch,
        train_loader
        ):

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    total_step = len(train_loader)
    correct = 0
    predicted = 0
    total = 0

    for epoch in range(num_epoch):
        correct = 0
        for i, (voice, label) in enumerate(train_loader):
            pred = model(voice)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 2 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epoch, i+1, total_step, loss.item()))
            
            # _, predicted = torch.max(pred.data, 1)
            # total += label.size(0)
            # correct += (predicted == label).sum().item()
            # print("Training accuracy = ", correct/total_step)
            




def set_random_cvs( 
                meas_input,
                dnpu_control_indeces,
                slop_length
                ):
    for i in range(len(dnpu_control_indeces)):
        rand_value = np.random.uniform(-0.2, 0.2, 1)
        ramp_up = np.linspace(0, rand_value[0], slop_length)
        plateau = np.linspace(rand_value[0], rand_value[0], np.shape(meas_input)[1] - 2 * slop_length)
        ramp_down = np.linspace(rand_value[0], 0, slop_length)
        meas_input[dnpu_control_indeces[i], :] = np.concatenate((ramp_up, plateau, ramp_down))

    return meas_input

def measurement(
                configs,
                num_projections,
                dnpu_input_index,
                dnpu_control_indeces,
                slope_length
                ):

    dataset, labels = dataset_ti_alpha.load_train_dataset(all_dataset=False)
    dataset = dataset_ti_alpha.remove_silence(dataset=dataset)

    dataset_after_projection = []
    labels_after_projection = []
    
    for d in range(1): # range(len(dataset)):
        for n_p in range(num_projections):
            meas_input = np.zeros((6, 2 * slope_length + np.shape(dataset[d])[0]))
            meas_input[dnpu_input_index,slope_length:-slope_length] = 20 * dataset[d][:]
            meas_input = set_random_cvs(meas_input, dnpu_control_indeces, slope_length)
            driver = get_driver(configs=configs["driver"])
            output = driver.forward_numpy(meas_input.T)
            driver.close_tasks()
            dataset_after_projection.append(output[slope_length:-slope_length])
            labels_after_projection.append(labels[d])
    
    return dataset_after_projection, labels_after_projection

def convert_to_torch_dataset(dataset_after_projection, labels_after_projection):
    tensor_x = torch.Tensor(dataset_after_projection)
    # tensor_y = torch.Tensor(labels_after_projection)

    le = preprocessing.LabelEncoder()
    tensor_y = le.fit_transform(labels_after_projection)

    dataset = TensorDataset(tensor_x, torch.as_tensor(tensor_y))
    dataloader_dataset = DataLoader(dataset)

    return dataset, dataloader_dataset


if __name__ == '__main__':
    from brainspy.utils.io import load_configs

    configs = load_configs('configs/defaults/processors/hw.yaml')
    slope_length = 10000

    dataset_after_projection, labels_after_projection = measurement(
                                                                    configs=configs,
                                                                    num_projections= 5, # 128,
                                                                    dnpu_input_index=1,
                                                                    dnpu_control_indeces=[0, 2, 3, 4, 5],
                                                                    slope_length=slope_length)

    # remove DC and  50 Hz noise
    for i in range(len(dataset_after_projection)):
        freq = np.fft.rfftfreq(n=np.shape(dataset_after_projection[i])[0], d=1/12500)
        f_transform = np.fft.rfft(dataset_after_projection[i][:,0])
        dataset_after_projection[i] = np.fft.irfft(f_transform[60:6000])

    # choosing 1024 samples from output data
    min_length = np.inf
    for i in range(len(dataset_after_projection)):
        if np.shape(dataset_after_projection[i])[0] < min_length:
            min_length = np.shape(dataset_after_projection[i])[0]
    idx = np.arange(0, min_length, dtype=int)
    idx = np.sort(np.random.choice(idx, 1024))

    for i in range(len(dataset_after_projection)):
        dataset_after_projection[i] = dataset_after_projection[i][idx]

    t_dataset, t_dataloader = convert_to_torch_dataset(
                                                        dataset_after_projection, 
                                                        labels_after_projection
                                                    )

    linear_layer = LinearLayer(
                            input_size=1024,
                            hidden_layer_size=512,
                            num_classes=26
    )

    train(
        model = linear_layer,
        num_epoch=20,
        train_loader= t_dataloader
    )

# plt.plot(np.fft.rfftfreq(n=np.shape(output)[0],d=1/12500)[55:-55],np.fft.rfft(output[:,0])[55:-55])
