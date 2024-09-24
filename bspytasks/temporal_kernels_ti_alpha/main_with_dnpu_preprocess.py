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

from torch.utils.data import DataLoader, Dataset, random_split

import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np

from tqdm import tqdm
from itertools import chain
from torchvision import transforms
from brainspy.utils.manager import get_driver

from main_with_dnpu_preprocess_hw_aware_train import apply_noise, noisy_Linear, noisy_Conv1d, train, hw_train, test, add_noise

_global_param = 32
_mean, _std = 0, 0

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

def set_random_control_voltages( 
                meas_input,
                dnpu_control_indeces,
                slope_length,
                projection_idx,
                rand_matrix #  (len(dnpu_control_indeces), 128) -> ((6, 128))
                ):
    for i in range(len(dnpu_control_indeces)):
        ramp_up = np.linspace(0, rand_matrix[i, projection_idx], slope_length)
        plateau = np.linspace(rand_matrix[i, projection_idx], rand_matrix[i, projection_idx], np.shape(meas_input)[1] - 2 * slope_length)
        ramp_down = np.linspace(rand_matrix[i, projection_idx], 0, slope_length)
        meas_input[dnpu_control_indeces[i], :] = np.concatenate((ramp_up, plateau, ramp_down))

    return meas_input

def load_audio_dataset(
    data_dir = None,
    min_max_scale = None
):
    dataset, label = [], []

    for subdir, _, files in chain.from_iterable(
        os.walk(path) for path in data_dir
    ):
        for file in files:
            tmp, _ = librosa.load(os.path.join(subdir, file), sr=None, mono=True, dtype=np.float32)
            if min_max_scale == True:
                scale = np.max(np.abs(tmp))
                tmp = tmp * (1/scale)
            tmp, _ = librosa.effects.trim(tmp, frame_length=512, hop_length=128, top_db=30)
            dataset.append(tmp)
            label.append(file[0])
    
    return dataset, label

def measurement(
    configs,
    n_output_channels, # number of dnpu configs
    slope_length,
    rest_length,
    dnpu_input_index,
    dnpu_control_indeces
):

    empty = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_in_house_arsenic/empty/"

    train_audios, train_labels = load_audio_dataset(
        data_dir        = (empty, "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spoken_digit_task/spoken_mnist/recordings/theo/train"),
        min_max_scale   = True
    )
    test_audios, test_labels = load_audio_dataset(
        data_dir        = (empty, "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spoken_digit_task/spoken_mnist/recordings/theo/test"),
        min_max_scale   = True
    )

    # in_len = 8000, k_size= 80, stride = 16
    dnpu_output_train = np.zeros((len(train_audios), n_output_channels, 496))
    dnpu_output_test = np.zeros((len(test_audios), n_output_channels, 496))

    driver = get_driver(configs["driver"])

    # Dividing random voltage of neighbouring electrodes by a factor of 2
    rand_matrix = np.random.uniform(
        -0.4, 
        0.4, 
        size = (len(dnpu_control_indeces), n_output_channels)
    )
    for i in range(0, len(rand_matrix)):
        if i == 0 or i == 5:
            rand_matrix[i][:] = rand_matrix[i][:] / 4

    for d in tqdm(range(len(train_audios)), desc="Measuring training data..."):
        for p_idx in tqdm(range(n_output_channels), desc = "Measuring projection"):

            # dnpu measurement input
            meas_inputs = np.zeros(
                (
                    len(dnpu_control_indeces) + 1,
                    len(train_audios[d]) + 2 * slope_length + rest_length
                )
            )

            meas_inputs[dnpu_input_index,slope_length + rest_length : -slope_length] = train_audios[d]
            meas_inputs = set_random_control_voltages(
                meas_input= meas_inputs,
                dnpu_control_indeces= dnpu_control_indeces,
                slope_length= slope_length,
                projection_idx= p_idx,
                rand_matrix= rand_matrix
            )

            output = driver.forward_numpy(meas_inputs.T)
            output = output[slope_length + rest_length : -slope_length, 0]

            avg = np.mean(output)

            output = output - avg

            output = butter_lowpass_filter(
                output, 3900, 4, 8000
            )

            # Padding output to 8000
            padded_output = np.zeros((8000))
            if len(output) < 8000:
                padded_output = np.pad(
                    output,
                    pad_width       = (0, 8000 - len(output)),
                    mode            = 'constant',
                    constant_values = 0.
                )
            else:
                padded_output = output[0:8000]
                print("Warning! Cropping data...")

            # assigning the measuremet output to the dnpu_output_train
            for k in range(0, 496):
                dnpu_output_train[d][p_idx][k] = padded_output[k * 16 + 80 - 1]
            
    np.save("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/theo/train/dnpu_output_train.npy", dnpu_output_train)
    np.save("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/theo/train/train_labels.npy", train_labels)

    for d in tqdm(range(len(test_audios)), desc = "Measuring test data: "):
        for p_idx in tqdm(range(n_output_channels), desc = "Measuring projection"):

            # dnpu measurement input
            meas_inputs = np.zeros(
                (
                    len(dnpu_control_indeces) + 1,
                    len(test_audios[d]) + 2 * slope_length + rest_length
                )
            )

            meas_inputs[dnpu_input_index,slope_length + rest_length : -slope_length] = test_audios[d]
            meas_inputs = set_random_control_voltages(
                meas_input= meas_inputs,
                dnpu_control_indeces= dnpu_control_indeces,
                slope_length= slope_length,
                projection_idx= p_idx,
                rand_matrix= rand_matrix
            )

            output = driver.forward_numpy(meas_inputs.T)
            output = output[slope_length + rest_length : -slope_length, 0]

            avg = np.mean(output)

            output = output - avg

            output = butter_lowpass_filter(
                output, 3900, 4, 8000
            )

            # Padding output to 8000
            padded_output = np.zeros((8000))
            padded_output = np.pad(
                output,
                pad_width       = (0, 8000 - len(output)),
                mode            = 'constant',
                constant_values = 0.
            )

            # assigning the measuremet output to the dnpu_output_train
            for k in range(0, 496):
                dnpu_output_test[d][p_idx][k] = padded_output[k * 16 + 80 - 1]
            
    np.save("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/theo/test/dnpu_output_test.npy", dnpu_output_test)
    np.save("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/theo/test/test_labels.npy", test_labels)

    driver.close_tasks()
    

class M4Compact(nn.Module):
    def __init__(self, input_ch, n_channels=64, hw_train = False) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_ch)

        if hw_train:
            self.conv1 = noisy_Conv1d(input_ch, n_channels, kernel_size= 8, device=device)
        else:
            self.conv1 = nn.Conv1d(input_ch, n_channels, kernel_size=8)

        self.bn2 = nn.BatchNorm1d(n_channels)
        self.pool1 = nn.MaxPool1d(8)

        if hw_train:
            self.conv2 = noisy_Conv1d(n_channels, n_channels, kernel_size=3, device=device)
        else:
            self.conv2 = nn.Conv1d(n_channels, n_channels, kernel_size=3)

        self.bn3   = nn.BatchNorm1d(n_channels)

        if hw_train:
            self.fc1 = noisy_Linear(n_channels, 10, device=device)
        else:
            self.fc1   = nn.Linear(n_channels, 10)
    
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

        for i in range(len(audio_data)):
            audio_data[i] = (audio_data[i] - _mean[i] / _std[i])

        return {
            'audio_data' : audio_data,
            'audio_label': audio_label
        }

# calculates mean and std
def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data in enumerate(dataloader):
        # Mean over batch, audio signal, but not over the channels
        audio_data = data[1]['audio_data']
        channels_sum += torch.mean(audio_data, dim=[0, 2])
        channels_squared_sum += torch.mean(audio_data**2, dim=[0, 2])
        num_batches += 1
    
    mean_temp = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std_temp = (channels_squared_sum / num_batches - mean_temp ** 2) ** 0.5

    return mean_temp, std_temp

# loads from .npy files
class DNPUAudioDataset(Dataset):
    def __init__(
        self,
        data_dir,
        label_dir,
        transform,
        projections_to_remove = [],
    ) -> None:
    
        self.transform = transform

        self.dataset_tmp = np.load(data_dir)
        self.label = np.load(label_dir)

        if projections_to_remove:
            # new dataset size -> 450, 32-rem, 496
            self.dataset = np.zeros((np.shape(self.dataset_tmp)[0], _global_param - len(projections_to_remove), np.shape(self.dataset_tmp)[2]))
            # loop over new number of projections
            cnt = 0
            for i in range(0, _global_param):
                if not(i in projections_to_remove):
                    self.dataset[:, cnt, :] = self.dataset_tmp[:, i, :]
                    cnt += 1
        else:
            self.dataset = self.dataset_tmp
        
        assert len(self.dataset) == len(self.label), "Error in loading data!"
    
    def __len__(self) -> None:
        return len(self.dataset)

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    # from brainspy.utils.io import load_configs

    empty = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_in_house_arsenic/empty/"

    # configs = load_configs(
    #     "configs/defaults/processors/hw.yaml"
    # )

    # np.random.seed(70)
    # measurement(
    #     configs=configs,
    #     n_output_channels       = 32,
    #     slope_length            = 200,
    #     rest_length             = 800,
    #     dnpu_input_index        = 2,
    #     dnpu_control_indeces    =[0, 1, 3, 4, 5, 6],
    # )

    projections_to_remove = []

    # Bellow is a bit complex type fo coding. I concatenate the whole training dataset to calculate mean and std (global variables)
    # Then, I use these variables to normalize both training AND test dataset.
    # dataset_for_mean = torch.utils.data.ConcatDataset(
    #                                 [
    #                                     DNPUAudioDataset(
    #                                         data_dir    = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/lucas/train/dnpu_output_train.npy",
    #                                         label_dir= "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/lucas/train/train_labels.npy",
    #                                         transform   = transforms.Compose([ToTensor()]),
    #                                     ),
    #                                     DNPUAudioDataset(
    #                                         data_dir    = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/george/train/dnpu_output_train.npy",
    #                                         label_dir= "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/george/train/train_labels.npy",
    #                                         transform   = transforms.Compose([ToTensor()]),
    #                                     ),
    #                                     DNPUAudioDataset(
    #                                         data_dir    = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/jackson/train/dnpu_output_train.npy",
    #                                         label_dir= "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/jackson/train/train_labels.npy",
    #                                         transform   = transforms.Compose([ToTensor()]),
    #                                     ),
    #                                     DNPUAudioDataset(
    #                                         data_dir    = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/nicolas/train/dnpu_output_train.npy",
    #                                         label_dir= "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/nicolas/train/train_labels.npy",
    #                                         transform   = transforms.Compose([ToTensor()]),
    #                                     ),
    #                                     DNPUAudioDataset(
    #                                         data_dir    = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/theo/train/dnpu_output_train.npy",
    #                                         label_dir= "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/theo/train/train_labels.npy",
    #                                         transform   = transforms.Compose([ToTensor()]),
    #                                         projections_to_remove= projections_to_remove
    #                                     ),
    #                                 ]
    # )

    # _mean, _std = get_mean_and_std(
    #     DataLoader(
    #         dataset_for_mean,
    #         batch_size= len(dataset_for_mean),
    #         shuffle= False,
    #         drop_last= False   
    #     )
    # )

    # transform = transforms.Compose([
    #     ToTensor(),
    #     # mean-std normalize
    #     Normalize()
    # ])

    # # loading from new measurement
    # train_set1 = DNPUAudioDataset(
    #     data_dir    = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/lucas/train/dnpu_output_train.npy",
    #     label_dir= "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/lucas/train/train_labels.npy",
    #     transform   = transform,
    # )
    # train_set2 = DNPUAudioDataset(
    #     data_dir    = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/george/train/dnpu_output_train.npy",
    #     label_dir= "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/george/train/train_labels.npy",
    #     transform   = transform,
    # )
    # train_set3 = DNPUAudioDataset(
    #     data_dir    = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/jackson/train/dnpu_output_train.npy",
    #     label_dir= "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/jackson/train/train_labels.npy",
    #     transform   = transform,
    # )
    # train_set4 = DNPUAudioDataset(
    #     data_dir    = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/nicolas/train/dnpu_output_train.npy",
    #     label_dir= "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/nicolas/train/train_labels.npy",
    #     transform   = transform,
    # )
    # train_set5 = DNPUAudioDataset(
    #     data_dir    = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/theo/train/dnpu_output_train.npy",
    #     label_dir= "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/theo/train/train_labels.npy",
    #     transform   = transform,
    #     projections_to_remove= projections_to_remove
    # )

    # test_set1 = DNPUAudioDataset(
    #     data_dir    = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/lucas/test/dnpu_output_test.npy",
    #     label_dir= "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/lucas/test/test_labels.npy",
    #     transform   = transform,
    # )
    # test_set2 = DNPUAudioDataset(
    #     data_dir    = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/george/test/dnpu_output_test.npy",
    #     label_dir= "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/george/test/test_labels.npy",
    #     transform   = transform,
    # )
    # test_set3 = DNPUAudioDataset(
    #     data_dir    = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/jackson/test/dnpu_output_test.npy",
    #     label_dir= "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/jackson/test/test_labels.npy",
    #     transform   = transform,
    # )
    # test_set4 = DNPUAudioDataset(
    #     data_dir    = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/nicolas/test/dnpu_output_test.npy",
    #     label_dir= "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/nicolas/test/test_labels.npy",
    #     transform   = transform,
    # )
    # test_set5 = DNPUAudioDataset(
    #     data_dir    = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/theo/test/dnpu_output_test.npy",
    #     label_dir= "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/theo/test/test_labels.npy",
    #     transform   = transform,
    #     projections_to_remove= projections_to_remove
    # )

    # train_loader = DataLoader(
    #     torch.utils.data.ConcatDataset([train_set1, train_set2, train_set3, train_set4, train_set5]),
    #     # train_set1,
    #     batch_size  = 32,
    #     shuffle     = True,
    #     drop_last   = False
    # )

    # test_loader = DataLoader(
    #     torch.utils.data.ConcatDataset([test_set1, test_set2, test_set3, test_set4, test_set5]),
    #     # test_set1,
    #     batch_size  = 32,
    #     shuffle     = False,
    #     drop_last   = False
    # )

    train_loader = DataLoader(
        np.load("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/trainset.npy", allow_pickle= True),
        batch_size= 64,
        shuffle= True,
        drop_last= False
    )
    test_loader = DataLoader(
        np.load("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/testset.npy", allow_pickle= True),
        batch_size= 64,
        shuffle= True,
        drop_last= False
    )

    model = M4Compact(
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
        batch_size      = 32,
    )

    # loading the model
    model = M4Compact(
        input_ch = 32,
        hw_train= True
    )
    model.load_state_dict(torch.load("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernels/models/HWAware_model_a2_5.pt"))
    model.to(device)

    hw_train(
        model           = model,
        num_epochs      = 500,
        train_loader    = train_loader,
        test_loader     = test_loader,
        device          = device,
        batch_size      = 32,
        weight_decay    = 10e-4
    )

    # Evaluation
    model = M4Compact(
        input_ch = 32,
        hw_train= False
    )
    model.load_state_dict(torch.load("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernels/models/HWAware_model_a2_5_noisy.pt"))
    model.to(device)
    model.eval()

    # - Obtain the non-noisy accuracy
    fp_acc = test(model, test_loader, device)

    # - Obtain accuracy under noise influence
    cs = [0.294462, -0.452322, 0.226837, 0.015175]
    accs = []
    for _ in range(250):
        noisy_model = add_noise(model, cs)
        accs.append(test(noisy_model, test_loader, device))
        print(f"FP acc. {fp_acc}% noisy acc {accs[-1]}%")
    
    print("Average of noise accuracy: ", np.mean(accs))
    print("Min/max of noise accuracy: ", np.min(accs), "/", np.max(accs))
    print("Accuracy fluctuation: ", np.max(accs)-np.min(accs) ,"%")

    _ = plt.hist(accs, bins='auto')
    plt.title("Distribution of validation accuracy after clipping and noise injection")
    plt.xlim(right=100)
    plt.xlabel("Accuracy (100%)")
    plt.ylabel("Distribution over 200 runs")
    plt.show()

