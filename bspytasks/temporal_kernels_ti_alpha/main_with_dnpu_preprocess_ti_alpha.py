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

_global_param = 32
_mean, _std = 0, 0

def butter_lowpass(cutoff, order, fs):
    return scipy.signal.butter( 
        N = order, 
        Wn = cutoff, 
        btype = 'lowpass', 
        analog=False,
        fs= fs
    )

def butter_lowpass_filter(data, cutoff, order, fs):
    b, a = butter_lowpass(cutoff, order = order, fs=fs)
    y = scipy.signal.filtfilt(
        b = b, 
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
    min_max_scale = None,
    low_pass_filter = None,
):
    dataset, label = [], []

    for subdir, _, files in chain.from_iterable(
        os.walk(path) for path in data_dir
    ):
        for file in files:
            tmp, _ = librosa.load(os.path.join(subdir, file), sr=12500, mono=True, dtype=np.float32)
            tmp, _ = librosa.effects.trim(tmp, frame_length = 128, hop_length = 8, top_db=25)

            if min_max_scale == True:
                scale = np.max(np.abs(tmp))
                tmp = tmp * (1/scale) * 1.
            
            if low_pass_filter == True:
                tmp = butter_lowpass_filter(
                                        tmp, 4500, 3, 12500
            )
            dataset.append(tmp)
            label.append(file[1])
    
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
        data_dir        = (empty, "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernels_ti_alpha/recordings/"),
        min_max_scale   = True,
        low_pass_filter = True    
    )

    driver = get_driver(configs["driver"])

    rand_matrix = np.random.uniform(
        -0.25, 
        0.25, 
        size = (len(dnpu_control_indeces), n_output_channels)
    )
    
    dnpu_output_train = []

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
                meas_input              = meas_inputs,
                dnpu_control_indeces    = dnpu_control_indeces,
                slope_length            = slope_length,
                projection_idx          = p_idx,
                rand_matrix             = rand_matrix
            )

            output = driver.forward_numpy(meas_inputs.T)
            output = output[slope_length + rest_length : -slope_length, 0]

            output = butter_lowpass_filter(
                output, 4500, 3, 12500
            )

            dnpu_output_train.append(output)
            
    np.save("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/in_elec_4_var/dnpu_output.npy", dnpu_output_train)
    np.save("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/in_elec_4_var/labels.npy", train_labels)

    driver.close_tasks()

def post_dnpu_down_sample(
    input, # (1300, 32, 19000) -> (1300, 32, 1180); 780
):
    # input -> 41600
    tmp = np.zeros((1300, 32, 12480))
    k = 0
    for i in range(len(input)):
        tmp[k][i%32] = np.pad(
            input[i] - np.mean(input[i]),
            pad_width = (0, 12480 - len(input[i])),
            mode = 'constant',
            constant_values= 0.
        )
        if i % 32 == 0 and i != 0:
            k += 1

    output = np.zeros((1300, 32, 780))
    for i in range(len(tmp)):
        output[i,:,:] = tmp[i,:,::16]

    return output

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

        self.dataset_tmp = np.load(data_dir, allow_pickle=True)
        self.label = np.load(label_dir)

        le = sklearn.preprocessing.LabelEncoder()
        le.fit(self.label)
        self.label = le.transform(self.label)

        self.dataset_tmp = post_dnpu_down_sample(self.dataset_tmp)

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
   
if __name__ == '__main__':

    # from brainspy.utils.io import load_configs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    empty = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_in_house_arsenic/empty/"

    # configs = load_configs(
    #     "configs/defaults/processors/hw.yaml"
    # )

    # np.random.seed(8)
    # measurement(
    #     configs                 = configs,
    #     n_output_channels       = 32,
    #     slope_length            = 200,
    #     rest_length             = 2000,
    #     dnpu_input_index        = 4,
    #     dnpu_control_indeces    =[0, 1, 2, 3, 5, 6],
    # )

    dataset =  DNPUAudioDataset(
        data_dir                = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/in_elec_4_var/dnpu_output.npy",
        label_dir               = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/in_elec_4_var/labels.npy",
        transform               = ToTensor(),
        projections_to_remove   = []
    )

    print("")
                                    
    # _mean, _std = get_mean_and_std(
    #     DataLoader(
    #         dataset_for_mean,
    #         batch_size= len(dataset_for_mean),
    #         shuffle= False,
    #         drop_last= False   
    #     )
    # )

    # transform = transforms.Compose(
    #     [ToTensor(), Normalize()]
    # )

    # batch_size = 32

    # dataset = DNPUAudioDataset(
    #     data_dir                = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/in_elec_4_ch_16/dnpu_output.npy",
    #     label_dir               = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/in_elec_4_ch_16/labels.npy",
    #     transform               = transform,
    #     projections_to_remove   = []
    # )

    # print("")

