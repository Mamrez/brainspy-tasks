from cgi import test
from distutils.command.config import config
from pickletools import optimize
from queue import Empty
from tracemalloc import reset_peak
from turtle import forward
from typing import NewType
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, SubsetRandomSampler, Dataset, random_split

import numpy as np
import gc, os, scipy, sys

from itertools import chain

from torchvision import transforms

import dataset_ti_spoken_digits

from brainspy.utils.manager import get_driver

from sklearn import preprocessing
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import seaborn

from torch.utils.tensorboard import SummaryWriter

class LinearLayer_ReLU(torch.nn.Module):
    def __init__(self, input_size, num_classes) -> None:
        super(LinearLayer_ReLU, self).__init__()
        self.linear_layer = torch.nn.Linear(input_size, num_classes)
        self.relu = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear_layer(x)
        return self.relu(x)

class LinearLayer(torch.nn.Module):
    def __init__(self, input_size, num_classes) -> None:
        super(LinearLayer, self).__init__()
        self.linear_layer = torch.nn.Linear(input_size, num_classes) 
    
    def forward(self, x):
        out = self.linear_layer(x)
        out = torch.log_softmax(out, dim=1)
        return out

class FCLayer(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_classes, drop_out_prob) -> None:
        super(FCLayer, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_layer_size)
        self.relu = torch.nn.ReLU()
        self.hidden_layer = torch.nn.Linear(hidden_layer_size, hidden_layer_size)
        self.dropout = torch.nn.Dropout(p=drop_out_prob)
        self.relu_2 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_layer_size, num_classes)
    
    def forward(self, x):

        out = self.fc1(x)
        out = self.relu(out)

        out = self.hidden_layer(out)
        out = self.relu_2(out)    
        out = self.dropout(out)  

        out = self.fc2(out)
        out = torch.log_softmax(out, dim=1)
        return out

class ConvLayer(torch.nn.Module):
    def __init__(self, num_classes, down_sample_no) -> None:
        super(ConvLayer, self).__init__()
        self.kernel_size = 256
        self.stride = 16
        self.out_channel = 8
        self.conv1 = torch.nn.Conv1d(
                                    in_channels     =   1,
                                    out_channels    =   self.out_channel ,
                                    kernel_size     =   self.kernel_size,
                                    stride          =   self.stride
        )
        self.relu = torch.nn.ReLU()
        self.flat = torch.nn.Flatten()
        self.fc1   = torch.nn.Linear(self.out_channel  * int((down_sample_no - self.kernel_size)/self.stride + 1), 256)
        self.fc2 = torch.nn.Linear(256, num_classes)
    
    def forward(self, x):
        out = self.conv1(x.reshape(batch_size, 1, down_sample_no))
        out = self.relu(out)
        out = self.flat(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = torch.log_softmax(out, dim=1)
        return out

class complexConvLayer(torch.nn.Module):
    def __init__(self, num_classes, down_sample_no) -> None:
        super(complexConvLayer, self).__init__()
        self.batch_norm = torch.nn.BatchNorm1d(num_features=1)
        self.layer_1 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(1),
            torch.nn.Conv1d(in_channels=1, out_channels=32 ,kernel_size=3, stride=1),
            torch.nn.SiLU()
        )
        self.layer_2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            torch.nn.SiLU()
        )
        self.maxpool_1 = torch.nn.MaxPool1d(2, 1)
        self.layer_3 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(32),
            torch.nn.Conv1d(32, 64, 3, 1),
            torch.nn.SiLU()
        )
        self.layer_4 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            torch.nn.SiLU()
        )
        self.maxpool_2 = torch.nn.MaxPool1d(2, 1)
        self.layer_5 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(64),
            torch.nn.Conv1d(64, 128, 3, 1),
            torch.nn.SiLU()
        )
        self.layer_6 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            torch.nn.SiLU()
        )
        self.maxpool_3 = torch.nn.MaxPool1d(2, 1)
        self.layer_7 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(128),
            torch.nn.Conv1d(128, 256, 3, 1),
            torch.nn.SiLU()
        )
        self.layer_8 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            torch.nn.SiLU()
        )
        self.maxpool_4 = torch.nn.MaxPool1d(2, 1)
        self.maxpool_flat = torch.nn.Sequential(
            torch.nn.MaxPool1d(12, 1),
            torch.nn.Flatten()
        )
        self.linear_1 = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.SiLU()
        )
        self.linear_2 = torch.nn.Sequential(
            torch.nn.Linear(128, 10),
            torch.nn.SiLU()
        )
    
    def forward(self, x):
        out = self.batch_norm(x)
        out = self.layer_1(out.reshape(batch_size, 1, down_sample_no))
        out = self.layer_2(out)
        out = self.maxpool_1(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.maxpool_2(out)
        out = self.layer_5(out)
        out = self.layer_6(out)
        out = self.maxpool_3(out)
        out = self.layer_7(out)
        out = self.layer_8(out)
        out = self.maxpool_4(out)
        out = self.maxpool_flat(out)
        out = self.linear_1(out)
        out = self.linear_2(out)


class ToTensor(object):
    def __call__(self, sample) -> object:
        audio_data, audio_label, projection_idx = sample['audio_data'], sample['audio_label'], sample['projection_idx']
        
        return {
            'audio_data'        : torch.tensor(audio_data, dtype=torch.float),
            'audio_label'       : torch.tensor(np.asarray(audio_label, dtype=np.float32), dtype=torch.float),
            'projection_idx'    : torch.tensor(np.asarray(projection_idx, dtype=np.float32), dtype=torch.float)
        }

class ProjectedAudioDataset(Dataset):
    def __init__(
                self, 
                data_dirs, 
                transform, 
                num_projections,
                top_projections,
                slope_length,
                rest_length,
                num_downsample,
                downsample_method
                ) -> None:

        # EXTRACTED FROM TRAINING DATASET ONLY
        self.mean = 0.01284
        self.std = 1.02609

        # self.mean, self.std = 0., 0.

        self.transform = transform
        self.num_downsample = num_downsample

        self.dataset_list = []
        self.label_list = []

        self.max_length = 0
        self.min_legnth = np.inf
        self.lengths = []

        # Loading dataset to memory
        print("Loading data to memory ...")

        for subdir, _, files in chain.from_iterable(
            os.walk(path) for path in data_dirs
        ):
            for file in files:
                if file[0] == 'd':
                    tmp = np.load(os.path.join(subdir, file))
                    assert num_projections == len(tmp), "Number of projections should be consistent with number of projected data"
                    for i in range(0, num_projections):
                        data = tmp[i][slope_length + rest_length : -slope_length, 0]
                        avg = np.mean(
                            tmp[i][slope_length + rest_length - 100 : slope_length + rest_length, 0]
                        )
                        avg2 = np.mean(
                            data
                        )
                        if top_projections != None:
                            if i in top_projections:
                                self.dataset_list.append(np.append((data - avg2), i))
                                # self.dataset_list.append(np.append(data, i))
                        else:
                            if len(data) < 10000 and len(data) > 2000:
                                self.dataset_list.append(np.append((data - avg2), i))
                                # self.dataset_list.append(np.append(data, i))
                        if len(data) > self.max_length:
                            self.max_length = len(data)
                        if len(data) < self.min_legnth:
                            self.min_legnth = len(data)
                        self.lengths.append(len(data))
                elif file[0] == 'l':
                    tmp = np.load(os.path.join(subdir, file))
                    assert num_projections == len(tmp), "Number of projections should be consisten with number of projected label"
                    for i in range(0, num_projections):
                        if top_projections != None:
                            if i in top_projections:
                                self.label_list.append(tmp[i])
                        else:
                            self.label_list.append(tmp[i])
                else:
                    sys.exit("Unknown type of data found in dataset!")
        
        self.len_dataset = len(self.dataset_list)
        self.dataset_numpy = np.zeros((
            np.shape(self.dataset_list)[0],
            # + 1 for keep tracking of the projection index
            num_downsample + 1
        ))
        self.label_numpy = np.zeros((
            self.len_dataset
        ))

        # Calculating mean and std
        # self.mean = np.average(self.dataset_numpy[:, :-1])
        # self.std  = np.std(self.dataset_numpy[:, :-1])
        # # Normalizing dataset
        # self.dataset_numpy[:, :-1] = ((self.dataset_numpy[:,:-1]) - self.mean)/self.std

        # means = []
        # stds = []

        # for i in range(len(self.dataset_list)):
        #     means.append(
        #         np.mean(self.dataset_list[i])
        #     )
        #     stds.append(
        #         np.std(self.dataset_list[i])
        #     )

        # self.mean = np.mean(means)
        # # self.std = np.mean(stds)
        # tmp = 0.
        # for i in range(len(stds)):
        #     tmp += stds[i]**2
        # tmp /= len(stds)
        # self.std = tmp ** 0.5

        # for i in range(len(self.dataset_list)):
        #     self.dataset_list[i] = (self.dataset_list[i] - self.mean)/self.std

        if downsample_method == 'variable':
            for i in range(0, self.len_dataset):
                projection_idx = self.dataset_list[i][-1]
                data = self.dataset_list[i][:-1]
                idx = np.arange(0, len(data), dtype=int)
                idx = np.sort(np.random.choice(idx, num_downsample))
                self.dataset_numpy[i] = np.append(
                    data[idx],
                    projection_idx
                )
                self.label_numpy[i] = self.label_list[i]
        elif downsample_method == 'zero_pad':
            idx = np.arange(0, self.max_length, dtype=int)
            idx = np.sort(np.random.choice(idx, num_downsample))
            for i in range(0, self.len_dataset):
                projection_idx = self.dataset_list[i][-1]
                self.dataset_list[i] = np.pad(
                    self.dataset_list[i][:-1], (0, self.max_length - len(self.dataset_list[i][:-1])), 'constant'
                )
                self.dataset_list[i] = self.dataset_list[i][idx]
                self.dataset_numpy[i] = np.append(
                    self.dataset_list[i],
                    projection_idx
                )
                self.label_numpy[i] = self.label_list[i]
        elif downsample_method == 'zero_pad_sym':
            idx = np.arange(0, self.max_length, dtype=int)
            idx = np.sort(np.random.choice(idx, num_downsample))
            for i in range(0, self.len_dataset):
                projection_idx = self.dataset_list[i][-1]
                self.dataset_list[i] = np.pad(
                    self.dataset_list[i][:-1], pad_width= (self.max_length - len(self.dataset_list[i][:-1]))//2
                )
                self.dataset_list[i] = self.dataset_list[i][idx]
                self.dataset_numpy[i] = np.append(
                    self.dataset_list[i],
                    projection_idx
                )
                self.label_numpy[i] = self.label_list[i]
        else:
            print("Downsample method UNKNOWN!")
        
        print("Loading completed successfully!")
        print("Lenght of dataset: ", self.len_dataset)
        print("---------------------------------------------------")
        print("Mean and Standard deviation of the dataset are: ", self.mean, "and ", self.std)
        print("Max length of data: ", self.max_length, "and, min length of data: ", self.min_legnth)
        self.max_length = 8000


    def __mean_std__(self):
        return {
            'mean'  : self.mean,
            'std'   : self.std
        }
        
    def __len__(self) -> None:
        return len(self.dataset_list)

    # Note: this method returns the audio data and the projection index separately
    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        
        audio_data = self.dataset_numpy[index, :-1]
        audio_label = self.label_numpy[index]
        projection_idx = self.dataset_numpy[index, -1]

        sample = {
            'audio_data'        : audio_data,
            'audio_label'       : audio_label,
            'projection_idx'    : projection_idx
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print("Reset trainable parameters of layer = ", layer)
            layer.reset_parameters()

def train_and_test(
        model,
        num_epoch,
        dataset,
        batch_size,
):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(torch.float)
    model.to(device)
    model = torch.compile(model)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_set, test_set = torch.utils.data.random_split(
                            dataset,
                            [
                                int(0.8 * len(dataset)), 
                                int(0.2 * len(dataset) + 1)
                            ]
    )

    train_dataloader = DataLoader(
        train_set,
        batch_size,
        drop_last= True,
        shuffle= True,
    )

    test_dataloader = DataLoader(
        test_set,
        batch_size,
        drop_last= True,
        shuffle= True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.01, weight_decay = .1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                                                optimizer, 
                                                max_lr = 0.01,
                                                steps_per_epoch = int(len(train_dataloader)),
                                                epochs = num_epoch,
                                                anneal_strategy = 'linear'
    )

    model.train()
    for epoch in range(0, num_epoch):
        print("Starting epoch: ", epoch + 1)
        current_loss = 0.
        for i, data in enumerate(train_dataloader):
            inputs = data['audio_data'].to(device)
            targets = data['audio_label'].type(torch.LongTensor).to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            current_loss += loss.item()
            if i % (batch_size) == (batch_size - 1):
                print("Loss after mini-batch %3d: %.3f" % (i+1, current_loss/(5 * batch_size)))
                current_loss = 0.
    
    print("Evaluating training procedure...")
    model.eval()
    correct, total = 0., 0.
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            inputs = data['audio_data'].to(device)
            targets = data['audio_label'].type(torch.LongTensor).to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted==targets).sum().item()
        print("Training accuracy: ", 100.*correct/total)

    torch.save(model.state_dict(), "saved_model.pt")
    np.save("test_set.npy", test_set)

    print(" ")

def test(
    model,
    dataset,
    device
    ):

    test_dataloader = DataLoader(
        dataset,
        batch_size= 1,
        shuffle= False,
        drop_last= False
    )
    print("Length of dataset: ", len(test_dataloader))
    correct, total = 0, 0
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            inputs = data['audio_data'].to(device)
            targets = data['audio_label'].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        # print("Number of tested data: ", i)
        print("Test accuracy: ", 100. * correct / total)

def kfold_cross_validation(
        model,
        num_epoch,
        dataset,
        num_projections,
        batch_size, 
        k_folds = 10,
    ):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(torch.float)
    model.to(device)
    model = torch.compile(model)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    # fold results
    results = {}
    kfold = KFold(
                n_splits    = k_folds,
                shuffle     = True
            )
    print('------------------------------------')
    # K-fold cross validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print("FOLD = ", fold)
        print('------------------------------------')
        train_sampler = SubsetRandomSampler(train_ids)
        test_sampler = SubsetRandomSampler(test_ids)

        trainloader = DataLoader(
                        dataset,
                        batch_size  =   batch_size,
                        sampler     =   train_sampler,
                        drop_last   =   True,
        )
        testloader = DataLoader(
                        dataset,
                        batch_size  =   batch_size,
                        sampler     =   test_sampler,
                        drop_last   =   True,
        )

        model.apply(reset_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr = 0.01, weight_decay = .1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                                                    optimizer, 
                                                    max_lr = 0.01,
                                                    steps_per_epoch = int(len(trainloader)),
                                                    epochs = num_epoch,
                                                    anneal_strategy = 'linear'
        )

        for epoch in range(0, num_epoch):
            print("Starting epoch ", epoch+1)
            current_loss = 0.
            for i, data in enumerate(trainloader, 0):
                inputs = data['audio_data'].to(device)
                targets = data['audio_label'].type(torch.LongTensor).to(device)
                optimizer.zero_grad()

                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                current_loss += loss.item()
                if i % (batch_size) == (batch_size - 1):
                    print("Loss after mini-batch %3d: %.3f" % (i+1, current_loss/(5 * batch_size)))
                    current_loss = 0.
        
        print("Start testing for fold: ", fold)
        correct, total = 0, 0
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs = data['audio_data'].to(device)
                targets = data['audio_label'].type(torch.LongTensor).to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted==targets).sum().item()
            print("Accuracy for fold %d: %d %%" %(fold, 100.*correct/total))
            print('------------------------------------')
            results[fold] = 100. * (correct / total)
        
    print(f"K-FOLD cross validation results for {k_folds} FOLDS")
    print('------------------------------------')
    sum = 0.
    for key, value in results.items():
        print(f"Fold {key}: {value}%")
        sum += value
    print(f"Average accuracy: {sum / len(results.items())} %")
    print("Saving model ... ")

    torch.save(model.state_dict(), "saved_model.pt")
    np.save("test_set.npy", test_sampler)

    print(" ")

def NNmodel(
    NNtype= 'LinearLayer',
    down_sample_no = 512,
    hidden_layer_size = 512,
    num_classes = 10,
    dropout_prob = 0.1
):
    if NNtype == 'Linear':
        tmp = LinearLayer(down_sample_no, num_classes)
    elif NNtype == 'Conv':
        tmp = ConvLayer(num_classes, down_sample_no) 
    elif NNtype == 'FC':
        tmp = FCLayer(down_sample_no, hidden_layer_size, num_classes, dropout_prob)
    
    return tmp

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

def measurement(
                configs,
                num_projections,
                dnpu_input_index,
                dnpu_control_indeces,
                slope_length,
                rest_length
                ):

    dataset_train_val, label_train_val, dataset_test, label_test = dataset_ti_spoken_digits.load_dataset()
    dataset_train_val, dataset_test = dataset_ti_spoken_digits.shift_frequency_for_female_speakers(dataset_train_val, dataset_test, 200, 12500)
    dataset_train_val, dataset_test = dataset_ti_spoken_digits.remove_silence_with_average(dataset_train_val, dataset_test)
    dataset_train_val, dataset_test = dataset_ti_spoken_digits.audio_low_pass_filter(dataset_train_val, dataset_test)
  
    cnt = 0
    driver = get_driver(configs=configs["driver"])
    rand_matrix = np.random.uniform(-0.25, 0.25, size=(len(dnpu_control_indeces), num_projections))
    
    for d in range(len(dataset_train_val)):
        dataset_train_val_after_projection = []
        label_train_val_after_projection = []
        for p_idx in range(num_projections):
            cnt += 1
            # 0 -> slope -> rest -> voice -> slope -> 0
            meas_input = np.zeros((len(dnpu_control_indeces) + 1, rest_length + 2 * slope_length + np.shape(dataset_train_val[d])[0]))
            meas_input[dnpu_input_index, slope_length + rest_length: -slope_length] = 20 * dataset_train_val[d][:]
            meas_input = set_random_control_voltages(
                                                    meas_input=             meas_input,
                                                    dnpu_control_indeces=   dnpu_control_indeces,
                                                    slope_length=           slope_length,
                                                    projection_idx=         p_idx,
                                                    rand_matrix=            rand_matrix)

            print("Completed pecentage of train: %.3f" %(100 * (cnt / (num_projections * (len(dataset_train_val))))), "%")
            # sys.stdout.write()
            # sys.stdout.flush()

            output = driver.forward_numpy(meas_input.T)

            dataset_train_val_after_projection.append(output)
            label_train_val_after_projection.append(label_train_val[d])
        
        path_dataset = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_digits_arsenic/train_val/f4/" + "data_" + str(d)
        path_label = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_digits_arsenic/train_val/f4/" + "label_" + str(d)
        np.save(path_dataset, dataset_train_val_after_projection)
        np.save(path_label, label_train_val_after_projection)
        del dataset_train_val_after_projection
        del label_train_val_after_projection
        gc.collect()

    cnt = 0
    for d in range(len(dataset_test)):
        dataset_test_after_projection = []
        label_test_after_projection = []
        for p_idx in range(num_projections):
            cnt += 1
            # 0 -> slope -> rest -> voice -> slope -> 0
            meas_input = np.zeros((len(dnpu_control_indeces) + 1, rest_length + 2 * slope_length + np.shape(dataset_test[d])[0]))
            meas_input[dnpu_input_index, slope_length + rest_length: -slope_length] = 20 * dataset_test[d][:]
            meas_input = set_random_control_voltages(
                                                    meas_input=             meas_input,
                                                    dnpu_control_indeces=   dnpu_control_indeces,
                                                    slope_length=           slope_length,
                                                    projection_idx=         p_idx,
                                                    rand_matrix=            rand_matrix)

            print("Completed pecentage of test: %.3f" %(100 * (cnt / (num_projections * (len(dataset_test))))), "%")

            output = driver.forward_numpy(meas_input.T)

            dataset_test_after_projection.append(output)
            label_test_after_projection.append(label_test[d])
        
        path_dataset = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_digits_arsenic/test/f4/" + "data_" + str(d)
        path_label = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_digits_arsenic/test/f4/" + "label_" + str(d)
        np.save(path_dataset, dataset_test_after_projection)
        np.save(path_label, label_test_after_projection)
        del dataset_test_after_projection
        del label_test_after_projection
        gc.collect()
    
    driver.close_tasks()
 


if __name__ == '__main__':
    from brainspy.utils.io import load_configs

    slope_length = 2000
    rest_length = 10000

    down_sample_no = 256

    hidden_layer_size = 128
    num_projections= 128

    batch_size = 128

    num_epoch = 50

    num_classes = 10
    normalizing_dataset = True
    train_with_all_projections = True
    new_number_of_projetions = 64
    zero_padding_downsample = True  

    # np.random.seed(5) 
    # configs = load_configs('configs/defaults/processors/hw.yaml')
    # measurement(
    #             configs             =   configs,
    #             num_projections     =   num_projections,
    #             dnpu_input_index    =   3,
    #             dnpu_control_indeces=   [0, 1, 2, 4, 5, 6],
    #             slope_length        =   slope_length,
    #             rest_length         =   rest_length
    #         )

    # load projected dataset
    dataset_train_val_after_projection = []
    label_train_val_after_projection = []
    dataset_test_after_projection = []
    label_test_after_projection = []

    projected_train_val_data_arsenic_f4 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_digits_arsenic/train_val/f4/"
    projected_test_data_arsenic_f4 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_digits_arsenic/test/f4/"
    projected_train_val_data_arsenic_f5 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_digits_arsenic/train_val/f5/"
    projected_test_data_arsenic_f5 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_digits_arsenic/test/f5/"
    projected_train_val_data_arsenic_f6 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_digits_arsenic/train_val/f6/"
    projected_test_data_arsenic_f6 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_digits_arsenic/test/f6/"
    projected_train_val_data_arsenic_f7 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_digits_arsenic/train_val/f7/"
    projected_test_data_arsenic_f7 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_digits_arsenic/test/f7/"
    projected_train_val_data_arsenic_f8 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_digits_arsenic/train_val/f8/"
    projected_test_data_arsenic_f8 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_digits_arsenic/test/f8/"

    empty = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/boron_roomTemp_30nm/ti_spoken_digits/test/empty/"

    dataset_path = (
        empty,
        projected_train_val_data_arsenic_f4,
        projected_train_val_data_arsenic_f5,
        projected_train_val_data_arsenic_f6,
        projected_train_val_data_arsenic_f7,
        projected_train_val_data_arsenic_f8,
    )

    test_dataset_path = (
        projected_test_data_arsenic_f4,
        projected_test_data_arsenic_f5,
        projected_test_data_arsenic_f6,
        projected_test_data_arsenic_f7,
        projected_test_data_arsenic_f8,

    )

    transform = transforms.Compose([
            ToTensor()
    ])

    dataset = ProjectedAudioDataset(
                data_dirs           = dataset_path,
                transform           = transform,
                num_projections     = 128,
                top_projections     = None,
                slope_length        = slope_length,
                rest_length         = rest_length,
                num_downsample      = down_sample_no,
                downsample_method  = 'zero_pad' # 'variable', 'zero_pad', 'zero_pad_sym'
                # NOTE: Variable length zero padding is logically incorrect,
                # Because it means varible low-pass filtering, high for low-durated audios, and low for high-duration adious
    )

    # test_dataset = ProjectedAudioDataset(
    #             data_dirs           = test_dataset_path,
    #             transform           = transform,
    #             num_projections     = 128,
    #             top_projections     = None,
    #             slope_length        = slope_length,
    #             rest_length         = rest_length,
    #             num_downsample      = down_sample_no,
    #             downsample_method  = 'zero_pad' # 'variable', 'zero_pad', 'zero_pad_sym'
    #             # NOTE: Variable length zero padding is logically incorrect,
    #             # Because it means varible low-pass filtering, high for low-durated audios, and low for high-duration adious
    # )
    
    classification_layer = NNmodel(
        NNtype= 'FC', # 'Conv', 'FC', 'Linear'
        down_sample_no= down_sample_no,
        hidden_layer_size = hidden_layer_size,
        num_classes= 10,
        dropout_prob= 0.5,
    )

    print("Number of learnable parameters are: ", sum(p.numel() for p in classification_layer.parameters()))

    # kfold_cross_validation(
    #     model           = classification_layer,
    #     num_epoch       = num_epoch,
    #     dataset         = dataset,
    #     num_projections = num_projections,
    #     batch_size      = batch_size,
    #     k_folds         = 5      
    # )

    train_and_test(
        model= classification_layer,
        num_epoch= num_epoch,
        dataset= dataset,
        batch_size= batch_size
    )

    classification_layer.load_state_dict(torch.load("saved_model.pt"))
    model = classification_layer.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    model.eval()
    test_dataset = np.load("test_set.npy", allow_pickle=True)
    test(
        model,
        test_dataset,
        device=  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    )

    # from librosa import display
    # fig, ax = plt.subplots()
    # S = np.abs(librosa.stft(self.dataset_list[0][:-1], n_fft=256))
    # img = librosa.display.specshow(librosa.amplitude_to_db(S,
    #                                                     ref=np.max),
    #                             y_axis='linear', x_axis='time', ax=ax)
    # ax.set_title('Power spectrogram')
    # fig.colorbar(img, ax=ax, format="%+2.0f dB")