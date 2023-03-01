from cgi import test
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, SubsetRandomSampler, Dataset, random_split

import numpy as np
import gc, os, scipy, sys

from itertools import chain
import torch.nn.functional as F

from torchvision import transforms

import spoken_mnist_dataset

from brainspy.utils.manager import get_driver

import sklearn
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import seaborn

from torch.utils.tensorboard import SummaryWriter

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

class M5(torch.nn.Module):
    def __init__(self, n_input=1, n_output=10, stride=16, n_channel=43):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = torch.nn.BatchNorm1d(n_channel)
        self.pool1 = torch.nn.MaxPool1d(4)
        self.conv2 = torch.nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = torch.nn.BatchNorm1d(n_channel)
        self.pool2 = torch.nn.MaxPool1d(4)
        self.conv3 = torch.nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = torch.nn.BatchNorm1d(2 * n_channel)
        self.pool3 = torch.nn.MaxPool1d(4)
        self.conv4 = torch.nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = torch.nn.BatchNorm1d(2 * n_channel)
        self.pool4 = torch.nn.MaxPool1d(4)
        self.fc1 = torch.nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x.reshape(128, 1, 8000))
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = x.resize(128, 10)
        return F.log_softmax(x)

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
        out = F.softmax(out, dim=1)
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
        self.kernel_size = 16
        self.stride = 8
        self.out_channel = 8
        self.conv1 = torch.nn.Conv1d(
                                    in_channels     =   1,
                                    out_channels    =   self.out_channel ,
                                    kernel_size     =   self.kernel_size,
                                    stride          =   self.stride
        )
        self.relu_1 = torch.nn.ReLU()
        # self.bn = torch.nn.BatchNorm1d(num_features=self.out_channel)
        self.flat = torch.nn.Flatten()
        self.fc1   = torch.nn.Linear(self.out_channel  * int((down_sample_no - self.kernel_size)/self.stride + 1), 32)
        self.relu_2 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(32, num_classes)
    
    def forward(self, x):
        out = self.conv1(x.reshape(batch_size, 1, down_sample_no))
        # out = self.bn(out)
        out = self.relu_1(out)
        out = self.flat(out)
        out = self.fc1(out)
        out = self.relu_2(out)
        out = self.fc2(out)
        out = torch.log_softmax(out, dim=1)
        return out

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

        self.transform = transform
        self.num_downsample = num_downsample
        self.mean = 0.
        self.std = 0.

        self.dataset_list = []
        self.label_list = []

        self.max_length = 0
        self.min_legnth = np.inf
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
                        
                        avg2 = np.mean(
                            tmp[i][slope_length + rest_length - 200: slope_length + rest_length - 50 , 0]
                        )
                        
                        data = butter_lowpass_filter(data=data, cutoff=128, order=5, fs=24000)

                        avg = np.mean(
                            data
                        )

                        if top_projections != None:
                            if i in top_projections:
                                self.dataset_list.append(np.append(200*(data - avg), i))
                        else:
                            # avg = np.mean(data) 
                            data = data - avg
                            scale = np.max(np.abs(data))
                            data = data * (1/scale)
                            self.dataset_list.append(
                                np.append(
                                    # sklearn.preprocessing.minmax_scale(data-avg, (-1,1)),# - np.mean(sklearn.preprocessing.minmax_scale(data, (-1,1))),
                                    data,
                                    i
                                )
                        )

                        if len(data) > self.max_length:
                            self.max_length = len(data)
                        if len(data) < self.min_legnth:
                            self.min_legnth = len(data)

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

        assert len(self.dataset_list) == len(self.label_list), "Error in loading data!"

        # tmp_label = []
        # tmp_data = []

        # for i in range(len(self.dataset_list)):
        #     if self.label_list[i] != '10':
        #         tmp_label.append(self.label_list[i])
        #         tmp_data.append(self.dataset_list[i])

        # self.dataset_list = tmp_data
        # self.label_list = tmp_label

        for i in range(len(self.dataset_list)):
            if self.label_list[i] == '00':
                self.label_list[i] = 0
            elif self.label_list[i] == '01':
                self.label_list[i] = 1
            elif self.label_list[i] == '02':
                self.label_list[i] = 1
            elif self.label_list[i] == '03':
                self.label_list[i] = 1
            elif self.label_list[i] == '04':
                self.label_list[i] = 1
            elif self.label_list[i] == '05':
                self.label_list[i] = 1
            elif self.label_list[i] == '06':
                self.label_list[i] = 1
            elif self.label_list[i] == '07':
                self.label_list[i] = 1
            elif self.label_list[i] == '08':
                self.label_list[i] = 1
            elif self.label_list[i] == '09':
                self.label_list[i] = 1
            elif self.label_list[i] == '10':
                self.label_list[i] = 1
            else:
                self.label_list[i] = 1
                print("EROOOORRRRRRR!!")

        assert len(self.dataset_list) == len(self.label_list), "Error in loading data!"

        # computing mean and std
        means, stds = [],[] 
        for i in range(len(self.dataset_list)):
            means.append(np.mean(self.dataset_list[i][:-1]))
            stds.append(np.std(self.dataset_list[i][:-1]))
        mean = np.mean(means)
        std = (np.sum(np.power(stds,2))/len(stds))**0.5

        # for i in range(len(self.dataset_list)):
        #     self.dataset_list[i] = (self.dataset_list[i]) / 0.001

        self.len_dataset = len(self.dataset_list)

        self.dataset_numpy = np.zeros((
            np.shape(self.dataset_list)[0],
            # + 1 for keep tracking of the projection index
            num_downsample + 1
        ))
        self.label_numpy = np.zeros((
            self.len_dataset
        ))

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

        # 128 * 512
        cnt = 0
        self.data_to_img = np.zeros((self.len_dataset//num_projections, num_projections, down_sample_no))
        for i in range(self.len_dataset//num_projections): # 22
            for j in range(num_projections): # 512
                self.data_to_img[i][j, :] = self.dataset_numpy[cnt,:-1]
                cnt += 1

        
        print("Loading completed successfully!")
        print("Lenght of dataset: ", self.len_dataset)
        print("Max length of data: ", self.max_length, "and, min length of data: ", self.min_legnth)

        print("---------------------------------------------------")
        
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

def accuracy_with_top_projections(
    projection_scores,
    num_tops = 5,
    targets = None,
    predictions = None
):  
    top_projection_indeces = np.argpartition(projection_scores, -num_tops)[-num_tops:]

    #  5 * 39
    predictions_of_tops = np.zeros((num_tops, len(targets)))
    for i in range(len(targets)):
        for j in range(0, num_tops):
            predictions_of_tops[j, i] = predictions[i, top_projection_indeces[j]]
    
    tmp = 0
    voted_predictions,voted_targets = [],[]
    for i in range(len(predictions)):
        counts = np.bincount(predictions_of_tops[:,i].astype(int))
        voted_predictions.append(np.argmax(counts))
        voted_targets.append(targets[i])
        if np.argmax(counts) == targets[i]:
            tmp += 1
    
    print("Confusion matrix for tops: ")
    print(sklearn.metrics.confusion_matrix(voted_targets, voted_predictions))
    return tmp/len(predictions)

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print("Reset trainable parameters of layer = ", layer)
            layer.reset_parameters()

def train_and_test(
        model,
        num_epoch,
        fold1,
        fold2,
        fold3,
        fold4,
        fold5,
        batch_size,
):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(torch.float)
    model.to(device)
    # model = torch.compile(model)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_set = torch.utils.data.ConcatDataset([fold1, fold2, fold3, fold4])

    train_loader = DataLoader(
        train_set,
        batch_size,
        shuffle= True,
    )

    test_loader = DataLoader(
        fold5,
        batch_size=batch_size,
        shuffle= False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr = 0.001, 
        weight_decay = .001
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr = 0.005,
        steps_per_epoch = int(len(train_loader)),
        epochs = num_epoch,
        anneal_strategy = 'linear'
    )

    model.train()
    for epoch in range(0, num_epoch):
        print("Starting epoch: ", epoch + 1)
        current_loss = 0.
        for i, data in enumerate(train_loader):
            inputs = data['audio_data'].to(device)
            targets = data['audio_label'].type(torch.LongTensor).to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                outputs = model(inputs)
                loss = loss_fn(outputs,targets) 

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            current_loss += loss.item()
            if i % (batch_size//4) == (batch_size//4 - 1):
                print("Loss after mini-batch %3d: %.3f" % (i+1, current_loss/(batch_size//4)))
                current_loss = 0.
    
    print("Evaluating training procedure...")
    model.eval()
    correct, total = 0., 0.
    correct_voted, total_voted = 0, 0
    # true positive, false positive
    projection_scores = np.zeros((128))
    voted_predictions, voted_targets  = [], []

    # 39 * 128
    predictions = np.zeros((len(test_loader), num_projections))
    targets_array = np.zeros((len(test_loader)))

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs = data['audio_data'].to(device)
            targets = data['audio_label'].type(torch.LongTensor).to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1) 
            total += targets.size(0)
            correct += (predicted==targets).sum().item()

            predictions[i] = predicted.cpu().numpy()
            targets_array[i] = targets[0].cpu().numpy()

            # Voting
            counts = np.bincount(predicted.cpu().numpy())
            total_voted += 1
            if np.argmax(counts) == targets[0].cpu():
                correct_voted += 1

            voted_predictions.append(np.argmax(counts))
            voted_targets.append(targets[0].cpu())

            for i in range(len(projection_scores)):
                if predicted[i] == targets[i]:
                    projection_scores[i] += 1
        
        print("Test accuracy: ", 100.*correct/total)
        print("Voted test accuracy:, " , 100 * correct_voted / total_voted)
        tmp = accuracy_with_top_projections(
            projection_scores= projection_scores,
            num_tops= 7,
            targets = targets_array,
            predictions= predictions
        )
        print("Accuracy with top projections: ", 100 * tmp)
        print("Fonfusion matrix: ")
        print(sklearn.metrics.confusion_matrix(voted_targets, voted_predictions))

    # torch.save(model.state_dict(), "saved_model.pt")
    # np.save("test_set.npy", test_set)

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
        testset,
        num_projections,
        batch_size, 
        k_folds = 10,
    ):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(torch.float)
    model.to(device)
    # model = torch.compile(model)
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
                        drop_last   =   True
        )
        testloader = DataLoader(
                        dataset,
                        batch_size  =   batch_size,
                        sampler     =   test_sampler,
                        drop_last   =   True
        )
        testsetloader = DataLoader(
            testset,
            batch_size= batch_size,
            shuffle= False
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
        model.train()
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
        
        print("Start evaluating for fold: ", fold)
        correct, total = 0, 0
        correct_t, total_t = 0, 0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs = data['audio_data'].to(device)
                targets = data['audio_label'].type(torch.LongTensor).to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted==targets).float().sum().item()
            print("Accuracy for fold %d: %d %%" %(fold, 100.*correct/total))
            print('------------------------------------')
            results[fold] = 100. * (correct / total)

            for i, data in enumerate(testsetloader, 0):
                inputs = data['audio_data'].to(device)
                targets = data['audio_label'].type(torch.LongTensor).to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_t += targets.size(0)
                correct_t += (predicted==targets).float().sum().item()
            print("Test accuracy for fold %d: %d %%" %(fold, 100.*correct_t/total_t))

        # print("Start testing for fold: ", fold)
        # correct, total = 0, 0
        # model.eval()
        # with torch.no_grad():
        #     for i, data in enumerate(testsetloader, 0):
        #         inputs = data['audio_data'].to(device)
        #         targets = data['audio_label'].type(torch.LongTensor).to(device)
        #         outputs = model(inputs)
        #         _, predicted = torch.max(outputs, 1)
        #         total += targets.size(0)
        #         correct += (predicted==targets).sum().item()
        #     print("Test accuracy for fold %d: %d %%" %(fold, 100.*correct/total))
        #     print('------------------------------------')
        #     # results[fold] = 100. * (correct / total)

    print(f"K-FOLD cross validation results for {k_folds} FOLDS")
    print('------------------------------------')
    sum = 0.
    for key, value in results.items():
        print(f"Fold {key}: {value}%")
        sum += value
    print(f"Average accuracy: {sum / len(results.items())} %")
    print("Saving model ... ")

    torch.save(model.state_dict(), "saved_model.pt")
    # np.save("test_set.npy", test_sampler)

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
    elif NNtype == 'M5':
        tmp = M5(n_input=1, n_output=num_classes, stride=16, n_channel=32)
    
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

    dataset_train_val, label_train_val, dataset_test, label_test = spoken_mnist_dataset.load_dataset()

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
            meas_input[dnpu_input_index, slope_length + rest_length: -slope_length] = dataset_train_val[d][:]
            meas_input = set_random_control_voltages(
                                                    meas_input=             meas_input,
                                                    dnpu_control_indeces=   dnpu_control_indeces,
                                                    slope_length=           slope_length,
                                                    projection_idx=         p_idx,
                                                    rand_matrix=            rand_matrix)

            output = driver.forward_numpy(meas_input.T)

            print("Completed pecentage of train: %.3f" %(100 * (cnt / (num_projections * (len(dataset_train_val))))), "%")

            dataset_train_val_after_projection.append(output)
            label_train_val_after_projection.append(label_train_val[d])

        path_dataset = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/train/" + "data_" + str(d)
        path_label = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/train/" + "label_" + str(d)

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
            meas_input[dnpu_input_index, slope_length + rest_length: -slope_length] = dataset_test[d][:]
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
        

        path_dataset = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/test/" + "data_" + str(d)
        path_label = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_spoken_mnist/test/" + "label_" + str(d)

        np.save(path_dataset, dataset_test_after_projection)
        np.save(path_label, label_test_after_projection)

        del dataset_test_after_projection
        del label_test_after_projection
        gc.collect()
    
    driver.close_tasks()
 

if __name__ == '__main__':
    from brainspy.utils.io import load_configs

    # these are for spoken mnist
    # slope_length = 200
    # rest_length = 1300

    slope_length = 100
    rest_length = 6000

    down_sample_no = 256

    hidden_layer_size = 256
    num_projections= 128

    batch_size = 128

    num_epoch = 20

    num_classes = 2

    train_with_all_projections = True
    new_number_of_projetions = 64
    zero_padding_downsample = True  

    # np.random.seed(25) 
    # configs = load_configs('configs/defaults/processors/hw.yaml')
    # measurement(
    #             configs             =   configs,
    #             num_projections     =   num_projections,
    #             dnpu_input_index    =   3,
    #             dnpu_control_indeces=   [0, 1, 2, 4, 5, 6],
    #             slope_length        =   slope_length,
    #             rest_length         =   rest_length
    # )

    # load projected dataset
    dataset_train_val_after_projection = []
    label_train_val_after_projection = []
    dataset_test_after_projection = []
    label_test_after_projection = []

    projected_train_val_data_arsenic_wilfred = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_in_house_arsenic/train/wilfred/"
    projected_test_data_arsenic_wilfred = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_in_house_arsenic/test/wilfred/"

    empty = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_in_house_arsenic/empty/"

    dataset_path = (
        empty,
        projected_train_val_data_arsenic_wilfred
    )

    test_dataset_path = (
        empty,
        projected_test_data_arsenic_wilfred

    )

    transform = transforms.Compose([
            ToTensor()
    ])

    fold1 = ProjectedAudioDataset(
                data_dirs           = 
                    ("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_in_house_arsenic/train/wilfred_pitch_shift/fold1/",empty),
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

    fold2 = ProjectedAudioDataset(
        data_dirs           = 
            ("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_in_house_arsenic/train/wilfred_pitch_shift/fold2/",empty),
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

    fold3 = ProjectedAudioDataset(
        data_dirs           = 
            ("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_in_house_arsenic/train/wilfred_pitch_shift/fold3/",empty),
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

    fold4 = ProjectedAudioDataset(
        data_dirs           = 
            ("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_in_house_arsenic/train/wilfred_pitch_shift/fold4/",empty),
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

    fold5 = ProjectedAudioDataset(
        data_dirs           = 
            ("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_in_house_arsenic/test/wilfred_pitch_shift/",empty),
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

    # dataset = ProjectedAudioDataset(
    #             data_dirs           = dataset_path,
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
        NNtype= 'Linear', # 'Conv', 'FC', 'Linear'
        down_sample_no= down_sample_no,
        hidden_layer_size = hidden_layer_size,
        num_classes= num_classes,
        dropout_prob= 0.5,
    )

    print("Number of learnable parameters are: ", sum(p.numel() for p in classification_layer.parameters()))

    # kfold_cross_validation(
    #     model           = classification_layer,
    #     num_epoch       = num_epoch,
    #     dataset         = test_dataset,
    #     testset         = dataset,
    #     num_projections = num_projections,
    #     batch_size      = batch_size,
    #     k_folds         = 10      
    # )

    train_and_test(
        model= classification_layer,
        num_epoch= num_epoch,
        fold1=fold1,
        fold2=fold2,
        fold3=fold3,
        fold4=fold4,
        fold5=fold5,
        batch_size= batch_size
    )

    # classification_layer.load_state_dict(
    #     torch.load(
    #         "saved_model_wilfred.pt", map_location='cuda:0' if torch.cuda.is_available() else 'cpu'
    #     )
    # )
    # model = classification_layer.to(
    #     device=torch.device(
    #         'cuda:0' if torch.cuda.is_available() else 'cpu'
    #     )
    # )
    # model.eval()
    # # test_dataset = np.load("test_set.npy", allow_pickle=True)
    # test(
    #     model,
    #     dataset,
    #     device=  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # )

    # from librosa import display
    # fig, ax = plt.subplots()
    # S = np.abs(librosa.stft(self.dataset_list[0][:-1], n_fft=256))
    # img = librosa.display.specshow(librosa.amplitude_to_db(S,
    #                                                     ref=np.max),
    #                             y_axis='linear', x_axis='time', ax=ax)
    # ax.set_title('Power spectrogram')
    # fig.colorbar(img, ax=ax, format="%+2.0f dB")