from cgi import test
from distutils.command.config import config
from pickletools import optimize
from re import T
import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import gc, os

from itertools import chain

from torchvision import transforms

import dataset_ti_spoken_digits

import scipy

from sklearn import preprocessing

import matplotlib.pyplot as plt

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


class LinearLayer(torch.nn.Module):
    def __init__(self, input_size, num_classes) -> None:
        super(LinearLayer, self).__init__()
        self.linear_layer = torch.nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.linear_layer(x)

class FCLayer(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_classes) -> None:
        super(FCLayer, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_layer_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_layer_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out


def train(
        model,
        num_epoch,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        num_projections
        ):

    loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=10e-6)

    # best_vloss = np.inf

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # writer = SummaryWriter("runs/fashion_trainer_{}".format(timestamp))

    for epoch in range(num_epoch):
        print("Training epoch {}: ".format(epoch + 1))
        
        #  Train step
        model.train(True)
        running_loss = 0.
        last_loss = 0.
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 32 == 31:
                last_loss = running_loss / 32
                print(" Batch {} loss: {}".format(i + 1, last_loss))
                # tb_x = epoch * len(train_dataloader) + i + 1
                # writer.add_scalar("Loss/train", last_loss, tb_x)
                running_loss = 0.

        avg_loss = last_loss

        # Validation step
        model.train(False)
        running_vloss = 0.0
        totalv = 0
        running_vaccuracy = 0.
        for i, vdata in enumerate(val_dataloader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss.item()

            _, vpredicted = torch.max(voutputs, 1)
            totalv += vlabels.size(0)
            running_vaccuracy += (vpredicted == vlabels).sum().item()
        
        avg_vloss = running_vloss / len(val_dataloader)


        # Test step
        model.train(False)
        running_test_loss = 0.0
        total_test = 0
        running_test_accuracy = 0.
        for i, test_data in enumerate(test_dataloader):
            test_inputs, test_labels = test_data
            test_outputs = model(test_inputs)
            test_loss = loss_fn(test_outputs, test_labels)
            running_test_loss += test_loss.item()
            _, predicted_test = torch.max(test_outputs, 0)   #torch.max(test_outputs, 1)
            total_test += 1 # test_labels.size(0)
            running_test_accuracy += (predicted_test == test_labels).sum().item()
        
        avg_test_loss = running_test_loss / len(test_dataloader)


        # Test step with voting/ensembling system --> Mechanism: 1; Majority
        model.train(False)
        total_test_voting = 0
        running_test_correct_count = 0
        accuracy_over_each_dataset_projection = []
        prediction_list = []
        for i, test_voting_data in enumerate(test_dataloader): # itterating over all test set
            test_voting_inputs, test_voting_labels = test_voting_data
            test_voting_outputs = model(test_voting_inputs)
            _, predicted_test_voting = torch.max(test_voting_outputs, 0)
            running_test_correct_count += (predicted_test_voting == test_voting_labels).sum().item()
            total_test_voting += 1
            prediction_list.append(predicted_test_voting)
            if i % num_projections == (num_projections - 1): # new class starts next round; 128 -> num. of projections
                # if running_test_correct_count > 40:
                #     temp = 1
                # else:
                #     temp = 0
                # accuracy_over_each_dataset_projection.append(temp)
                # running_test_correct_count = 0
                # total_test_voting = 0
                # Second approach
                highest_chance_prediction = np.bincount(prediction_list).argmax()
                if highest_chance_prediction == test_voting_labels:
                    temp = 1
                else:
                    temp = 0
                accuracy_over_each_dataset_projection.append(temp)
                running_test_correct_count = 0
                total_test_voting = 0
                prediction_list = []

                

        print("Train loss {}, Val. loss {}, Test loss {}, Val. acc. {}%, Test acc. {}%, Voting test acc. {}%".format(
                                                                    avg_loss, avg_vloss, avg_test_loss,
                                                                    (100 * running_vaccuracy / totalv),
                                                                    (100 * running_test_accuracy / total_test),
                                                                    (100 * np.average(accuracy_over_each_dataset_projection)))
        )


        # writer.add_scalars(
        #                 "Validation loss vs. Validation accuracy",
        #                 {"Validation": avg_vloss, "Accuracy":(100 * running_accuracy / total)},
        #                 epoch + 1
        #                 )
        # writer.add_scalars(
        #             "Test accuracy: ",
        #             {"Test accuracyc": (100 * running_accuracy / total)},
        #             epoch + 1
        #             )
        # writer.flush()

        # if avg_vloss < best_vloss:
        #     best_vloss = avg_vloss
        #     # model_path = "model_{}_{}".format(timestamp, epoch)
        #     torch.save(model.state_dict(), "best_trained_model.pt")
        


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


def butter_bandpass(cutoff_low, cutoff_high, order, fs):
    return scipy.signal.butter(
                                N = order,
                                btype = 'bandpass',
                                Wn = [cutoff_low, cutoff_high],
                                analog = False,
                                fs = fs
    )

def butter_bandpass_filter(data, cutoff_low, cutoff_high, order, fs):
    b, a = butter_bandpass(
                            cutoff_low=cutoff_low,
                            cutoff_high= cutoff_high,
                            order = order,
                            fs = fs
                        )
    return scipy.signal.lfilter(
                                b = b,
                                a = a,
                                x = data
    )

def butter_lowpass(cutoff, order, fs):
    return scipy.signal.butter( N = order, 
                                Wn = cutoff, 
                                btype = 'low', 
                                analog=False,
                                fs= fs)

def butter_lowpass_filter(data, cutoff, order, fs):
    b, a = butter_lowpass(cutoff, order = order, fs=fs)
    y = scipy.signal.lfilter(b = b, 
                            a = a, 
                            x = data)
    return y



def measurement_with_filters(
                            num_projections,
                            filter_type
    ):

    dataset_train_val, label_train_val, dataset_test, label_test = dataset_ti_spoken_digits.load_dataset()
    dataset_train_val, dataset_test = dataset_ti_spoken_digits.remove_silence(dataset_train_val, dataset_test)
    
    cnt = 0
    rand_matrix_of_cutoffs = np.random.uniform(20, 6000, size=(128))

    for d in range(len(dataset_train_val)):
        dataset_train_val_after_projection = []
        label_train_val_after_projection = []
        for p_idx in range(num_projections):
            cnt += 1
            if filter_type == "low_pass":
                dataset_train_val_after_projection.append(
                                                butter_lowpass_filter(
                                                                    data    = dataset_train_val[d][:],
                                                                    cutoff  = rand_matrix_of_cutoffs[p_idx],
                                                                    fs      = 12500,
                                                                    order   = 5
                                                )  
                )
                label_train_val_after_projection.append(label_train_val[d])
            elif filter_type == "bandpass":
                low = rand_matrix_of_cutoffs[p_idx]
                high = rand_matrix_of_cutoffs[p_idx - 1]
                if low > high:
                    temp = low
                    low = high
                    high = temp
                dataset_train_val_after_projection.append(
                                                butter_bandpass_filter(
                                                    data    = dataset_train_val[d][:],
                                                    cutoff_low  = low,
                                                    cutoff_high = high,
                                                    fs      = 12500,
                                                    order   = 5
                                                )
                )
                label_train_val_after_projection.append(label_train_val[d])

        path_dataset = "tmp/sanity_checks/random_low_pass/train_val/" + "data_" + str(d)
        path_label = "tmp/sanity_checks/random_low_pass/train_val/" + "label_" + str(d)
        np.save(path_dataset, dataset_train_val_after_projection)
        np.save(path_label, label_train_val_after_projection)
        del dataset_train_val_after_projection
        del label_train_val_after_projection
        gc.collect()

    for d in range(len(dataset_test)):
        dataset_test_after_projection = []
        label_test_after_projection = []
        for p_idx in range(num_projections):
            cnt += 1
            if filter_type == "low_pass":
                dataset_test_after_projection.append(
                                                butter_lowpass_filter(
                                                                    data    = dataset_test[d][:],
                                                                    cutoff  = rand_matrix_of_cutoffs[p_idx],
                                                                    fs      = 12500,
                                                                    order   = 5
                                                )  
                )
                label_test_after_projection.append(label_test[d])
            elif filter_type == "bandpass":
                low = rand_matrix_of_cutoffs[p_idx]
                high = rand_matrix_of_cutoffs[p_idx - 1]
                if low > high:
                    temp = low
                    low = high
                    high = temp
                dataset_test_after_projection.append(
                                                butter_bandpass_filter(
                                                    data    = dataset_test[d][:],
                                                    cutoff_low  = low,
                                                    cutoff_high = high,
                                                    fs      = 12500,
                                                    order   = 5
                                                )
                )
                label_test_after_projection.append(label_test[d])


        path_dataset = "tmp/sanity_checks/random_low_pass/test/" + "data_" + str(d)
        path_label = "tmp/sanity_checks/random_low_pass/test/" + "label_" + str(d)
        np.save(path_dataset, dataset_test_after_projection)
        np.save(path_label, label_test_after_projection)
        del dataset_test_after_projection
        del label_test_after_projection
        gc.collect()

def convert_to_torch_dataset(
                            train,
                            dataset, 
                            label
                            ):

    tensor_x = torch.Tensor(dataset)
    # tensor_y = torch.Tensor(labels_after_projection)

    # torch.from_numpy

    le = preprocessing.LabelEncoder()
    tensor_y = le.fit_transform(label)

    dataset = TensorDataset(tensor_x, torch.as_tensor(tensor_y))

    if train == True:
        val_dataset_size = int(0.2 * len(dataset))
        train_dataset_size = len(dataset) - val_dataset_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_dataset_size, val_dataset_size])

        train_dataloader = DataLoader(train_dataset,
                                    batch_size=64, 
                                    shuffle=True,
                                    drop_last= True
        )

        val_dataloader = DataLoader(val_dataset,
                                    batch_size= 64,
                                    shuffle= True,
                                    drop_last= True
        )
        return train_dataloader, val_dataloader
    else:
        test_dataloader = DataLoader(
                                    dataset,
                                    shuffle= False,
                                    batch_size= None
                                    )
        return test_dataloader

def training_accuracy(model, dataset):
    # model.eval()
    correct_number = 0
    num_samples = 0

    with torch.no_grad():
        for x,y in dataset:
            scores = model(x)
            _, preds = scores.max(1)
            correct_number += (preds == y).sum()
            num_samples += preds.size(0)
        print(f'Got {correct_number} / {num_samples} with accuracy {float(correct_number)/float(num_samples)*100:.2f}%')


def test_accuracy(dataset, model):
    # model.eval()
    correct_number = 0
    num_samples = 0

    with torch.no_grad():
        for x,y in dataset:
            scores = model(x)
            _, preds = scores.max(1)
            correct_number += (preds == y).sum()
            num_samples += preds.size(0)
        print(f'Got {correct_number} / {num_samples} with accuracy {float(correct_number)/float(num_samples) * 100:.2f}')

def downsample_varible_input_length(
                                down_sample_no,
                                input
    ):
    for i in range(len(input)):
        len_data = len(input[i])
        idx = np.arange(0, len_data, dtype=int)
        idx = np.sort(np.random.choice(idx, down_sample_no))
        input[i] = input[i][idx]
    return input

if __name__ == '__main__':

    # np.random.seed(1) 
    num_projections = 128

    measurement_with_filters(
            num_projections = num_projections,
            filter_type     = "bandpass" # "low_pass", "bandpass"
    )


    # load projected dataset
    dataset_train_val_after_projection = []
    label_train_val_after_projection = []
    dataset_test_after_projection = []
    label_test_after_projection = []

    sanity_with_lowpass_train_val = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/sanity_checks/random_low_pass/train_val/"
    sanity_with_lowpass_test = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/sanity_checks/random_low_pass/test/"

    for subdir, _, files in os.walk(sanity_with_lowpass_train_val):# chain.from_iterable(os.walk(path) for path in dataset_paths):
        for file in files:
            if file[0] == 'd':
                temp = np.load(os.path.join(subdir, file))
                for i in range(np.shape(temp)[0]):
                    dataset_train_val_after_projection.append(temp[i])
            elif file[0] == 'l':
                temp = np.load(os.path.join(subdir, file)) 
                for i in range(np.shape(temp)[0]):
                    label_train_val_after_projection.append(temp[0])

    for subdir, _, files in os.walk(sanity_with_lowpass_test):# chain.from_iterable(os.walk(path) for path in dataset_paths):
        for file in files:
            if file[0] == 'd':
                temp = np.load(os.path.join(subdir, file))
                for i in range(np.shape(temp)[0]):
                    dataset_test_after_projection.append(temp[i])
            elif file[0] == 'l':
                temp = np.load(os.path.join(subdir, file)) 
                for i in range(np.shape(temp)[0]):
                    label_test_after_projection.append(temp[0])

    
    # Low pass filter
    for i in range(len(dataset_train_val_after_projection)):
        dataset_train_val_after_projection[i] = butter_lowpass_filter(
                                                                data = 10 * dataset_train_val_after_projection[i][:],
                                                                cutoff= 6000,
                                                                fs = 12500,
                                                                order= 4
        )

    for i in range(len(dataset_test_after_projection)):
        dataset_test_after_projection[i] = butter_lowpass_filter(
                                                                data = 10 * dataset_test_after_projection[i][:],
                                                                cutoff= 6000,
                                                                fs = 12500,
                                                                order= 4
        )

    down_sample_no = 256
    dataset_train_val_after_projection = downsample_varible_input_length(
                                                                    down_sample_no = down_sample_no,
                                                                    input = dataset_train_val_after_projection
    )

    dataset_test_after_projection = downsample_varible_input_length(
                                                                    down_sample_no = down_sample_no,
                                                                    input = dataset_test_after_projection
    )


    # Conversion to PyTorch dataset; train
    train_dataloader, val_dataloader = convert_to_torch_dataset(
                                                    train   = True,
                                                    dataset = dataset_train_val_after_projection, 
                                                    label   = label_train_val_after_projection
                                                    )
                                    
    # Conversion to PyTorch dataset; test
    test_dataloader = convert_to_torch_dataset(
                                        train   = False,
                                        dataset = dataset_test_after_projection, 
                                        label   = label_test_after_projection
                                        )

    classification_layer = LinearLayer(
                            input_size  =    down_sample_no,
                            num_classes =    10
    )


    linear_layer = LinearLayer(
                            input_size= down_sample_no,
                            num_classes= 10
    )

    # torch.save(t_dataloader, 'f2_data_128_projection_test.pt')

    train(
        model               = classification_layer,
        num_epoch           = 500,
        train_dataloader    = train_dataloader,
        val_dataloader      = val_dataloader,
        test_dataloader     = test_dataloader,
        num_projections     = num_projections
    )

    # # linear_layer.load_state_dict(torch.load('model_20220811_143742_0.pt'))
    # # linear_layer.eval()
    # # training_accuracy(
    # #                 linear_layer,
    # #                 train_dataloader # torch.load('f2_data_128_projection_test.pt')
    # # )

    # # fc_layer = FCLayer(
    # #             input_size=1024,
    # #             hidden_layer_size=512,
    # #             num_classes=26
    # # )

    # # fc_layer.load_state_dict(torch.load('trained_model.pt'))
    # # fc_layer.eval()
    
    # # test_accuracy(
    # #             torch.load('f1_data_projected_test.pt'),
    # #             fc_layer
    # # )

    

