from cgi import test
from distutils.command.config import config
import enum
from pickletools import optimize
from queue import Empty
from tracemalloc import reset_peak
from turtle import forward
from typing import NewType
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, SubsetRandomSampler

import numpy as np
import gc, os, scipy, sys, time

from itertools import chain, dropwhile

from torchvision import transforms

import dataset_ti_spoken_digits

from brainspy.utils.manager import get_driver

from sklearn import preprocessing
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

from datetime import datetime

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
        # self.max_pool = torch.nn.AvgPool1d(kernel_size=5, stride=1)
        # 682 -> maxpool k=3
        # 409 -> maxpool k=5
        self.linear_layer = torch.nn.Linear(input_size, num_classes) 
    
    def forward(self, x):
        return self.linear_layer(x)

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
        return out

class conv_layer(torch.nn.Module):
    def __init__(self, input_size, num_classes) -> None:
        super(conv_layer, self).__init__()
        self.conv1 = torch.nn.Conv1d(
                                    in_channels=    1,
                                    out_channels=   64,
                                    kernel_size=    3,
                                    stride=         1
        )
        self.relu = torch.nn.ReLU()
        self.fc   = torch.nn.Linear(1024, num_classes)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.fc(out)

        return out

def train(
        model,
        num_epoch,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        num_projections,
        batch_size
        ):

    loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    # optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_deacay=10e-5)
    # optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.01, weight_decay = 0.35)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    max_lr = 0.01,
                                                    steps_per_epoch = int(len(train_dataloader)),
                                                    epochs = num_epoch,
                                                    anneal_strategy = 'linear'
                                                )


    best_vloss = np.inf
    best_test_accuracy = 0.0

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # writer = SummaryWriter("runs/fashion_trainer_{}".format(timestamp))

    # To store each projection store
    projection_score = np.zeros((num_projections)) 


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
            scheduler.step()

            running_loss += loss.item()
            if i % batch_size == (batch_size-1):
                last_loss = running_loss / batch_size
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

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            torch.save(model.state_dict(), "top_val_model.pt")

        # Normal Test step
        model.train(False)
        running_test_loss = 0.0
        total_test = 0
        running_test_accuracy = 0.
        for i, test_data in enumerate(test_dataloader):
            test_inputs, test_labels = test_data
            test_outputs = model(test_inputs)
            test_loss = loss_fn(test_outputs, test_labels)
            running_test_loss += test_loss.item()
            _, predicted_test = torch.max(test_outputs, 1)
            total_test += test_labels.size(0)
            running_test_accuracy += (predicted_test == test_labels).sum().item()
        
        avg_test_loss = running_test_loss / len(test_dataloader)


        # Test step with voting/ensembling system --> Mechanism: 1; Majority
        # Batch is set to None for majory voting; processing each test dataset one-by-one
        model.train(False)
        total_test_voting = 0
        running_test_correct_count = 0
        accuracy_over_each_dataset_projection = []
        prediction_list = []
        projection_score_list = []
        for i, test_voting_data in enumerate(test_dataloader): # itterating over all test set
            test_voting_inputs, test_voting_labels = test_voting_data
            test_voting_outputs = model(test_voting_inputs)
            _, predicted_test_voting = torch.max(test_voting_outputs, 1)
            running_test_correct_count += (predicted_test_voting == test_voting_labels).sum().item()
            total_test_voting += 1
            prediction_list.append(predicted_test_voting)
            if i % num_projections == (num_projections - 1): # new class starts next round; 128 -> num. of projections
                highest_chance_prediction = np.bincount(prediction_list).argmax()
                if highest_chance_prediction == test_voting_labels:
                    temp = 1
                else:
                    temp = 0
                accuracy_over_each_dataset_projection.append(temp)
                # Prediction score handling 
                for j in range(len(projection_score)):
                    if prediction_list[j] == test_voting_labels:
                        projection_score[j] += 1
                    else:
                        projection_score[j] -= 1
                # projection_score_list.append(projection_score)
                running_test_correct_count = 0
                total_test_voting = 0
                prediction_list = []
                # projection_score = np.zeros((num_projections))

        if (100 * np.average(accuracy_over_each_dataset_projection)) >= best_test_accuracy:
            best_test_accuracy = 100 * np.average(accuracy_over_each_dataset_projection)


        print("Train loss {:0.2f}, Val. loss {:0.2f}, Test loss {:0.2f}, Val. acc.{:0.2f}%, Test acc. {:0.2f}%, Voting test acc. {:0.2f}%, Best voting test acc. {:0.2f}%".format(
                                                                    avg_loss, avg_vloss, avg_test_loss,
                                                                    (100 * running_vaccuracy / totalv),
                                                                    (100 * running_test_accuracy / total_test),
                                                                    (100 * np.average(accuracy_over_each_dataset_projection)),
                                                                    best_test_accuracy
                                                            )
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



    return projection_score
        

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print("Reset trainable parameters of layer = ", layer)
            layer.reset_parameters()

def kfold_cross_validation(
        model,
        num_epoch,
        train_dataset,
        val_dataset,
        test_dataset,
        num_projections,
        batch_size, 
        k_folds = 10,
    ):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    # fold results
    results = {}

    kfold = KFold(
                n_splits    = k_folds,
                shuffle     = True
            )
    
    dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

    acuuracy_of_projection_indeces = np.zeros((num_projections, 2))

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
                        sampler     =   train_sampler
        )
        testloader = DataLoader(
                        dataset,
                        batch_size  =   batch_size,
                        sampler     =   test_sampler
        )

        model.apply(reset_weights)

        optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_decay = 0.35)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    max_lr = 0.01,
                                                    steps_per_epoch = int(len(train_dataloader)),
                                                    epochs = num_epoch,
                                                    anneal_strategy = 'linear'
                                                )

        for epoch in range(0, num_epoch):
            if epoch % 10 == 0:
                print("Starting epoch ", epoch+1)
            current_loss = 0.
            for i, data in enumerate(trainloader):
                inputs, targets = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs[:,:-1])
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()
                current_loss += loss.item()
                if i % (batch_size) == (batch_size - 1):
                    print("Loss after mini-batch %3d: %.3f" % (i+1, current_loss/(5 * batch_size)))
                    current_loss = 0.
        
        print("Training process completed, saving model ...")

        # Save model

        print("Start testing...")

        # Evaluation for this fold
        correct, total = 0, 0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                inputs, targets = data[0].to(device), data[1].to(device)
                outputs = model(inputs[:,:-1])
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted==targets).sum().item()
            print("Accuracy for fold %d: %d %%" %(fold, 100.*correct/total))
            print('------------------------------------')
            results[fold] = 100. * (correct / total)

        # A list to keep track of projection scores; 1 -> correct, 0 -> incorrect
        # [projection_idx, 0/1, target_digit]
        projection_score_list = []
        with torch.no_grad():
            for i, data in enumerate(testloader):
                inputs, targets = data[0].to(device), data[1].to(device)
                outputs = model(inputs[:,:-1])
                _, predicted = torch.max(outputs, 1)

                for i in range(predicted.size(0)):
                    if predicted[i] == targets[i]:
                        projection_score_list.append(
                            [inputs[i, -1], 1, targets[i]]
                        )
                    else:
                        projection_score_list.append(
                            [inputs[i, -1], 0, targets[i]]
                        )

            # Sorting projection_score_list based on target digit
            # [[projection_idx, corr./incorr., target_digit]]
            accuracy_list_sort_by_target = []
            for i in range(0, num_classes):
                temp = []
                for j in range(len(projection_score_list)):
                    if projection_score_list[j][2] == i:
                        temp.append([projection_score_list[j][0], projection_score_list[j][1], i])
                accuracy_list_sort_by_target.append(temp)
            
            # Voting mechanism
            # [Correct predictions, total predictions]
            voting_accuracy_of_digits = []
            for i in range(0, num_classes):
                temp = 0
                for j in range(len(accuracy_list_sort_by_target[i])):
                    temp += accuracy_list_sort_by_target[i][j][1]
                voting_accuracy_of_digits.append([temp, len(accuracy_list_sort_by_target[i])])
                # if temp >= len(accuracy_list_sort_by_target[i])//2:
                    # voting_accuracy_of_digits.append(1)
                # else:
                    # voting_accuracy_of_digits.append(0)

            # Sorting proejction score list based on projections
            # Here we can find best projections
            accuracy_list_sort_by_projection_idx = []
            for i in range(0, num_projections):
                temp = []
                for j in range(len(projection_score_list)):
                    if projection_score_list[j][0] == i:
                        temp.append([projection_score_list[j][0], projection_score_list[j][1], projection_score_list[j][2]])
                accuracy_list_sort_by_projection_idx.append(temp)
        
    print(f"K-FOLD cross validation results for {k_folds} FOLDS")
    print('------------------------------------')
    sum = 0.
    for key, value in results.items():
        print(f"Fold {key}: {value}%")
        sum += value
    print(f"Average accuracy: {sum / len(results.items())} %")


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
    
def convert_to_torch_dataset(
                            train,
                            dataset, 
                            label,
                            batch_size,
                            normalize = True,
                            # For K-Fold cross validation this should be FALSE!!
                            convert_to_dataloader = True,
                            ):

    if normalize == True:
        # mean = np.mean(dataset[:][:-1])
        # std = np.std(dataset[:][:-1])
        # dataset = (dataset - mean) / std

        np_dataset = np.zeros((np.shape(dataset)[0], np.shape(dataset)[1]))

        for i in range(len(dataset)):
            np_dataset[i] = dataset[i]
        

        mean = np.mean(np_dataset[:,:-1])
        std = np.std(np_dataset[:,:-1])

        np_dataset[:,:-1] = ((np_dataset[:,:-1]) - mean)/std

    tensor_x = torch.Tensor(np_dataset)
    le = preprocessing.LabelEncoder()
    tensor_y = le.fit_transform(label)

    dataset = TensorDataset(tensor_x, torch.as_tensor(tensor_y))

    if train == True:
        val_dataset_size = int(0.15 * len(dataset))
        train_dataset_size = len(dataset) - val_dataset_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_dataset_size, val_dataset_size])

        if convert_to_dataloader == True:
            train_dataloader = DataLoader(train_dataset,
                                        batch_size=batch_size, 
                                        shuffle=True,
                                        drop_last= True
            )
            val_dataloader = DataLoader(val_dataset,
                                        batch_size= 1,
                                        shuffle= True,
                                        drop_last= True
            )
        else:
            train_dataloader = train_dataset
            val_dataloader = val_dataset

        return train_dataloader, val_dataloader

    else:
        if convert_to_dataloader == True:
            test_dataloader = DataLoader(
                                        dataset,
                                        shuffle= True,
                                        batch_size= 1
                                        )
        else:
            test_dataloader = dataset

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


def test_with_top_projections(dataset, model, num_projections):
    total_test_voting = 0
    running_test_correct_count = 0
    accuracy_over_each_dataset_projection = []
    prediction_list = []
    predictions_for_cf_matrix = []
    labels_for_cf_matrix = []
    for i, test_voting_data in enumerate(dataset): # itterating over all test set
        test_voting_inputs, test_voting_labels = test_voting_data
        test_voting_outputs = model(test_voting_inputs)
        _, predicted_test_voting = torch.max(test_voting_outputs, 1)
        running_test_correct_count += (predicted_test_voting == test_voting_labels).sum().item()
        total_test_voting += 1
        prediction_list.append(predicted_test_voting)
        if i % num_projections == (num_projections - 1): # new class starts next round; 128 -> num. of projections
            highest_chance_prediction = np.bincount(prediction_list).argmax()
            predictions_for_cf_matrix.append(highest_chance_prediction)
            labels_for_cf_matrix.append(test_voting_labels)
            if highest_chance_prediction == test_voting_labels:
                temp = 1
            else:
                temp = 0
            accuracy_over_each_dataset_projection.append(temp)
            running_test_correct_count = 0
            total_test_voting = 0
            prediction_list = []
        
    print("Test accuracy with best projections: {:0.2f}".format(np.average(accuracy_over_each_dataset_projection)))
    # cf_matrix = confusion_matrix(np.array(labels_for_cf_matrix), np.array(predictions_for_cf_matrix))
    len_matrix = len(labels_for_cf_matrix)
    a = np.zeros((len_matrix))
    b = np.zeros((len_matrix))
    for i in range(len(a)):
        a[i] = labels_for_cf_matrix[i].numpy()
    
    b = np.asarray(predictions_for_cf_matrix)
    cf_matrix = confusion_matrix(a, b)
    ax = seaborn.heatmap(cf_matrix/cf_matrix.sum(axis=0), annot=True, cmap='Blues', fmt='0.1%')
    plt.show()


def downsample_with_zero_padding(down_sample_no,
                                input
    ):
    max_length = 0
    for i in range(len(input)):
        if len(input[i][:-1]) > max_length:
            max_length = len(input[i][:-1])
    idx = np.arange(0, max_length, dtype=int)
    idx = np.sort(np.random.choice(idx, down_sample_no))
    for i in range(len(input)):
        projection_idx = input[i][-1]
        input[i] = np.pad(input[i][:-1], (0, max_length - len(input[i][:-1])), 'constant')
        input[i] = input[i][idx]
        input[i] = np.append(input[i], projection_idx)
    
    return input
        
    

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

def butter_highpass(cutoff, order, fs):
    return scipy.signal.butter( N = order, 
                                Wn = cutoff, 
                                btype = 'high', 
                                analog=False,
                                fs= fs)

def butter_highpass_filter(data, cutoff, order, fs):
    b, a = butter_highpass(cutoff, order = order, fs=fs)
    y = scipy.signal.lfilter(b = b, 
                            a = a, 
                            x = data)
    return y


if __name__ == '__main__':
    from brainspy.utils.io import load_configs

    slope_length = 2000
    rest_length = 10000

    hidden_layer_size = 512
    num_projections= 128
    batch_size = 64
    down_sample_no = 512
    num_epoch = 5
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

    projected_train_val_data_m3 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/boron_roomTemp_30nm/ti_spoken_digits/train_val/m3/"
    projected_test_data_m3      = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/boron_roomTemp_30nm/ti_spoken_digits/test/m3/"
    projected_train_val_data_m4 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/boron_roomTemp_30nm/ti_spoken_digits/train_val/m4/"
    projected_test_data_m4      = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/boron_roomTemp_30nm/ti_spoken_digits/test/m4/"
    projected_train_val_data_m5 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/boron_roomTemp_30nm/ti_spoken_digits/train_val/m5/"
    projected_test_data_m5      = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/boron_roomTemp_30nm/ti_spoken_digits/test/m5/"
    projected_train_val_data_f5 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/boron_roomTemp_30nm/ti_spoken_digits/train_val/f5/"
    projected_test_data_f5      = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/boron_roomTemp_30nm/ti_spoken_digits/test/f5/"
    projected_train_val_data_f6 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/boron_roomTemp_30nm/ti_spoken_digits/train_val/f6_frequency_shift/"
    projected_test_data_f6      = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/boron_roomTemp_30nm/ti_spoken_digits/test/f6_frequency_shift/"
    projected_train_val_data_f7 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/boron_roomTemp_30nm/ti_spoken_digits/train_val/f7_frequency_shift/"
    projected_test_data_f7      = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/boron_roomTemp_30nm/ti_spoken_digits/test/f7_frequency_shift/"


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

    dataset_train_val_path = (
                        empty,
                        projected_train_val_data_arsenic_f4,
                        projected_train_val_data_arsenic_f5,
                        projected_train_val_data_arsenic_f6,
                        projected_train_val_data_arsenic_f7,
                        projected_train_val_data_arsenic_f8
    )

    dataset_test_path = (
                        empty,
                        projected_test_data_arsenic_f4,
                        projected_test_data_arsenic_f5,
                        projected_test_data_arsenic_f6,
                        projected_test_data_arsenic_f7,
                        projected_test_data_arsenic_f8
    )

    # Top projections; load
    if train_with_all_projections == False:
        num_projections = new_number_of_projetions
        tmp = np.load("top_projections.npy")
        top_projections = np.argpartition(tmp, -num_projections)[-num_projections:]

    # Loading training dataset
    for subdir, _, files in chain.from_iterable(os.walk(path) for path in dataset_train_val_path):
        for file in files:
            if file[0] == 'd':
                temp = np.load(os.path.join(subdir, file))
                for i in range(len(temp)):
                    if train_with_all_projections == False:
                        if i in top_projections: # only picks the top projections
                            dataset_train_val_after_projection.append(temp[i][slope_length+rest_length:-slope_length] - (np.average(temp[i][slope_length+rest_length - 50:slope_length+rest_length])))
                    else:
                        dataset_train_val_after_projection.append(
                                np.append(temp[i][slope_length+rest_length:-slope_length, 0] - (np.average(temp[i][slope_length+rest_length - 50:slope_length+rest_length, 0])), i)
                                )
            elif file[0] == 'l':
                temp = np.load(os.path.join(subdir, file)) 
                for i in range(len(temp)):
                    if train_with_all_projections == False:
                        if i in top_projections: # only picks the top projections
                            label_train_val_after_projection.append(temp[0])
                    else:
                        label_train_val_after_projection.append(temp[0])

    # Loading test dataset
    for subdir, _, files in chain.from_iterable(os.walk(path) for path in dataset_test_path):
        for file in files:
            if file[0] == 'd':
                temp = np.load(os.path.join(subdir, file))
                for i in range(len(temp)):
                    if train_with_all_projections == False:
                        if i in top_projections: # only picks the top projections
                            dataset_test_after_projection.append(temp[i][slope_length+rest_length:-slope_length] - (np.average(temp[i][slope_length+rest_length - 50:slope_length+rest_length])))
                    else:
                        dataset_test_after_projection.append(
                            np.append(temp[i][slope_length+rest_length:-slope_length, 0] - (np.average(temp[i][slope_length+rest_length - 50:slope_length+rest_length, 0])), i)
                            )
            elif file[0] == 'l':
                temp = np.load(os.path.join(subdir, file)) 
                for i in range(len(temp)):
                    if train_with_all_projections == False:
                        if i in top_projections: # only picks the top projections
                            label_test_after_projection.append(temp[0])
                    else:
                        label_test_after_projection.append(temp[0])

    # NOT WORKING APPROACH: filtering between 20 Hz and 6.5 Khz does not lead to good accuracy.
    # temp_train = []
    # temp_test = []
    # for i in range(len(dataset_train_val_after_projection)):
    #     freq = np.fft.rfftfreq(n=len(dataset_train_val_after_projection[i]), d=1/12500)
    #     f_transform = np.fft.rfft(dataset_train_val_after_projection[i][:,0])
    #     for j in range(len(freq)):
    #         if freq[j] >= 20:
    #             f_start = j
    #             break
    #     for j in range(len(freq)):
    #         if freq[- 1 - j] >= 6000:
    #             pass
    #         else:
    #             f_stop = j
    #             break
    #     temp_train.append(10 * np.fft.irfft(f_transform[f_start : -f_stop]))

    # for i in range(len(dataset_test_after_projection)):
    #     freq = np.fft.rfftfreq(n=len(dataset_test_after_projection[i]), d=1/12500)
    #     f_transform = np.fft.rfft(dataset_test_after_projection[i][:,0])
    #     for j in range(len(freq)):
    #         if freq[j] >= 20:
    #             f_start = j
    #             break
    #     for j in range(len(freq)):
    #         if freq[- 1 - j] >= 6000:
    #             pass
    #         else:
    #             f_stop = j
    #             break
    #     temp_test.append(10 * np.fft.irfft(f_transform[f_start : -f_stop]))


    # # Low pass filter
    # for i in range(len(dataset_train_val_after_projection)):
    #     dataset_train_val_after_projection[i] = butter_lowpass_filter(
    #                                                             data    = 10 * dataset_train_val_after_projection[i][:],
    #                                                             cutoff  = 4000,
    #                                                             fs      = 25000,
    #                                                             order   = 5
    #     )    

    # for i in range(len(dataset_test_after_projection)):
    #     dataset_test_after_projection[i] = butter_lowpass_filter(
    #                                                             data    = 10 * dataset_test_after_projection[i][:],
    #                                                             cutoff  = 4000,
    #                                                             fs      = 25000,
    #                                                             order   = 5
    #     )

    # # Removing DC; train val
    # for i in range(len(dataset_train_val_after_projection)):
    #     dataset_train_val_after_projection[i][:,-1] = 10 * (dataset_train_val_after_projection[i][:-1])

    # # # # Removing DC; test
    # for i in range(len(dataset_test_after_projection)):
    #     # dataset_test_after_projection[i] = 10 * (dataset_test_after_projection[i][:,0] - np.mean(dataset_test_after_projection[i][:,0]))
    #     dataset_test_after_projection[i][:,-1] = 10 * (dataset_test_after_projection[i][:-1])

    # Downsampling method used here should be revisited...
    if zero_padding_downsample == True:
        dataset_train_val_after_projection = downsample_with_zero_padding(
                                        down_sample_no= down_sample_no,
                                        input = dataset_train_val_after_projection)
        dataset_test_after_projection = downsample_with_zero_padding(
                                        down_sample_no= down_sample_no,
                                        input = dataset_test_after_projection)
    else:
        dataset_train_val_after_projection = downsample_varible_input_length(
                                        down_sample_no = down_sample_no,
                                        input = dataset_train_val_after_projection)
        dataset_test_after_projection = downsample_varible_input_length(
                                        down_sample_no = down_sample_no,
                                        input = dataset_test_after_projection)

    # adding some data from test to train
    # for i in range(int(0.5 * len(dataset_test_after_projection))):
    #     idx = np.random.randint(0, len(dataset_test_after_projection), dtype=int)
    #     dataset_train_val_after_projection.append(dataset_test_after_projection[idx])
    #     label_train_val_after_projection.append(label_test_after_projection[idx])
    #     del dataset_test_after_projection[idx]
    #     del label_test_after_projection[idx]


    # Conversion to PyTorch dataset; train
    train_dataloader, val_dataloader = convert_to_torch_dataset(
                                                    train   = True,
                                                    dataset = dataset_train_val_after_projection, 
                                                    label   = label_train_val_after_projection,
                                                    batch_size = batch_size,
                                                    normalize= normalizing_dataset,
                                                    # For KFOLD this should be FALSE
                                                    convert_to_dataloader= False
                                                    )
                                    
    # Conversion to PyTorch dataset; test
    test_dataloader = convert_to_torch_dataset(
                                        train   = False,
                                        dataset = dataset_test_after_projection, 
                                        label   = label_test_after_projection,
                                        batch_size = batch_size,
                                        normalize= normalizing_dataset,
                                        # For KFOLD this should be FALSE
                                        convert_to_dataloader= False
                                        )

    # classification_layer = LinearLayer(
    #                         input_size  =    down_sample_no,
    #                         num_classes =    num_classes
    # )

    

    classification_layer = FCLayer( 
                            input_size          = down_sample_no,
                            hidden_layer_size   = hidden_layer_size,
                            num_classes         = num_classes,
                            drop_out_prob       = 0.5    
    )

    # classification_layer = conv_layer(
    #                                 input_size= down_sample_no,
    #                                 num_classes= 10
    # )

    # projection_score = train(
    #                         model               = classification_layer,
    #                         num_epoch           = num_epoch,
    #                         train_dataloader    = train_dataloader,
    #                         val_dataloader      = val_dataloader,
    #                         test_dataloader     = test_dataloader,
    #                         num_projections     = num_projections,
    #                         batch_size          = batch_size
    #                     )

    # if train_with_all_projections == True:
    #     np.save("top_projections.npy", projection_score)

    
    # classification_layer.load_state_dict(torch.load('top_val_model.pt'))
    # classification_layer.eval()
    # test_with_top_projections(
    #                         test_dataloader,
    #                         classification_layer,
    #                         num_projections
    # )

    kfold_cross_validation(
                        model           = classification_layer,
                        num_epoch       = num_epoch,
                        train_dataset   = train_dataloader,
                        val_dataset     = val_dataloader,
                        test_dataset    = test_dataloader,
                        num_projections = num_projections,
                        batch_size      = batch_size,
                        k_folds         = 10       
    )
