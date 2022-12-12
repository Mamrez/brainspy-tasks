from cgi import test
from distutils.command.config import config
from pickletools import optimize
from turtle import forward
import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import gc, os, scipy

from itertools import chain

from torchvision import transforms

import dataset_ti_spoken_digits

from brainspy.utils.manager import get_driver

from sklearn import preprocessing

import matplotlib.pyplot as plt

from datetime import datetime

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
        # self.max_pool = torch.nn.AvgPool1d(kernel_size=3)
        # 682 -> maxpool k=3
        # 409 -> maxpool k=5
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
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    # optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=10e-6)

    # best_vloss = np.inf

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
            _, predicted_test = torch.max(test_outputs, 1)
            total_test += 1 
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
    return projection_score
        


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
    dataset_train_val, dataset_test = dataset_ti_spoken_digits.remove_silence(dataset_train_val, dataset_test)
  
    cnt = 0
    driver = get_driver(configs=configs["driver"])
    rand_matrix = np.random.uniform(-0.2, 0.2, size=(len(dnpu_control_indeces), num_projections))

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

            output = driver.forward_numpy(meas_input.T)

            dataset_train_val_after_projection.append(output[slope_length+rest_length:-slope_length])
            label_train_val_after_projection.append(label_train_val[d])
        
        path_dataset = "tmp/projected_ti_alpha/boron_roomTemp_30nm/ti_spoken_digits/train_val/m1_256pro_0_2v/" + "data_" + str(d)
        path_label = "tmp/projected_ti_alpha/boron_roomTemp_30nm/ti_spoken_digits/train_val/m1_256pro_0_2v/" + "label_" + str(d)
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

            print("Completed pecentage of test: %.3f" %(100 * (cnt / (num_projections * (len(dataset_train_val))))), "%")

            output = driver.forward_numpy(meas_input.T)

            dataset_test_after_projection.append(output[slope_length+rest_length:-slope_length])
            label_test_after_projection.append(label_test[d])
        
        path_dataset = "tmp/projected_ti_alpha/boron_roomTemp_30nm/ti_spoken_digits/test/m1_256pro_0_2v/" + "data_" + str(d)
        path_label = "tmp/projected_ti_alpha/boron_roomTemp_30nm/ti_spoken_digits/test/m1_256pro_0_2v/" + "label_" + str(d)
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
                            batch_size
                            ):

    tensor_x = torch.Tensor(dataset)
    # tensor_x = torch.from_numpy(np.array(dataset))

    # torch.from_numpy

    le = preprocessing.LabelEncoder()
    tensor_y = le.fit_transform(label)

    dataset = TensorDataset(tensor_x, torch.as_tensor(tensor_y))

    if train == True:
        val_dataset_size = int(0.2 * len(dataset))
        train_dataset_size = len(dataset) - val_dataset_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_dataset_size, val_dataset_size])

        train_dataloader = DataLoader(train_dataset,
                                    batch_size=batch_size, 
                                    shuffle=True,
                                    drop_last= True
        )

        val_dataloader = DataLoader(val_dataset,
                                    batch_size= batch_size,
                                    shuffle= True,
                                    drop_last= True
        )
        return train_dataloader, val_dataloader
    else:
        test_dataloader = DataLoader(
                                    dataset,
                                    shuffle= False,
                                    batch_size= 1
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


if __name__ == '__main__':
    from brainspy.utils.io import load_configs

    np.random.seed(1) 

    train_with_all_projections = True
    slope_length = 800
    rest_lenth = 5000

    num_projections= 128
    batch_size = 64
    down_sample_no = 512
    num_epoch = 100
    num_classes = 26

    # configs = load_configs('configs/defaults/processors/hw.yaml')
    # measurement(
    #             configs=configs,
    #             num_projections= num_projections,
    #             dnpu_input_index = 3,
    #             dnpu_control_indeces = [0, 1, 2, 4, 5, 6],
    #             slope_length = slope_length,
    #             rest_length= rest_lenth
    #         )

    # load projected dataset
    dataset_train_val_after_projection = []
    label_train_val_after_projection = []
    dataset_test_after_projection = []
    label_test_after_projection = []

    projected_train_val_data_m1 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/boron_roomTemp_30nm/mANDf8_128_projections_elec3_limited_cv_with_rest/"
    projected_test_data_m1      = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/boron_roomTemp_30nm/mANDf8_128_projections_elec3_limited_cv_with_rest/"

    dataset_paths = (
                        projected_train_val_data_m1,
                        projected_test_data_m1
                )

    # Top projections; load
    if train_with_all_projections == False:
        num_projections = 64
        tmp = np.load("top_projections.npy")
        top_projections = np.argpartition(tmp, -num_projections)[-num_projections:]

    # Loading training dataset
    for subdir, _, files in os.walk(projected_train_val_data_m1): # chain.from_iterable(os.walk(path) for path in dataset_paths):
        for file in files:
            if file[0] == 'd':
                temp = np.load(os.path.join(subdir, file))
                for i in range(len(temp)):
                    if train_with_all_projections == False:
                        if i in top_projections: # only picks the top projections
                            dataset_train_val_after_projection.append(temp[i])
                    else:
                        dataset_train_val_after_projection.append(temp[i])
            elif file[0] == 'l':
                temp = np.load(os.path.join(subdir, file)) 
                for i in range(len(temp)):
                    if train_with_all_projections == False:
                        if i in top_projections: # only picks the top projections
                            label_train_val_after_projection.append(temp[0])
                    else:
                        label_train_val_after_projection.append(temp[0])

    # Loading test dataset
    for subdir, _, files in os.walk(projected_test_data_m1): # chain.from_iterable(os.walk(path) for path in dataset_paths):
        for file in files:
            if file[0] == 'd':
                temp = np.load(os.path.join(subdir, file))
                for i in range(len(temp)):
                    if train_with_all_projections == False:
                        if i in top_projections: # only picks the top projections
                            dataset_test_after_projection.append(temp[i])
                    else:
                        dataset_test_after_projection.append(temp[i])
            elif file[0] == 'l':
                temp = np.load(os.path.join(subdir, file)) 
                for i in range(len(temp)):
                    if train_with_all_projections == False:
                        if i in top_projections: # only picks the top projections
                            label_test_after_projection.append(temp[0])
                    else:
                        label_test_after_projection.append(temp[0])

    # NOT WORKING APPROACH: filtering between 20 Hz and 6.5 Khz does not lead to good accuracy.
    # Low pass filter
    for i in range(len(dataset_train_val_after_projection)):
        dataset_train_val_after_projection[i] = butter_lowpass_filter(
                                                                data = 10 * dataset_train_val_after_projection[i][:],
                                                                cutoff= 5000,
                                                                fs = 12500,
                                                                order= 5
        )

    for i in range(len(dataset_test_after_projection)):
        dataset_test_after_projection[i] = butter_lowpass_filter(
                                                                data = 10 * dataset_test_after_projection[i][:],
                                                                cutoff= 5000,
                                                                fs = 12500,
                                                                order= 5
        )

    # Removing DC; train val
    for i in range(len(dataset_train_val_after_projection)):
        dataset_train_val_after_projection[i] = 10 * (dataset_train_val_after_projection[i][:,0] - np.mean(dataset_train_val_after_projection[i][:,0]))

    # Removing DC; test
    for i in range(len(dataset_test_after_projection)):
        dataset_test_after_projection[i] = 10 * (dataset_test_after_projection[i][:,0] - np.mean(dataset_test_after_projection[i][:,0]))

    # Downsampling method used here should be revisited...
    # Either we should go for 1-d convs. or using scipy decimate (or resample)
    # Another solution is max-over-time pooling
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
                                                    label   = label_train_val_after_projection,
                                                    batch_size = batch_size
                                                    )
                                    
    # Conversion to PyTorch dataset; test
    test_dataloader = convert_to_torch_dataset(
                                        train   = False,
                                        dataset = dataset_test_after_projection, 
                                        label   = label_test_after_projection,
                                        batch_size = batch_size
                                        )



    classification_layer = LinearLayer(
                            input_size  =    down_sample_no,
                            num_classes =    num_classes
    )

    # classification_layer = FCLayer(
    #                         input_size= down_sample_no,
    #                         hidden_layer_size=  256,
    #                         num_classes= 10      
    # )

    # classification_layer = conv_layer(
    #                                 input_size= down_sample_no,
    #                                 num_classes= 10
    # )


    # torch.save(t_dataloader, 'f2_data_128_projection_test.pt')

    projection_score = train(
                            model               = classification_layer,
                            num_epoch           = num_epoch,
                            train_dataloader    = train_dataloader,
                            val_dataloader      = val_dataloader,
                            test_dataloader     = test_dataloader,
                            num_projections     = num_projections,
                            batch_size          = batch_size
                        )

    if train_with_all_projections == True:
        np.save("top_projections.npy", projection_score)

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

    

