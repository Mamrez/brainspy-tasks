from cgi import test
from distutils.command.config import config
from pickletools import optimize
import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import gc, os

from itertools import chain

from torchvision import transforms

import dataset_hbs_challenge

from brainspy.utils.manager import get_driver

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
        val_dataloader
        ):

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # optimizer = torch.optim.NAdam(model.parameters(), lr = 0.001)


    best_vloss = np.inf

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("runs/fashion_trainer_{}".format(timestamp))

    for epoch in range(num_epoch):
        print("Epoch {}: ".format(epoch + 1))
        
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
                print(" batch {} loss: {}".format(i + 1, last_loss))
                tb_x = epoch * len(train_dataloader) + i + 1
                writer.add_scalar("Loss/train", last_loss, tb_x)
                running_loss = 0.

        avg_loss = last_loss

        # Validation step
        model.train(False)
        running_vloss = 0.0
        total = 0
        running_accuracy = 0.
        for i, vdata in enumerate(val_dataloader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss.item()

            _, predicted = torch.max(voutputs, 1)
            total += vlabels.size(0)
            running_accuracy += (predicted == vlabels).sum().item()
        
        avg_vloss = running_vloss / len(val_dataloader)

        print("Train loss {}, Validation loss {}, Validation accuracy {}%".format(avg_loss, avg_vloss, (100 * running_accuracy / total)))

        writer.add_scalars(
                        "Validation loss vs. Validation accuracy",
                        {"Validation": avg_vloss, "Accuracy":(100 * running_accuracy / total)},
                        epoch + 1
                        )
        # writer.add_scalars(
        #             "Test accuracy: ",
        #             {"Test accuracyc": (100 * running_accuracy / total)},
        #             epoch + 1
        #             )
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            # model_path = "model_{}_{}".format(timestamp, epoch)
            torch.save(model.state_dict(), "best_trained_model.pt")
        


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

    dataset, labels = dataset_hbs_challenge.load_train_dataset()
    dataset = dataset_hbs_challenge.low_pass_filter(dataset=dataset)
    
    cnt = 0
    driver = get_driver(configs=configs["driver"])
    rand_matrix = np.random.uniform(-0.25, 0.25, size=(len(dnpu_control_indeces), num_projections))
    for d in range(len(dataset)):
        dataset_after_projection = []
        labels_after_projection = []
        for p_idx in range(num_projections):
            cnt += 1
            # 0 -> slope -> rest -> voice -> slope -> 0
            meas_input = np.zeros((len(dnpu_control_indeces) + 1, rest_length + 2 * slope_length + np.shape(dataset[d])[0]))
            meas_input[dnpu_input_index, slope_length + rest_length: -slope_length] = 1.5 * dataset[d][:]
            
            meas_input = set_random_control_voltages(
                                                    meas_input=             meas_input,
                                                    dnpu_control_indeces=   dnpu_control_indeces,
                                                    slope_length=           slope_length,
                                                    projection_idx=         p_idx,
                                                    rand_matrix=            rand_matrix)

            print("Completed pecentage: %.3f" %(100 * (cnt / (num_projections * (len(dataset))))), "%")

            output = driver.forward_numpy(meas_input.T)

            dataset_after_projection.append(output[slope_length+rest_length:-slope_length])
            labels_after_projection.append(labels[d])
        
        path_dataset = "tmp/projected_ti_alpha/boron_roomTemp_30nm/hbs_challenge_datasetA/" + "data_" + str(d)
        path_label = "tmp/projected_ti_alpha/boron_roomTemp_30nm/hbs_challenge_datasetA/" + "label_" + str(d)
        np.save(path_dataset, dataset_after_projection)
        np.save(path_label, labels_after_projection)
        del dataset_after_projection
        del labels_after_projection
        gc.collect()
    
    driver.close_tasks()
    
    # return dataset_after_projection, labels_after_projection

def convert_to_torch_dataset(dataset_after_projection, labels_after_projection):
    tensor_x = torch.Tensor(dataset_after_projection)
    # tensor_y = torch.Tensor(labels_after_projection)

    # torch.from_numpy

    le = preprocessing.LabelEncoder()
    tensor_y = le.fit_transform(labels_after_projection)

    dataset = TensorDataset(tensor_x, torch.as_tensor(tensor_y))

    val_dataset_size = int(0.25 * len(dataset))
    train_dataset_size = len(dataset) - val_dataset_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_dataset_size, val_dataset_size])

    train_dataloader = DataLoader(train_dataset,
                                batch_size=64, 
                                shuffle=True
                                )

    val_dataloader = DataLoader(val_dataset,
                                shuffle= True
                                )

    return train_dataloader, val_dataloader

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

if __name__ == '__main__':
    # from brainspy.utils.io import load_configs

    # np.random.seed(1) 

    # configs = load_configs('configs/defaults/processors/hw_hbs_challenge.yaml')
    # slope_length = 1000
    # rest_lenth = 6000

    # measurement(
    #             configs=configs,
    #             num_projections= 128,
    #             dnpu_input_index = 3,
    #             dnpu_control_indeces = [0, 1, 2, 4, 5, 6],
    #             slope_length = slope_length,
    #             rest_length= rest_lenth
    #         )

    # measurement_for_test(

    # )

    # load projected dataset
    dataset_after_projection = []
    labels_after_projection = []

    projected_training_data = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/boron_roomTemp_30nm/hbs_challenge_datasetA/"

    for subdir, _, files in os.walk(projected_training_data):
        for file in files:
            if file[0] == 'd':
                temp = np.load(os.path.join(subdir, file))
                for i in range(np.shape(temp)[0]):
                    dataset_after_projection.append(temp[i])
            elif file[0] == 'l':
                temp = np.load(os.path.join(subdir, file)) 
                for i in range(np.shape(temp)[0]):
                    labels_after_projection.append(temp[0])

    for i in range(len(dataset_after_projection)):
        mean = np.mean(dataset_after_projection[i][:,0])
        dc_removed = dataset_after_projection[i][:,0] - mean
        dataset_after_projection[i] = 10 * dc_removed

    # downsample
    down_sample_no = 512
    min_length = np.inf
    for i in range(len(dataset_after_projection)):
        if np.shape(dataset_after_projection[i])[0] < min_length:
            min_length = np.shape(dataset_after_projection[i])[0]
    idx = np.arange(0, min_length, dtype=int)
    idx = np.sort(np.random.choice(idx, down_sample_no))

    for i in range(len(dataset_after_projection)):
        dataset_after_projection[i] = dataset_after_projection[i][idx]

    train_dataloader, val_dataloader = convert_to_torch_dataset(
                                                    dataset_after_projection, 
                                                    labels_after_projection
                                                    )

    linear_layer = LinearLayer(
                            input_size= down_sample_no,
                            num_classes= 4
    )

    # torch.save(t_dataloader, 'f2_data_128_projection_test.pt')

    train(
        model = linear_layer,
        num_epoch= 500,
        train_dataloader= train_dataloader,
        val_dataloader = val_dataloader
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

    

