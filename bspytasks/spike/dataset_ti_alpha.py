import os
import numpy as np
import librosa

def load_train_dataset(
                        all_dataset: True):
    dataset = []
    label = []
    if all_dataset == True:
        dataset_ti_alpha_train_dir = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_alpha/train/"
    else:
        dataset_ti_alpha_train_dir = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_alpha/train/f1/"
    for subdir, dirs, files in os.walk(dataset_ti_alpha_train_dir):
        for file in files:
            temp, _ = librosa.load(os.path.join(subdir, file),sr=12500,dtype=np.float32)
            dataset.append(temp)
            label.append(file[1])
    
    return dataset, label

def remove_silence(dataset):
    for v in range(len(dataset)):
        t_start = 0
        for i in range(np.shape(dataset[v])[0]):
            if dataset[v][i] <= 1e-3:
                t_start += 1
            else:
                break
        t_stop = np.shape(dataset[v])[-1] - 1
        for i in range(np.shape(dataset[v])[0]):
            if dataset[v][t_stop - i] <= 1e-3:
                t_stop -= 1
            else:
                break
        dataset[v] = dataset[v][t_start:t_stop]
    return dataset


if __name__ == '__main__':
    dataset, label = load_train_dataset(all_dataset= False)
    dataset = remove_silence(dataset)
    print("hi")
