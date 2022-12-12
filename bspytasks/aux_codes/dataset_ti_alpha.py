import os
import numpy as np
import librosa
from itertools import chain

def load_train_dataset():
    dataset = []
    label = []

    dataset_ti_alpha_train_dir_m8 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_alpha/train/m8/"
    dataset_ti_alpha_train_dir_f8 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_alpha/train/f8/"

    dataset_paths = (dataset_ti_alpha_train_dir_m8, 
                    dataset_ti_alpha_train_dir_f8)

    for subdir, _, files in chain.from_iterable(os.walk(path) for path in dataset_paths):
        for file in files:
            temp, _ = librosa.load(os.path.join(subdir, file),sr=12500,dtype=np.float32)
            dataset.append(temp)
            label.append(file[1])

    return dataset, label

def remove_silence(dataset):
    for v in range(len(dataset)):
        t_start = 0
        for i in range(np.shape(dataset[v])[0]):
            if dataset[v][i] <= 5e-4:
                t_start += 1
            else:
                break
        # t_stop = np.shape(dataset[v])[-1]
        for i in range(np.shape(dataset[v])[0]):
            if dataset[v][-1 - i] <= 5e-4:
                pass
            else:
                t_stop = i
                break
        dataset[v] = dataset[v][4 + t_start: np.shape(dataset[v])[0] - t_stop + 4]
    return dataset


if __name__ == '__main__':
    dataset, label = load_train_dataset(all_dataset= False)
    dataset = remove_silence(dataset)
    print("hi")
