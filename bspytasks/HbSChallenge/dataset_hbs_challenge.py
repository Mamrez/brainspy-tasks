import os
import numpy as np
import librosa
# from itertools import chain
import scipy
import matplotlib.pyplot as plt

def load_train_dataset():
    dataset = []
    label = []

    dataset_hbs_challenge_normal        = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/HbSChallenge/dataset/dataset_A/Atraining_normal/Atraining_normal"
    dataset_hbs_challenge_murmur        = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/HbSChallenge/dataset/dataset_A/Atraining_murmur/Atraining_murmur"
    dataset_hbs_challenge_extrasystole  = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/HbSChallenge/dataset/dataset_A/Atraining_extrahs/Atraining_extrahls"
    dataset_hbs_challenge_artifacts     = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/HbSChallenge/dataset/dataset_A/Atraining_artifact/Atraining_artifact"


    dataset_paths = [
                    dataset_hbs_challenge_normal,
                    dataset_hbs_challenge_murmur,
                    dataset_hbs_challenge_extrasystole,
                    dataset_hbs_challenge_artifacts
    ]

    for i in range(len(dataset_paths)):
        for subdir, _, files in os.walk(dataset_paths[i]):
            for file in files:
                # print(librosa.get_samplerate(os.path.join(subdir, file))) -> 4000
                temp, sr = librosa.load(os.path.join(subdir, file), sr = 4000, dtype=np.float32)
                if librosa.get_duration(y=temp, sr=sr) >= 2.0:
                    dataset.append(temp[100:8100]) # cropping two seconds
                    if subdir[-6:-1] == "norma":
                        label.append("normal")
                    if subdir[-6:-1] == "murmu":
                        label.append("murmur")
                    if subdir[-6:-1] == "trahl":
                        label.append("extra")
                    if subdir[-6:-1] == "tifac":
                        label.append("artifact")

    return dataset, label

def butter_lowpass(cutoff, fs, order = 5):
    return scipy.signal.butter(order, cutoff, fs=fs, btype = 'low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order = 5):
    b, a = butter_lowpass(cutoff, fs, order = order)
    y = scipy.signal.lfilter(b, a, data)
    return y

# filtering out frequencies bellow 195 Hz
def low_pass_filter(dataset):
    lowpass_filtered_dataset = []
    for i in range(len(dataset)):
        lowpass_filtered_dataset.append(
                                        butter_lowpass_filter(
                                            data = dataset[i],
                                            cutoff = 195,
                                            fs= 4000,
                                            order= 5
                                        )
        )
    return lowpass_filtered_dataset

if __name__ == '__main__':
    dataset, label = load_train_dataset()
    dataset = low_pass_filter(dataset)
    print("hi")
