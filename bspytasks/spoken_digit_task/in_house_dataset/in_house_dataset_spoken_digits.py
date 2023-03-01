import os
import numpy as np
import librosa
from itertools import chain
import scipy
import matplotlib.pyplot as plt
from pydub import AudioSegment
import sklearn
import pyrubberband as pyrb

def load_dataset():

    dataset_train = []
    dataset_test = []
    label_train = []
    label_test = []

    in_house_dataset_train_wilfred = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spoken_digit_task/in_house_dataset/wilfred_spoken_digits/train"
    in_house_dataset_test_wilfred = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spoken_digit_task/in_house_dataset/wilfred_spoken_digits/test"


    dataset_paths = (
        in_house_dataset_train_wilfred,
        in_house_dataset_test_wilfred
    )

    for subdir, _, files in chain.from_iterable(os.walk(path) for path in dataset_paths):
        for file in files:
            if subdir[-5:] == "train":
                temp, _ = librosa.load(os.path.join(subdir, file), sr=24000, mono = True, dtype=np.float32)
                dataset_train.append(temp)
                label_train.append(file[0:2])
            if subdir[-4:] == "test":
                temp, _ = librosa.load(os.path.join(subdir, file), sr=24000, mono = True, dtype=np.float32)
                dataset_test.append(temp)
                label_test.append(file[0:2])

    return dataset_train, label_train, dataset_test, label_test

def remove_silence(dataset_train_val, dataset_test):
    for v in range(len(dataset_train_val)):
        t_start = 0
        for i in range(np.shape(dataset_train_val[v])[0]):
            if dataset_train_val[v][i] <= 1e-3:
                t_start += 1
            else:
                break
        # t_stop = np.shape(dataset[v])[-1]
        for i in range(np.shape(dataset_train_val[v])[0]):
            if dataset_train_val[v][-1 - i] <= 1e-3:
                pass
            else:
                t_stop = i
                break
        dataset_train_val[v] = dataset_train_val[v][4 + t_start: np.shape(dataset_train_val[v])[0] - t_stop + 4]
    for v in range(len(dataset_test)):
        t_start = 0
        for i in range(np.shape(dataset_test[v])[0]):
            if dataset_test[v][i] <= 1e-3:
                t_start += 1
            else:
                break
        # t_stop = np.shape(dataset[v])[-1]
        for i in range(np.shape(dataset_test[v])[0]):
            if dataset_test[v][-1 - i] <= 1e-3:
                pass
            else:
                t_stop = i
                break
        dataset_test[v] = dataset_test[v][4 + t_start: np.shape(dataset_test[v])[0] - t_stop + 4]
    return dataset_train_val, dataset_test

def remove_silence_with_average(dataset_train_val, dataset_test, threshold):
    for v in range(len(dataset_train_val)):
        t_start = 0
        for i in range(len(dataset_train_val[v])):
            if (i > 5):
                if np.average(dataset_train_val[v][i-5:i]) < threshold:
                    t_start = i
                else:
                    break
        for i in range(len(dataset_train_val[v]), 0, -1):
            if (i < len(dataset_train_val[v]) - 5):
                if np.average(dataset_train_val[v][i:i+5]) < threshold:
                    t_stop = i
                else:
                    break
        dataset_train_val[v] = dataset_train_val[v][t_start - 50 : t_stop + 50]
        if len(dataset_train_val[v]) > 12000:
            print("")
    for v in range(len(dataset_test)):
        t_start = 0
        for i in range(len(dataset_test[v])):
            if (i > 5):
                if np.average(dataset_test[v][i-5:i]) < threshold:
                    t_start = i
                else:
                    break
        for i in range(len(dataset_test[v]), 0, -1):
            if (i < len(dataset_test[v]) - 5):
                if np.average(dataset_test[v][i:i+5]) < threshold:
                    t_stop = i
                else:
                    break
        dataset_test[v] = dataset_test[v][t_start - 50 : t_stop + 50]
        if len(dataset_test[v]) > 12000:
            print("")
        if len(dataset_test[v]) < 1000:
            print("Warning!")

    return dataset_train_val, dataset_test

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

def audio_low_pass_filter(dataset_train_val, dataset_test):
    for i in range(len(dataset_train_val)):
        dataset_train_val[i] = butter_lowpass_filter(
                                            data = dataset_train_val[i],
                                            cutoff= 4500,
                                            fs = 24000,
                                            order= 5
        )
    for i in range(len(dataset_test)):
        dataset_test[i] = butter_lowpass_filter(
                                            data = dataset_test[i],
                                            cutoff= 4500,
                                            fs = 24000,
                                            order= 5
        )

    return dataset_train_val, dataset_test

def shift_frequency(
        dataset_train_val,
        dataset_test,
        frequency_shift = 125, # in Hz
        fs = 24000
):
    for i in range(len(dataset_train_val)):
        freqs = np.fft.rfftfreq(len(dataset_train_val[i]), d = 1/fs)
        for j in range(len(freqs)):
            if freqs[j] >= frequency_shift:
                f_start = j
                break
        f_transform = np.fft.rfft(dataset_train_val[i])
        shifted_in_frequency = np.zeros((len(f_transform)), dtype=np.complex_)
        shifted_in_frequency[0:-f_start] = f_transform[f_start:]
        dataset_train_val[i] = np.fft.irfft(shifted_in_frequency)
        
    
    for i in range(len(dataset_test)):
        freqs = np.fft.rfftfreq(len(dataset_test[i]), d = 1/fs)
        for j in range(len(freqs)):
            if freqs[j] >= frequency_shift:
                f_start = j
                break
        f_transform = np.fft.rfft(dataset_test[i])
        shifted_in_frequency = np.zeros((len(f_transform)), dtype=np.complex_)
        shifted_in_frequency[0:-f_start] = f_transform[f_start:]
        dataset_test[i] = np.fft.irfft(shifted_in_frequency)
       

    return dataset_train_val, dataset_test

def normalize(dataset):
    for i in range(len(dataset)):
        dataset[i] = sklearn.preprocessing.minmax_scale(dataset[i], feature_range = (-1, 1))
        # dataset[i] = (dataset[i] - np.mean(dataset[i])) / np.std(dataset[i])
        dataset[i] = dataset[i] - np.mean(dataset[i])
    return dataset

def pitch_shift(
    dataset,
    fs=24000,
):
    for i in range(len(dataset)):
        # dataset[i] = pyrb.pitch_shift(dataset[i], sr=fs, n_steps=-6)
        dataset[i] = librosa.effects.pitch_shift(
            dataset[i],
            sr=fs,
            n_steps=-12,
            bins_per_octave=12
        )
    return dataset

if __name__ == '__main__':
    dataset_train, label_train, dataset_test, label_test = load_dataset()

    dataset_train, dataset_test = audio_low_pass_filter(dataset_train, dataset_test)

    dataset_train = pitch_shift(dataset_train)
    dataset_test = pitch_shift(dataset_test)

    dataset_train = normalize(dataset_train)
    dataset_test = normalize(dataset_test)


    print("hi")
