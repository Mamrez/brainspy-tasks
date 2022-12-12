import os
import numpy as np
import librosa
from itertools import chain
import scipy
import matplotlib.pyplot as plt

def load_dataset():
    dataset_train_val = []
    dataset_test = []
    label_train_val = []
    label_test = []

    dataset_ti_spoken_digits_train_dir_m1 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/train/m1"
    dataset_ti_spoken_digits_train_dir_f1 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/train/f1"    
    dataset_ti_spoken_digits_train_dir_m2 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/train/m2"    
    dataset_ti_spoken_digits_train_dir_f2 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/train/f2"    
    dataset_ti_spoken_digits_train_dir_m3 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/train/m3"    
    dataset_ti_spoken_digits_train_dir_f3 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/train/f3"    
    dataset_ti_spoken_digits_train_dir_m4 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/train/m4"    
    dataset_ti_spoken_digits_train_dir_f4 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/train/f4"
    dataset_ti_spoken_digits_train_dir_m5 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/train/m5"
    dataset_ti_spoken_digits_train_dir_f5 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/train/f5"
    dataset_ti_spoken_digits_train_dir_m6 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/train/m6"
    dataset_ti_spoken_digits_train_dir_f6 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/train/f6"    
    dataset_ti_spoken_digits_train_dir_m7 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/train/m7"    
    dataset_ti_spoken_digits_train_dir_f7 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/train/f7"
    dataset_ti_spoken_digits_train_dir_m8 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/train/m8"
    dataset_ti_spoken_digits_train_dir_f8 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/train/f8"

    dataset_ti_spoken_digits_test_dir_m1 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/test/m1"
    dataset_ti_spoken_digits_test_dir_f1 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/test/f1"    
    dataset_ti_spoken_digits_test_dir_m2 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/test/m2"    
    dataset_ti_spoken_digits_test_dir_f2 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/test/f2"
    dataset_ti_spoken_digits_test_dir_m3 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/test/m3"
    dataset_ti_spoken_digits_test_dir_m4 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/test/m4"    
    dataset_ti_spoken_digits_test_dir_m5 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/test/m5"
    dataset_ti_spoken_digits_test_dir_f4 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/test/f4"    
    dataset_ti_spoken_digits_test_dir_f5 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/test/f5"    
    dataset_ti_spoken_digits_test_dir_f6 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/test/f6"
    dataset_ti_spoken_digits_test_dir_f7 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/test/f7"
    dataset_ti_spoken_digits_test_dir_f8 = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/spike/ti46/ti_spoken_digits/test/f8"

    


    dataset_paths = (#dataset_ti_spoken_digits_train_dir_m1, 
                    # dataset_ti_spoken_digits_train_dir_f1,
                    # dataset_ti_spoken_digits_train_dir_m2,
                    # dataset_ti_spoken_digits_train_dir_f2,
                    # dataset_ti_spoken_digits_train_dir_m3,
                    # dataset_ti_spoken_digits_train_dir_m4,
                    # dataset_ti_spoken_digits_train_dir_f3,
                    # dataset_ti_spoken_digits_train_dir_m4,
                    dataset_ti_spoken_digits_train_dir_f4,
                    # dataset_ti_spoken_digits_train_dir_m5,
                    # dataset_ti_spoken_digits_train_dir_f5,
                    # dataset_ti_spoken_digits_train_dir_m6,
                    # dataset_ti_spoken_digits_train_dir_f6,
                    # dataset_ti_spoken_digits_train_dir_m7,
                    # dataset_ti_spoken_digits_train_dir_f7,
                    # dataset_ti_spoken_digits_train_dir_f8,
                    # dataset_ti_spoken_digits_train_dir_m8,
                    # dataset_ti_spoken_digits_train_dir_f8,
                    # dataset_ti_spoken_digits_test_dir_m1, 
                    # dataset_ti_spoken_digits_test_dir_f1,
                    # dataset_ti_spoken_digits_test_dir_m2,
                    # dataset_ti_spoken_digits_test_dir_f2,
                    # dataset_ti_spoken_digits_test_dir_m3,
                    # dataset_ti_spoken_digits_test_dir_m4,
                    # dataset_ti_spoken_digits_test_dir_m5,
                    dataset_ti_spoken_digits_test_dir_f4,
                    # dataset_ti_spoken_digits_test_dir_f5,
                    # dataset_ti_spoken_digits_test_dir_f6,
                    # dataset_ti_spoken_digits_test_dir_f7
                    # dataset_ti_spoken_digits_test_dir_f8

                )

    for subdir, _, files in chain.from_iterable(os.walk(path) for path in dataset_paths):
        for file in files:
            if subdir[-8:-3] == "train":
                temp, _ = librosa.load(os.path.join(subdir, file), sr=12500,dtype=np.float32)
                dataset_train_val.append(temp)
                label_train_val.append(file[1])
            if subdir[-7:-3] == "test":
                temp, _ = librosa.load(os.path.join(subdir, file), sr=12500,dtype=np.float32)
                dataset_test.append(temp)
                label_test.append(file[1])

    return dataset_train_val, label_train_val, dataset_test, label_test

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

def remove_silence_with_average(dataset_train_val, dataset_test):
    for v in range(len(dataset_train_val)):
        t_start = 0
        for i in range(len(dataset_train_val[v])):
            if (i > 5):
                if np.average(dataset_train_val[v][i-5:i]) < 2e-3:
                    t_start = i
                else:
                    break
        for i in range(len(dataset_train_val[v]), 0, -1):
            if (i < len(dataset_train_val[v]) - 5):
                if np.average(dataset_train_val[v][i:i+5]) < 2e-3:
                    t_stop = i
                else:
                    break
        dataset_train_val[v] = dataset_train_val[v][t_start -5 : t_stop + 5]
    for v in range(len(dataset_test)):
        t_start = 0
        for i in range(len(dataset_test[v])):
            if (i > 5):
                if np.average(dataset_test[v][i-5:i]) < 2e-3:
                    t_start = i
                else:
                    break
        for i in range(len(dataset_test[v]), 0, -1):
            if (i < len(dataset_test[v]) - 5):
                if np.average(dataset_test[v][i:i+5]) < 2e-3:
                    t_stop = i
                else:
                    break
        dataset_test[v] = dataset_test[v][t_start -5 : t_stop + 5]

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
                                            cutoff= 4000,
                                            fs =12500,
                                            order= 10
        )
    for i in range(len(dataset_test)):
        dataset_test[i] = butter_lowpass_filter(
                                            data = dataset_test[i],
                                            cutoff= 4000,
                                            fs =12500,
                                            order= 10
        )

    return dataset_train_val, dataset_test

def shift_frequency_for_female_speakers(dataset_train_val,
                                    dataset_test,
                                    frequency_shift = 125, # in Hz
                                    fs = 12500
                                    ):
    for i in range(len(dataset_train_val)):
        freqs = np.fft.rfftfreq(len(dataset_train_val[i]), d = 1/12500)
        for j in range(len(freqs)):
            if freqs[j] >= frequency_shift:
                f_start = j
                break
        f_transform = np.fft.rfft(dataset_train_val[i])
        shifted_in_frequency = np.zeros((len(f_transform)), dtype=np.complex_)
        shifted_in_frequency[0:-f_start] = f_transform[f_start:]
        dataset_train_val[i] = np.fft.irfft(shifted_in_frequency)
        
    
    for i in range(len(dataset_test)):
        freqs = np.fft.rfftfreq(len(dataset_test[i]), d = 1/12500)
        for j in range(len(freqs)):
            if freqs[j] >= frequency_shift:
                f_start = j
                break
        f_transform = np.fft.rfft(dataset_test[i])
        shifted_in_frequency = np.zeros((len(f_transform)), dtype=np.complex_)
        shifted_in_frequency[0:-f_start] = f_transform[f_start:]
        dataset_test[i] = np.fft.irfft(shifted_in_frequency)
       

    return dataset_train_val, dataset_test

if __name__ == '__main__':
    dataset_train_val, label_train_val, dataset_test, label_test = load_dataset()
    dataset_train_val, dataset_test = audio_low_pass_filter(dataset_train_val, dataset_test)

    dataset_train_val, dataset_test = shift_frequency_for_female_speakers(dataset_train_val, dataset_test)

    # dataset_train_val, dataset_test  = remove_silence(dataset_train_val, dataset_test)
    dataset_train_val, dataset_test  = remove_silence_with_average(dataset_train_val, dataset_test)
    

    print("hi")
