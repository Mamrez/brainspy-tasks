from cProfile import label
from itertools import cycle
from os import ftruncate
import numpy as np
import scipy

from scipy.signal import chirp
from brainspy.utils.manager import get_driver

import librosa
import librosa.display

import matplotlib.pyplot as plt
from matplotlib import mlab

from brainspy.utils.io import load_configs
from brainspy.utils.manager import get_driver

def set_random_control_voltages( 
                meas_input,
                dnpu_control_indeces,
                slope_length,
                rand_matrix
                ):
    for i in range(len(dnpu_control_indeces)):
        ramp_up = np.linspace(0, rand_matrix[i], slope_length)
        plateau = np.linspace(rand_matrix[i], rand_matrix[i], np.shape(meas_input)[1] - 2 * slope_length)
        ramp_down = np.linspace(rand_matrix[i], 0, slope_length)
        meas_input[dnpu_control_indeces[i], :] = np.concatenate((ramp_up, plateau, ramp_down))

    return meas_input

def run_measurement(
        configs,
        freqs,
        random_control_voltages: bool,
        fs: int,
        T : float,
        normalize: bool,
        save: bool,
        slope_length : int,
        rest_length: int,
        amplitude = 0.75,
        PATH = None
):
    rand_matrix = np.random.uniform(
        -0.85, 
        0.85, 
        size = 6
    )
    driver = get_driver(configs["driver"])

    for f in freqs:

        meas_input = np.zeros((7, int(2 * slope_length + rest_length + T * fs)))
        t = np.linspace(0, T, int(T * fs))

        input_signal = amplitude * np.sin(2 * np.pi * f * t)

        if random_control_voltages == True:
            meas_input = set_random_control_voltages(
                meas_input = meas_input,
                dnpu_control_indeces = [0, 1, 2, 4, 5, 6],
                slope_length = slope_length,
                rand_matrix = rand_matrix
            )
    
        meas_input[3, slope_length + rest_length : -slope_length] = input_signal
        output = driver.forward_numpy(meas_input.T)

        output = output - np.mean(output[slope_length+rest_length-150:slope_length+rest_length-10])
        output = output[slope_length+rest_length:-slope_length,0]
    
        if normalize:
            output = (output / np.max(np.abs(output))) * amplitude
    
        if save:
            np.save(PATH+"_frequency_"+str(f), output)

    driver.close_tasks()

def bode_analysis(
        freqs,
        PATH,
        folders,
        T,
        fs
    ):
    
    
    t = np.linspace(0, T, int(T * fs))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xscale('log')
    
    if folders == 'all':
        pass
    else:
        for folder in folders:
            output_freq_components = []
            input_freq_components = []
            for f in freqs:
                output = np.load(PATH[:-2]+str(folder)+"/_frequency_"+str(f)+".npy")
                input = 0.75 * np.sin(2 * np.pi * f * t)

                specs_out = np.abs(np.fft.rfft(output))
                fft_freqs_out = np.fft.rfftfreq(len(output), 1/fs)

                specs_in = np.abs(np.fft.rfft(input))
                fft_freqs_in = np.fft.rfftfreq(len(input), 1/fs)

                idx = np.where(fft_freqs_in == f)

                output_freq_components.append(specs_out[idx])
                input_freq_components.append(specs_in[idx])
            
            res = []
            for j in range(len(input_freq_components)):
                res.append(output_freq_components[j]/input_freq_components[j])
            
            # ax.scatter(freqs, 20*np.log10(res))
            plt.plot(freqs, 20*np.log10(res), marker = '*')
    plt.show()

    pass


if __name__ == '__main__':

    freqs = np.arange(10, 1250, 2)

    configs = load_configs(
			'configs/defaults/processors/hw.yaml'
		)

    PATH = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/frequency_analysis/bode_plot/cv_rand_1/"

    # for i in range(0, 10):
    #     run_measurement(
    #         configs = configs,
    #         freqs = freqs,
    #         random_control_voltages = True,
    #         fs = 25000,
    #         T = 0.5,
    #         normalize = True,
    #         save = True,
    #         slope_length = 5000,
    #         rest_length = 12500,
    #         amplitude = 0.75,
    #         PATH = PATH[:-2]+str(i+17)+"/"
    #     )

    bode_analysis(
        freqs, 
        PATH,
        # either 'all' or a folder number
        folders = [20, 19, 17, 16, 5], 
        T = 0.5,
        fs = 25000
    )
