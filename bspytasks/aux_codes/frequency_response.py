from cProfile import label
from itertools import cycle
from os import ftruncate
import numpy as np
import scipy

from scipy.signal import chirp, spectrogram
from brainspy.utils.manager import get_driver


import matplotlib.pyplot as plt
from matplotlib import mlab


def set_random_control_voltages( 
                meas_input,
                dnpu_control_indeces,
                slop_length,
                magnitudes
                ):
    for i in range(len(dnpu_control_indeces)):
        rand_value = np.random.uniform(magnitudes[0], magnitudes[1], 1)
        ramp_up = np.linspace(0, rand_value[0], slop_length)
        plateau = np.linspace(rand_value[0], rand_value[0], np.shape(meas_input)[1] - 2 * slop_length)
        ramp_down = np.linspace(rand_value[0], 0, slop_length)
        meas_input[dnpu_control_indeces[i], :] = np.concatenate((ramp_up, plateau, ramp_down))

    return meas_input

def specgram3d(y, srate=12500, ax=None, title=None, slope_length= 12500):
    ax = plt.axes(projection='3d')
    ax.set_title(title, loc='center', wrap=True)
    cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    for i in range(len(y)):
        spec, freqs, t = mlab.specgram(y[i][slope_length:-slope_length,0], Fs=srate)
        X, Y, Z = t[None, :], freqs[:, None],  20.0 * np.log10(spec)
        ax.plot_surface(X, Y, Z, cmap=cmaps[i])
    ax.set_ylabel('frequencies (Hz)')
    ax.set_xlabel('time (s)')
    ax.set_zlabel('amplitude (dB)')
    plt.show()
    
def chirp_analysis(
                    slope_length,
                    T,
                    fs,
                    start_freq,
                    stop_freq,
                    num_projections,
                    driver,
                    chirp_signal_method,
                    repetitions = 3,
                    rest_length = 8000):

    meas_input = np.zeros((7, int(rest_length + 2 * slope_length + T * fs)))
    t = np.linspace(0, T, int(T * fs))
    outputs = []
    for i in range(num_projections):
        chirp_signal = chirp(
                            t = t,
                            f0 = start_freq,
                            t1 = T,
                            f1 = stop_freq,
                            method= chirp_signal_method,
                            phi = 90)
        chirp_signal_tilde = (start_freq / T) * np.exp(-t/T) * np.flip(chirp_signal)
        meas_input[3, rest_length + slope_length:-slope_length] = 1.2 * chirp_signal
        if i != 0:
            meas_input = set_random_control_voltages(
                                            meas_input=meas_input,
                                            dnpu_control_indeces=[0, 1, 2, 4, 5, 6],
                                            slop_length=slope_length,
                                            magnitudes=[-.85, .85]
            )
        for j in range(repetitions):
            outputs.append(driver.forward_numpy(meas_input.T))    
    driver.close_tasks()

    # f_freqs = np.fft.rfftfreq(len(chirp_signal), d = 1/12500)
    # fig2, ax2 = plt.subplots(subplot_kw={'projection': '3d'})
    # specgram3d(outputs, srate=12500, ax=ax2, slope_length=slope_length)

    
    plt.figure()
    # plt.gca().set_xlim(0, 600)
    plt.xscale("log")
    freqs = np.fft.rfftfreq(len(chirp_signal), 1/25000)
    (s, f) = plt.psd(chirp_signal/20, Fs=25000, NFFT=int(2 ** np.ceil(np.log2(len(chirp_signal)))))
    for i in range(num_projections):
        signal = outputs[i][slope_length+rest_length:-slope_length,0] - np.average(outputs[i][slope_length+rest_length-500:slope_length+rest_length-100,0])
        (s, f) = plt.psd(signal/20, Fs=25000, NFFT=int(2 ** np.ceil(np.log2(len(signal)))))

    threshold = 6.02 * 16 + 1.76
    threshold += 10 * np.log(fs/2)
    threshold = np.linspace(-threshold, -threshold, len(f))
    plt.plot(threshold)
    plt.show()

    # plt.figure()
    # # plt.gca().set_xlim(0, 600)
    # plt.xscale("log")
    # s, f = scipy.signal.welch(chirp_signal/20, fs=25000, nperseg=len(chirp_signal))
    # plt.plot(s, 10 * np.log(f))
    # for i in range(num_projections):
    #     signal = outputs[i][slope_length+rest_length:-slope_length,0] - np.average(outputs[i][slope_length+rest_length-500:slope_length+rest_length-100,0])
    #     s, f = scipy.signal.welch(signal/20, fs=25000, nperseg=len(signal))
    #     plt.plot(s, 10*np.log(f))

    # # threshold = np.linspace(-136.06, -136.06, len(f))
    # plt.plot(threshold)
    

    print("")


    # for i in range(len(outputs)):
    #     f_magnitude = np.abs(np.fft.rfft(outputs[i][slope_length:-slope_length,0]))
    #     plt.plot(f_freqs, f_magnitude)
    # plt.show()
    # fig1, ax1 = plt.subplots()
    # specgram2d(outputs[0][:,0], srate=12500, ax=ax1)
    print("")



# def specgram2d(y, srate=12500, ax=None, title=None):
# #   if not ax:
#     ax = plt.axes()
#     ax.set_title(title, loc='center', wrap=True)
#     spec, freqs, t, im = ax.specgram(y, Fs=srate, scale='dB')
#     ax.set_xlabel('time (s)')
#     ax.set_ylabel('frequencies (Hz)')
#     cbar = plt.colorbar(im, ax=ax)
#     cbar.set_label('Amplitude (dB)')
#     cbar.minorticks_on()
#     return spec, freqs, t, im

def bode_analysis(
                    slope_length,
                    fs,
                    start_freq,
                    stop_freq,
                    num_steps,
                    num_projections,
                    driver):

    sine_freqs_steps = np.linspace(start_freq, stop_freq, num_steps, dtype=int)
    cycle_durations = 1 / sine_freqs_steps

    sine_cycles = []
    for i in range(len(sine_freqs_steps)):
        t = np.linspace(0, cycle_durations[i], int(fs * cycle_durations[i]))
        sine_cycles.append(np.sin(2 * np.pi * sine_freqs_steps[i] * t))
    
    outputs = []
    for i in range(num_projections):
        for j in range(len(sine_cycles)):
            meas_input = np.zeros((6, int(2 * slope_length + len(sine_cycles[j]))))
            if i != 0:
                meas_input = set_random_control_voltages(
                                            meas_input=meas_input,
                                            dnpu_control_indeces=[0, 2, 3, 4, 5],
                                            slop_length=slope_length,
                                            magnitudes=[-0.3, 0.3])
            meas_input[1, slope_length:-slope_length] = sine_cycles[j]
            outputs.append(driver.forward_numpy(meas_input.T))
    
    plt.figure()
    for i in range(len(outputs)):
        freqs_output = np.fft.rfftfreq(np.shape(outputs[i][:,0])[0], d = 1/12500)
        plt.plot(freqs_output, np.fft.rfft(outputs[i][slope_length:-slope_length,0]), label=str("Output FFT, sine input freq = " + sine_freqs_steps[i]))
        # freqs_input = np.fft.rfftfreq(np.shape(sine_cycles[i][:])[0], d=1/12500)
        # plt.plot(freqs_input, np.fft.rfft(sine_cycles[i]))
    plt.legend()
    plt.show()

    # plt.figure()
    # plt.xscale("log")
    # for i in range(len(outputs)):
    #     for j in range(len(sine_cycles)):
    #         plt.plot(sine_freqs_steps[j], outputs[i * num_projections + j, slope_length:-slope_length]/sine_cycles[j])


if __name__ == '__main__':

    from brainspy.utils.io import load_configs
    from brainspy.utils.manager import get_driver

    # np.random.seed(0)  

    configs = load_configs('configs/defaults/processors/hw_freq_analysis.yaml')
    driver = get_driver(configs=configs["driver"])
    # configs['driver']['instruments_setup']['activation_sampling_frequency'] = 62500

    chirp_analysis(
                    slope_length=4000,
                    T= 5,
                    fs= 25000,
                    start_freq= 10,
                    stop_freq= 300,
                    num_projections= 5,
                    driver=driver,
                    chirp_signal_method = 'logarithmic',
                    repetitions = 1
    )

    # bode_analysis(
    #             slope_length= 12500,
    #             fs = 12500,
    #             start_freq= 60,
    #             stop_freq= 2000,
    #             num_steps = 10,
    #             num_projections= 4,
    #             driver= driver
    # )
    
    print("")