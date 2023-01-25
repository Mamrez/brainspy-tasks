from cProfile import label
from itertools import cycle
from os import ftruncate
import numpy as np
import scipy

from scipy.signal import chirp
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

def butter_lowpass(cutoff, fs, order = 5):
    return scipy.signal.butter(order, cutoff, fs=fs, btype = 'low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order = 5):
    b, a = butter_lowpass(cutoff, fs, order = order)
    y = scipy.signal.lfilter(b, a, data)
    return y

def butter_highpass(cutoff, fs, order = 5):
    return scipy.signal.butter(order, cutoff, fs=fs, btype = 'high', analog=False)

def butter_highpass_filter(data, cutoff, fs, order = 5):
    b, a = butter_highpass(cutoff, fs, order = order)
    y = scipy.signal.lfilter(b, a, data)
    return y    

def filter_amplify(inputs):
    temp = []
    outputs = []
    for i in range(len(inputs)):
        temp.append(
                butter_lowpass_filter(
                    data    = inputs[i],
                    cutoff  = 6000,
                    fs      = 12500,
                    order   = 5
                )
        )
        outputs.append( 10 * (
                                butter_highpass_filter(
                                    data    = temp[i],
                                    cutoff  = 60,
                                    fs      = 12500,
                                    order = 5
                                )
                            )
        )

    return outputs

def chirp_analysis(
                    slope_length,
                    T,
                    fs,
                    start_freq,
                    stop_freq,
                    num_projections,
                    driver,
                    chirp_signal_method,
                    plot = True
                ):

    meas_input = np.zeros((7, int(2 * slope_length + T * fs)))

    t = np.linspace(0, T, int(T * fs))

    outputs = []
    chirp_signal = chirp(
                            t = t,
                            f0 = start_freq,
                            t1 = T,
                            f1 = stop_freq,
                            method= chirp_signal_method,
                            phi = 0
                        )
    meas_input[3, slope_length:-slope_length] = chirp_signal

    chirp_signal_freqs = np.fft.rfftfreq(n=len(chirp_signal), d = 1/fs)
    chirp_signal_rfft = np.fft.rfft(chirp_signal)
        
    for i in range(num_projections):
        if i != 0:
            meas_input = set_random_control_voltages(
                                            meas_input=             meas_input,
                                            dnpu_control_indeces=   [0, 1, 2, 4, 5, 6],
                                            slop_length=            slope_length,
                                            magnitudes=             [-.25, .25]
            )
        outputs.append(driver.forward_numpy(meas_input.T))    
    driver.close_tasks()

    filtered_amplified_outputs = filter_amplify(outputs)

    if plot == True:
        fig, axs = plt.subplots(2, 1)
        axs[0].set_xscale("log")
        axs[0].plot(chirp_signal_freqs, np.abs(chirp_signal_rfft))
        axs[1].specgram(chirp_signal, Fs=fs)
        axs[0].set_title("Fourier transform of chirp signal")
        axs[1].set_title("Spectogram of chirp signal")
        axs[0].set_xlabel("Frequency")
        axs[0].set_ylabel("Frequency component magnitude")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Frequency component magnitude")

    print("hi")

    # f_freqs = np.fft.rfftfreq(len(chirp_signal), d = 1/12500)
    # fig2, ax2 = plt.subplots(subplot_kw={'projection': '3d'})
    # specgram3d(outputs, srate=12500, ax=ax2, slope_length=slope_length)

    # if plot == True:
    #     plt.figure()
    #     fig2, axs2 = plt.subplots(num_projections, 1)
    #     output_freqs = np.fft.rfftfreq(n=len())
    #     for i in range(len(num_projections)):

    # plt.figure()
    # plt.xscale("log")
    # plt.yscale("log")
    # freqs = np.fft.rfftfreq(len(outputs[0][slope_length:-slope_length,0]), d=1/fs)    
    # plt.plot(freqs, np.abs(np.fft.rfft(outputs[1][slope_length:-slope_length,0])))


    # for i in range(len(outputs)):
    #     f_magnitude = np.abs(np.fft.rfft(outputs[i][slope_length:-slope_length,0]))
    #     plt.plot(f_freqs, f_magnitude)
    # plt.show()
    # fig1, ax1 = plt.subplots()
    # specgram2d(outputs[0][:,0], srate=12500, ax=ax1)



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

def chirp_phase_analysis(
    slope_length,
    T,
    fs,
    start_freq,
    stop_freq,
    num_projections,
    driver,
    chirp_signal_method
):

    meas_input = np.zeros((7, int(2 * slope_length + T * fs)))

    t = np.linspace(0, T, int(T * fs))

    outputs = []
    chirp_signal = chirp(
                            t = t,
                            f0 = start_freq,
                            t1 = T,
                            f1 = stop_freq,
                            method= chirp_signal_method,
                            phi = -90
                        )
    meas_input[3, slope_length:-slope_length] = chirp_signal

    for i in range(num_projections):
        if i != 0:
            meas_input = set_random_control_voltages(
                                            meas_input=             meas_input,
                                            dnpu_control_indeces=   [0, 1, 2, 4, 5, 6],
                                            slop_length=            slope_length,
                                            magnitudes=             [-.25, .25]
            )
        outputs.append(driver.forward_numpy(meas_input.T))    
    driver.close_tasks()

    output = outputs[0][slope_length : -slope_length, 0]

    plt.phase_spectrum(output, Fs=fs)

    print("hi")

def phase_diff(in1, in2, T, fs, freq):
    corr = scipy.signal.correlate(in1, in2)
    t_corr = np.linspace(-T, T, 2 * int(T * fs) - 1)

    max_corr = np.argmax(corr)

    phase_diff = 360. * freq * t_corr[max_corr]

    # while(phase_diff > 180):
    #     phase_diff -= 180

    # while(phase_diff < -180):
    #     phase_diff += 180


    # phases = (phases + np.pi) % (2 * np.pi) - np.pi

    phase_diff = (phase_diff + 180) % 360 - 180

    return phase_diff

def fixed_freq_phase_analysis_with_Marc(
    slope_length,
    rest_length,
    T,
    fs,
    start_freq,
    stop_freq,
    freqs_no,
    voltage_sweep,
    driver
):
    meas_input = np.zeros((7, int(2 * slope_length + rest_length + T * fs)))
    t = np.linspace(0, T, int(T * fs))

    freqs = np.linspace(start_freq, stop_freq, freqs_no)

    outputs = []
    phase_diff_list = []
    

    # meas_input = set_random_control_voltages(
    #     meas_input=             meas_input,
    #     dnpu_control_indeces=   [0, 1, 2, 4, 5, 6],
    #     slop_length=            slope_length,
    #     magnitudes=             [-.25, .25]
    # )

    if voltage_sweep == False:
        for i in range(0, freqs_no):
            sin_wave = np.sin(2 * np.pi * freqs[i] * t)
            meas_input[3, slope_length + rest_length : -slope_length] = sin_wave
            tmp = driver.forward_numpy(meas_input.T)
            outputs.append(tmp[slope_length + rest_length : - slope_length, 0])
            phase_diff_list.append(
                phase_diff(
                    sin_wave,
                    outputs[i],
                    T,
                    fs,
                    freqs[i]
                )
            )
        driver.close_tasks() 
        plt.scatter(freqs, phase_diff_list)
        plt.show()
    elif voltage_sweep == True:
        voltages = np.linspace(0., 0.8, freqs_no)
        sin_wave = np.sin(2 * np.pi * 250 * t)
        for i in range(0, len(voltages)):
            ramp_1 = np.linspace(0, voltages[i], slope_length)
            plateau = np.linspace(voltages[i], voltages[i], int(rest_length + T * fs))
            ramp_2 = np.linspace(voltages[i], 0, slope_length)
            meas_input[3, slope_length + rest_length : -slope_length] = sin_wave
            meas_input[0,  :] = np.concatenate((ramp_1, plateau, ramp_2))
            tmp = driver.forward_numpy(meas_input.T)
            outputs.append(tmp[slope_length + rest_length : - slope_length, 0])
            phase_diff_list.append(
                phase_diff(
                    sin_wave,
                    outputs[i],
                    T,
                    fs,
                    freqs[i]
                )
            )
        driver.close_tasks() 
        plt.scatter(voltages, phase_diff_list)
        plt.show()

        

    print("hi")


if __name__ == '__main__':

    from brainspy.utils.io import load_configs
    from brainspy.utils.manager import get_driver

    np.random.seed(0)  

    configs = load_configs('configs/defaults/processors/hw.yaml')
    driver = get_driver(configs=configs["driver"])

    fixed_freq_phase_analysis_with_Marc(
        slope_length= 5000,
        rest_length = 5000,
        T= 0.5,
        fs = 25000,
        start_freq = 20,
        stop_freq = 2000,
        freqs_no = 20,
        voltage_sweep= True,  
        driver= driver,
    )

    # chirp_phase_analysis(
    #     slope_length= 10000,
    #     T  = 3,
    #     fs = 25000,
    #     start_freq= 20,
    #     stop_freq= 5000,
    #     num_projections= 1,
    #     driver= driver,
    #     chirp_signal_method= 'linear'
    # )

    # chirp_analysis(
    #                 slope_length=25000,
    #                 T= 3,
    #                 fs= 25000,
    #                 start_freq= 10,
    #                 stop_freq= 2000,
    #                 num_projections= 5,
    #                 driver=driver,
    #                 chirp_signal_method = 'logarithmic' #'linear' 
    # )

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