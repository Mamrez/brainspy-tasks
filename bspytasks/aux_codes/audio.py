
from cProfile import label
from msilib.schema import AdvtUISequence
from brainspy.utils.manager import get_driver

    

def ramping(start, stop, slope_points_no, plateau_length):
    ramp_up = np.linspace(start, stop, slope_points_no)

    plateau = np.linspace(stop, stop, plateau_length)

    ramp_down = np.linspace(stop, start, slope_points_no)

    return np.concatenate((ramp_up, plateau, ramp_down))

def set_control_electrodes(
                            inputs,
                            control_indeces,
                            control_voltages,
                            meas_length,
                            slope_length
                        ):
    # cv_iter = iter(control_voltages)
    # for i in control_indeces:
    #     inputs[i, :] = ramping(0, next(cv_iter), slope_length, meas_length - 2 * slope_length)
    for i in range(len(control_indeces)):
        inputs[control_indeces[i], :] = ramping(0, control_voltages[i], slope_length, meas_length - 2 * slope_length)
    return inputs

def run_task(
            input_index,
            control_indeces,
            control_voltages,
            audio_file,
            slope_length,
            configs
            ):

    outputs = []
    for i in range(len(audio_file)):
        meas_length = audio_file[0].shape[0] + 2 * slope_length
        inputs = np.zeros((6, meas_length))

        inputs[input_index, slope_length : meas_length - slope_length] = 2 * audio_file[i][:, 0]
        inputs = set_control_electrodes(inputs, control_indeces, control_voltages, meas_length, slope_length)

        driver = driver = get_driver(configs["driver"])
        output = driver.forward_numpy(inputs.T)
        driver.close_tasks()


        outputs.append(output)


    return outputs


def solution(s):
    # s : input -> ""
    # return : int 
    
    divid = 0
    
    # init solution
    rep_list = s[0]
    
    for i in range(1, len(s)):
        if s[i] != rep_list[0]:
            rep_list += s[i]
        else:
            # checking if this is the repetition
            if rep_list == s[i : i + len(rep_list)]:
                divid = i
                if rep_list == s[-divid:-1]:
                  break
                else:
                  rep_list += s[i]
            else:
                rep_list += s[i]
    if divid != 0:
        return int(len(s)/divid)
    else:
        return int(divid)

if __name__ == '__main__':
    
    s = "sldkjghassssssldkjghassssssldkjghassssssldkjghasssss"
    
    d = solution(s)

    print(len(s) / d)

    print("")
    
    
    # import matplotlib.pyplot as plt
    # import numpy as np

    # import scipy
    # from scipy.io import wavfile

    # from brainspy.utils.io import load_configs

    # s_rate, a_audio = wavfile.read('bspytasks/spike/a_audio_48KSPS_normalized.wav')
    # s_rate_b, b_audio = wavfile.read('bspytasks/spike/b_audio_48KSPS_normalized.wav')
    # b_audio = b_audio / 32767

    # audios = [a_audio, b_audio]

    # time = np.linspace(0., (a_audio.shape[0]/s_rate), a_audio.shape[0])
    
    # configs = load_configs('configs/defaults/processors/hw.yaml')

    # control_voltages = np.random.uniform(-0.8, 0.8, 5)
    # # control_voltages = np.zeros((5))

    # # output = run_task(  input_index=1,
    # #                     control_indeces=[0, 2, 3, 4, 5],
    # #                     control_voltages = control_voltages/10,
    # #                     audio_file= audios,
    # #                     slope_length= 10,
    # #                     configs= configs
    # # )

    # output_spectrum = []
    # outputs = []
    # for i in range(2):
    #     output = run_task(
    #                         input_index=1,
    #                         control_indeces=[0, 2, 3, 4, 5],
    #                         control_voltages= np.array([0.05, -0.09, -0.001, 0.085, -0.0091]), # np.random.uniform(-0.4, 0.4, 5), #
    #                         audio_file= [audios[0]], # a_audio
    #                         slope_length= 10,
    #                         configs=configs
    #     )
    #     output_spectrum.append(plt.magnitude_spectrum(output[0][:, 0], Fs=48000)[0][55:20000])
    #     outputs.append(output)


    # # audio_a_spectrum = plt.magnitude_spectrum(audios[0][0,:], Fs=48000)[0]

    # plt.figure()
    # for i in range(len(output_spectrum)):
    #     plt.plot(output_spectrum[i])
    # plt.show()

    # plt.figure(1)
    # fig = plt.subplot(211)
    # for i in range(len(outputs)):
    #     pxx, freqs, bins, im = fig.specgram(outputs[i][0][:,0], Fs = 48000)
    #     fig.set_xlabel('Time')
    #     fig.set_ylabel('Frequency')

    # plt.show()

    # print("")






