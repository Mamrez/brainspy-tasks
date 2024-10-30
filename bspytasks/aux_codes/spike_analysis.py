from this import s
import torch
import numpy as np
from scipy.optimize import curve_fit

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def ramping(start, stop, slope_points_no, plateau_length):
    ramp_up = np.linspace(start, stop, slope_points_no)

    plateau = np.linspace(stop, stop, plateau_length)

    ramp_down = np.linspace(stop, start, slope_points_no)

    return np.concatenate((ramp_up, plateau, ramp_down))

def set_normal_control_voltages(inputs,
                        control_voltages,
                        slope_length,
                        cv_indeces,
                        meas_length
                        ):
    cv_iter = iter(control_voltages)
    for i in cv_indeces:
        inputs[i,:] = ramping(0, next(cv_iter), slope_length, meas_length - 2* slope_length)
    return inputs

def control_spikes(
    inputs,
    control_indeces,
    intervals = [2, 2, 2, 2],
    magnitudes = [0.7, 0.7, 0.7, 0.7],
    meas_length = 5000
    ):
    for i in range(4):
        for j in range(meas_length):
            if (j) % intervals[i] == 0:
                if (j >= 20) and (j <= (meas_length - 20)):
                    inputs[control_indeces[i], j] = magnitudes[i]

    return inputs

def input_spikes(inputs,
                    input_indeces,
                    intervals,
                    magnitude,
                    slope_length,
                    meas_length
                    ):

    each_spike_logic_length = (meas_length - 2 * slope_length) // 4

    for i in range(each_spike_logic_length): # (0, 0)
        if i%intervals[0] == 0:
            inputs[input_indeces[0], i + slope_length] = magnitude
            inputs[input_indeces[1], i + slope_length] = magnitude 

    for i in range(each_spike_logic_length): # (0, 1)
        if i%intervals[0] == 0:
            inputs[input_indeces[0], i + slope_length + each_spike_logic_length] = magnitude
        if i%intervals[1] == 0:
            inputs[input_indeces[1], i + slope_length + each_spike_logic_length + 1] = magnitude 

    for i in range(each_spike_logic_length): # (1, 0)
        if i%intervals[1] == 0:
            inputs[input_indeces[0], i + slope_length + 2 * each_spike_logic_length] = magnitude
        if i%intervals[0] == 0:
            inputs[input_indeces[1], i + slope_length + 2 * each_spike_logic_length + 1] = magnitude

    for i in range(each_spike_logic_length): # (1, 1)
        if i%intervals[1] == 0:
            inputs[input_indeces[0], i + slope_length + 3 * each_spike_logic_length] = magnitude
            inputs[input_indeces[1], i + slope_length + 3 * each_spike_logic_length] = magnitude

    return inputs
    

def place_normal_inputs(inputs,
                        indeces,
                        meas_length,
                        slope_length
                        ):

    tmp = meas_length // 4

    inputs[indeces[0], 0 : tmp] = ramping(0, -0.7, slope_length, tmp - 2 * slope_length)
    inputs[indeces[1], 0 : tmp] = ramping(0, -0.7, slope_length, tmp - 2 * slope_length)

    inputs[indeces[0], tmp : 2 * tmp] = ramping(0, -0.7, slope_length, tmp - 2 * slope_length)
    inputs[indeces[1], tmp : 2 * tmp] = ramping(0, 0.7, slope_length, tmp - 2 * slope_length)

    inputs[indeces[0], 2 * tmp : 3 * tmp] = ramping(0, 0.7, slope_length, tmp - 2 * slope_length)
    inputs[indeces[1], 2 * tmp : 3 * tmp] = ramping(0, -0.7, slope_length, tmp - 2 * slope_length)

    inputs[indeces[0], 3 * tmp : 4 * tmp] = ramping(0, 0.7, slope_length, tmp - 2 * slope_length)
    inputs[indeces[1], 3 * tmp : 4 * tmp] = ramping(0, 0.7, slope_length, tmp - 2 * slope_length)
  
    return inputs

def f(x, m, q):
    return m*x + q

def exp(x, tau, A, t0, bias):
    ex = np.exp(-(x-t0)/tau)
    return bias + (A * (1 - ex))

def recurrent_memory_expriment():
    configs = load_configs('configs/defaults/processors/hw_recurrent_memory_experiment.yaml')
    T = 0.4
    input_to_device = np.zeros((7, int(T * 25000)))

    # width = 200
    # distance = 40

    # one_pulse = np.zeros((width + distance))
    # one_pulse[width:2*width] = 0.8

    # for i in range(20):
    #     if i % 9 == 0:
    #         input_to_device[3, i*(width+distance) : (i+1)*(width+distance)] = one_pulse


    # input_to_device[3, 10:20] = .75
    # input_to_device[3, 25:35] = .75
    # input_to_device[3, 40:50] = .75

    input_to_device[3, 200:1000] = .8
    input_to_device[3, 4000:5000] = .8
    input_to_device[3, 5100:6100] = .8
    # input_to_device[3, 70*25:90*25] = .0
    # input_to_device[3, 90*25:110*25] = .8
    # input_to_device[3, 110*25:130*25] = .0
    # input_to_device[3, 130*25:150*25] = .8
    # input_to_device[3, 150*25:170*25] = .0
    # input_to_device[3, 170*25:190*25] = .8
    # input_to_device[3, 190*25:210*25] = .0
    
    driver = get_driver(configs["driver"])
    output = driver.forward_numpy(input_to_device.T)
    driver.close_tasks()

    plt.figure()
    plt.plot(output/np.max(output))
    plt.plot(np.repeat(input_to_device[3]/np.max(input_to_device[3]), 1))
    plt.show()

    # remove first 10 point to improve fit

    cut = 20

    t_croped = np.linspace(cut/25000, 1000/25000, 1000-cut)
    croped_output_1 = output[4000+cut:5000,0]
    croped_output_2 = output[5100+cut:6100,0]

    # log_croped_output_1 = np.log(1 - croped_output_1[cut:])
    # log_croped_output_2 = np.log(1 - croped_output_2[cut:])

    par_1, cov_1 = curve_fit(exp, t_croped, croped_output_1)
    par_2, cov_2 = curve_fit(exp, t_croped, croped_output_2, p0 = par_1)

    # tau_1 = 1/par_1
    # tau_2 = 1/par_2

    # error_tau_1 = (tau_1**2) * np.sqrt(cov_1.diagonal())
    # error_tau_2 = (tau_2**2) * np.sqrt(cov_2.diagonal())

    plt.figure()
    plt.plot(t_croped, croped_output_1)
    plt.plot(t_croped, exp(t_croped, *par_1))

    # plt.figure(3)
    # plt.plot(t_croped, log_croped_output_1)
    # plt.plot(t_croped, f(t_croped, *par_1))

    # plt.figure(4)
    # plt.plot(t_croped, log_croped_output_2)
    # plt.plot(t_croped, f(t_croped, *par_2))

    plt.figure()
    plt.plot(t_croped, croped_output_2)
    plt.plot(t_croped, exp(t_croped, *par_2))

    pass

def exp_discharge(x, tau, A, bias):
    return bias + (A * np.exp(-x/tau))

def time_constant_fits(input):

    t = np.linspace(0, 750/25000, 500)

    n = 9
    # exp = exp_discharge
    # tau, A, t0, bias
    # par_init = np.array([0.020, 0.025, 0, 0])
    par_init = np.array([0.020, 0.04, 0.04])
    par, cov = curve_fit(exp_discharge, t, input[250 + n*500 : 750 + n*500, 0] , p0 = par_init)
    plt.plot(t, input[250 + n*500 : 750 + n*500, 0])
    plt.plot(t, exp_discharge(t, *par))
    print(par)


    plt.show()

    pass


if __name__ == '__main__':
    from brainspy.utils.io import load_configs
    from brainspy.utils.manager import get_driver
    import matplotlib.pyplot as plt

    time_constant_fits(input = np.load("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/recurrent_time_constants_1.npy"))

    recurrent_memory_expriment()
    
    configs = load_configs('configs/defaults/processors/hw.yaml')

    XOR_1 = [ 0.2763, -0.3771,  0.3211,  0.4243] # XOR 1 

    cv_indeces = [0, 2, 4, 5]

    meas_length = 20000
    slope_length = 2000
    inputs = np.zeros((6,meas_length))

    # inputs = set_normal_control_voltages(inputs, XOR_1, slope_length=slope_length,
    #                              cv_indeces=cv_indeces, meas_length=meas_length)
                    
    inputs = place_normal_inputs(inputs, [1, 3], meas_length, slope_length)
    
    # inputs = control_spikes(inputs,
    #                         cv_indeces,
    #                         [40, 40, 40, 40], # -> XOR_2
    #                         [0.7, -0.7, 0.7, 0.7], # -> XOR_2
    #                         meas_length
    #                         )

    inputs = input_spikes(inputs, [1,3], 
                            intervals=[8,2], 
                            magnitude=0.7, 
                            slope_length=slope_length, 
                            meas_length=meas_length)
        

    driver = get_driver(configs["driver"])
    output = driver.forward_numpy(inputs.T)
    driver.close_tasks()

    if configs['driver']['instruments_setup']['average_io_point_difference'] == True:
        plt.plot(output[slope_length:meas_length-slope_length])
    else:
        plt.plot(output)
    plt.show()

    print("hi")
