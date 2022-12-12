from this import s
import torch
import numpy as np

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



if __name__ == '__main__':
    from brainspy.utils.io import load_configs
    from brainspy.utils.manager import get_driver
    import matplotlib.pyplot as plt

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
