import os
import numpy as np
import matplotlib.pyplot as plt
from brainspy.utils.io import load_configs

from brainspy.utils.manager import get_driver

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

configs = load_configs('configs/utils/brains_ivcurve_template.yaml')
n_pts = 2000
perturbated_electrode_index = 1
device = 'A'
pos_voltage = configs['driver']['instruments_setup'][device]['activation_voltage_ranges'][perturbated_electrode_index][1]
neg_voltage = configs['driver']['instruments_setup'][device]['activation_voltage_ranges'][perturbated_electrode_index][0]

# pos_voltage = .2
# neg_voltage = -.5

def iv(pos_voltage, neg_voltage):  # [-0.75, 0.4],
    ramp_up = np.linspace(0, pos_voltage, (n_pts-200)//4)
    ramp_down = np.linspace(pos_voltage, neg_voltage, (n_pts-200)//2)
    ramp_up_2 = np.linspace(neg_voltage, 0, (n_pts-200)//4)
    return np.concatenate((ramp_up, ramp_down, ramp_up_2))


def ramping(bias_value, slope_points_no, meas_length):
    ramp_up = np.linspace(0, bias_value, slope_points_no)

    plateau = np.linspace(bias_value, bias_value, meas_length - 2* slope_points_no)

    ramp_down = np.linspace(bias_value, 0, slope_points_no)

    return np.concatenate((ramp_up, plateau, ramp_down))

def time_constant_meas(configs, 
                        meas_electrode=1,
                        plateau_voltage=-0.05,
                        slope_points_no=10,
                        meas_length=50):
    inputs = np.zeros((7, meas_length))

    inputs[meas_electrode, :] = ramping(plateau_voltage, slope_points_no, meas_length)

    driver = get_driver(configs["driver"])
    output = driver.forward_numpy(inputs.T)
    driver.close_tasks()

    plt.plot(output)
    plt.plot(inputs.T[:, meas_electrode])
    plt.xlabel('points')
    plt.ylabel('V [V]')
    plt.show()
    # np.save('output.npz', output)

def NDR_finder(configs):
    
    inputs = np.zeros((7, n_pts))

    # bias_dict = np.array([0.2,0.2,0.2,0.2,0,0,0])
    bias_dict = np.zeros(7)
    
    for j in range(7):
        bias_dict[j] = np.random.uniform(configs['driver']['instruments_setup'][device]['activation_voltage_ranges'][j][0], configs['driver']['instruments_setup'][device]['activation_voltage_ranges'][j][1])
        #  bias_dict[j] = np.random.uniform(-1.0, 1.0)
       
    # print(bias_dict)

    for i in range(7):
        # inputs[i,:] = ramping(bias_dict[i])
        if i != perturbated_electrode_index:
            inputs[i, :] = ramping(bias_dict[i], slope_points_no=50, meas_length=n_pts)


    inputs[perturbated_electrode_index, 100:(n_pts-100)] = iv(
                                                                pos_voltage= pos_voltage,
                                                                neg_voltage= neg_voltage
    )

    driver = get_driver(configs["driver"])
    output = driver.forward_numpy(inputs.T)
    driver.close_tasks()

    plt.plot(output)
    plt.plot(inputs[perturbated_electrode_index, 100:(n_pts-100)], output[100:(n_pts-100)]) #100:4900 inputs[perturbated_electrode_index, :], 
    plt.xlabel('V [V]')
    plt.ylabel('I [nA]')
    plt.show()

    # np.savetxt('Backgating/IV_elec_0_bias_'+str(bias_dict[6])+'V.txt', output[100:(n_pts-100)])
    # np.savetxt('Backgating/x.txt', inputs[perturbated_electrode_index][100:(n_pts-100)])

    print("")

def spike_gen(spike_mag, 
            spike_no, meas_length,
            interval=2):
    tmp = np.zeros((1, meas_length))
    for i in range(spike_no):
        # if i%2==1:
        for j in range(1000):
            tmp[0, j+1+i*interval] = spike_mag
       
    return tmp

def spike_pattern(configs, 
                        meas_electrode=4,
                        meas_length=2000):

    inputs = np.zeros((7, meas_length))

    inputs[meas_electrode, :] = spike_gen(spike_mag=0.5, spike_no=1, meas_length=meas_length, interval=1)

    driver = get_driver(configs["driver"])
    output = driver.forward_numpy(inputs.T)
    driver.close_tasks()

    plt.plot(output)
    # plt.plot(inputs.T[:, meas_electrode])
    plt.xlabel('points')
    plt.ylabel('V [V]')
    plt.show()
    # np.save('output.npz', output)

if __name__ == '__main__':

    configs = load_configs(
        'configs/defaults/processors/hw_freq_analysis.yaml'
    )

    # time_constant_meas(configs)
    # spike_pattern(configs)
    NDR_finder(configs)
