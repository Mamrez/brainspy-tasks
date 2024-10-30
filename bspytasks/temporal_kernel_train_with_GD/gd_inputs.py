import numpy as np
import yaml
from yaml.loader import SafeLoader
import matplotlib.pyplot as plt



def ramp(low, high, configs):

    #Defines the way to ramp the voltages, introducing a waiting time when the desired voltage is reached.

    ramp = np.linspace(low, high, configs["ramping_points"])
    wait = np.ones(configs["waiting_points"]) * high
    return np.concatenate((ramp, wait))



def input_task_boolean (controls: np.array, configs):

    #Returns the input signal for the boolean task only (no controls perturbations), must be used together with input_perturbation

    n_elecs = int(len(controls))
    fs = configs["driver"]["instruments_setup"]["activation_sampling_frequency"]
    # Number of cases to be considered (4 for boolean gates)
    n_cases = int(len(configs["target"]))
    # Number of points for the single case
    n_seg = int(configs["segment_time"] * fs)
    # Total number of points, comprehensive of ramping up and down
    n_pts = int(n_cases * (n_seg) + (n_cases + 1) * (configs["ramping_points"] + configs["waiting_points"]))
    inputs = np.zeros((n_elecs, n_pts))

    for i in range(n_elecs):

        # Creating input signal for the first input electrode
        if (i == np.asarray(configs["inputs"][0])) :

            # Defining up and down levels
            high = configs["driver"]["instruments_setup"]["activation_voltage_ranges"][i][1]
            low = configs["driver"]["instruments_setup"]["activation_voltage_ranges"][i][0]

            ramp_from_0 = ramp(0, low, configs)
            ramp_to_0 = ramp(high, 0, configs)
            ramp_up = ramp(low, high, configs)
            ramp_down = ramp(high, low, configs)
            one = np.ones(n_seg) * high
            zero = np.ones(n_seg) * low	
            
            inputs[i] = np.concatenate((ramp_from_0, zero, ramp_up, one, ramp_down, zero, ramp_up, one, ramp_to_0))

        else:
            # Creating input signal for the second input electrode
            if (i == np.asarray(configs["inputs"][1])) :

                # Defining up and down levels
                high = configs["driver"]["instruments_setup"]["activation_voltage_ranges"][i][1]
                low = configs["driver"]["instruments_setup"]["activation_voltage_ranges"][i][0]

                ramp_from_0 = ramp(0, low, configs)
                ramp_to_0 = ramp(high, 0, configs)
                ramp_up = ramp(low, high, configs)
                one = np.ones(2* n_seg + configs["ramping_points"] + configs["waiting_points"]) * high
                zero = np.ones(2* n_seg + configs["ramping_points"] + configs["waiting_points"]) * low	

                inputs[i] = np.concatenate((ramp_from_0, zero, ramp_up, one, ramp_to_0))

            else:
                
                # Creates the inputs for the control electrodes
                ramp_from_0 = ramp(0, controls[i], configs)
                hold = np.ones(len(inputs[i]) - len(ramp_from_0)) * controls[i]

                 #The control voltages do not return to 0, because the perturbations are still to be applied
                inputs[i] = np.concatenate((ramp_from_0, hold))  

    return inputs



def input_perturbation (input, controls: np.array, configs):

    #Returns the inputs for the controls perturbations, must be used with input_task_boolean

    n_elecs = int(len(controls)+1)
    fs = configs["driver"]["instruments_setup"]["activation_sampling_frequency"]
    # Number of points for the sinusoidal perturbation
    n_pert = int(fs * (configs["perturbation_time"]))
    # Total number of points
    n_pts = int(n_pert + 2 * (configs["ramping_points"] + configs["waiting_points"]))
    inputs = np.zeros((n_elecs, n_pts))   
    t = np.linspace(0, configs["perturbation_time"], n_pert) 
    
    kk = 0
    for i in range(n_elecs):

        if i not in configs["inputs"]:
            # the inputs DO start zero
            ramp_up = ramp(0, controls[i+kk], configs)
            pert  = controls[i+kk] + configs["perturbation_amplitudes"][i] * np.sin(2 * np.pi * configs["perturbation_frequencies"][i] * t) 
            ramp_to_0 = ramp(pert[-1], 0, configs)
            inputs[i] =  np.concatenate((ramp_up, pert, ramp_to_0))
        else:
            kk -= 1
            ramp_up = ramp(0, input, configs)
            plateau = np.linspace(input, input, n_pert)
            ramp_down = ramp(input, 0, configs)
            inputs[i] = np.concatenate((ramp_up, plateau, ramp_down))
    return inputs



def input_combined (controls: np.array, configs):

    # Returns the combined input signal for the boolean task and the controls perturbations. The perturbation of the controls and the measurements are performed at the same time for each segment.
    # In this way it is possible to speed up significantly the evolution process, but this comes to the cost of risking losing accuracy for the gradient evaluation. 

    n_elecs = int(len(controls))
    fs = configs["driver"]["instruments_setup"]["activation_sampling_frequency"]

    # Number of points for the sinusoidal perturbation
    n_cases = int(len(np.asarray(configs["target"])))

    # Number of points per case
    n_seg = int(configs["segment_time"] * fs)

    # Total number of points
    n_pts = int(n_cases * (n_seg) + (n_cases + 1) * (configs["ramping_points"] + configs["waiting_points"]))
    inputs = np.zeros((n_elecs, n_pts))
    t = np.linspace(0, configs["segment_time"], n_seg) 

    for i in range(n_elecs):

        # Creating input signal for the first input electrode
        if (i == np.asarray(configs["inputs"][0])) :

            # Defining up and down levels
            high = configs["driver"]["instruments_setup"]["activation_voltage_ranges"][i][1]
            low = configs["driver"]["instruments_setup"]["activation_voltage_ranges"][i][0]

            ramp_from_0 = ramp(0, low, configs)
            ramp_to_0 = ramp(high, 0, configs)
            ramp_up = ramp(low, high, configs)
            ramp_down = ramp(high, low, configs)
            one = np.ones(n_seg) * high
            zero = np.ones(n_seg) * low	

            inputs[i] = np.concatenate((ramp_from_0, zero, ramp_up, one, ramp_down, zero, ramp_up, one, ramp_to_0))

        else:

            # Creating input signal for the second input electrode
            if (i == np.asarray(configs["inputs"][1])):

                # Defining up and down levels
                high = configs["driver"]["instruments_setup"]["activation_voltage_ranges"][i][1]
                low = configs["driver"]["instruments_setup"]["activation_voltage_ranges"][i][0]

                ramp_from_0 = ramp(0, low, configs)
                ramp_to_0 = ramp(high, 0, configs)
                ramp_up = ramp(low, high, configs)
                one = np.ones(2* n_seg + configs["ramping_points"] + configs["waiting_points"]) * high
                zero = np.ones(2* n_seg + configs["ramping_points"] + configs["waiting_points"]) * low	

                inputs[i] = np.concatenate((ramp_from_0, zero, ramp_up, one, ramp_to_0))

            else:

                # Creates the inputs for the control electrodes
                ramp_from_0 = ramp(0, controls[i], configs)
                pert = controls[i] + configs["perturbation_amplitudes"][i] * np.sin(2 * np.pi * configs["perturbation_frequencies"][i] * t)
                ramp_to_control = ramp(pert[-1], controls[i], configs)
                ramp_to_0 = ramp(pert[-1], 0, configs)
                inputs[i] = np.concatenate((ramp_from_0, pert, ramp_to_control, pert, ramp_to_control, pert, ramp_to_control, pert, ramp_to_0))  

    return inputs




def input_sequential (controls: np.array, configs):

    # Returns the input signal for the boolean task and the controls perturbations. Every time the inputs voltages are changed the inputs are kept fixed for evaluate the loss function and then perturbed for extracting the gradient.
    

    n_elecs = int(len(controls))
    fs = configs["driver"]["instruments_setup"]["activation_sampling_frequency"]

    # Number of cases to be considered (4 for boolean gates)
    n_cases = int(len(np.asarray(configs["target"])))

    # Number of points per case
    n_seg = int(configs["segment_time"] * fs)

    # Number of points for each perturbation
    n_pert = int(configs["perturbation_time"] * fs)

    # Total number of points
    n_pts = int(n_cases * (n_seg + n_pert) + (n_cases + 1) * (configs["ramping_points"] + configs["waiting_points"]))
    inputs = np.zeros((n_elecs, n_pts))
    t = np.linspace(0, configs["perturbation_time"], n_pert) 

    for i in range(n_elecs):

        # Creating input signal for the first input electrode
        if (i == np.asarray(configs["inputs"][0])) :

            # Defining up and down levels
            high = configs["driver"]["instruments_setup"]["activation_voltage_ranges"][i][1]
            low = configs["driver"]["instruments_setup"]["activation_voltage_ranges"][i][0]

            ramp_from_0 = ramp(0, low, configs)
            ramp_to_0 = ramp(high, 0, configs)
            ramp_up = ramp(low, high, configs)
            ramp_down = ramp(high, low, configs)
            one = np.ones(n_seg + n_pert) * high
            zero = np.ones(n_seg + n_pert) * low	

            inputs[i] = np.concatenate((ramp_from_0, zero, ramp_up, one, ramp_down, zero, ramp_up, one, ramp_to_0))

        else:

            # Creating input signal for the second input electrode
            if (i == np.asarray(configs["inputs"][1])):

                # Defining up and down levels
                high = configs["driver"]["instruments_setup"]["activation_voltage_ranges"][i][1]
                low = configs["driver"]["instruments_setup"]["activation_voltage_ranges"][i][0]

                ramp_from_0 = ramp(0, low, configs)
                ramp_to_0 = ramp(high, 0, configs)
                ramp_up = ramp(low, high, configs)
                one = np.ones(2* (n_seg + n_pert) + configs["ramping_points"] + configs["waiting_points"]) * high
                zero = np.ones(2* (n_seg + n_pert) + configs["ramping_points"] + configs["waiting_points"]) * low	

                inputs[i] = np.concatenate((ramp_from_0, zero, ramp_up, one, ramp_to_0))

            else:

                # Creates the inputs for the control electrodes
                ramp_from_0 = ramp(0, controls[i], configs)
                pert = controls[i] + configs["perturbation_amplitudes"][i] * np.sin(2 * np.pi * configs["perturbation_frequencies"][i] * t)
                hold = np.ones(n_seg) * controls[i] 
                ramp_to_control = ramp(pert[-1], controls[i], configs)
                ramp_to_0 = ramp(pert[-1], 0, configs)
                meas = np.concatenate((hold, pert))
                inputs[i] = np.concatenate((ramp_from_0, meas, ramp_to_control, meas, ramp_to_control, meas, ramp_to_control, meas, ramp_to_0))  

    return inputs

def input_sequential_inverted (controls: np.array, configs):

    # Returns the input signal for the boolean task and the controls perturbations. Every time the inputs voltages are changed the inputs are perturbed for extracting the gradient and then kept fixed for evaluate the loss function.
    # This is the standard method for creating the input signal.

    n_elecs = int(len(controls))
    fs = configs["driver"]["instruments_setup"]["activation_sampling_frequency"]

    # Number of cases to be considered (4 for boolean gates)
    n_cases = int(len(np.asarray(configs["target"])))

    # Number of points per case
    n_seg = int(configs["segment_time"] * fs)

    # Number of points for each perturbation
    n_pert = int(configs["perturbation_time"] * fs)

    # Total number of points
    n_pts = int(n_cases * (n_seg + n_pert) + (2 * n_cases + 1) * (configs["ramping_points"] + configs["waiting_points"]))
    inputs = np.zeros((n_elecs, n_pts))
    t = np.linspace(0, configs["perturbation_time"], n_pert) 

    for i in range(n_elecs):

        # Creating input signal for the first input electrode
        if (i == np.asarray(configs["inputs"][0])) :

            # Defining up and down levels
            high = configs["driver"]["instruments_setup"]["activation_voltage_ranges"][i][1]
            low = configs["driver"]["instruments_setup"]["activation_voltage_ranges"][i][0]

            ramp_from_0 = ramp(0, low, configs)
            ramp_to_0 = ramp(high, 0, configs)
            ramp_up = ramp(low, high, configs)
            ramp_down = ramp(high, low, configs)
            one = np.ones(n_seg + n_pert + configs["ramping_points"] + configs["waiting_points"]) * high
            zero = np.ones(n_seg + n_pert + configs["ramping_points"] + configs["waiting_points"]) * low	

            inputs[i] = np.concatenate((ramp_from_0, zero, ramp_up, one, ramp_down, zero, ramp_up, one, ramp_to_0))

        else:

            # Creating input signal for the second input electrode
            if (i == np.asarray(configs["inputs"][1])):

                # Defining up and down levels
                high = configs["driver"]["instruments_setup"]["activation_voltage_ranges"][i][1]
                low = configs["driver"]["instruments_setup"]["activation_voltage_ranges"][i][0]

                ramp_from_0 = ramp(0, low, configs)
                ramp_to_0 = ramp(high, 0, configs)
                ramp_up = ramp(low, high, configs)
                one = np.ones((n_cases//2)* (n_seg + n_pert) + (n_cases//2 + 1) * (configs["ramping_points"] + configs["waiting_points"])) * high
                zero = np.ones((n_cases//2)* (n_seg + n_pert) + (n_cases//2 + 1) * (configs["ramping_points"] + configs["waiting_points"])) * low	

                inputs[i] = np.concatenate((ramp_from_0, zero, ramp_up, one, ramp_to_0))

            else:

                # Creates the inputs for the control electrodes
                ramp_from_0 = ramp(0, controls[i], configs)
                pert = controls[i] + configs["perturbation_amplitudes"][i] * np.sin(2 * np.pi * configs["perturbation_frequencies"][i] * t)
                hold = np.ones(n_seg) * controls[i] 
                ramp_to_control = ramp(pert[-1], controls[i], configs)
                ramp_to_0 = ramp(pert[-1], 0, configs)
                meas = np.concatenate((pert, ramp_to_control, hold))
                wait = np.ones(configs["ramping_points"] + configs["waiting_points"]) * controls[i] 
                inputs[i] = np.concatenate((ramp_from_0, meas, wait, meas, wait, meas, wait, meas, ramp_to_0))  

    return inputs







# This main can be used for testing the inputs generator
if __name__ == "__main__":

    with open("gd_configs.yml") as file:
        configs = yaml.load(file, Loader=SafeLoader)

    controls = np.array([0.4186033, -0.33180682,  0.69622607, -0.43978884,  1.60150471, -0.41809101,0.35081436])

    inputs_1  = input_task_boolean(controls, configs)
    inputs_2 = input_perturbation(controls, configs)
    input = np.zeros((len(inputs_1), (len(inputs_1[0]) + len(inputs_2[0]))))
    input_comb = input_combined(controls, configs)
    input_seq = input_sequential_inverted(controls, configs)
    for i in range(len(inputs_2)):
        input[i]  = np.concatenate((inputs_1[i], inputs_2[i]))
        plt.figure(1)
        plt.plot(input[i])
        plt.figure(2)
        plt.plot(input_comb[i])
        plt.figure(3)
        plt.plot(input_seq[i])
    print(input_comb.shape)
    print(input_seq.shape)
    plt.show()