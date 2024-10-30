import numpy as np
import yaml
import pickle
from yaml.loader import SafeLoader
import matplotlib.pyplot as plt
from brainspy.utils.manager import get_driver
from brainspy.utils.io import create_directory, create_directory_timestamp
import os 
from datetime import datetime
import gd_inputs
import gd



def mask_separate_task (output: np.array, configs):

    # Returns the output signal only for the boolean task segments. To be used when the inputs for the task and the perturbation are separated,
    # i.e. when the input_task_boolean() and input_perturbation() functions from gd_inputs.py are used.

    # Number of cases to be considered (4 for boolean gates)
    n_cases = int(len(configs["target"]))

    # Number of points per case
    n_seg = int(configs["segment_time"] * configs['driver']['instruments_setup']['activation_sampling_frequency'])

    # mask contains the indeces of the points to be kept (ramping up and down is removed)
    mask = np.zeros(n_seg * n_cases)

    for i in range(n_cases):
        start = (i+1)*(configs['ramping_points']+configs['waiting_points']) + i*n_seg
        stop = (i+1)*(configs['ramping_points']+configs['waiting_points']+n_seg)
        mask[i*n_seg : (i+1)*n_seg] = np.arange(start, stop)
     
    mask = mask.astype(int)
    return output[mask]



def mask_separate_pert (output: np.array, configs):

    # Returns the output signal only for the inputs perturbation part. To be used when the inputs for the task and the perturbation are separated,
    # i.e. when the input_task_boolean() and input_perturbation() functions from gd_inputs.py are used.

    # Number of cases to be considered (4 for boolean gates)
    n_cases = int(len(configs["target"]))

    # Number of points for the sinusoidal perturbation
    n_pert = int(configs['driver']['instruments_setup']['activation_sampling_frequency'] * (configs["perturbation_time"]))

    # Number of points for per case
    n_seg = int(configs["segment_time"] * configs['driver']['instruments_setup']['activation_sampling_frequency'])
    start = (n_cases * n_seg) + (n_cases+1) * (configs['ramping_points'] + configs['waiting_points'])
    stop = start + n_pert

    # mask contains the indeces of the points to be kept (ramping up and down is removed)
    mask = (np.arange(start, stop)).astype(int)
    
    return output[mask]



def mask_combined (output: np.array, configs):

    # Returns the output signal for the boolean task segments. To be used when the inputs for the task and the perturbation are combined
    # i.e. when the function input_combined() from gd_inputs.py is used.

    # Number of cases to be considered (4 for boolean gates)
    n_cases = int(len(configs["target"]))

    # Number of points for per case
    n_seg = int(configs["segment_time"] * configs['driver']['instruments_setup']['activation_sampling_frequency'])

    # mask contains the indeces of the points to be kept (ramping up and down is removed)
    mask = np.zeros(n_seg * n_cases)
    for i in range(n_cases):
        start = (i+1)*(configs['ramping_points']+configs['waiting_points']) + i*n_seg
        stop = (i+1)*(configs['ramping_points']+configs['waiting_points']+n_seg)
        mask[i*n_seg : (i+1)*n_seg] = np.arange(start, stop)
     
    mask = mask.astype(int)
    return output[mask]



def mask_sequential_task (output: np.array, configs):

    # Returns the output signal only for the boolean task segments. To be used when the inputs for the task and the perturbation are sequential
    # i.e. when the function input_sequential() from gd_inputs.py is used.

    # Number of cases to be considered (4 for boolean gates)
    n_cases = int(len(configs["target"]))

    # Number of points for per case
    n_seg = int(configs["segment_time"] * configs['driver']['instruments_setup']['activation_sampling_frequency'])

    # Number of points for the sinusoidal perturbation
    n_pert = int(configs['driver']['instruments_setup']['activation_sampling_frequency'] * (configs["perturbation_time"]))

    # mask contains the indeces of the points to be kept (ramping up and down is removed)
    mask = np.zeros(n_seg * n_cases)
    for i in range(n_cases):
        start = (i+1) * (configs['ramping_points']+configs['waiting_points']) + i * (n_pert + n_seg)
        stop = start + n_seg
        mask[i*n_seg : (i+1)*n_seg] = (np.arange(start, stop))
    
    mask = mask.astype(int)
    return output[mask]



def mask_sequential_pert (output: np.array, configs):

    #Returns the output signal only for the perturbation part of the measurement. To be used when the inputs for the task and the perturbation are sequential
    # i.e. when the function input_sequential() from gd_inputs.py is used.

    # Number of cases to be considered (4 for boolean gates)
    n_cases = int(len(configs["target"]))

    # Number of points for per case
    n_seg = int(configs["segment_time"] * configs['driver']['instruments_setup']['activation_sampling_frequency'])

    # Number of points for the sinusoidal perturbation
    n_pert = int(configs['driver']['instruments_setup']['activation_sampling_frequency'] * (configs["perturbation_time"]))

    # mask contains the indeces of the points to be kept (ramping up and down is removed)
    mask = np.zeros(n_pert * n_cases)
    for i in range(n_cases):
        start = (i+1) * (configs['ramping_points']+configs['waiting_points']+n_seg) + i * (n_pert)
        stop = start + n_pert
        mask[i*n_pert : (i+1)*n_pert] = (np.arange(start, stop))
    
    mask = mask.astype(int)
    return output[mask]




def mask_sequential_inverted_task (output: np.array, configs):

    # Returns the output signal only for the boolean task segments. To be used when the inputs for the task and the perturbation are sequential but inverted
    # i.e. when the function input_sequential_inverted() from gd_inputs.py is used.

    # # Number of cases to be considered (4 for boolean gates)
    n_cases = int(len(configs["target"]))

    # Number of points for per case
    n_seg = int(configs["segment_time"] * configs['driver']['instruments_setup']['activation_sampling_frequency'])

    # Number of points for the sinusoidal perturbation
    n_pert = int(configs['driver']['instruments_setup']['activation_sampling_frequency'] * (configs["perturbation_time"]))

    # mask contains the indeces of the points to be kept (ramping up and down is removed)
    mask = np.zeros(n_seg * n_cases)
    for i in range(n_cases):
        start = (2*(i+1)) * (configs['ramping_points']+configs['waiting_points']) + (i+1) * n_pert + i * (n_seg)
        stop = start + n_seg
        mask[i*n_seg : (i+1)*n_seg] = (np.arange(start, stop))
    
    mask = mask.astype(int)
    return output[mask]



def mask_sequential_inverted_pert (output: np.array, configs):

    # Returns the output signal only for the perturbation part of the measurement. To be used when the inputs for the task and the perturbation are sequential but inverted,
    # i.e. when the function input_sequential_inverted() from gd_inputs.py is used.

    # Number of cases to be considered (4 for boolean gates)
    n_cases = int(len(configs["target"]))

    # Number of points for per case
    n_seg = int(configs["segment_time"] * configs['driver']['instruments_setup']['activation_sampling_frequency'])

    # Number of points for the sinusoidal perturbation
    n_pert = int(configs['driver']['instruments_setup']['activation_sampling_frequency'] * (configs["perturbation_time"]))

    # mask contains the indeces of the points to be kept (ramping up and down is removed)
    mask = np.zeros(n_pert * n_cases)
    for i in range(n_cases):
        start = (2*i+1) * (configs['ramping_points']+configs['waiting_points']) + i * (n_pert + n_seg)
        stop = start + n_pert
        mask[i*n_pert : (i+1)*n_pert] = (np.arange(start, stop))
    
    mask = mask.astype(int)
    return output[mask]





def optimizer(configs, driver):

    # Evolves the control voltages, minimizing the error function.

    evolution = {}

    n_it = configs['max_iterations']
    n_elec = len(configs['driver']['instruments_setup']['activation_voltage_ranges'])

    # Number of control electrodes.
    n_ctrl = n_elec - len(configs['inputs'])
    a = (np.arange(0, n_elec)).astype(int)

    #Indices of control electrodes.
    ctrl = np.delete(a, configs['inputs'])      
    
    # controls contains the value of the control voltages applied for every iteration.
    controls = np.zeros((n_it+1, n_elec))

    # Random initialization using the function initialize().
    controls[0] = initialize(configs)

    # grad stores the value of the gradient dE/dV for each control and for each iteration.
    grad = np.zeros((n_it, n_elec))

    # loss contains the value of the loss function for every iteration.
    loss = np.zeros(n_it)

    # coor contains the value of the correlation for every iteration.
    corr = np.zeros(n_it)

    # In data the raw output of the device is stored.
    data = np.zeros((n_it, int(len(configs['target']) * (configs['driver']['instruments_setup']['activation_sampling_frequency'] * configs['segment_time'])))) 

    flag = 0

    # Parameter for the adam optimizer.
    m = np.zeros(n_ctrl)
    v = np.zeros(n_ctrl)
    beta1_t = 1
    beta2_t = 1

    # Adapting the target to the lenght of the output signal (if necessary).
    if (configs['average_per_segment'] == False):
        targ = gd.long_target(configs) 

    else:
        if (configs['average_per_segment'] == True):
            targ = np.array(configs['target'])

    for i in range (n_it):

        print("Iteration N. {}".format(i+1))
        flag = i


        # Creating the input signal with one of the functions defined in gd_inputs.py and masking the output with the corresponding mask function.
        if (configs['input_shape'] == 'separated'):

            input = np.concatenate((gd_inputs.input_task_boolean(controls[i], configs), gd_inputs.input_perturbation(controls[i], configs)))
            output = driver.forward_numpy(input.T)
            task = mask_separate_task(output, configs)[:,0]
            pert = mask_separate_pert(output, configs)[:,0]
            data[i] = task
            check_clip = gd.average_I(targ, task, configs)

            if (configs['average_per_segment'] == True):
                task = check_clip     

            grad[i] = gd.dE_dV_separeted_inputs(task, targ, pert, configs)

        else:
            if(configs['input_shape'] == 'combined'):

                input = gd_inputs.input_combined(controls[i], configs)
                output = driver.forward_numpy(input.T)
                task = mask_combined(output, configs)[:,0]
                pert = mask_combined(output, configs)[:,0]
                data[i] = task
                check_clip = gd.average_I(targ, task, configs)

                if (configs['average_per_segment'] == True):
                    task = check_clip
                    
            else:
                if(configs['input_shape'] == 'sequential'):

                    input = gd_inputs.input_sequential(controls[i], configs)
                    output = driver.forward_numpy(input.T)
                    task = mask_sequential_task(output, configs)[:,0]
                    pert = mask_sequential_pert(output, configs)[:,0]
                    data[i] = task
                    check_clip = gd.average_I(targ, task, configs)

                    if (configs['average_per_segment'] == True):
                        task = check_clip

                else:
                    if(configs['input_shape'] == 'sequential_inverted'):

                        input = gd_inputs.input_sequential_inverted(controls[i], configs)
                        output = driver.forward_numpy(input.T)
                        task = mask_sequential_inverted_task(output, configs)[:,0]
                        pert = mask_sequential_inverted_pert(output, configs)[:,0]
                        data[i] = task
                        check_clip = gd.average_I(targ, task, configs)

                        if (configs['average_per_segment'] == True):
                            task = check_clip

                    else:

                        print('Specify input shape.')
                        break

        # Computing the gradient of the loss function with respect to the control voltages.    
        grad[i] = gd.dE_dV_mixed(task, targ, pert, configs)

        # Calculating the loss function and the correlation.
        E = gd.loss_function(task, targ, configs)
        loss[i] = E
        print("Loss value = {}".format(E))
        corr[i] = gd.correlation(task, targ)

        # If the loss is below the treshold, the evolution stops.
        if (E < configs['loss_threshold']):
            print('')
            break      

        if (i != n_it-1):

            # Reinizialize controls if the output is clipping.
            if (configs['prevent_clipping'] == True):  
                if any(abs(k) > configs['clipping_threshold'] for k in check_clip):
                    controls[i+1, ctrl] = np.random.rand(n_ctrl) * (np.array(configs['driver']['instruments_setup']['activation_voltage_ranges'])[ctrl][:,1] - np.array(configs['driver']['instruments_setup']['activation_voltage_ranges'])[ctrl][:,0]) * 0.9 + np.array(configs['driver']['instruments_setup']['activation_voltage_ranges'])[ctrl][:,0] + np.array(configs['perturbation_amplitudes'])[ctrl]
                    print("Reinitialize.")
                    if (configs['optimizer'] == 'adam'):
                        m = np.zeros(n_ctrl)
                        v = np.zeros(n_ctrl)
                        beta1_t = 1
                        beta2_t = 1


                else:

                    # Updating controls values.
                    if (configs['optimizer'] == 'standard'):
                        controls[i+1][ctrl] = controls[i][ctrl] - configs['alpha'] * grad[i][ctrl]

                    else:
                        if (configs['optimizer'] == 'adam'):

                                m = configs['beta1'] * m + (1 - configs['beta1']) * grad[i][ctrl]
                                v = configs['beta2'] * v + (1 - configs['beta2']) * (np.square(grad[i][ctrl]))
                                beta1_t = beta1_t * configs['beta1']
                                beta2_t = beta2_t * configs['beta2']
                                alpha_t  = configs['alpha'] * np.sqrt(1 - beta2_t)/np.sqrt(1 - beta1_t)
                                m1 = alpha_t * m
                                v1 = (np.sqrt(v) + np.ones(len(ctrl)) *configs['epsilon'])
                                steps = np.divide(m1, v1)
                                controls[i+1][ctrl] = controls[i][ctrl] - steps

                        else:

                            print('Specify optimizer.')
                            break

                    for j in ctrl:      #Check for staying in the specified voltage ranges 

                        controls[i+1][j] = min(configs['driver']['instruments_setup']['activation_voltage_ranges'][j][1] - configs['perturbation_amplitudes'][j], controls[i+1][j])
                        controls[i+1][j] = max(configs['driver']['instruments_setup']['activation_voltage_ranges'][j][0] + configs['perturbation_amplitudes'][j], controls[i+1][j])

        else:

            controls[i+1] = controls[i]

    driver.close_tasks()


    evolution['controls'] = controls[:flag+1]
    evolution['loss_function'] = loss[:flag+1]
    evolution['gradient'] = grad[:flag+1]
    evolution['correlation'] = corr[:flag+1]
    evolution['data'] = data[:flag+1]
    evolution['ctrl'] = ctrl

    return evolution


def learn_rate_decay_step(iter, configs):

    a_0 = configs['alpha']
    step = 0.99 * configs['alpha']/configs['max_iterations']
    return a_0 - step*iter


def learn_rate_decay_cos(iter, configs):
    max_a = configs['alpha']
    min_a = 0.01 * configs['alpha']
    period = 0.05 * configs['max_iterations']
    return min_a + 0.5 * (max_a - min_a) * (1 + np.cos(np.pi * (iter/period)))



def accuracy(evolution, configs):

    #Computes the accuracy for the last iteration

    x = evolution['data'][-1] - np.mean(evolution['data'][-1])
    norm_data = x/(max(abs(x)))
    targ = gd.long_target(configs)
    acc  = 0

    for i in range(len(targ)):

        if (norm_data[i] >= 0):
            acc += targ[i]
        else:
            acc += (1 - targ[i])

    print('Accuracy is {}/1'.format(acc/len(targ)) )
    
    return acc/len(targ)



def accuracy_2 (evolution, configs):

    targ = gd.long_target(configs)
    high = np.where(targ == 1)
    low = np.where(targ == 0)
    h_m = np.mean(evolution['data'][-1][high])
    l_m = np.mean(evolution['data'][-1][low])
    x = evolution['data'][-1] - 0.5*(h_m + l_m)
    norm_data = x/(max(abs(x)))
    acc  = 0

    for i in range(len(targ)):

        if (norm_data[i] >= 0):
            acc += targ[i]
        else:
            acc += (1 - targ[i])

    print('Accuracy is {}/1'.format(acc/len(targ)) )
    
    return acc/len(targ)





def initialize(configs):

    #Initialize control voltages to random values

    n_elec = len(configs['driver']['instruments_setup']['activation_voltage_ranges'])
    n_ctrl = n_elec - len(configs['inputs'])
    a = (np.arange(0, n_elec)).astype(int)
    ctrl = np.delete(a, configs['inputs'])
    controls_0 = np.zeros(n_elec)
    controls_0[ctrl] = np.random.rand(n_ctrl) * (np.array(configs['driver']['instruments_setup']['activation_voltage_ranges'])[ctrl][:,1] - np.array(configs['driver']['instruments_setup']['activation_voltage_ranges'])[ctrl][:,0])*0.95 + np.array(configs['driver']['instruments_setup']['activation_voltage_ranges'])[ctrl][:,0] + np.array(configs['perturbation_amplitudes'])[ctrl]

    return controls_0




def plot_evolution(evolution, configs, path):

    #Plots the results

    plt.figure(1)
    plt.plot(evolution['loss_function'])
    plt.ylabel('Loss function')
    plt.xlabel('N. iterations')
    plt.savefig(path+'/Loss function')

    targ = gd.long_target(configs)
    t = np.linspace(0, len(evolution['data'][-1]), 1000)
    high = np.where(targ == 1)
    low = np.where(targ == 0)
    h_m = np.mean(evolution['data'][-1][high])
    l_m = np.mean(evolution['data'][-1][low])
    m = 0.5*(h_m + l_m) * np.ones(1000)
    plt.figure(2)
    plt.plot(evolution['data'][-1])
    plt.plot(t, m, linestyle = '--', color ='red')
    plt.ylabel('Current [nA]')
    plt.savefig(path+'/Output')

    x = evolution['data'][-1] - 0.5*(h_m + l_m)
    norm_data = x/(max(abs(x)))
    targ = gd.long_target(configs)
    thr = np.zeros(1000)
    n = np.arange(len(x))
    plt.figure(3)
    plt.plot(n, targ, color = 'green', label = 'Target')
    plt.errorbar(n, norm_data, label = 'Data')
    plt.plot(t, thr, linestyle = '--', color ='red')
    plt.legend(loc = 'upper right')
    plt.ylabel('Normalized current')
    plt.savefig(path+'/Normalized output')

    plt.figure(4)
    plt.plot(evolution['correlation'])
    plt.ylabel('Correlation')
    plt.xlabel('N. iterations')
    plt.savefig(path+'/Correlation')

    grad = np.transpose(evolution['gradient'])
    controls = np.transpose(evolution['controls'])
    plt.figure(5)
    for i in evolution['ctrl']:
        plt.figure(5)
        plt.plot(grad[i], label = 'Elec '+str(i))
        

        plt.figure(6)
        plt.plot(controls[i], label = 'Elec '+str(i))
    
    plt.figure(5)
    plt.legend(loc = 'upper right')
    plt.ylabel('dE/dV [V$^{-1}$]')
    plt.xlabel('N. interation')
    plt.savefig(path+'/Gradient')

    plt.figure(6)
    plt.legend(loc = 'upper right')
    plt.ylabel('V [V]')
    plt.xlabel('N. interation')
    plt.savefig(path+'/Controls')



    return



if __name__ == "__main__":

    with open("gd_configs.yml") as file:
        configs = yaml.load(file, Loader=SafeLoader)

    driver = get_driver(configs["driver"])
    evolution = optimizer(configs, driver)
    acc = accuracy_2(evolution, configs)

    path = 'C:/Users/Lorenzo/Documents/github/brainspy-tasks/GD_boolean/'+configs['optimizer']+'/Sigmoid_scale_'+str(configs['sigmoid_scale'])+'/Learning_rate'+str(configs['alpha'])+'/'+str(configs['target'])+'_'+str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(path)

    with open(path+'/configs.yaml', 'w') as conf_file:
        yaml.dump(configs, conf_file)
    
    with open(path + '/data.pickle', 'wb') as outfile:
        pickle.dump(evolution, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    plot_evolution(evolution, configs, path)
    plt.show()

 

        
















# if __name__ == "__main__":

#     with open("gd_configs.yml") as file:
#         configs = yaml.load(file, Loader=SafeLoader)

#     controls = np.array([0.4186033, -0.33180682,  0.69622607, -0.43978884,  1.60150471, -0.41809101,0.35081436])

#     # inputs_1  = gd_inputs.input_task_boolean(controls, configs)
#     # inputs_2 = gd_inputs.input_perturbation(controls, configs)
#     # input = np.zeros((len(inputs_1), (len(inputs_1[0]) + len(inputs_2[0]))))
#     input = gd_inputs.input_sequential(controls, configs)
#     for i in range(len(input)):
#         #input[i]  = np.concatenate((inputs_1[i], inputs_2[i]))
#         plt.figure(1)
#         plt.plot(input[i])
#     #input = gd_inputs.input_combined(controls, configs)

#     driver = get_driver(configs["driver"])
#     output = driver.forward_numpy(input.T)
#     driver.close_tasks()

#     plt.figure(4)
#     plt.plot(output)

#     task = mask_sequential_task(output, configs)
#     pert = mask_sequential_pert(output, configs)
#     plt.figure(2)
#     plt.plot(task)
#     plt.figure(3)
#     plt.plot(pert)

#     # output = mask_combined(output, configs)
#     # plt.figure(2)
#     # plt.plot(output)

#     plt.show()


    
