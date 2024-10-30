import numpy as np
from scipy.signal import butter, lfilter
import yaml
from yaml.loader import SafeLoader


def correlation (x: np.array, y: np.array):

    #Returns Pearson correlation coefficient for two arrays

    assert x.shape == y.shape, 'Target and data have different dimensions.'

    vx = x - np.mean(x)
    vy = y - np.mean(y)
    sum_vx = np.sum(vx**2)
    sum_vy = np.sum(vy**2)
    sum_vxvy = np.sum(vx*vy)

    return sum_vxvy/(np.sqrt(sum_vx)*np.sqrt(sum_vy))



def sigmoid (x, configs):
    return 1/(1 + np.e**(-x/configs['sigmoid_scale']))



def sep (I: np.array, targ: np.array):

    # Defines the separation between the high and low states

    up  = np.where(targ == 1)
    down = np.where(targ == 0)
    return np.mean(I[up]) - np.mean(I[down])



def loss_function (I: np.array, targ: np.array, configs):

    #Defines the loss function to minimize (other loss functions could be implemented)

    rho  = correlation(I, targ)
    sig = sigmoid(sep(I, targ), configs)
    return (1 - rho)/sig


def long_target(configs):

    # Creates an array of 0s ore 1s for each point of the target, so that the output of the device and the target have the same lenght.
    # This function is meant for boolean gates.

    n = len(configs['target'])
    n_seg = int(configs['segment_time'] * configs['driver']['instruments_setup']['activation_sampling_frequency'])
    n_pts = int(n_seg * n)
    t = np.zeros(n_pts)
    for i in range(n):
        t[i*n_seg: (i+1)*n_seg] = np.ones(n_seg) * configs['target'][i]

    return t



def long_target_ring(targ: np.array, configs):

    # Equivalent of the previous function for ring (and square) classifier 

    n = len(targ)
    n_seg = int(configs['segment_time'] * configs['driver']['instruments_setup']['activation_sampling_frequency'])
    n_pts = int(n_seg * n)
    t = np.zeros(n_pts)
    for i in range(n):
        t[i*n_seg: (i+1)*n_seg] = np.ones(n_seg) * targ[i]

    return t





def average_I(targ: np.array, I: np.array, configs):

    # Makes an average of the output signal over a segment of the target

    n = len(targ)
    n_seg = int(configs['segment_time'] * configs['driver']['instruments_setup']['activation_sampling_frequency'])
    output = np.zeros(n)
    for i in range(n):
        output[i] = np.mean(I[i*n_seg : (i+1)*n_seg])

    return output


    


def dE_dI (I: np.array, targ: np.array, configs):
    
    # Returns the derivate of the loss function with respect to the output current
    # This derivative is computed specifically for the corrsigfit loss function; if one wants to change the loss function, a new derivated must be calculated.

    n = len(I)
    derivative = np.zeros(n)
    for i in range(n):

        #Derivative of the correlation coefficient
        drho = (targ[i] - np.mean(targ))/(n * np.std(I) * np.std(targ)) - (correlation(I, targ) * (I[i] - np.mean(I)))/(n * (np.std(I)**2))  

        #Derivative of the separation between high and low states
        dsep = (targ[i])/(np.sum(targ)) - (1- targ[i])/(n - np.sum(targ)) 

        #Derivative of the loss function with respect to the output current for the single segment
        der = -drho/sigmoid(sep(I, targ), configs) - ((1 - correlation(I, targ)) * dsep * np.e**(- sep(I, targ))) 
        derivative[i] = der

    return derivative


# The following pair of functions implements the digital low-pass filter as a butterworth filter (filter order is specified in the configs file)

def butter_lowpass(configs, analog=False):
    nyq = 0.5 * configs['driver']['instruments_setup']['activation_sampling_frequency']
    highcut = configs['cutoff_frequency'] / nyq
    b, a = butter(configs['filter_order'], highcut, btype='lowpass', analog=analog)
    return b, a


def butter_lowpass_filter(data, configs):
    b, a = butter_lowpass(configs)
    y = lfilter(b, a, data)
    return y


def dI_dV (output: np.array, configs):

    #Returns the derivative of the output current with respect to the control voltages

    freqs = configs['perturbation_frequencies']
    n_elec = len(freqs) 
    a = (np.arange(0, n_elec)).astype(int)

    # Control electrodes 
    ctrl = np.delete(a, configs['inputs']) 
    derivative = np.array([])
    # (len(ctrl))
    n_pts = len(output)

    # Time of the measurement for a single evolution step
    time = n_pts/configs['driver']['instruments_setup']['activation_sampling_frequency']
    t = np.linspace(0, time, n_pts)

    for i in ctrl:

        # Reference signals (in and out of phase)
        lock_in_ref = np.sin(2 * np.pi * freqs[i] * t)
        lock_in_ref_phase = np.sin(2 * np.pi * freqs[i] * t + np.pi/2)

        # Signal mixing
        mixed_signal = lock_in_ref * output
        mixed_signal_phase = lock_in_ref_phase * output

        # Low-pass filtering
        X = butter_lowpass_filter(mixed_signal,configs)
        Y = butter_lowpass_filter(mixed_signal_phase, configs)

        # Extraction of X and Y components 
        # phase_out = np.arctan2(Y[-1], X[-1]) * 180 / np.pi
        phase_out = np.arctan2(np.mean(Y[-250:-1]), np.mean(X[-250:-1])) * 180 / np.pi

        sign = 2 * (abs(phase_out) < configs['phase_threshold']) - 1

        # Value of the derivative for each control electrode
        # grad = sign * 2 * np.sqrt(X[-1]**2 + Y[-1]**2) / configs['perturbation_amplitudes'][i]
        grad = sign * 2 * np.sqrt(np.mean(X[-250:-1])**2 + np.mean(Y[-250:-1])**2) / configs['perturbation_amplitudes'][i]

        derivative = np.append(derivative, grad)

        # derivative = np.append(derivative, X[-1])
        # derivative[i] = grad

    return derivative


def dE_dV_separeted_inputs (I: np.array, targ: np.array, output: np.array, configs):
    
    #Multiplies dE/dI and dI/dV to obtain the gradient with respect to the inputs voltages. To be used when the inputs for the boolean task and the control perturbations are separated
    
    n_elec = len(configs['perturbation_frequencies']) 
    dE = dE_dI(I, targ, configs)
    dI = dI_dV(output, configs)
    dEdV = np.zeros(n_elec)
    n_cases = len(np.asarray(configs['target']))
    
    # Two separate cases (it is possible to choose between the two using the configs file): 
    # 1) The amount of points for the output current equals the amount of segments (e.g. 4, for boolean gates) i.e. an average is taken for the current over every segment using the function average_I().
    if (len(I) == n_cases):
        for i in range (n_cases):
            dEdV += dE[i] * dI

    # 2) No average has been taken and the target has been reshaped to be the same lenght as the output (the function long_target() has been used)
    else:
        #Number of points per segment
        n_seg = len(I)//n_cases         
        for i in range (n_cases):
            dEdV += np.mean(dE[i*n_seg : (i+1)*n_seg]) * dI

    return dEdV



def dE_dV_mixed (I: np.array, targ: np.array, output: np.array, configs):
    
    #Multiplies dE/dI and dI/dV to obtain the gradient with respect to the inputs voltages. To be used when the inputs for the boolean task and the control perturbations are combined or sequential
    
    n_elec = len(np.asarray(configs['perturbation_frequencies']))
    dE = dE_dI(I, targ, configs)
    dEdV = np.zeros(n_elec)
    n_cases = len(targ)

    assert len(output) % n_cases == 0, 'Number of output points is not a multiple of the number of cases.'
    n_pert = len(output) // n_cases
    
    # Two separate cases (it is possible to choose between the two using the configs file): 
    # 1) The amount of points for the output current equals the amount of segments (e.g. 4, for boolean gates) i.e. an average is taken for the current over every segment using the function average_I().
    if (len(I) == n_cases):
        for i in range (n_cases):
            dI = dI_dV(output[i * n_pert : (i+1) * n_pert], configs)
            dEdV += dE[i] * dI

    # 2) No average has been taken and the target has been reshaped to be the same lenght as the output (the function long_target() has been used)
    else:
        #Number of points per segment
        n_seg = len(I)//n_cases         
        for i in range (n_cases):
            dI = dI_dV(output[i * n_pert : (i+1) * n_pert], configs)
            dEdV += np.mean(dE[i*n_seg : (i+1)*n_seg]) * dI

    return dEdV, dI




# This commented main could be used for testing the functions in this file 

# if __name__ == '__main__':

#     global sig_scale
#     sig_scale = 1

#     I = np.array([20, 35, 36, 24])
#     targ = np.array([0, 1, 1, 0])

#     targ = long_target(I, targ)


#     grad = dE_dI(I, targ)
#     print('dE = '+str(grad))
#     print('mean = '+str(np.mean(grad)))
#     print('std = '+str(np.std(grad)))
#     print('E = '+str(loss_function(I, targ)))
#     print('rho = '+str(correlation(I, targ)))
#     print('sig = '+str(sigmoid(sep(I, targ))))    




