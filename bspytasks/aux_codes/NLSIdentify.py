from tkinter import PhotoImage
import weakref
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
from brainspy.utils.io import load_configs
from brainspy.utils.manager import get_driver
import math

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

def measurement(
                driver,
                T,
                fs,
                start_freq,
                stop_freq,
                num_projections,
				chirp_signal,
                rest_length     = 10000,
                slope_length    = 4000,
				input_elec_idx 	= 3,
				
        ):

	meas_input = np.zeros((6, int(rest_length + 2 * slope_length + T * fs)))
	t = np.linspace(0, T, int(T * fs))
	outputs = []
	# chirp_signal = scipy.signal.chirp(
	# 							t       = t,
	# 							f0      = start_freq,
	# 							t1      = T,
	# 							f1      = stop_freq,
	# 							method  = 'logarithmic',
	# 							phi     = 90
	# 	)
	for i in range(num_projections):
		meas_input[input_elec_idx, rest_length + slope_length : -slope_length] = chirp_signal
		if i != 0:
			meas_input = set_random_control_voltages(
					meas_input 				=	meas_input,
					dnpu_control_indeces	=	[0, 1, 2, 4, 5],
					slop_length				=	slope_length,
					magnitudes				=	[-.85, .85]
			)
		outputs.append(driver.forward_numpy(meas_input.T))
	driver.close_tasks()

	np.save("raw_measured_output.npy", outputs)
	temp = []
	for i in range(len(outputs)):
		temp.append(
			outputs[i][slope_length + rest_length : -slope_length, 0] - np.mean(outputs[i][slope_length + rest_length -1000 : slope_length + rest_length - 100, 0])
		)
	name = "croped_measured_output_" + str(start_freq) + "_" + str(stop_freq) + ".npy"
	np.save(name, temp)


if __name__ == '__main__':

	do_measurement = False
	plots = False

	fs = 500000
	start_freq = 16
	stop_freq = 4096
	num_projections = 2
	rest_length = 20000
	slope_length = 4000
	input_elec_idx = 3
	holding_shift_in_time = False
	N = 4
	# idx = 0 --> all zero control voltages
	idx = 0


	np.random.seed(0)

	L = 0.5
	T = L * np.log(stop_freq/start_freq)
	t = np.linspace(0, T, int(T * fs))

	chirp_signal = np.sin(
						2 * np.pi * start_freq * L * (np.exp(t / L) - 1)
	)
	
	chirp_signal_tilde = (start_freq / L) * np.exp(-t/L) * np.flip(chirp_signal)

	if do_measurement == True:  
		configs = load_configs(
			'configs/defaults/processors/hw_freq_analysis.yaml'
		)
		driver = get_driver(
			configs=configs["driver"]
		)
		measurement(
				driver		=	driver,
				T 			=	T,
				fs			=	fs,
				start_freq	=	start_freq,
				stop_freq	=	stop_freq,
				num_projections	=	num_projections,
				chirp_signal = chirp_signal,
				rest_length	=	rest_length,
				slope_length=	slope_length,
				input_elec_idx	=	input_elec_idx,
				
		)

	name = "croped_measured_output_" + str(start_freq) + "_" + str(stop_freq) + ".npy"
	measured_output = np.load(name)

	conv_results = []
	for i in range(len(measured_output)):
		temp = measured_output[i] - np.mean(measured_output[i])
		conv_results.append(
			scipy.signal.convolve(
				chirp_signal_tilde,
				temp,
				mode = 'full'
			)
		)
	t_conv = np.linspace(-T, T, int(2 * len(chirp_signal_tilde) - 1))
	if plots == True:
		plt.figure()
		for i in range(len(conv_results)):
			plt.plot(t_conv, conv_results[i])
		plt.show()
	
	# Center of h_m(t)'s for 10
	delta_ts = []
	for i in range(N+1):
		delta_ts.append(
					-L * np.log(i+1)
		)
	
	# separating h_m(t)'s
	# 0 -> linear
	# N -> N'th nonlinear part
	hm_t = []
	# Following lines are the implementation where shifts in time are held
	if holding_shift_in_time == True:
		for i in range(N):
			temp = np.zeros((len(t_conv)))
			center_point = int(((len(t_conv))/(2*T))*(delta_ts[i]) + (len(t_conv)/2))
			point_to_next = int(((len(t_conv))/(2*T))*(delta_ts[i+1]) + (len(t_conv)/2))
			diff = np.abs(int(point_to_next - center_point))
			temp[int(center_point - diff/2) : int(center_point + diff/2)] = conv_results[idx][int(center_point - diff/2) : int(center_point + diff/2)]
			hm_t.append(
					temp
			)
	else:
		for i in range(N):
			temp = np.zeros((len(t_conv)))
			center_point = int(((len(t_conv))/(2*T))*(delta_ts[i]) + (len(t_conv)/2))
			point_to_next = int(((len(t_conv))/(2*T))*(delta_ts[i+1]) + (len(t_conv)/2))
			diff = np.abs(int(point_to_next - center_point))
			len_crop = len(conv_results[idx][int(center_point - diff/2):int(center_point + diff/2)])
			temp[int(len(t_conv)/2 - len_crop/2) : int(len(t_conv)/2 + len_crop/2)] = conv_results[idx][int(center_point - diff/2) : int(center_point + diff/2)]
			hm_t.append(
					temp
			)
		# # Without shift in time
		# for i in range(len(delta_ts)):
		# 	temp = np.zeros((len(t_conv)))
		# 	center_time = delta_ts[i]
		# 	if i != len(delta_ts) - 1:
		# 		time_to_next = delta_ts[i+1] - delta_ts[i]
		# 	else:
		# 		time_to_next = delta_ts[i] - delta_ts[i-1]
		# 	t_idx_1 = 0.
		# 	t_idx_2 = 0.
		# 	for i in range(len(t_conv)):
		# 		if np.abs(t_conv[i] - (center_time - time_to_next/2)) <= 0.001:
		# 			t_idx_1 = i
		# 		if np.abs(t_conv[i] - (center_time + time_to_next/2) <= 0.001):
		# 			t_idx_2 = i
		# 	len_crop = len(conv_results[idx][t_idx_2:t_idx_1])
		# 	temp[int(len(t_conv)/2 - len_crop/2) : int(len(t_conv)/2 + len_crop/2)] = conv_results[idx][t_idx_2:t_idx_1]
		# 	hm_t.append(
		# 		temp
		# 	)

	
	Hm_f = []
	Gn_f = []
	for i in range(len(hm_t)):
		Hm_f.append(
				np.fft.rfft(hm_t[i])
		)

	A_matrix = np.zeros((N, N), dtype=np.complex)
	for n in range(N):
		for m in range(N):
			if n >= m and (n + m)%2 == 0:
				A_matrix[n, m] = (((-1) ** (2 * n + (1 - m)/2)) / (2**(n - 1))) * math.comb(n, (n-m)//2)
	
	A_matrix_T = np.matrix.transpose(A_matrix)
	A_matrix_T_inv = np.linalg.inv(A_matrix_T)

	for m in range(N):
		temp = np.zeros((np.shape(Hm_f)[1]))
		temp = np.dot(A_matrix_T_inv[m,:], Hm_f[:])
		Gn_f.append(temp)
		
	# Gn_f.append(Hm_f[0] + 3 * Hm_f[2] + 5 * Hm_f[4])
	# Gn_f.append(2 * np.complex(0,1) * Hm_f[1] + 8 * np.complex(0,1) * Hm_f[3])
	# Gn_f.append(-4 * Hm_f[2] -20 * Hm_f[4])
	# Gn_f.append(-8 * np.complex(0,1) * Hm_f[3])
	# Gn_f.append(16 * Hm_f[4])
	
	gn_t = []
	for i in range(N):
		gn_t.append(np.fft.irfft(Gn_f[i]))

	freqs = np.fft.rfftfreq(2 * len(Gn_f[0]), 1/fs)
	Gn_f_dB = 20*np.log10(np.abs(Gn_f[0])/np.max(np.abs(Gn_f[0])))
	# plt.plot(freqs[1:], Gn_f_dB)

	# output reconstruction
	t = np.linspace(0, 1, int(1 * fs))
	x_t_sin = 1.2 * np.sin(2 * np.pi * 1000 * t)
	x_t_sin[-1] = 0.

	y_t = np.zeros((len(scipy.signal.convolve(gn_t[0], x_t_sin))))
	for i in range(N):
		# convolver = np.zeros((len(gn_t[i])))
		# convolver[np.argmax(gn_t[i])] = np.max(gn_t[i])
		# y_t += scipy.signal.convolve(convolver, x_t_sin**(i+1))
		y_t += scipy.signal.convolve(gn_t[i], x_t_sin**(i+1))


	# meaure device
	configs = load_configs(
			'configs/defaults/processors/hw_freq_analysis.yaml'
		)
	driver = get_driver(
		configs=configs["driver"]
	)

	meas_input = np.zeros((6, int(rest_length + 2 * slope_length + 1 * fs)))
	meas_input[input_elec_idx, rest_length + slope_length : -slope_length] = x_t_sin
	meas_input = set_random_control_voltages(
					meas_input 				=	meas_input,
					dnpu_control_indeces	=	[0, 1, 2, 4, 5],
					slop_length				=	slope_length,
					magnitudes				=	[-.85, .85]
			)

	output = driver.forward_numpy(meas_input.T)
	driver.close_tasks()

	freqs = np.fft.rfftfreq(len(y_t), 1/fs)
	freqs_2 = np.fft.rfftfreq(len(output[:,0]), 1/fs)
	plt.figure()
	plt.plot(freqs, np.abs(np.fft.rfft(y_t)))
	plt.xlim((start_freq, stop_freq))
	plt.figure()
	plt.plot(freqs_2, np.abs(np.fft.rfft(output[:,0] - np.mean(output[:,0]))))
	plt.xlim((start_freq, stop_freq))


	print(":")




	

	
	
	


    
