from doctest import OutputChecker
import numpy as np
import matplotlib.pyplot as plt
from brainspy.utils.io import load_configs
from brainspy.utils.manager import get_driver


def plot_diff_sin(
				configs,
				fs = 10000,
				T = 1,
				freqs = [100, 250, 500, 1000],
				no_cvs = 5,
				slope_length = 2000,
				rest_length = 4000,
				dnpu_control_indeces = [0, 1, 2, 4, 5],
):
	meas_input = np.zeros((6, int(int(fs * T) + 2 * slope_length + rest_length)))

	configs['driver']['instruments_setup']['activation_sampling_frequency'] = fs
	configs['driver']['instruments_setup']['readout_sampling_frequency'] = 100000
	driver = get_driver(
		configs=configs["driver"]
	)

	outputs = []
	t = np.linspace(0, T, int(fs * T))
	# set random control voltages
	for i in range(len(freqs)):
		sinus = np.sin(2 * np.pi * freqs[i] * t)
		meas_input[3, slope_length + rest_length : -slope_length] = sinus
		for j in range(no_cvs):
			if j != 0:
				for i in range(len(dnpu_control_indeces)):
					rand_value = np.random.uniform(-0.55, 0.55)
					ramp_up = np.linspace(0, rand_value, slope_length)
					plateau = np.linspace(rand_value, rand_value, int(fs * T) + rest_length)
					ramp_down = np.linspace(rand_value, 0, slope_length)
					meas_input[dnpu_control_indeces[i],:] = np.concatenate((ramp_up, plateau, ramp_down))
			outputs.append(driver.forward_numpy(meas_input.T))
	driver.close_tasks()

	for i in range(len(freqs)):
		for j  in range(no_cvs):
			plt.plot(outputs[i+j][-slope_length - 400 : -slope_length - 200,0] - np.mean(outputs[i+j][-slope_length - 500 : -slope_length - 100,0]))

	# plt.plot(outputs[4][-slope_length - 388 : -slope_length - 185,0] - np.mean(outputs[4][-slope_length - 500 : -slope_length - 100,0]))


	print(" ")


def plot_iv(
			configs,
			no_plots = 5,
			slope_length = 2000,
			rest_length = 5000,
			plateau_length = 8000,
			dnpu_control_indeces = [0, 1, 2, 4, 5]

):	
	meas_input = np.zeros((6, int(plateau_length + 2 * slope_length + rest_length)))

	ramp_down_1 = np.linspace(0, -1.8, plateau_length//4)
	ramp_up 	= np.linspace(-1.8, 1.8, plateau_length//2)
	ramp_down_2 = np.linspace(1.8, 0, plateau_length//4)

	temp = np.concatenate((ramp_down_1, ramp_up, ramp_down_2))
	meas_input[3, slope_length + rest_length : -slope_length] = temp

	configs['driver']['instruments_setup']['activation_sampling_frequency'] = 100
	configs['driver']['instruments_setup']['readout_sampling_frequency'] = 10000
	driver = get_driver(
		configs=configs["driver"]
	)

	outputs = []
	# set random control voltages
	for i in range(no_plots):
		if i != 0:
			for i in range(len(dnpu_control_indeces)):
				rand_value = np.random.uniform(-0.55, 0.55)
				ramp_up = np.linspace(0, rand_value, slope_length)
				plateau = np.linspace(rand_value, rand_value, rest_length + plateau_length)
				ramp_down = np.linspace(rand_value, 0, slope_length)
				meas_input[dnpu_control_indeces[i], :] = np.concatenate((ramp_up, plateau, ramp_down))
		outputs.append(driver.forward_numpy(meas_input.T))

	driver.close_tasks()
	for i in range(no_plots):
		plt.plot(meas_input[3,slope_length + rest_length : -slope_length], outputs[i][slope_length + rest_length : -slope_length,0] - np.mean(outputs[i][slope_length + rest_length - 500 : slope_length + rest_length - 100,0]))
	plt.show()

def calculate_time_constant(
						input,
						slope_length = 1000,
						rest_length = 1500,
						plateau_length = 5000,

):	
	signal = input[slope_length + rest_length -1 : -slope_length,0] - np.mean(
																	input[slope_length + rest_length - 100 : slope_length + rest_length - 20,0]
																	)
	time_constant_index = 0
	# steady state value
	
	ss_voltage = np.average(signal[len(signal)//2 : len(signal)//2 + 100])

	for i in range(len(signal)):
		if np.abs((signal[i] - 0.632*ss_voltage)) <= 0.005:
			time_constant_index = i

	if time_constant_index == 0:
		for i in range(len(signal)):
			if np.abs((signal[i] - 0.632*ss_voltage)) <= 0.01:
				time_constant_index = i

	return time_constant_index

def time_constant_analysis(
						configs,
						slope_length = 1000,
						rest_length = 1500,
						plateau_length = 5000,

						input_elec_index = 3,
						input_step_voltage = 1.,

						bias_electrode_index = 0,
						bias_voltage = .4,
						dnpu_control_indeces = [0, 1, 2, 4, 5, 6]

):

	configs['driver']['instruments_setup']['activation_sampling_frequency'] = 1000
	configs['driver']['instruments_setup']['readout_sampling_frequency'] = 100000

	meas_input = np.zeros((7, int(plateau_length + 2 * slope_length + rest_length)))

	meas_input[input_elec_index, slope_length + rest_length : -slope_length] = input_step_voltage * np.ones((plateau_length))
	
	# ramp_up = np.linspace(0, bias_voltage, slope_length)
	# plateau = np.linspace(bias_voltage, bias_voltage, rest_length + plateau_length)
	# ramp_down = np.linspace(bias_voltage, 0, slope_length)

	# meas_input[bias_electrode_index, :] = np.concatenate((ramp_up, plateau, ramp_down))

	driver = get_driver(
		configs=configs["driver"]
	)

	time_constant_indeces = []
	for i in range(500):
	# set random control voltages
		for i in range(len(dnpu_control_indeces)):
			rand_value = np.random.uniform(-0.5, 0.5)
			ramp_up = np.linspace(0, rand_value, slope_length)
			plateau = np.linspace(rand_value, rand_value, rest_length + plateau_length)
			ramp_down = np.linspace(rand_value, 0, slope_length)
			meas_input[dnpu_control_indeces[i], :] = np.concatenate((ramp_up, plateau, ramp_down))
		output = driver.forward_numpy(meas_input.T)
		time_constant = calculate_time_constant(
											input = 		output,
											slope_length = 	1000,
											rest_length = 	1500,
											plateau_length =5000,
						)
		time_constant_indeces.append(time_constant)
	
	driver.close_tasks()

	counts, bins = np.histogram(time_constant_indeces, 20)
	plt.figure(dpi=150)
	plt.plot(bins[:-1], counts, 'o-.', color='blue')
	plt.xticks(np.arange(min(bins[:-1]), max(bins[:-1])+1, 5))
	plt.legend(["Measured time constant in milliseconds"])
	plt.xlabel("Bins (mS)", fontsize=13)
	plt.ylabel("Counts", fontsize=13)


	print(" ")

def spike_pattern_response(
			configs,
			slope_length = 100,
			rest_length = 1500,
			plateau_length = 2000,
			random_cvs = 20,
			dnpu_control_indeces = [0, 1, 2, 4, 5, 6]
):

	configs['driver']['instruments_setup']['activation_sampling_frequency'] = 1000
	configs['driver']['instruments_setup']['readout_sampling_frequency'] = 100000

	meas_input = np.zeros((7, int(plateau_length + 2 * slope_length + rest_length)))

	# no control voltages
	
	spike_train_length = 50
	
	interval = [2, 3, 5, 10]

	driver = get_driver(
		configs=configs["driver"]
	)
	outputs = []

	# set random control voltages
	for i in range(random_cvs):
		if i != 0:
			for i in range(len(dnpu_control_indeces)):
				rand_value = np.random.uniform(-0.55, 0.55)
				ramp_up = np.linspace(0, rand_value, slope_length)
				plateau = np.linspace(rand_value, rand_value, rest_length + plateau_length)
				ramp_down = np.linspace(rand_value, 0, slope_length)
				meas_input[dnpu_control_indeces[i], :] = np.concatenate((ramp_up, plateau, ramp_down))
		for k in range(len(interval)):
			inputs = np.zeros((plateau_length))
			spike_train = np.zeros((spike_train_length))
			for i in range(spike_train_length):
				if i % interval[k] == 0:
					spike_train[i] = 1.0

			inputs[0:spike_train_length] = spike_train

			meas_input[3, slope_length + rest_length : -slope_length] = inputs

			outputs.append(driver.forward_numpy(meas_input.T))

	print(" ")


if __name__ == '__main__':

	# np.random.seed(0)

	fs = 50000
	num_projections = 2
	input_elec_idx = 3

	

	# meaure device
	configs = load_configs(
			'configs/defaults/processors/hw_freq_analysis.yaml'
		)

	# time_constant_analysis(configs)

	spike_pattern_response(configs)
	
	# plot_iv(configs)

	# plot_diff_sin(
	# 				configs 		= configs,
	# )


	print(":")





    
