import numpy as np
import sysidentpy as si
import matplotlib.pyplot as plt
from scipy.signal import chirp
import pandas as pd

from sysidentpy.metrics import mean_squared_error
from sysidentpy.utils.generate_data import get_siso_data

from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
# from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation, compute_cross_corr



# loading data from files
slope_length = 4000
rest_length = 8000
T = 5
fs = 25000
start_freq = 10
stop_freq = 300

t = np.linspace(0, T, int(T * fs))
chirp_signal = chirp(
                    t = t,
                    f0 = start_freq,
                    t1 = T,
                    f1 = stop_freq,
                    method= 'linear',
                    phi = 90
                )

device_output = np.load("outputs.npy")


y_train = device_output[0][slope_length+rest_length:-slope_length,0] - np.average(device_output[0][slope_length+rest_length-500:slope_length+rest_length-100,0])
x_train = chirp_signal

basic_function = Polynomial(degree=3)
model = FROLS(
            order_selection =   True,
            n_info_values   =   20,
            extended_least_squares= False,
            ylag=10,
            xlag=10,
            info_criteria='aic',
            estimator='least_squares',
            basis_function=basic_function
)

model.fit(X=x_train.reshape(-1,1), y=y_train.reshape(-1,1))

r = pd.DataFrame(
        results(
                model.final_model, model.theta, model.err,
                model.n_terms, err_precision=3, dtype='sci'
        ),
        columns=['Regressors', 'Parameters', 'ERR']
) 

print(r)
