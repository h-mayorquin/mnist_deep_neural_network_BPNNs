import numpy as np
from load_binary_files import training_ims, training_labels
from init_functions import connectivity

    
# Select quantity of data to use
N_data_total = len(training_labels)
percentage = 0.8
N_to_use = int(percentage * N_data_total)

# Decide how much data to use
X = training_ims[0:N_to_use]
Y = training_labels[0:N_to_use]
N_hypercolumns = X.shape[1]
units_per_hypercolumn = 2

w, p = connectivity(N_hypercolumns, units_per_hypercolumn, X, low_noise=0.001)

folder = './data/'
name = 'saved_connectivity'
name_to_save_matrix = folder + name + '_matrix'
name_to_save_p = folder + name + '_p'
np.save(name_to_save_matrix, w)
np.save(name_to_save_p, p)
