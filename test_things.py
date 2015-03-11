import numpy as np
from load_binary_files import training_ims, training_labels
from init_functions import connectivity

# Select quantity of data to use
N_data_total = len(training_labels)
percentage = 0.001
N_to_use = int(percentage * N_data_total)

# Decide how much data to use
X = training_ims[0:N_to_use]
Y = training_labels[0:N_to_use]
N_hypercolumns = X.shape[1]
units_per_hypercolumn = 2

p_matrix, p = connectivity(N_hypercolumns, units_per_hypercolumn, X)
