import numpy as np
from load_binary_files import training_ims, training_labels
from prob_functions import p_joint

# Select quantity of data to use
N_data_total = len(training_labels)
percentage = 0.001
N_to_use = int(percentage * N_data_total)

# Decide how much data to use
X = training_ims[0:N_to_use]
Y = training_labels[0:N_to_use]
N_hypercolumns = X.shape[1]
units_per_hypercolumn = 2
low_noise = 10e-10

p = np.zeros((N_hypercolumns, units_per_hypercolumn))

for i in range(units_per_hypercolumn):
    p[:, i] = np.sum(X == i, axis=0)

p = p * 1.0 / X.shape[0]
# Add low noise
p[p < low_noise] = low_noise
# Now we need to normalize
p = p / p.sum(axis=1)[:, np.newaxis]


distribution = {0: 0, 1: 1, 'no': 3}
p_j = p_joint(N_hypercolumns, units_per_hypercolumn, X, distribution)

