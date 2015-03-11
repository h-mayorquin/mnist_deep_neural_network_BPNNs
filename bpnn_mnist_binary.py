import numpy as np
import matplotlib.pyplot as plt
from load_binary_files import training_ims, training_labels
from init_functions import connectivity
from corrupt_functions import distort_binnary
    
# Select quantity of data to use
N_data_total = len(training_labels)
percentage = 0.01
N_to_use = int(percentage * N_data_total)

# Decide how much data to use
X = training_ims[0:N_to_use]
Y = training_labels[0:N_to_use]
N_hypercolumns = X.shape[1]
units_per_hypercolumn = 2

w, p = connectivity(N_hypercolumns, units_per_hypercolumn, X)

log_beta = np.log(p)

# Now let's distort a pattern
percentage = 0.1
index = 1

dis_pattern, pattern = distort_binnary(index, percentage, X)

iterations = 10

s = np.zeros((N_hypercolumns, units_per_hypercolumn))
random = np.random.rand(N_hypercolumns, units_per_hypercolumn)

o = random

aux = np.vstack((dis_pattern, dis_pattern))
aux[1, :] = 1 - aux[1, :]

o = aux.T

# We will make o a probability
o = o / np.sum(o, axis=1)[:, np.newaxis]

for iter in xrange(iterations):
    print 'iter', iter
    print '----------------'
    for h_index in xrange(N_hypercolumns):
        quantity_0 = 0
        quantity_1 = 0
        
        for h_out in xrange(N_hypercolumns):
            quantity_0 += np.log(w[h_index, h_out, 0, 0] * o[h_out, 0]
                                 + w[h_index, h_out, 0, 1] * o[h_out, 1])
            quantity_1 += np.log(w[h_index, h_out, 1, 0] * o[h_out, 0]
                                 + w[h_index, h_out, 1, 1] * o[h_out, 1])

        s[h_index, 0] = log_beta[h_index, 0] + quantity_0
        s[h_index, 1] = log_beta[h_index, 1] + quantity_1

    o = np.exp(s / np.abs(np.max(s)))
    o = o / np.sum(o, axis=1)[:, np.newaxis]
