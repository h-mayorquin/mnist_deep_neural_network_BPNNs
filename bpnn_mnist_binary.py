import numpy as np
import matplotlib.pyplot as plt
from load_binary_files import training_ims, training_labels
from init_functions import connectivity
from corrupt_functions import distort_binnary
from copy import deepcopy
    
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

pattern = deepcopy(X[index])
dis_pattern = distort_binnary(index, percentage, X)
save_dis = deepcopy(dis_pattern)

iterations = 2

aux = np.vstack((dis_pattern, dis_pattern))
aux[1, :] = 1 - aux[1, :]  # Make the 0 1's

o = aux.T

# We will make o a probability
o = o / np.sum(o, axis=1)[:, np.newaxis]
s = np.zeros(o.shape, dtype=float)

noise = 10e-10

for iter in xrange(iterations):
    print 'iter', iter
    print '----------------'
    for h_index in xrange(N_hypercolumns):
        quantity_0 = 0
        quantity_1 = 0
        
        for h_out in xrange(N_hypercolumns):
            aux_1 = (w[h_index, h_out, 0, 0] * o[h_out, 0] +
                     w[h_index, h_out, 0, 1] * o[h_out, 1])
            aux_2 = (w[h_index, h_out, 1, 0] * o[h_out, 0] +
                     w[h_index, h_out, 1, 1] * o[h_out, 1])

            if (aux_1 > 0):
                quantity_0 += np.log(aux_1)
            else:
                quantity_0 += noise
            
            if (aux_2 > 0):
                quantity_1 += np.log(aux_2)
            else:
                quantity_1 += noise

        if False:
            print '1 log , quant', log_beta[h_index, 0], quantity_0
            print '2 log, quant', log_beta[h_index, 1], quantity_1

        x_1 = log_beta[h_index, 0] + quantity_0
        x_2 = log_beta[h_index, 1] + quantity_1
        s[h_index, 0] = x_1
        s[h_index, 1] = x_2

    # o = np.exp(s / np.abs(np.max(s)))
    o = np.exp(s)
    o[o < noise] = noise

    o = o / np.sum(o, axis=1)[:, np.newaxis]


# Plot

output = 1 - np.argmax(o, axis=1)
plt.subplot(1, 3, 1)
plt.imshow(pattern.reshape((28, 28)))
plt.title('Original')

plt.subplot(1, 3, 2)
plt.imshow(output.reshape((28, 28)))
plt.title('Output')

plt.subplot(1, 3, 3)
plt.imshow(save_dis.reshape((28, 28)))
plt.title('Distorted')

plt.show()

print 'The thing did move?', np.sum(save_dis - output)
