import numpy as np
import matplotlib.pyplot as plt
from load_binary_files import training_ims, training_labels
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
noise = 1e-10

directory = './data/'
name_matrix = 'saved_connectivity_matrix.npy'
name_prob = 'saved_connectivity_p.npy'
w = np.load(directory + name_matrix)
p = np.load(directory + name_prob)

beta = p
log_beta = np.log(p)

# Now let's distort a pattern
percentage = 0.1
index = 1

pattern = deepcopy(X[index])
dis_pattern = distort_binnary(index, percentage, X)
save_dis = deepcopy(dis_pattern)

aux = np.vstack((dis_pattern, dis_pattern))
aux[1, :] = 1 - aux[1, :]  # Make the 0 1's

o = aux.T.astype(float)
# Eliminate cells with very low probability
o[o < noise] = noise
zeros = np.sum(o == 0)
# We will make o a probability
o = o / np.sum(o, axis=1)[:, np.newaxis]
s = np.zeros(o.shape, dtype=float)

double_bias = np.log(noise)
double_bias = noise


iterations = 50
to_store = np.zeros((6, N_hypercolumns, iterations))
G = 0.0001
counter_aux_1 = 0
counter_aux_2 = 0
counter = 0
statistics1 = []
statistics2 = []
statistics3 = []

for iter in xrange(iterations):
    print 'iter', iter
    print '----------------'
    for h_index in xrange(N_hypercolumns):
        quantity_0 = 1
        quantity_1 = 1
        print '-------------- Iterate for a particular support ----------'
        for h_out in xrange(N_hypercolumns):
            aux_1 = (w[h_index, h_out, 0, 0] * o[h_out, 0] +
                     w[h_index, h_out, 0, 1] * o[h_out, 1])
            aux_2 = (w[h_index, h_out, 1, 0] * o[h_out, 0] +
                     w[h_index, h_out, 1, 1] * o[h_out, 1])

            if (aux_1 > 0):
                quantity_0 *= aux_1
                print 'quantity_0', quantity_0
                counter_aux_1 = 0
            else:
                quantity_0 *= double_bias
                counter_aux_1 = 1

            if (aux_2 > 0):
                quantity_1 *= aux_2
                counter_aux_2 = 0
                print 'quantity_1', quantity_1
            else:
                quantity_1 *= double_bias
                counter_aux_2 = 1
            
            counter += counter_aux_1 * counter_aux_2
            if False:
                print '1 log , quant', log_beta[h_index, 0], quantity_0
                print '2 log, quant', log_beta[h_index, 1], quantity_1

        x_1 = beta[h_index, 0] * quantity_0
        x_2 = beta[h_index, 1] * quantity_1
        statistics1.append(quantity_0)
        statistics1.append(quantity_1)
        s[h_index, 0] = x_1
        s[h_index, 1] = x_2
        print 'xs', x_1, x_2
        statistics2.append(x_1)
        statistics2.append(x_2)
        statistics3.append(np.abs(x_2 - x_1))

    plt.plot(s)
    plt.savefig('./figures/figure' + str(iter) + '.png')
    plt.hold(False)
    o = s

    to_store[0, :, iter] = o[:, 0]
    to_store[1, :, iter] = o[:, 1]

    o[o < noise] = noise

    to_store[4, :, iter] = o[:, 0]
    to_store[5, :, iter] = o[:, 1]

    o = o / np.sum(o, axis=1)[:, np.newaxis]
    to_store[2, :, iter] = o[:, 0]
    to_store[3, :, iter] = o[:, 1]


# Plot
if iterations * 2 < N_hypercolumns:
    plot_to_N = iterations * 2
else:
    plot_to_N = N_hypercolumns

output = 1 - np.argmax(o, axis=1)

plot = True
if plot:

    plt.subplot(3, 2, 1)
    plt.imshow(to_store[0, 0:plot_to_N, ...])
    plt.title('0 before normalization and noise')
    plt.colorbar()

    plt.subplot(3, 2, 2)
    plt.imshow(to_store[1, 0:plot_to_N, ...])
    plt.title('1 before normalization and noise')
    plt.colorbar()

    plt.subplot(3, 2, 3)
    plt.imshow(to_store[2, 0:plot_to_N, ...])
    plt.title('0 after normalization and noise')
    c = plt.colorbar()
    c.set_clim(0, 1)

    plt.subplot(3, 2, 4)
    plt.imshow(to_store[3, 0:plot_to_N, ...])
    plt.title('1 after normalization and noise')
    c = plt.colorbar()
    c.set_clim(0, 1)

    plt.subplot(3, 2, 5)
    plt.imshow(to_store[4, 0:plot_to_N, ...])
    plt.title('0 after noise')
    plt.colorbar()

    plt.subplot(3, 2, 6)
    plt.imshow(to_store[5, 0:plot_to_N, ...])
    plt.title('1 after noise')
    plt.colorbar()

    plt.show()

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
