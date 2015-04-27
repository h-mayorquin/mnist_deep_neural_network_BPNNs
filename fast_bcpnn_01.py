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

# Load the data
directory = './data/'
name_matrix = 'saved_connectivity_matrix.npy'
name_prob = 'saved_connectivity_p.npy'
w = np.load(directory + name_matrix)
p = np.load(directory + name_prob)

log_beta = np.log(p)

# Now let's distort a pattern
percentage = 1
index = 10

pattern = deepcopy(X[index])
dis_pattern = distort_binnary(index, percentage, X)
save_dis = deepcopy(dis_pattern)

aux = np.vstack((dis_pattern, dis_pattern))
aux = np.vstack((pattern, pattern))
aux[1, :] = 1 - aux[1, :]  # Make the 0 1's

o = aux.T
o = np.random.rand(784, 2)
save_dis = np.argmax(o, axis=1)

# We will make a probability
o = o / np.sum(o, axis=1)[:, np.newaxis]
s = np.zeros(o.shape, dtype=float)

noise = 10e-10

double_bias = np.log(noise)

iterations = 10
to_store = np.zeros((6, N_hypercolumns, iterations))
G = 5.0
plot = True

statistics1 = []
statistics2 = []
statistics3 = []

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

        x_1 = log_beta[h_index, 0] + np.log(aux_1)
        x_2 = log_beta[h_index, 1] + np.log(aux_2)

        s[h_index, 0] = x_1
        s[h_index, 1] = x_2
        statistics1.append(x_1)
        statistics1.append(x_1)
        statistics2.append(np.abs(x_2 - x_1))

    o = np.exp(G * s)

    plt.subplot(1, 3, 1)
    plt.plot(s)
    plt.ylim([-10, 1])

    plt.subplot(1, 3, 2)
    plt.plot(o)
    plt.ylim([-0.1, 1.1])

    to_store[0, :, iter] = o[:, 0]
    to_store[1, :, iter] = o[:, 1]

    to_store[4, :, iter] = o[:, 0]
    to_store[5, :, iter] = o[:, 1]

    denominator = np.sum(o, axis=1)
    statistics3.append(denominator)
    o = o / denominator[:, np.newaxis]
    plt.subplot(1, 3, 3)
    plt.plot(o)

    to_store[2, :, iter] = o[:, 0]
    to_store[3, :, iter] = o[:, 1]

    plt.savefig('./figures/figure' + str(iter) + '.png')
    plt.hold(False)


# Plot
if iterations * 2 < N_hypercolumns:
    plot_to_N = iterations * 2
else:
    plot_to_N = N_hypercolumns

output = 1 - np.argmax(o, axis=1)


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
