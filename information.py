import numpy as np
import matplotlib.pyplot as plt
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


def p_independent(N_hypercolumns, units_per_hypercolumn,
                  X, low_noise=10e-10, normalize=True):
    """
    This and that
    """

    p = np.zeros((N_hypercolumns, units_per_hypercolumn))

    for i in range(units_per_hypercolumn):
        p[:, i] = np.sum(X == i, axis=0)

    if normalize:
        p = p * 1.0 / X.shape[0]
        # Add low noise
        p[p < low_noise] = low_noise
        # Now we need to normalize
        p = p / p.sum(axis=1)[:, np.newaxis]

    return p

p = p_independent(N_hypercolumns, units_per_hypercolumn, X)
distribution = {0: 0, 1: 1, 'no': 3}
p_j = p_joint(N_hypercolumns, units_per_hypercolumn, X, distribution)
# Add noise
aux = np.copy(p_j)
p_j[p_j < low_noise] = low_noise
# Normalize
normalize = p_j.sum(axis=(2, 3))
p_j = p_j / normalize[..., np.newaxis, np.newaxis]


def calculate_information(w, p, i, j):
    log_1 = np.log(w[i, j, 0, 0] / (p[i, 0] * p[j, 0]))
    log_2 = np.log(w[i, j, 0, 1] / (p[i, 0] * p[j, 1]))
    log_3 = np.log(w[i, j, 1, 0] / (p[i, 1] * p[j, 0]))
    log_4 = np.log(w[i, j, 1, 1] / (p[i, 1] * p[j, 1]))
    x = (w[i, j, 0, 0] * log_1 +
         w[i, j, 0, 1] * log_2 +
         w[i, j, 1, 0] * log_3 +
         w[i, j, 1, 1] * log_4)

    return x


def calculate_joint_entr(w, i, j):
    log_1 = np.log(w[i, j, 0, 0])
    log_2 = np.log(w[i, j, 0, 1])
    log_3 = np.log(w[i, j, 1, 0])
    log_4 = np.log(w[i, j, 1, 1])

    x = (w[i, j, 0, 0] * log_1 +
         w[i, j, 0, 1] * log_2 +
         w[i, j, 1, 0] * log_3 +
         w[i, j, 1, 1] * log_4)

    return x


# Calculate the information
def information_matrix(w, p):
    N = p.shape[0]
    I = np.zeros((N, N))
    for i in xrange(N):
        for j in xrange(N):
            I[i, j] = calculate_information(w, p, i, j)

    return I


def joint_entr_matrix(w, p):
    N = p.shape[0]
    I = np.zeros((N, N))
    for i in xrange(N):
        for j in xrange(N):
            I[i, j] = calculate_joint_entr(w, i, j)

    return I

calculate = True
if calculate:
    I = information_matrix(p_j, p)
    J = -joint_entr_matrix(p_j, p)

D = 1 - I / J

# Visualize mutual information
plot = True
if plot:
    to_plot = D
    plt.imshow(to_plot, interpolation='nearest', cmap='hot')
    plt.colorbar()
    plt.show()
