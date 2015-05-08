import numpy as np
import matplotlib.pyplot as plt
from load_binary_files import training_ims, training_labels
from prob_functions import joint, p_independent
from information_functions import entropy, joint_entropy, mutual_information
from information_functions import mutual_information2

# Select quantity of data to use
N_data_total = len(training_labels)
percentage = 0.1
N_to_use = int(percentage * N_data_total)

# Decide how much data to use
X = training_ims[0:N_to_use]
Y = training_labels[0:N_to_use]
N_hypercolumns = X.shape[1]
units_per_hypercolumn = 2
distribution = {0: 0, 1: 1, 'no': 3}

# Normalize
normalize = True
low_noise = 10e-10
# Coolective
collective = True
N1 = 28 * 1
N2 = N1 + 28 * 3
N1 = 0
N2 = N_hypercolumns

# Single example
i = 0
j = 0

p = p_independent(N_hypercolumns, units_per_hypercolumn, X)
p_i = p[i, :]
p_j = p[j, :]
p_joint = joint(i, j, X, distribution, units_per_hypercolumn)


if normalize:
    # Joint
    p_joint[p_joint < low_noise] = low_noise
    p_joint = p_joint / p_joint.sum()
    p_i = p_joint.sum(axis=1)
    p_j = p_joint.sum(axis=0)

# Calculate the entropies
x1 = entropy(p_i)
x2 = entropy(p_j)
x3 = joint_entropy(p_joint)

MI = mutual_information(p_i, p_j, p_joint)
MI2 = mutual_information2(p_i, p_j, p_joint)
MI_alt = x1 + x2 - x3

D1 = 1 - MI / x3
D2 = 1 - MI2 / x3
print 'MI', MI
print 'MI2', MI2
print 'MI_alt', MI_alt
print np.isclose(MI, MI2)
print 'distances 1, 2', D1, D2

if collective:
    # Values fo the map

    d_matrix = np.zeros((N_hypercolumns, N_hypercolumns))
    d_matrix2 = np.zeros((N_hypercolumns, N_hypercolumns))
    d_matrix3 = np.zeros((N_hypercolumns, N_hypercolumns))
    works = np.zeros((N_hypercolumns, N_hypercolumns))
    je = np.zeros((N_hypercolumns, N_hypercolumns))

    domain = range(N1, N2)

    for i in domain:
        print 'i', i
        for j in domain:
            p = p_independent(N_hypercolumns, units_per_hypercolumn, X)
            p_i = p[i, :]
            p_j = p[j, :]
            p_joint = joint(i, j, X, distribution, units_per_hypercolumn)

            if np.any(p_joint == 0):
                # Joint
                p_joint[p_joint < low_noise] = low_noise
                p_joint = p_joint / p_joint.sum()
                p_i = p_joint.sum(axis=1)
                p_j = p_joint.sum(axis=0)
                
            x1 = entropy(p_i)
            x2 = entropy(p_j)
            x3 = joint_entropy(p_joint)

            MI = mutual_information(p_i, p_j, p_joint)
            MI2 = mutual_information2(p_i, p_j, p_joint)

            D1 = 1 - MI / x3
            D2 = 1 - MI2 / x3

            D3 = x3 - MI

            d_matrix[i, j] = D1
            d_matrix2[i, j] = D2
            d_matrix3[i, j] = D3

            works[i, j] = np.isclose(MI, MI2)
            je[i, j] = x3

    plt.subplot(1, 3, 1)
    plt.imshow(d_matrix[N1:N2, N1:N2], interpolation='nearest')
    plt.title('Distance 1')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(d_matrix2[N1:N2, N1:N2], interpolation='nearest')
    plt.colorbar()
    plt.title('Distance 2')

    plt.subplot(1, 3, 3)
    plt.imshow(d_matrix3[N1:N2, N1:N2], interpolation='nearest')
    plt.colorbar()
    plt.title('Alternative Distance')

    plt.show()

# Here we save the data

folder = './data/'
name = 'information_distances'
file_name = folder + name + str(percentage)
np.save(file_name, d_matrix3)
