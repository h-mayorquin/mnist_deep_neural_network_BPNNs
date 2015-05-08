import numpy as np
import matplotlib.pyplot as plt
from load_binary_files import training_ims, training_labels
from prob_functions import joint, p_independent
from information_functions import entropy, joint_entropy, mutual_information
from information_functions import mutual_information2

# Select quantity of data to use
N_data_total = len(training_labels)
percentage = 0.001
N_to_use = int(percentage * N_data_total)

# Decide how much data to use
X = training_ims[0:N_to_use]
Y = training_labels[0:N_to_use]
N_hypercolumns = X.shape[1]
units_per_hypercolumn = 2
distribution = {0: 0, 1: 1, 'no': 3}

powers = np.linspace(1, 20, 40)
noises = np.power(10, -powers)
difference = np.zeros(noises.size)

# Normalize
low_noise = 10e-10


# Single example
i = 0
j = 0

p = p_independent(N_hypercolumns, units_per_hypercolumn, X)
p_i = p[i, :]
p_j = p[j, :]
p_joint = joint(i, j, X, distribution, units_per_hypercolumn)
aux = np.copy(p_joint)

if np.any(p_joint == 0):
    # Joint
    p_joint[p_joint < low_noise] = low_noise
    sum = p_joint.sum()
    p_joint = p_joint / sum
    p_i = p_joint.sum(axis=1)
    p_j = p_joint.sum(axis=0)

# Calculate the entropies
x1 = entropy(p_i)
x2 = entropy(p_j)
x3 = joint_entropy(p_joint)

MI = mutual_information(p_i, p_j, p_joint)
MI2 = mutual_information2(p_i, p_j, p_joint)
MI_alt = x1 + x2 - x3

D1 = x3 - MI
D2 = x3 - MI2
print 'MI', MI
print 'MI2', MI2
print 'MI_alt', MI_alt
print np.isclose(MI, MI2)
print 'distances 1, 2', D1, D2
