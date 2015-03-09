import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from load_binary_files import training_ims, training_labels

# Select quantity of data to use
N_data_total = len(training_labels)
percentage = 0.05
N_to_use = int(percentage * N_data_total)

# Decide how much data to use
X = training_ims[0:N_to_use]
Y = training_labels[0:N_to_use]
image_size = X.shape[1]

# Now let's distort a pattern
pattern = 3
percentage_to_corrupt = 0.1
N_to_corrupt = int(percentage_to_corrupt * image_size)
index_set = np.arange(0, image_size)
corrup_index = np.random.choice(index_set, N_to_corrupt, replace=False)
stable = deepcopy(X[pattern])
dis_pattern = X[pattern]
dis_pattern[corrup_index] = 1 - dis_pattern[corrup_index]

plt.subplot(1, 2, 1)
plt.imshow(stable.reshape((28, 28)))

plt.subplot(1, 2, 2)
plt.imshow(dis_pattern.reshape((28, 28)))
plt.show()


def distort_binnary(pattern_number, percentage_to_corrupt, X):
    """
    Ths functions takes the pattern number (index) and returns both
    the pattern as it is distorted version (specificied as a percentage)

    arguments:
    pattern_numnber: The label -index- of the image
    percentage: specifies how much the pattern will be corrupt
    """
    N_to_corrupt = int(percentage_to_corrupt * image_size)
    index_set = np.arange(0, image_size)
    corrup_index = np.random.choice(index_set, N_to_corrupt, replace=False)
    stable = deepcopy(X[pattern_number])
    dis_pattern = X[pattern_number]
    dis_pattern[corrup_index] = 1 - dis_pattern[corrup_index]

    return dis_pattern, stable
