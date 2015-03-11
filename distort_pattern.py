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

index = 0
size = 28
percentage = 0.1
pattern = deepcopy(X[index])
dis_pattern = distort_binnary(index, percentage, X)

plt.subplot(1, 3, 1)
plt.title('Corrupted')
plt.imshow(dis_pattern.reshape((size, size)))
plt.subplot(1, 3, 2)
plt.title('Normal pattern')
plt.imshow(pattern.reshape((size, size)))

plt.subplot(1, 3, 3)
plt.title('Normal should be')
plt.imshow(X[index].reshape((size, size)))

plt.show()
