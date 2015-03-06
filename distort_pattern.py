import numpy as np
import matplotlib.pyplot as plt
from load_files import training_ims, training_labels
    

# Transform everything into array
training_ims = np.array(training_ims)
training_labels = np.array(training_labels)

# Select quantity of data to use
N_data_total = len(training_labels)
percentage = 0.05
N_to_use = int(percentage * N_data_total)

size = 28

pattern = training_ims[0].reshape((size, size))

plt.subplot(2, 1, 1)
plt.imshow(pattern)

distortion = np.random.randint(0, 256, pattern.shape)
distorted_pattern = pattern + distortion

plt.subplot(2, 1, 2)
plt.imshow(distorted_pattern)

plt.show()
