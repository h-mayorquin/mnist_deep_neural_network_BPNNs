import numpy as np
import matplotlib.pyplot as plt
from load_files import training_ims, training_labels

# Visualize an example just for testing
label = training_labels[0]
image = training_ims[0]
image = np.array(image)
image = np.reshape(image, (28, 28))

plt.imshow(image)
plt.show()
