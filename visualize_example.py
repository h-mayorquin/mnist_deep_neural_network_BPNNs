import numpy as np
import matplotlib.pyplot as plt
from load_files import training_ims, training_labels

# Visualize an example just for testing
index = 14

label = training_labels[1]
image = training_ims[1]
image = np.array(image)
image = np.reshape(image, (28, 28))


image[image > 0] = 1

plt.imshow(image)
plt.title('this should be the number = ' + str(label))
plt.show()
