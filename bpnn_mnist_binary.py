import numpy as np
import matplotlib.pyplot as plt
from load_binary_files import training_ims, training_labels
from init_functions import p_matrix, w_connectivity
from corrupt import distort_binnary
    
# Select quantity of data to use
N_data_total = len(training_labels)
percentage = 0.05
N_to_use = int(percentage * N_data_total)

# Decide how much data to use
X = training_ims[0:N_to_use]
Y = training_labels[0:N_to_use]
image_size = X.shape[1]

p_vector_1 = np.mean(X, axis=0)
p_vector_0 = np.mean(1 - X, axis=0)

p_matrix_1 = p_matrix(X)
p_matrix_0 = p_matrix(1 - X)

beta_1 = p_vector_1
beta_1[beta_1 == 0] = 1.0 / N_to_use ** 2
beta_0 = p_vector_0
beta_0[beta_0 == 0] = 1.0 / N_to_use ** 2

w_1 = w_connectivity(p_vector_1, p_matrix_1, N_to_use)
w_0 = w_connectivity(p_vector_0, p_matrix_0, N_to_use)

# Now let's distort a pattern
percentage = 0.1
index = 1

dis_pattern, pattern = distort_binnary(index, percentage, X)

log_beta_1 = np.log(beta_1)
log_beta_0 = np.log(beta_0)
G = 1.1

o_1 = dis_pattern
o_0 = dis_pattern

iterations = 1000

for t in range(iterations):
    s_1 = log_beta_1 + np.log(np.dot(w_1, o_1))
    s_0 = log_beta_0 + np.log(np.dot(w_0, o_0))
    o_1 = np.exp(G * s_1)
    o_0 = np.exp(G * s_0)
    o_1 *= 1.0 / (o_1 + o_0)
    o_0 *= 1.0 / (o_1 + o_0)


plt.subplot(1, 3, 1)
plt.imshow(pattern.reshape((28, 28)))
plt.title('Original')16  * 16  

plt.subplot(1, 3, 2)
plt.imshow(o_0.reshape((28, 28)))
plt.title('O_1')

plt.subplot(1, 3, 3)
plt.imshow(o_1.reshape((28, 28)))
plt.title('O_0')

plt.show()


