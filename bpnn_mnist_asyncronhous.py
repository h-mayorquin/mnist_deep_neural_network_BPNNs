import numpy as np
import matplotlib.pyplot as plt
from load_files import training_ims, training_labels
from init_functions import p_matrix, w_connectivity
    

# Transform everything into array
training_ims = np.array(training_ims)
training_labels = np.array(training_labels)

# Select quantity of data to use
N_data_total = len(training_labels)
percentage = 0.05
N_to_use = int(percentage * N_data_total)

# Decide how much data to use
X = training_ims[0:N_to_use]
Y = training_labels[0:N_to_use]

# First we need to calculate the probabilities w and beta
p_vector = np.mean(X, axis=0)
p_matrix = p_matrix(X)

beta = p_vector
beta[beta == 0] = 1.0 / N_to_use ** 2
w = w_connectivity(p_vector, p_matrix, N_to_use)

# Now we create a distorted pattern
pattern = 3
distorted_pattern = X[pattern] + np.random.randint(0, 256, len(X[pattern]))

# Now we need to update the pattern
o = distorted_pattern.astype('float')
s = np.zeros_like(o)

log_beta = np.log(beta)
G = 0.01

iterations = 100

for t in range(iterations):
    print t
    for index, element in enumerate(o):
        aux_2 = np.log(np.dot(w, o))[index]
        s[index] = log_beta[index] + aux_2
        aux1 = np.sum(np.exp(G * s))
        print 'denominator', aux1
        aux_3 = np.exp(G * s[index]) * 1.0 / aux1
        print 'o value', aux_3
        o[index] = aux_3

# Print the shit out of it
plt.subplot(1, 3, 1)
plt.imshow(X[pattern].reshape((28, 28)))
plt.title('Original')

plt.subplot(1, 3, 2)
plt.imshow(o.reshape((28, 28)))
plt.title('O')

plt.subplot(1, 3, 3)
plt.imshow(s.reshape((28, 28)))
plt.title('S')

plt.show()
