import numpy as np
import matplotlib.pyplot as plt

# Calculate probabilities
directory = './data/'
name_matrix = 'saved_connectivity_matrix.npy'
name_prob = 'saved_connectivity_p.npy'

# Obtain probabilities
w = np.load(directory + name_matrix)
p = np.load(directory + name_prob)


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


def calculate_joint_entr(w, p, i, j):
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
            I[i, j] = calculate_joint_entr(w, p, i, j)

    return I
calculate = True

if calculate:
    I = information_matrix(w, p)
    J = joint_entr_matrix(w, p)

# Visualize mutual information
plot = False
if plot:
    to_plot = I[200:400, 200:400]
    plt.imshow(to_plot, interpolation='nearest', cmap='hot')
    plt.colorbar()
    plt.show()
