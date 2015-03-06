import numpy as np
from load_files import training_ims
import time

# Transform everything into array
training_ims = np.array(training_ims)

# Select quantity of data to use
N_data_total = training_ims.shape[0]
percentage = 0.08
N_to_use = int(percentage * N_data_total)

# Decide how much data to use
X = training_ims[0:N_to_use]

# First we need to calculate the probabilities w and beta


def p_matrix_1(X):
    """
    Calculates the outer product of the matrix with a loop
    """

    average = 0
    for index, pattern in enumerate(X):
        outter = np.outer(pattern, pattern)
        average += outter

    p_matrix = average * 1.0 / X.shape[0]
    return p_matrix


def p_matrix_2(X):
    """
    Calculates the outer product of a function with broadcasting
    """

    average = 0
    for index, pattern in enumerate(X):
        outter = pattern[:, np.newaxis] * pattern[np.newaxis, :]
        average += outter

    p_matrix = average * 1.0 / X.shape[0]
    
    return p_matrix


def p_matrix_3(X):
    """
    Calculating the outer product using vectorization
    """
    return np.mean(X[:, :, np.newaxis] * X[:, np.newaxis, :], axis=0)


# Now we profile
start = time.time()
p_matrix_1(X)
end = time.time()
print end - start

start = time.time()
p_matrix_2(X)
end = time.time()
print end - start

start = time.time()
p_matrix_3(X)
end = time.time()
print end - start


