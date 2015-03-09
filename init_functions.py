import numpy as np


def p_matrix(X):
    """
    Calculates the outer product of a function with broadcasting
    """
    average = 0
    for index, pattern in enumerate(X):
        outter = np.outer(pattern, pattern)
        average += outter

    p_matrix = average * 1.0 / X.shape[0]

    return p_matrix


def w_connectivity(p_vector, p_matrix, N_to_use):
    """
    Constructs the matrix of connectivity w
    """

    w = np.zeros_like(p_matrix)

    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            if (p_vector[i] == 0) or (p_vector[j] == 0):
                w[i, j] = 0
            elif (p_matrix[i, j] == 0):
                w[i, j] = 1.0 / N_to_use
            else:
                w[i, j] = p_matrix[i, j] / (p_vector[i] * p_vector[j])

    return w
