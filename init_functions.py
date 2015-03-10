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


def connectivity(N_hypercolumns, units_per_hypercolumn, X):
    """
    Calculates the connectivity matrix for a network with
    N_hypercolumns number of hypercolumns and neurons_per_hyper
    neurons per hypercolumn. It returns a tensor where the first
    two dimensions correspond to the hyercolumn and the last two
    dimensions correspond to the neurons in the hypercolum
    """

    w = np.zeros((N_hypercolumns, N_hypercolumns,
                  units_per_hypercolumn, units_per_hypercolumn))

    for h_pre in range(N_hypercolumns):
        for h_post in range(h_pre + 1, N_hypercolumns):

            aux = hypercolumn_connection(h_pre, h_post,
                                         units_per_hypercolumn, X)

            w[h_pre, h_post, ...] = aux
            w[h_post, h_pre, ...] = aux

    return w


def hypercolumn_connection(h_pre, h_post, units_per_hypercolumn, X):
    """
    Creates the connection between two hypercolumns, in this case
    fromm the h_pre hypercolumn to the h_post hypercolumn. Neurons
    hyper is the number of features or units per hypercolum
    """

    h_connectivity = np.zeros((units_per_hypercolumn, units_per_hypercolumn))

    for n_i in range(units_per_hypercolumn):
        for n_j in range(units_per_hypercolumn):
            h_connectivity[n_i, n_j] = unitary_connection(h_pre, h_post,
                                                          n_i, n_j, X)

    return h_connectivity


def unitary_connection(h_pre, h_post, n_pre, n_post, X):
    """
    Gives the connectivity value between the n_pre unit
    in the h_pre hypercolumn and the n_post unit in the h_post column
    """

    hits_pre = X[:, h_pre] == n_pre
    hits_post = X[:, h_post] == n_post

    return np.sum(hits_pre * hits_post)
