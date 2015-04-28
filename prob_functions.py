import numpy as np


def coincidences(i, j, k, l, X, distribution):
    """
    Caculates the number of coincidences between two distributions
    """
    hits_1 = X[..., i] == distribution[k]
    hits_2 = X[..., j] == distribution[l]

    return np.sum(hits_1 * hits_2)


def joint(i, j, X, distribution, units_per_hypercolumn):
    """
    This and that
    """
    hypercolums = range(units_per_hypercolumn)
    joint_probability = np.zeros((units_per_hypercolumn,
                                  units_per_hypercolumn))

    for k in hypercolums:
        for l in hypercolums:
            joint_probability[k, l] = coincidences(i, j, k, l, X, distribution)

    return joint_probability * 1.0 / X.shape[0]


def p_joint(N_hypercolumns, units_per_hypercolumn, X, distribution):
    """
    This and that
    """
    p_joint = np.zeros((N_hypercolumns, N_hypercolumns,
                        units_per_hypercolumn, units_per_hypercolumn))

    for i in xrange(N_hypercolumns):
        for j in xrange(i, N_hypercolumns):
            aux = joint(i, j, X, distribution,
                        units_per_hypercolumn)

            p_joint[i, j, ...] = aux
            p_joint[j, i, ...] = aux.T

    return p_joint
