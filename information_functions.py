import numpy as np


def entropy(prob_dist):
    """
    Calculates the entropy of a distribution

    Arguments
    -----------------------
    prob_dist (numpy array):
    A probability distribution (prob_dist.sum() = 1)

    Returns
    -----------------------
    entropy (scalar):
    The entropy in bites of the probability distribution

    Examples
    -----------------------
    >>> p_example_1 = np.array((0.5, 0.5))
    >>> entropy(p_example_1)
    1.0
    >>> p_example_2 = np.array((0.1, 0.9))
    >>> entropy(p_example_2)
    0.46899559358928122
    >>> entropy(np.array((0.5, 0.3))) == entropy(np.array((0.3, 0.5)))
    True
    """
    return -np.sum(prob_dist * np.log2(prob_dist))


def joint_entropy(prob_dist):
    """
    Calculates mutual entropy of a prob_dist

    Arguments
    -----------------------------
    prob_dist (numpy array):
    A joint probability distribution (prob_dist.sum() = 1)
    the sum over the first index gives the second prob distribution
    that is, we marginalize the first variable.

    Returns
    -----------------------------
    joint_entropy (scalar):
    The joint entropy of the probability distribution

    Examples
    -----------------------------
    Maximal for uninformative distribution:
    >>> p_joint= np.array((0.25, 0.25, 0.25, 0.25)).reshape((2, 2))
    >>> p1 = p_joint.sum(axis=1)
    >>> p2 = p_joint.sum(axis=0)
    >>> joint_entropy(p_joint)
    2.0

    The joint entropy is bigger thatn the individual entropies:

    >>> joint_entropy(p_joint) > entropy(p1)
    True
    >>> joint_entropy(p_joint) > entropy(p2)
    True

    On the other hand is smaller than the sum of the individual
    entropies:

    >>> joint_entropy(p_joint) <= entropy(p1) + entropy(p2)
    True
    """
    return -np.sum(prob_dist * np.log2(prob_dist))


def mutual_information(prob1, prob2, prob_joint):
    """
    Calculates mutual information between two random variables

    Arguments
    ------------------
    prob1 (numpy array):
    The probability distribution of the first variable
    prob1.sum() should be 1
    prob2 (numpy array):
    The probability distrubiont of the second variable
    Again, prob2.sum() should be 1

    prob_joint (two dimensional numpy array):
    The joint probability, marginazling over the
    different axes should give prob1 and prob2

    Returns
    ------------------
    mutual information (scalar):
    The mutual information between two variables

    Examples
    ------------------
    A mixed joint:
    >>> p_joint = np.array((0.3, 0.1, 0.2, 0.4)).reshape((2, 2))
    >>> p1 = p_joint.sum(axis=1)
    >>> p2 = p_joint.sum(axis=0)
    >>> mutual_information(p1, p2, p_joint)
    0.12451124978365299

    An uninformative joint:
    >>> p_joint = np.array((0.25, 0.25, 0.25, 0.25)).reshape((2, 2))
    >>> p1 = p_joint.sum(axis=1)
    >>> p2 = p_joint.sum(axis=0)
    >>> mutual_information(p1, p2, p_joint)
    0.0

    A very coupled joint:
    >>> p_joint = np.array((0.4, 0.05, 0.05, 0.4)).reshape((2, 2))
    >>> p1 = p_joint.sum(axis=1)
    >>> p2 = p_joint.sum(axis=0)
    >>> mutual_information(p1, p2, p_joint)
    0.58387028280246378

    Using the alternative definition  of mutual information
    >>> p_joint = np.array((0.4, 0.2, 0.1, 0.3)).reshape((2, 2))
    >>> p1 = p_joint.sum(axis=1)
    >>> p2 = p_joint.sum(axis=0)
    >>> MI = mutual_information(p1, p2, p_joint)
    >>> x1 = entropy(p1)
    >>> x2 = entropy(p2)
    >>> x3 = joint_entropy(p_joint)
    >>> np.isclose(MI, x1 + x2 - x3)
    True

    """
    outer = np.outer(prob1, prob2)

    return np.sum(prob_joint * np.log2(prob_joint / outer))


def mutual_information2(prob1, prob2, prob_joint):
    """
    This and that
    """

    x1 = entropy(prob1)
    x2 = entropy(prob2)
    x3 = joint_entropy(prob_joint)

    return x1 + x2 - x3
