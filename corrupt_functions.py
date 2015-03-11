import numpy as np
from copy import deepcopy


def distort_binnary(pattern_number, percentage_to_corrupt, X):
    """
    Ths functions takes the pattern number (index) and returns both
    the pattern as it is distorted version (specificied as a percentage)

    arguments:
    pattern_numnber: The label -index- of the image
    percentage: specifies how much the pattern will be
    corrupt (0.5 is 50 per cent)
    """
    image_size = X.shape[1]

    # Take the patterns

    dis_pattern = deepcopy(X[pattern_number])

    N_to_corrupt = int(percentage_to_corrupt * image_size)
    index_set = np.arange(0, image_size)
    corrupt_index = np.random.choice(index_set, N_to_corrupt, replace=False)

    dis_pattern[corrupt_index] = 1 - dis_pattern[corrupt_index]

    return dis_pattern
