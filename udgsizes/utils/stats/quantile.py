import numpy as np


def quantile_threshold(values, q):
    """ Used for identifying models within a confidence interval defined by q. """
    values_sorted = values.copy().reshape(-1)
    values_sorted.sort()
    values_sorted = values_sorted[::-1]
    csum = np.cumsum(values_sorted)
    total = csum[-1]
    idx = np.argmin(abs(csum - q*total))
    return values_sorted[idx]
