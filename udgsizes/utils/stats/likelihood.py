import numpy as np


def unlog_likelihood(log_values):
    """
    """
    return np.exp(log_values - np.nanmax(log_values))
