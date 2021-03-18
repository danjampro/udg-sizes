import numpy as np


def unlog_likelihood(log_values):
    """
    """
    log_values = log_values - np.log(np.exp(log_values).sum())
    return np.exp(log_values - np.nanmax(log_values))
