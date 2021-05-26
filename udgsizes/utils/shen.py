# https://ui.adsabs.harvard.edu/abs/2003MNRAS.343..978S/abstract
import numpy as np

M0 = 3.98E+10
GAMMA = 0.1
ALPHA = 0.14
BETA = 0.39
SIGMA_1 = 0.47
SIGMA_2 = 0.34


def logmstar_to_mean_rec(logmstar):
    mstar = 10 ** logmstar
    return GAMMA * mstar ** ALPHA * (1 + (mstar / M0)) ** (BETA - ALPHA)


def logmstar_sigma(logmstar):
    """ Return the natural log-normal dispersion parameter in the stellar mass - size relation.
    Equation 19 from Shen et al. (2003).
    Args:
        logmstar (float): The log10 stellar mass.
    Returns:
        float: The dispersion parameter.
    """
    mstar = 10 ** logmstar
    norm = 1 + (mstar / M0) ** 2
    return SIGMA_2 + (SIGMA_1 - SIGMA_2) / norm


def apply_rec_offset(rec_phys_mean, rec_phys_offset):
    return np.exp(np.log(rec_phys_mean) + rec_phys_offset)
