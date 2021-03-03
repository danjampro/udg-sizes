# https://ui.adsabs.harvard.edu/abs/2003MNRAS.343..978S/abstract

M0 = 3.98E+10
GAMMA = 0.1
ALPHA = 0.14
BETA = 0.39
SIGMA_1 = 0.47
SIGMA_2 = 0.34


def logmstar_to_mean_rec(logmstar):
    mstar = 10 ** logmstar
    return GAMMA * mstar ** ALPHA * (1 + (mstar / M0)) ** (BETA - ALPHA)
