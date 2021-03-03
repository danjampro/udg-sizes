import numpy as np

LOG10 = np.log(10)


def schechter(logmstar, a, phi, logm0, min):
    if logmstar <= min:
        return 0
    A = LOG10 * phi
    B = (10 ** (logmstar - logm0)) ** (a + 1)
    C = np.exp(-10 ** (logmstar-logm0))
    return A*B*C


def schechter_baldry(logmstar, a, phi, logm0, min, cosmo):
    logmstar = logmstar + 2 * np.log10(cosmo.H0.value/70)
    return schechter(logmstar, a, phi, logm0, min=min)
