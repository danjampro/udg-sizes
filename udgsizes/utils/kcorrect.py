""" From Chilingarian + 12, http://kcor.sai.msu.ru/getthecode/"""

coeffs_gr_g = [[0, 0, 0, 0],
               [-2.45204, 4.10188, 10.5258, -13.5889],
               [56.7969, -140.913, 144.572, 57.2155],
               [-466.949, 222.789, -917.46, -78.0591],
               [2906.77, 1500.8, 1689.97, 30.889],
               [-10453.7, -4419.56, -1011.01, 0],
               [17568, 3236.68, 0, 0],
               [-10820.7, 0, 0, 0]]

coeffs_gr_r = [[0, 0, 0, 0],
               [1.83285, -2.71446, 4.97336, -3.66864],
               [-19.7595, 10.5033, 18.8196, 6.07785],
               [33.6059, -120.713, -49.299, 0],
               [144.371, 216.453, 0, 0],
               [-295.39, 0, 0, 0]]


def _calculate(coeffs, colour_obs, redshift):
    """
    """
    kcor = 0.0
    for x, a in enumerate(coeffs):
        for y, b in enumerate(coeffs[x]):
            kcor += coeffs[x][y] * redshift ** x * colour_obs ** y
    return kcor


def k_gr_g(gr_obs, redshift):
    """
    Return the K-correction for the g-band based on g-r color.
    """
    return _calculate(coeffs_gr_g, gr_obs, redshift)


def k_gr_r(gr_obs, redshift):
    """
    Return the K-correction for the r-band based on g-r color.
    """
    return _calculate(coeffs_gr_r, gr_obs, redshift)
