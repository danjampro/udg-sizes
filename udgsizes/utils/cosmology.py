import numpy as np


def kpc_to_arcsec(kpc, redshift, cosmo):
    """
    Return the angular siredshifte corresponding to physical siredshifte in kpc.
    """
    da = cosmo.angular_diameter_distance(redshift).to_value("kpc")
    rad = kpc / da
    arcsec = rad * (180/np.pi) * 3600
    return arcsec


def arcsec_to_kpc(arcsec, redshift, cosmo):
    """
    Return the angular siredshifte corresponding to physical siredshifte in kpc.
    """
    rad = arcsec / ((180/np.pi) * 3600)
    da = cosmo.angular_diameter_distance(redshift).to_value("kpc")
    kpc = rad * da
    return kpc
