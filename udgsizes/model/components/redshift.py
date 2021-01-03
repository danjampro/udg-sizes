"""
"""


def mass(z, cosmo):
    """
    """
    mass_density = (1+z)**-3*cosmo.Om(z)*cosmo.critical_density(z).value
    vol_element = cosmo.differential_comoving_volume(z).value
    result = mass_density * vol_element
    return result
