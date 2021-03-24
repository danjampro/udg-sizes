from deepscan import sersic

ABSMAG_SUN = 4.65


def calculate_logml(uae, rec, logmstar, redshift, cosmo):
    """ Return log10 (ML ratio)
    """
    # Calculate apparent magnitude
    mag = sersic.meanSB2mag(uae=uae, re=rec, q=1)

    # Calculate abs mag
    mag_abs = mag - cosmo.distmod(redshift).value

    # Calculate ML ratio
    return 0.4 * (mag_abs - ABSMAG_SUN) + logmstar


def calculate_logml_taylor(uae, rec, logmstar, redshift, cosmo):
    """ Return log10 (ML ratio)
    """
    # Calculate apparent magnitude
    mag = sersic.meanSB2mag(uae=uae, re=rec, q=1)

    # Calculate abs mag
    mag_abs = mag - cosmo.distmod(redshift).value

    # Calculate ML ratio
    return 0.4 * mag_abs + logmstar


def logmstar_to_absmag(logmstar, logml):
    """
    """
    return ABSMAG_SUN - 2.5 * (logmstar - logml)


def logmstar_to_absmag_taylor(logmstar, logml):
    """
    """
    return - 2.5 * (logmstar - logml)
