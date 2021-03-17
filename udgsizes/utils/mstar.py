import numpy as np
from deepscan import sersic

from udgsizes.base import UdgSizesBase
from udgsizes.utils.dimming import SBDimming


class SbCalculator(UdgSizesBase):
    """
    """

    def __init__(self, population_name, mlratio, absmag_sun=4.65, cosmo=None, **kwargs):
        super().__init__()
        self._pop_name = population_name
        self._mlratio = mlratio
        self._absmag_sun = absmag_sun
        if cosmo is None:
            cosmo = self.config["cosmology"]
        self._cosmo = cosmo
        self._dimmer = SBDimming(population_name=population_name, **kwargs)

    def logmstar_from_uae_phys(self, uae_phys, rec, redshift):
        """
        """
        mag = sersic.meanSB2mag(uae_phys, re=rec, q=1)
        absmag = mag - self._cosmo.distmod(redshift).value
        logmstar = (self._absmag_sun - absmag) / 2.5 + np.log10(self._mlratio)
        return logmstar

    def calculate_uae_phys(self, logmstar, rec, redshift):
        """
        """
        # Calculate apparent magnitude
        mag = self._absmag_sun - 2.5 * (logmstar - np.log10(self._mlratio))
        mag += self._cosmo.distmod(redshift).value
        # Calculate apparent surface brightness
        return sersic.mag2meanSB(mag, re=rec, q=1)

    def calculate_uae(self, logmstar, rec, redshift):
        """
        """
        # Calculate apparent magnitude
        uae_phys = self.calculate_uae_phys(logmstar, rec, redshift)
        # Apply k-correction
        uae = uae_phys + self._dimmer(redshift)
        return uae


class EmpiricalSBCalculator
