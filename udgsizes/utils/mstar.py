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

    def calculate_uae(self, logmstar, rec, redshift):
        """
        """
        # Calculate apparent magnitude
        mag = self._absmag_sun - 2.5 * (logmstar - np.log10(self._mlratio))
        mag += self._cosmo.distmod(redshift).value
        # Apply k-correction
        mag += self._dimmer(redshift)
        # Calculate apparent surface brightness
        uae = sersic.mag2meanSB(mag, re=rec, q=1)
        return uae
