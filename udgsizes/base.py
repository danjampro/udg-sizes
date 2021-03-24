""" Base class for all huntsman-drp classes which provides the default logger and config."""
import numpy as np
from scipy.interpolate import interp1d

from udgsizes.core import get_config, get_logger
from udgsizes.utils import cosmology as cosmo_utils


class UdgSizesBase():

    def __init__(self, config=None, logger=None, zmin=0.0001, zmax=2, z_samples=500):
        """
        """
        self.logger = get_logger() if logger is None else logger
        self.config = get_config() if config is None else config

        self._cosmo = self.config["cosmology"]

        self._zmin = zmin
        self._zmax = zmax
        zz = np.linspace(self._zmin, self._zmax, z_samples)
        self._cosmo_interps = self._make_cosmo_interps(zz)

    def distmod(self, redshift):
        """
        """
        dm = self._cosmo_interps["distmod"](redshift)
        if not hasattr(redshift, "__len__"):
            return float(dm)
        return dm

    def kpc_to_arcsec(self, kpc, redshift):
        """
        """
        return kpc * self._cosmo_interps["arcsec_per_kpc"](redshift)

    def _make_cosmo_interps(self, zz):
        """
        """
        interps = {}

        interps["distmod"] = interp1d(zz, self._cosmo.distmod(zz).to_value("mag"))

        arcsec_per_kpc = cosmo_utils.kpc_to_arcsec(1, zz, cosmo=self._cosmo)
        interps["arcsec_per_kpc"] = interp1d(zz, arcsec_per_kpc)

        return interps
