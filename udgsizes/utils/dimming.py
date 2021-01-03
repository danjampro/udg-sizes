"""
Use the FSPS package to create interpolated k-corrections assuming stellar
pops of target populations.
"""
import os
import dill as pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import fsps

from udgsizes.base import UdgSizesBase


class SBDimming(UdgSizesBase):
    """ This class is required to quickly calulate surface brightness dimming (in magnitudes)
    based on a simple stellar population model. It calculates the dimming as a function of redshift
    using FSPS and interpolates it so it can be evalulated quickly. The class is callable.
    """

    def __init__(self, population_name, band=None, z0=0.0001, zmax=2, instrument="suprimecam",
                 remake=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.population_name = population_name
        if band is None:
            band = self.config["defaults"]["band"]
        self.band = band

        self._z0 = z0
        self._zmax = zmax
        self._instrument = instrument
        self._pop_config = self.config["dimming"][population_name]
        self._cosmo = self.config["cosmology"]

        path = os.path.join(self.config["directories"]["data"], "sb-dimming")
        basename = f"kcorrect_interp_{self.population_name}_{self.band}.pkl"
        interp_filename = os.path.join(path, basename)

        # Make or load the interpolated dimming model
        if remake:
            self._interp = self._make_interp(interp_filename)
        else:
            try:
                self._interp = self._load_interp(interp_filename)
            except FileNotFoundError:
                self._interp = self._make_interp(interp_filename)

    def __call__(self, redshift):
        """
        """
        try:
            return self._interp(redshift)
        except ValueError:
            return 0

    def make_checkplot(self, ax=None, show=True, xlabel="redshift", ylabel="dimming [mag]",
                       n_samples=100, **kwargs):
        """
        """
        if ax is None:
            fig, ax = plt.subplots()
        zz = np.linspace(self._z0, self._zmax, n_samples)
        ax.plot(zz, self(zz), linestyle='-', color='b', **kwargs)
        ax.plot(zz, 10*np.log10(1+zz), linestyle='--', color='k', **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if show:
            plt.show(block=False)
        return ax

    def _load_interp(self, filename):
        """
        """
        with open(filename, 'rb') as f:
            interp = pickle.load(f)
        return interp

    def _make_interp(self, filename, n_samples=500):
        """
        """
        self.logger.debug(f"Creating new FSPS model for '{self.population_name}' dimming model in"
                          f" {self.band} band.")

        band_name = f"{self._instrument}_{self.band[-1].lower()}"

        # Make stellar population object
        age = self._pop_config['age']
        fsps_config = self._pop_config['fsps']
        sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, **fsps_config)

        # Use SP to calculate projected mags
        zz = np.linspace(self._z0, self._zmax, n_samples)
        mags = np.array([sp.get_mags(tage=age, bands=[band_name], redshift=z)[0] for z in zz])

        # Use reference redshift to calculate magnitude difference
        mag0 = sp.get_mags(tage=age, bands=[band_name], redshift=self._z0)[0]
        magdiff = mags - mag0

        # Calculate the SB dimming by removing contribution from distance modulus
        distmod = self._cosmo.distmod(self._z0) - self._cosmo.distmod(zz)
        dimming = magdiff + distmod.to_value("mag")

        # Make and save the interp
        interp = interp1d(x=zz, y=dimming, bounds_error=True)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(interp, f)

        return interp


class Reddening(UdgSizesBase):
    """ Colours change when stellar pops are redshifted. Typically (but not always) they become
    redder.
    """

    def __init__(self, population_name, band1="HSC-G", band2="HSC-R", **kwargs):
        super().__init__(**kwargs)
        self.bands = [band1, band2]

        self._dimmers = []
        for band in self.bands:
            self._dimmers.append(SBDimming(population_name=population_name, band=band, **kwargs))

    def __call__(self, redshift):
        """
        """
        try:
            return self._dimmers[0](redshift) - self._dimmers[1](redshift)
        except ValueError:
            return 0

    def make_checkplot(self, ax=None, show=True, n_samples=100, z0=0.0001, zmax=2, **kwargs):
        """
        """
        if ax is None:
            fig, ax = plt.subplots()
        zz = np.linspace(z0, zmax, n_samples)
        ax.plot(zz, self(zz), linestyle='-', color='k', **kwargs)
        ax.set_xlabel("redshift")
        ax.set_ylabel(f"{self.bands[0]} - {self.bands[1]} reddening")
        if show:
            plt.show(block=False)
        return ax
