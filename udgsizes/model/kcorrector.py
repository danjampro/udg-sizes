import numpy as np
from scipy.interpolate import NearestNDInterpolator

from udgsizes.base import UdgSizesBase
from udgsizes.utils import kcorrect


class KCorrector(UdgSizesBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate_kr(self, colour_rest, redshift, **kwargs):
        """ Return the r-band k-correction in magnitudes. """
        raise NotImplementedError

    def calculate_kgr(self, colour_rest, redshift, **kwargs):
        """ Return the (g-r) colour k-correction in magnitudes. """
        raise NotImplementedError


class EmpiricalKCorrector(KCorrector):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._redshift_min = 0
        self._redshift_max = 1
        self._colour_obs_min = -0.5
        self._colour_obs_max = 1.5
        self._n_samples = 100

        colour_obs = np.linspace(self._colour_obs_min, self._colour_obs_max, self._n_samples,
                                 dtype="float32")
        redshift = np.linspace(self._redshift_min, self._redshift_max, self._n_samples,
                               dtype="float32")
        points_obs = np.vstack([colour_obs, redshift]).T

        # Calculate k-corrections from grid of observable quantities
        self._kr = self._calculate_kr(points_obs)
        self._kg = self._calculate_kg(points_obs)

        # Calculate rest frame colours from observable quantities and k-corrections
        colour_rest = colour_obs + self._kg - self._kr

        self._points = np.vstack([colour_rest, redshift]).T
        self._kr_interp = NearestNDInterpolator(self._points, self._kr)
        self._kg_interp = NearestNDInterpolator(self._points, self._kg)

    # Public methods

    def calculate_kr(self, colour_rest, redshift):
        """
        """
        return self._kr_interp([colour_rest, redshift])[0]

    def calculate_kgr(self, colour_rest, redshift):
        """
        """
        point = [colour_rest, redshift]
        return self._kg_interp(point)[0] - self._kr_interp(point)[0]

    # Private methods

    def _calculate_kr(self, points_obs):
        """
        """
        return np.array([kcorrect.k_gr_g(gr, redshift=z) for z, gr in points_obs])

    def _calculate_kg(self, points_obs):
        """
        """
        return np.array([kcorrect.k_gr_r(gr, redshift=z) for z, gr in points_obs])
