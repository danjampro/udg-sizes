import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator

from udgsizes.base import UdgSizesBase
from udgsizes.utils import kcorrect
from udgsizes.obs.sample import load_gama_masses


class KCorrector(UdgSizesBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate_kr(self, colour_rest, redshift, **kwargs):
        """ Return the r-band k-correction in magnitudes. """
        raise NotImplementedError

    def calculate_kgr(self, colour_rest, redshift, **kwargs):
        """ Return the (g-r) colour k-correction in magnitudes. """
        raise NotImplementedError


class DummyKcorrector(KCorrector):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate_kr(self, *args, **kwargs):
        """ Return the r-band k-correction in magnitudes. """
        return 0

    def calculate_kgr(self, *args, **kwargs):
        """ Return the (g-r) colour k-correction in magnitudes. """
        return 0


class EmpiricalKCorrector(KCorrector):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._colour_obs_min = -0.5
        self._colour_obs_max = 1.5
        self._n_samples = 200

        points_obs = self._get_points_obs()
        cc_obs = points_obs[:, 0]
        zz = points_obs[:, 1]

        # Calculate k-corrections from grid of observable quantities
        self._kr = self._calculate_kr(points_obs)
        self._kg = self._calculate_kg(points_obs)

        # Calculate rest frame colours from observable quantities and k-corrections
        colour_rest = cc_obs - self._kg + self._kr

        self._points = np.vstack([colour_rest, zz]).T
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
        return np.array([kcorrect.k_gr_r(gr, redshift=z) for gr, z in points_obs])

    def _calculate_kg(self, points_obs):
        """
        """
        return np.array([kcorrect.k_gr_g(gr, redshift=z) for gr, z in points_obs])

    def _get_points_obs(self):
        """
        """
        colour_obs = np.linspace(self._colour_obs_min, self._colour_obs_max, self._n_samples,
                                 dtype="float32")
        redshift = np.linspace(self._zmin, self._zmax, self._n_samples, dtype="float32")
        cc_obs, zz = np.meshgrid(colour_obs, redshift)
        cc_obs = cc_obs.reshape(-1)
        zz = zz.reshape(-1)

        points_obs = np.vstack([cc_obs, zz]).T

        return points_obs


class InvEmpiricalKCorrector(EmpiricalKCorrector):
    """ Calculate k-corrections on the assumption that the difference between the k-correction
    implied using the rest-frame colour as the observed colour is small.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate_kr(self, colour_rest, redshift):
        """
        """
        return self._calculate_kr([[colour_rest, redshift]])[0]

    def calculate_kgr(self, colour_rest, redshift):
        """
        """
        point = [colour_rest, redshift]
        return self._calculate_kg([point])[0] - self._calculate_kr([point])[0]


class GamaKCorrector(KCorrector):
    """ Calculate k-corrections based on mean relation of GAMA galaxies with redshift. """

    def __init__(self, lambdar=True, zmax=0.4, bins_z=20, bins_gr=4, gr_min=-0.1, gr_max=1.0,
                 **kwargs):
        super().__init__(**kwargs)

        self._lambdar = lambdar
        self._zmax = zmax
        self._gr_min = gr_min
        self._gr_max = gr_max

        dfg = load_gama_masses(lambdar=lambdar, z_max=zmax)
        self._redshift = dfg["redshift"].values
        self._gr_rest = dfg["gr"].values
        self._kr = dfg["kcorr_r"].values
        self._kg = dfg["kcorr_g"].values

        cond = (self._redshift >= self._zmin) & (self._redshift < self._zmax)
        cond &= (self._gr_rest >= self._gr_min) & (self._gr_rest < self._gr_max)

        self._redshift = self._redshift[cond]
        self._gr_rest = self._gr_rest[cond]
        self._kr = self._kr[cond]
        self._kg = self._kg[cond]

        histkwargs = {"bins": (bins_z, bins_gr), "x": self._redshift, "y": self._gr_rest,
                      "range": ((self._zmin, self._zmax), (self._gr_min, self._gr_max)),
                      "statistic": "median"}

        self._kg_av, e1, e2, _ = binned_statistic_2d(values=self._kg, **histkwargs)
        self._kr_av, e1, e2, _ = binned_statistic_2d(values=self._kr, **histkwargs)

        c1 = 0.5 * (e1[1:] + e1[:-1])
        c2 = 0.5 * (e2[1:] + e2[:-1])

        yy, xx = np.meshgrid(c2, c1)
        points = np.vstack([xx.reshape(-1), yy.reshape(-1)]).T

        self._interp_kg = LinearNDInterpolator(points, self._kg_av.reshape(-1), fill_value=0)
        self._interp_kr = LinearNDInterpolator(points, self._kr_av.reshape(-1), fill_value=0)

    def calculate_kr(self, colour_rest, redshift):
        """
        """
        return self._interp_kr([redshift, colour_rest])[0]

    def calculate_kgr(self, colour_rest, redshift):
        """
        """
        point = [redshift, colour_rest]
        return self._interp_kg(point)[0] - self._interp_kr(point)[0]

    def summary_plot(self):

        fig, ax = plt.subplots()

        ax.plot(self._redshift, self._kg - self._kr, "k+", markersize=1)

        zz = np.linspace(0, self._zmax, 50)
        for gr in (0.1, 0.2, 0.4, 0.6):
            kgr = [self.calculate_kgr(gr, z) for z in zz]
            ax.plot(zz, kgr, "-")

        plt.show(block=False)
