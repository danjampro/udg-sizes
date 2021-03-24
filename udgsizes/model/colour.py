import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic, norm
import matplotlib.pyplot as plt

from astropy.stats import sigma_clipped_stats

from udgsizes.base import UdgSizesBase
from udgsizes.obs.sample import load_gama_masses, load_leisman_udgs

COLOUR_MEAS_ERROR = 0.06  # Fiducial colour measurement error from GAMA
COLOUR_MIN = 0.35  # Minimum average rest frame colour based on known late type dwarfs


class ColourModel(UdgSizesBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def clipped_median(data):
    return sigma_clipped_stats(data)[1]


def clipped_std(data):
    return sigma_clipped_stats(data)[2]


class EmpiricalColourModel(ColourModel):

    def __init__(self, bins=9, logmstar_min=6.5, logmstar_max=11, lambdar=False, use_leisman=False,
                 **kwargs):
        super().__init__(**kwargs)

        self._lambdar = lambdar

        dfg = load_gama_masses(lambdar=self._lambdar)
        colour = dfg["gr"].values
        logmstar = dfg["logmstar"].values

        cond_leisman = np.zeros_like(colour, dtype="bool")

        dfl = load_leisman_udgs()
        colour = np.hstack([colour, dfl["gr"].values])
        logmstar = np.hstack([logmstar, dfl["logmstar"].values])
        cond_leisman = np.hstack([cond_leisman, np.ones(dfl.shape[0], dtype="bool")])

        cond = (logmstar >= logmstar_min) & (logmstar < logmstar_max)
        self._logmstar = logmstar[cond]
        self._colour = colour[cond]
        self._cond_leisman = cond_leisman[cond]

        cond_fit = np.ones_like(self._cond_leisman)
        if use_leisman:
            cond_fit[self._cond_leisman] = False

        histkwargs = {"bins": bins, "range": (logmstar_min, logmstar_max)}
        self._means, edges, _ = binned_statistic(self._logmstar[cond_fit], self._colour[cond_fit],
                                                 statistic=clipped_median, **histkwargs)
        self._stds, edges, _ = binned_statistic(self._logmstar[cond_fit], self._colour[cond_fit],
                                                statistic=clipped_std, **histkwargs)

        # Calculate intrinsic dispersion given total and measurement error
        self.sigma = np.sqrt(self._stds.mean() ** 2 - COLOUR_MEAS_ERROR ** 2)

        self._centres = 0.5 * (edges[1:] + edges[:-1])

        self._interp = interp1d(self._centres, self._means, fill_value="extrapolate")

        self.offset_pdf = norm(loc=0, scale=self.sigma).pdf

    def get_mean_colour_rest(self, logmstar):
        """ """
        colour = self._interp(logmstar)
        return max(colour, COLOUR_MIN)

    def summary_plot(self):
        """
        """
        df = load_gama_masses(lambdar=self._lambdar)
        logmstar = df["logmstar"].values

        fig, ax = plt.subplots()
        ax.hist2d(logmstar, df["gr"].values, cmap="binary", bins=70)

        ax.plot(self._centres, self._means, "ro")
        ax.plot(self._centres, self._means + self._stds, "r--")
        ax.plot(self._centres, self._means - self._stds, "r--")

        xx = np.linspace(logmstar.min(), logmstar.max(), 100)
        ax.plot(xx, [self.get_mean_colour_rest(_) for _ in xx], "b-")
        ax.plot(xx, [self.get_mean_colour_rest(_) + self.sigma for _ in xx], "b--")
        ax.plot(xx, [self.get_mean_colour_rest(_) - self.sigma for _ in xx], "b--")

        cond = (self._logmstar < 8.5) & (self._cond_leisman == 0)
        ax.plot(self._logmstar[cond], self._colour[cond], "ko", markersize=0.5, alpha=0.2)
        cond = (self._logmstar < 8.5) & (self._cond_leisman == 1)
        ax.plot(self._logmstar[cond], self._colour[cond], "bo", markersize=1)

        ax.set_xlim(6, 12)
        ax.set_ylim(-0.1, 1)

        plt.show(block=False)
