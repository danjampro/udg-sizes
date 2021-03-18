import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic, norm
import matplotlib.pyplot as plt

from udgsizes.base import UdgSizesBase
from udgsizes.obs.sample import load_gama_masses


class ColourModel(UdgSizesBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class EmpiricalColourModel(ColourModel):

    def __init__(self, bins=10, logmstar_min=8, logmstar_max=11, lambdar=False, **kwargs):
        super().__init__(**kwargs)

        self._lambdar = lambdar

        df = load_gama_masses(lambdar=self._lambdar)

        colour = df["gr"].values
        logmstar = df["logmstar"].values

        cond = (logmstar >= logmstar_min) & (logmstar < logmstar_max)
        logmstar = logmstar[cond]
        colour = colour[cond]

        self._means, edges, _ = binned_statistic(logmstar, colour, bins=bins)
        self._stds, edges, _ = binned_statistic(logmstar, colour, bins=bins, statistic="std")

        self._centres = 0.5 * (edges[1:] + edges[:-1])

        self._interp = interp1d(self._centres, self._means, fill_value="extrapolate")

        self.offset_pdf = norm(loc=0, scale=self.sigma).pdf

    @property
    def sigma(self):
        return self._stds.mean()

    def get_mean_colour_rest(self, logmass, logmass_min=5):
        """ """
        logmass = max(logmass, logmass_min)
        return self._interp(logmass)

    def summary_plot(self):
        """
        """
        df = load_gama_masses(lambdar=self._lambdar)
        logmstar = df["logmstar"].values

        fig, ax = plt.subplots()
        ax.hist2d(logmstar, df["gr"].values, cmap="binary", bins=50)

        ax.plot(self._centres, self._means, "ro")
        ax.plot(self._centres, self._means + self._stds, "r--")
        ax.plot(self._centres, self._means - self._stds, "r--")

        xx = np.linspace(logmstar.min(), logmstar.max(), 100)
        ax.plot(xx, [self.get_mean_colour_rest(_) for _ in xx], "b-")
        ax.plot(xx, [self.get_mean_colour_rest(_) + self.sigma for _ in xx], "b--")
        ax.plot(xx, [self.get_mean_colour_rest(_) - self.sigma for _ in xx], "b--")

        ax.set_xlim(6, 12)
        ax.set_ylim(-0.1, 1)

        plt.show(block=False)
