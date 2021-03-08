import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from deepscan import sersic

from udgsizes.base import UdgSizesBase
from udgsizes.utils.dimming import SBDimming
from udgsizes.obs.sample import load_gama_masses


def ml_ratio_i(gi_rest):
    """ https://ui.adsabs.harvard.edu/abs/2011MNRAS.418.1587T/abstract
    """
    return -0.68 + 0.70 * gi_rest


def ml_ratio_r(ri_rest, gi_rest):
    """
    """
    return ml_ratio_i(gi_rest) + ri_rest / 2.5


def get_logmstar_bins(config):
    """
    """
    binning_config = config["colour_model"]["binning"]["logmstar"]
    bins = np.arange(binning_config["min"], binning_config["max"], binning_config["step"])
    return bins


def get_colour_bins(config):
    """
    """
    binning_config = config["colour_model"]["binning"]["colour"]
    bins = np.arange(binning_config["min"], binning_config["max"], binning_config["step"])
    return bins


def get_bin_indices(values, bins):
    """
    """
    return np.array([get_bin_index(_, bins) for _ in values])


def get_bin_index(value, bins):
    """ Truncate at lower bin.
    """
    return max(np.digitize([value], bins=bins)[0] - 1, 0)


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


class EmpiricalSBCalculator(UdgSizesBase):
    """
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._cosmo = self.config["cosmology"]

        selection_config = self.config["colour_model"]["selection"]
        df = load_gama_masses(config=self.config, **selection_config)

        self._logmstar = df["logmstar"].values
        self._colour = df["gr"].values
        self._index = df["n"].values
        self._logmstar_absmag_ratio = df["logmstar_absmag_r"].values
        self._redshift = df["redshift"].values
        self._kcorrs_r = df["kcorr_r"].values
        self._kcorrs_g = df["kcorr_g"].values

        self._logmstar_bins = get_logmstar_bins(config=self.config)
        self._logmstar_bin_indices = get_bin_indices(self._logmstar, bins=self._logmstar_bins)

        self._ml_polys = {}
        self._create_ml_model()

        self._kdes_colour_index = {}
        self._create_colour_index_model()

        self._mass_colour_tree = None
        self._create_kcorr_model()

    # Properties

    @property
    def n_logmstar_bins(self):
        return self._logmstar_bins.size

    @property
    def colour_range(self):
        return self._colour.min(), self._colour.max()

    @property
    def index_range(self):
        return self._index.min(), self._index.max()

    # Public methods

    def calculate_uae_phys(self, logmstar, rec, redshift, colour_rest):
        """
        """
        # Calculate absolute mag in r-band
        idx = get_bin_index(logmstar, bins=self._logmstar_bins)
        absmag = logmstar / np.polyval(self._ml_polys[idx], colour_rest)

        # Calculate apparent magnitude
        mag = absmag + self._cosmo.distmod(redshift).value

        # Calculate apparent surface brightness
        return sersic.mag2meanSB(mag, re=rec, q=1)

    def calculate_uae(self, logmstar, rec, redshift, colour_rest):
        """
        """
        # Calculate apparent magnitude
        uae_phys = self.calculate_uae_phys(logmstar, rec, redshift, colour_rest=colour_rest)

        # Apply k-correction
        uae = uae_phys + self.get_k_correction_r(logmstar, colour_rest, redshift=redshift)

        return uae

    def colour_index_likelihood(self, logmstar, colour_rest, index):
        """
        """
        idx = get_bin_index(logmstar, bins=self._logmstar_bins)
        kde = self._kdes_colour_index[idx]
        return kde.pdf([colour_rest, index])

    def get_k_correction_r(self, logmstar, colour_rest, redshift):
        """
        """
        idx = self._mass_colour_tree.query([logmstar, colour_rest, redshift], k=1)[1]
        return self._kcorrs_r[idx]

    def get_k_correction_gr(self, logmstar, colour_rest, redshift):
        """
        """
        idx = self._mass_colour_tree.query([logmstar, colour_rest, redshift], k=1)[1]
        kr = self._kcorrs_r[idx]
        kg = self._kcorrs_g[idx]
        return kg - kr

    # Plotting methods

    def summary_plot_ml(self):
        """
        """
        fig, ax = plt.subplots(figsize=(self.n_logmstar_bins * 3, 3))
        cc = np.linspace(self.colour_range[0], self.colour_range[1], 10)
        for idx in range(self.n_logmstar_bins):
            ax = plt.subplot(1, self.n_logmstar_bins, idx + 1)
            cond = self._logmstar_bin_indices == idx
            ax.plot(self._colour[cond], self._logmstar_absmag_ratio[cond], "k+", markersize=1,
                    alpha=0.5)
            ax.plot(cc, np.polyval(self._ml_polys[idx], cc), "b--")
            ax.set_xlim(-0.2, 1.1)
            ax.set_ylim(-0.55, -0.43)
        plt.show(block=False)

    def summary_plot_colour_index(self, nbins=20):
        """
        """
        plt.figure(figsize=(self.n_logmstar_bins * 3, 6))

        for idx in range(self.n_logmstar_bins):
            cond = self._logmstar_bin_indices == idx

            ax0 = plt.subplot(2, self._logmstar_bins.size, idx + 1)
            ax0.hist2d(self._colour[cond], self._index[cond], density=True, bins=nbins,
                       range=(self.colour_range, self.index_range))

            xx, yy = np.meshgrid(np.linspace(*self.colour_range, 100),
                                 np.linspace(*self.index_range, 100))
            values = np.vstack([xx.reshape(-1), yy.reshape(-1)])
            kde_values = self._kdes_colour_index[idx](values).reshape(xx.shape)

            ax1 = plt.subplot(2, self._logmstar_bins.size, self._logmstar_bins.size + idx + 1)
            ax1.imshow(kde_values, extent=(*self.colour_range, *self.index_range),
                       origin="lower")
        plt.show(block=False)

    # Private methods

    def _create_colour_index_model(self):
        """
        """
        for idx in range(self.n_logmstar_bins):
            cond = self._logmstar_bin_indices == idx
            values = np.vstack([self._colour[cond], self._index[cond]])
            self._kdes_colour_index[idx] = gaussian_kde(values)

    def _create_ml_model(self):
        """
        """
        for idx in range(self.n_logmstar_bins):
            cond = self._logmstar_bin_indices == idx
            p = np.polyfit(self._colour[cond], self._logmstar_absmag_ratio[cond], 1)
            self._ml_polys[idx] = p

    def _create_kcorr_model(self):
        """
        """
        values = np.vstack([self._logmstar, self._colour, self._redshift]).T
        self._mass_colour_tree = cKDTree(values)
