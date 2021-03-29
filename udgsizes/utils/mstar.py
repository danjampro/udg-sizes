import numpy as np
import matplotlib.pyplot as plt

from deepscan import sersic

from udgsizes.base import UdgSizesBase
from udgsizes.utils.dimming import SBDimming
from udgsizes.obs.sample import load_gama_masses


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

    def calculate_uae(self, **kwargs):
        """
        """
        # Calculate apparent magnitude
        uae_phys = self.calculate_uae_phys(**kwargs)
        # Apply k-correction
        uae = uae_phys + self._dimmer(kwargs.get("redshift"))
        return uae


class EmpiricalSBCalculator(UdgSizesBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        selection_config = self.config["ml_model"]["selection"]
        df = load_gama_masses(config=self.config, **selection_config)

        self._logmstar = df["logmstar"].values
        self._colour = df["gr"].values
        self._logmstar_absmag_ratio = df["logmstar_absmag_r"].values
        self._redshift = df["redshift"].values

        self._logmstar_bins = self._get_logmstar_bins()
        self._logmstar_bin_indices = self._get_bin_indices(self._logmstar, bins=self._logmstar_bins)

        self._ml_polys = {}
        self._create_ml_model()

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
        """ Return the r-band rest-frame re-averaged surface brightness in mag/arcsec-2.
        Args:
            logmstar (float): log10 stellar mass in solar masses.
            rec: The circularised effective radius in arcseconds.
            redshift (float): The redshift.
            colour_rest (float): The rest-frame g-r colour in magnitudes.
        Returns:
            float: The rest-frame surface brightness in mag/arcsec-2.
        """
        # Calculate absolute mag in r-band
        idx = self._get_bin_index(logmstar, bins=self._logmstar_bins)
        absmag = logmstar / np.polyval(self._ml_polys[idx], colour_rest)

        # Calculate apparent magnitude
        mag = absmag + self.distmod(redshift)

        # Calculate apparent surface brightness
        return sersic.mag2meanSB(mag, re=rec, q=1)

    def calculate_logml_ab(self, logmstar, colour_rest):
        """ Luminosity in units of AB mag=0 as in Taylor+11.
        """
        # Calculate absolute mag in r-band
        idx = self._get_bin_index(logmstar, bins=self._logmstar_bins)

        absmag = logmstar / np.polyval(self._ml_polys[idx], colour_rest)
        return 0.4 * (absmag) + logmstar

    # Plotting

    def summary_plot_ml(self, figwidth=12, xrng=(-0.2, 1.1), yrng=(-0.55, -0.43)):
        """
        """
        figheight = figwidth / self.n_logmstar_bins
        fig, ax = plt.subplots(figsize=(figwidth, figheight))

        aspect = (xrng[1] - xrng[0]) / (yrng[1] - yrng[0])

        cc = np.linspace(self.colour_range[0], self.colour_range[1], 10)

        for idx in range(self.n_logmstar_bins):
            ax = plt.subplot(1, self.n_logmstar_bins, idx + 1)

            cond = self._logmstar_bin_indices == idx
            ax.plot(self._colour[cond], self._logmstar_absmag_ratio[cond], "k+", markersize=1,
                    alpha=0.5)
            ax.plot(cc, np.polyval(self._ml_polys[idx], cc), "b--")

            ax.set_xlim(xrng)
            ax.set_ylim(yrng)
            ax.set_aspect(aspect)

        plt.show(block=False)

    # Private methods

    def _create_ml_model(self):
        """
        The ratio of log-stellar mass over absolute magnitude is fit with a power law for each
        stellar mass bin.
        """
        for idx in range(self.n_logmstar_bins):
            cond = self._logmstar_bin_indices == idx
            p = np.polyfit(self._colour[cond], self._logmstar_absmag_ratio[cond], 1)
            self._ml_polys[idx] = p

    def _get_logmstar_bins(self):
        """
        """
        binning_config = self.config["ml_model"]["binning"]["logmstar"]
        bins = np.arange(binning_config["min"], binning_config["max"], binning_config["step"])
        return bins

    def _get_bin_indices(self, values, bins):
        """
        """
        return np.array([self._get_bin_index(_, bins) for _ in values])

    def _get_bin_index(self, value, bins):
        """ Truncate at lower bin.
        """
        return max(np.digitize([value], bins=bins)[0] - 1, 0)
