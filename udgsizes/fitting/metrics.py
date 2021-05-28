from contextlib import suppress
import numpy as np
from scipy import stats

import powerlaw

from udgsizes.base import UdgSizesBase
from udgsizes.obs.sample import load_sample
from udgsizes.utils.stats.kstest import kstest_2d
from udgsizes.utils.selection import parameter_ranges
from udgsizes.utils.stats.likelihood import fit_colour_gaussian
from udgsizes.utils.stats.kde import TransformedGaussianPDF, TransformedKDE


class MetricEvaluator(UdgSizesBase):
    """ A class to calculate statistical metrics to compare model samples to observations. """

    _metric_names = ('log_likelihood_poisson', 'log_likelihood_colour', 'kstest_2d',
                     'udg_power_law', "n_udg",  "n_dwarf", "n_selected", "n_total",
                     "kstest_rec_obs_jig",  "kstest_uae_obs_jig", "kstest_colour_obs",
                     "log_likelihood_gauss_3d", "log_likelihood_kde_3d")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dfobs = load_sample(config=self.config, logger=self.logger, select=True)

    def evaluate(self, df, dfo=None, metrics_ignore=None):
        """
        """
        if metrics_ignore is None:
            metrics_ignore = []

        if dfo is None:
            dfo = self._dfobs

        result = {}
        for metric_name in self._metric_names:
            if metric_name in metrics_ignore:
                continue

            _metric_name = "_" + metric_name
            result[metric_name] = getattr(self, _metric_name)(df, dfo=dfo)

        with suppress(KeyError):
            result["log_likelihood"] = result["log_likelihood_poisson"] + \
                result["log_likelihood_colour"]

        return result

    def _kstest_2d(self, df, dfo):
        """
        """
        cond = df["selected_jig"].values == 1
        x1 = df['uae_obs_jig'].values[cond]
        y1 = df['rec_obs_jig'].values[cond]
        x2 = dfo['mueff_av'].values
        y2 = dfo['rec_arcsec'].values
        return kstest_2d(x1, y1, x2, y2)

    def _log_likelihood_poisson(self, df, dfo, n_bins=10):
        """ Bin the model samples in 2D and renormalise to match the number of observations. This
        fixes the rate paramter of the Poisson distribution in each bin. The likelihood is
        evaluated by calculating the Poisson probability of the observed counts in each bin.
        """
        cond = df["selected_jig"].values == 1
        range = parameter_ranges['uae'], parameter_ranges['rec']

        uae_obs = dfo["mueff_av"].values
        rec_obs = dfo["rec_arcsec"].values
        obs, xedges, yedges = np.histogram2d(uae_obs, rec_obs, range=range, bins=n_bins)

        uae_mod = df["uae_obs_jig"].values[cond]
        rec_mod = df["rec_obs_jig"].values[cond]
        model, _, _ = np.histogram2d(uae_mod, rec_mod, range=range, bins=n_bins, density=True)

        # Rescale model by number of observations
        model = model.astype("float") * dfo.shape[0]

        # Calculate Poisson probability for each bin
        obs = obs.reshape(-1).astype("float")
        model = model.reshape(-1)
        probs = stats.poisson(mu=model).pmf(obs)

        # Return overall log likelihood
        return np.log(probs).sum()

    def _log_likelihood_colour(self, df, dfo):
        """ Calculate the log-likelihood of the colours assuming a Gaussian model.
        """
        pdf = fit_colour_gaussian(df["colour_obs"].values)
        return np.log(pdf(dfo["g_r"].values)).sum()

    def _n_dwarf(self, df, dfo):
        return df["is_dwarf"].sum()

    def _n_udg(self, df, dfo):
        return df["is_udg"].sum()

    def _n_selected(self, df, dfo):
        return df["selected_jig"].sum()

    def _n_total(self, df, dfo):
        return df.shape[0]

    def _udg_power_law(self, df, dfo, rec_phys_min=1.5):
        """
        """
        rec_phys = df["rec_phys"].values[df["is_udg"].values == 1]
        fit_result = powerlaw.Fit(rec_phys, xmin=rec_phys_min)
        return fit_result.power_law.alpha

    def _kstest_rec_obs_jig(self, df, dfo):
        vobs = dfo["rec_arcsec"]
        vmod = df["rec_obs_jig"].values
        return stats.kstest(vobs, vmod)[1]

    def _kstest_uae_obs_jig(self, df, dfo):
        vobs = dfo["mueff_av"]
        vmod = df["uae_obs_jig"].values
        return stats.kstest(vobs, vmod)[1]

    def _kstest_colour_obs(self, df, dfo):
        vobs = dfo["g_r"]
        vmod = df["colour_obs"].values
        return stats.kstest(vobs, vmod)[1]

    def _log_likelihood_gauss_3d(self, df, dfo):

        cond = df["selected_jig"].values == 1
        df = df[cond].reset_index(drop=True)

        pdf = TransformedGaussianPDF(df)
        probs = pdf.evaluate(dfo)

        return np.log(probs).sum()

    def _log_likelihood_kde_3d(self, df, dfo):

        cond = df["selected_jig"].values == 1
        df = df[cond].reset_index(drop=True)

        pdf = TransformedKDE(df)
        probs = pdf.evaluate(dfo)

        return np.log(probs).sum()
