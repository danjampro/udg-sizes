import numpy as np
from scipy import stats

from udgsizes.base import UdgSizesBase
from udgsizes.obs.sample import load_sample
from udgsizes.utils.selection import select_samples
from udgsizes.utils.kstest import kstest_2d
from udgsizes.utils.selection import parameter_ranges


class MetricEvaluator(UdgSizesBase):
    """ A class to calculate statistical metrics to compare model samples to observations. """

    _metric_names = ('kstest_2d', 'poisson_likelihood_2d')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._observations = load_sample(config=self.config, logger=self.logger, select=True)

    def evaluate(self, df, metrics_ignore=None):
        """
        """
        if metrics_ignore is None:
            metrics_ignore = []
        result = {}
        for metric_name in self._metric_names:
            if metric_name in metrics_ignore:
                self.logger.debug(f"Skipping metric: {metric_name}.")
                continue
            _metric_name = "_" + metric_name
            result[metric_name] = getattr(self, _metric_name)(df)
        return result

    def _kstest_2d(self, df):
        """
        """
        cond = df["selected_jig"].values == 1
        x1 = df['uae_obs_jig'].values[cond]
        y1 = df['rec_obs_jig'].values[cond]
        x2 = self._observations['mueff_av'].values
        y2 = self._observations['rec_arcsec'].values
        return kstest_2d(x1, y1, x2, y2)

    def _poisson_likelihood_2d(self, df, n_bins=10):
        """ Bin the model samples in 2D and renormalise to match the number of observations. This
        fixes the rate paramter of the Poisson distribution in each bin. The likelihood is
        evaluated by calculating the Poisson probability of the observed counts in each bin.
        """
        cond = df["selected_jig"].values == 1
        range = parameter_ranges['uae'], parameter_ranges['rec']

        uae_obs = self._observations["mueff_av"].values
        rec_obs = self._observations["rec_arcsec"].values
        obs, xedges, yedges = np.histogram2d(uae_obs, rec_obs, range=range, bins=n_bins)

        uae_mod = df["uae_obs_jig"].values[cond]
        rec_mod = df["rec_obs_jig"].values[cond]
        model, _, _ = np.histogram2d(uae_mod, rec_mod, range=range, bins=n_bins, density=True)

        # Rescale model by number of observations
        model = model.astype("float") * self._observations.shape[0]

        # Calculate Poisson probability for each bin
        obs = obs.reshape(-1).astype("float")
        model = model.reshape(-1)
        probs = stats.poisson(mu=model).pmf(obs)

        # Return overall log likelihood
        return np.log(probs).sum()

    def _select_model_samples(self, df):
        """
        """
        cond = select_samples(uae=df['uae_obs_jig'].values, rec=df['rec_obs_jig'].values)
        return df[cond].reset_index(drop=True)
