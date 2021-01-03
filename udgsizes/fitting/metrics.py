import numpy as np
from scipy import stats

from udgsizes.base import UdgSizesBase
from udgsizes.obs.sample import load_sample
from udgsizes.utils.selection import select_samples
from udgsizes.utils.kstest import kstest_2d
from udgsizes.utils.selection import parameter_ranges


class MetricEvaluator(UdgSizesBase):
    """ A class to calculate statistical metrics to compare model samples to observations. """

    _metric_names = ('_kstest_2d', '_poisson_likelihood_2d')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._observations = load_sample(config=self.config, logger=self.logger, select=True)

    def evaluate(self, df):
        """
        """
        result = {}
        for metric_name in self._metric_names:
            if metric_name.startswith("_"):
                _metric_name = metric_name[1:]
            else:
                _metric_name = metric_name
            result[_metric_name] = getattr(self, metric_name)(df)
        return result

    def _kstest_2d(self, df):
        """
        """
        x1 = df['uae_obs_jig'].values
        y1 = df['rec_obs_jig'].values
        x2 = self._observations['mueff_av'].values
        y2 = self._observations['rec_arcsec'].values
        return kstest_2d(x1, y1, x2, y2)

    def _poisson_likelihood_2d(self, df, n_bins=10):
        """ Bin the model samples in 2D and renormalise to match the number of observations. This
        fixes the rate paramter of the Poisson distribution in each bin. The likelihood is
        evaluated by calculating the Poisson probability of the observed counts in each bin.
        """
        range = parameter_ranges['uae'], parameter_ranges['rec']

        uae_obs = self._observations["mueff_av"].values
        rec_obs = self._observations["rec_arcsec"].values
        obs, xedges, yedges = np.histogram2d(uae_obs, rec_obs, range=range, bins=n_bins)

        uae_mod = df["uae_obs_jig"].values
        rec_mod = df["rec_obs_jig"].values
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
