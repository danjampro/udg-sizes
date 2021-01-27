from collections import abc
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import LinearNDInterpolator

from udgsizes.utils.selection import parameter_ranges
from udgsizes.fitting.grid import ParameterGrid
from udgsizes.fitting.metrics import MetricEvaluator


class InterpolatedGrid(ParameterGrid):
    def __init__(self, model_name, bins=10, *args, **kwargs):
        super().__init__(model_name=model_name, *args, **kwargs)
        self._bins = int(bins)
        self._interps = None
        self._evaluator = InterpolatedMetricEvaluator(config=self.config, logger=self.logger)

    def sample(self, *args, **kwargs):
        """
        """
        result = super().sample(*args, **kwargs)
        self._interpolate()
        return result

    def create_model(self, param_dict):
        """
        """
        params = self._parameter_dict_to_list(param_dict)
        model = np.empty((self._bins, self._bins))
        for i in range(self._bins):
            for j in range(self._bins):
                interp = self._interps[i * self._bins + j]
                model[i, j] = interp(*params)
        return model

    def evaluate(self, oversample=4, nproc=None, save=True, **kwargs):
        """
        """
        if nproc is None:
            nproc = self.config["defaults"]["nproc"]
        self.logger.debug(f"Evaluating metrics using {nproc} processes.")

        permutations = self._get_permuations(oversample=oversample)

        fn = partial(self.evaluate_one, **kwargs)
        with Pool(nproc) as pool:
            result = pool.map(fn, [p for p in permutations])

        df = pd.concat(result, axis=1).T
        if save:
            self.logger.debug(f"Saving metrics to {self._metric_filename}.")
            df.to_csv(self._metric_filename)

        return df

    def evaluate_one(self, params, metric="poisson_likelihood_2d", **kwargs):
        """
        """
        if isinstance(params, abc.Mapping):
            param_dict = params
        else:
            param_dict = self._permutation_to_dict(params)

        # Load model interp
        model = self.create_model(param_dict)

        # Evaluate metrics
        result = pd.Series(self._evaluator.evaluate(model=model, **kwargs))

        # Include hyper parameters in result
        for quantity_name, params in param_dict.items():
            for par_name, par_value in params.items():
                result[f"{quantity_name}_{par_name}"] = par_value

        return result

    def _interpolate(self, xkey="uae_obs_jig", ykey="rec_obs_jig"):
        """
        """
        self.logger.info("Interpolating grid.")

        # Get paramter permuations in a nxD array
        points = np.array(self.permutations)

        values = np.empty((self.n_permutations, self._bins, self._bins))

        # Make 2D histograms - one interp per histogram element
        prange = parameter_ranges['uae'], parameter_ranges['rec']
        for i in range(self.n_permutations):

            # Get data for this grid element
            df = self.load_sample(i, select=True)
            xx = df[xkey].values
            yy = df[ykey].values

            # Make 2D histogram
            hist, xe, ye = np.histogram2d(xx, yy, range=prange, bins=self._bins, density=True)
            values[i, :, :] = hist

        self._interps = []
        for i in range(self._bins):
            for j in range(self._bins):
                self._interps.append(LinearNDInterpolator(points=points, values=values[:, i, j]))


class InterpolatedMetricEvaluator(MetricEvaluator):

    _metric_names = "poisson_likelihood_2d",

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, model, metrics_ignore=None):
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
            result[metric_name] = getattr(self, _metric_name)(model=model)
        return result

    def _poisson_likelihood_2d(self, model, bins=10):
        """ Bin the model samples in 2D and renormalise to match the number of observations. This
        fixes the rate paramter of the Poisson distribution in each bin. The likelihood is
        evaluated by calculating the Poisson probability of the observed counts in each bin.
        """
        range = parameter_ranges['uae'], parameter_ranges['rec']

        uae_obs = self._observations["mueff_av"].values
        rec_obs = self._observations["rec_arcsec"].values
        obs, xedges, yedges = np.histogram2d(uae_obs, rec_obs, range=range, bins=bins)

        # Rescale model by number of observations
        model = model.astype("float") * self._observations.shape[0]

        # Calculate Poisson probability for each bin
        obs = obs.reshape(-1).astype("float")
        model = model.reshape(-1)
        probs = stats.poisson(mu=model).pmf(obs)

        # Return overall log likelihood
        return np.log(probs).sum()
