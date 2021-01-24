""" Grid of model parameters """
import os
import shutil
from functools import partial
from multiprocessing import Pool
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from udgsizes.base import UdgSizesBase
from udgsizes.model.utils import create_model
from udgsizes.fitting.metrics import MetricEvaluator
from udgsizes.utils.selection import select_samples
from udgsizes.utils.stats.confidence import confidence_threshold
from udgsizes.fitting.utils.plotting import fit_summary_plot


class ParameterGrid(UdgSizesBase):
    """ N-dimensional nested parameter grid.
    """

    def __init__(self, model_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self._model = None
        model_class = self.config["models"][model_name]["type"]

        # Setup the directory to store model samples
        self._datadir = os.path.join(self.config["directories"]["data"], "models", "grid",
                                     self.model_name)
        self._metric_filename = os.path.join(self._datadir, "metrics.csv")

        grid_config = self.config["grid"][model_class]
        self._quantity_names = list(grid_config["parameters"].keys())
        self._sample_kwargs = grid_config["sampling"]

        # Create the metric evaluator object
        self._evaluator = MetricEvaluator(config=self.config, logger=self.logger)

        # Get parameter permuations per quantity
        self._par_names = {}
        self._par_values = {}
        for quantity_name, par_config in grid_config["parameters"].items():
            self._par_names[quantity_name] = list(par_config.keys())
            par_arrays = []
            for par_name, par_range in par_config.items():
                min = par_range['min']
                max = par_range['max']
                step = par_range['step']
                par_arrays.append(list(np.arange(min, max+step, step)))
            self._par_values[quantity_name] = list(itertools.product(*par_arrays))

        self.permumtations = list(itertools.product(*list(self._par_values.values())))
        self.n_permutations = len(self.permumtations)

        self.logger.debug(f"Created ParameterGrid with {self.n_permutations} "
                          "parameter permutations.")

    def sample(self, nproc=None, overwrite=False):
        """
        """
        if nproc is None:
            nproc = self.config["defaults"]["nproc"]
        self._setup_datadir(overwrite)

        self.logger.debug(f"Sampling model grid with {self.n_permutations} parameter permutations"
                          f" using {nproc} processes.")
        with Pool(nproc) as pool:
            pool.map(self._sample, np.arange(self.n_permutations))

        self.logger.debug("Finished sampling parameter grid.")

    def evaluate(self, nproc=None, save=True, **kwargs):
        """
        """
        if nproc is None:
            nproc = self.config["defaults"]["nproc"]
        self.logger.debug(f"Evaluating metrics using {nproc} processes.")

        fn = partial(self._evaluate, **kwargs)
        with Pool(nproc) as pool:
            result = pool.map(fn, np.arange(self.n_permutations))

        df = pd.concat(result, axis=1).T
        if save:
            self.logger.debug(f"Saving metrics to {self._metric_filename}.")
            df.to_csv(self._metric_filename)

        return df

    def summary_plot(self, index=None, metric="poisson_likelihood_2d", **kwargs):
        """
        """
        if index is None:
            index = self._get_best_index(metric=metric)
        df = self.load_sample(index, select=True)
        return fit_summary_plot(df=df, config=self.config, logger=self.logger, **kwargs)

    def slice_plot(self, df=None, x_key="rec_phys_alpha", z_key="uae_phys_k",
                   metric="poisson_likelihood_2d",  show=True):
        """
        """
        if df is None:
            df = self.load_metrics()
        x = df[x_key].values
        y = df[metric].values
        z = df[z_key].values

        fig, ax = plt.subplots()
        for uz in np.unique(z):
            cond = z == uz
            ax.plot(x[cond], y[cond], '-', alpha=0.5, linewidth=1)

        ax.set_xlabel(x_key)
        ax.set_ylabel(metric)

        if show:
            plt.show(block=False)
        return fig, ax

    def load_best_sample(self, metric="poisson_likelihood_2d", **kwargs):
        """
        """
        index_best = self._get_best_index(metric=metric)
        return self.load_sample(index=index_best, **kwargs)

    def load_sample(self, index, select=True):
        """
        """
        df = pd.read_csv(self._get_sample_filename(index))
        if select:
            cond = select_samples(uae=df['uae_obs_jig'].values, rec=df['rec_obs_jig'].values)
            df = df[cond].reset_index(drop=True)
        return df

    def load_metrics(self):
        """
        """
        return pd.read_csv(self._metric_filename)

    def identify_confident(self, metric="poisson_likelihood_2d", q=0.9, as_bool_array=False):
        """ Identify best models within confidence interval. """
        df = self.load_metrics()
        values = df[metric].values
        values = np.exp(values+1000)
        threshold = confidence_threshold(values, q=q)
        cond = values >= threshold
        if as_bool_array:
            return cond
        return df[cond].reset_index(drop=True)

    def load_confident_samples(self, **kwargs):
        """ Identify best models within confidence interval, returning a generator. """
        cond = self.identify_confident(as_bool_array=True, **kwargs)
        return (self.load_sample(i) for i in range(self.n_permutations) if cond[i])

    def get_best(self, metric="poisson_likelihood_2d", **kwargs):
        """
        """
        df = self.load_metrics()
        index = self._get_best_index(df=df, metric=metric, **kwargs)
        return df.iloc[index]

    def _setup_datadir(self, overwrite):
        """
        """
        if os.path.isdir(self._datadir):
            if not overwrite:
                raise FileExistsError(f"Grid directory already exists: {self._datadir}."
                                      " Pass overwrite=True to overwrite.")
            else:
                self.logger.warning(f"Removing existing grid data: {self._datadir}.")
                shutil.rmtree(self._datadir)
        os.makedirs(self._datadir)

    def _get_sample_filename(self, permutation_number):
        """
        """
        basename = f"sample_{permutation_number}.csv"
        return os.path.join(self._datadir, basename)

    def _get_parameter_dict(self, index):
        """
        """
        par_array = self.permumtations[index]
        pdict = {q: p for q, p in zip(self._quantity_names, par_array)}
        return pdict

    def _get_best_index(self, metric, df=None, func=np.argmax):
        """
        """
        if df is None:
            df = self.load_metrics()
        return func(df[metric].values)

    def _sample(self, index):
        """
        """
        # Create the model instance
        # Need to do this here so we don't run into pickle/multiprocessing problems
        model = create_model(self.model_name, config=self.config, logger=self.logger)

        # Package parameter permutation
        hyper_params = self._get_parameter_dict(index)

        # Get filename for this permutation
        filename = self._get_sample_filename(index)

        # Sample the model for this parameter permutation
        model.sample(hyper_params=hyper_params, filename=filename, **self._sample_kwargs)

    def _evaluate(self, index, **kwargs):
        """
        """
        filename = self._get_sample_filename(index)

        # Load model data
        df = pd.read_csv(filename)

        # Evaluate metrics
        result = pd.Series(self._evaluator.evaluate(df, **kwargs))

        # Include hyper parameters in result
        # TODO: Make this neater
        hyper_params = self._get_parameter_dict(index)
        for quantity_name, par_values in hyper_params.items():
            for par_name, par_value in zip(self._par_names[quantity_name], par_values):
                result[f"{quantity_name}_{par_name}"] = par_value

        return result
