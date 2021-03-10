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
from udgsizes.core import get_config
from udgsizes.model.utils import create_model, get_model_config
from udgsizes.fitting.metrics import MetricEvaluator
from udgsizes.utils.selection import select_samples
from udgsizes.utils.stats.confidence import confidence_threshold
from udgsizes.fitting.utils.plotting import fit_summary_plot, plot_2d_hist


def _get_datadir(model_name, config=None):
    """
    """
    if config is None:
        config = get_config()
    return os.path.join(config["directories"]["data"], "models", "grid", model_name)


def _get_metric_filename(model_name, **kwargs):
    """
    """
    datadir = _get_datadir(model_name, **kwargs)
    return os.path.join(datadir, "metrics.csv")


def load_metrics(model_name, **kwargs):
    """
    """
    filename = _get_metric_filename(model_name, **kwargs)
    return pd.read_csv(filename)


class ParameterGrid(UdgSizesBase):
    """ N-dimensional nested parameter grid.
    """
    _default_metric = "kstest_2d"

    def __init__(self, model_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self._model = None
        model_class = get_model_config(model_name, config=self.config)["type"]

        # Setup the directory to store model samples
        self._datadir = _get_datadir(model_name=self.model_name, config=self.config)
        self._metric_filename = _get_metric_filename(self.model_name, config=self.config)

        self._grid_config = self.config["grid"][model_class]
        self.quantity_names = list(self._grid_config["parameters"].keys())

        # Create the metric evaluator object
        self._evaluator = MetricEvaluator(config=self.config, logger=self.logger)

        self.parameter_names = {}
        for quantity_name, quantity_config in self._grid_config["parameters"].items():
            self.parameter_names[quantity_name] = []
            for parameter_name, parameter_config in quantity_config.items():
                self.parameter_names[quantity_name].append(parameter_name)

        self.permutations = self._get_permuations()
        self.n_permutations = len(self.permutations)

        self.logger.debug(f"Created ParameterGrid with {self.n_permutations} "
                          "parameter permutations.")

    def _get_permuations(self, oversample=1):
        """
        """
        parameter_values = []
        for quantity_name, quantity_config in self._grid_config["parameters"].items():
            for parameter_name, parameter_config in quantity_config.items():
                min = parameter_config['min']
                max = parameter_config['max']
                step = parameter_config['step'] / oversample
                parameter_values.append(np.arange(min, max+step, step))
        permutations = list(itertools.product(*parameter_values))
        return permutations

    def check_initial_values(self):
        """ Make sure a viable set of initial values exists for each model. """
        for par_array in self.permutations:
            hyper_params = self._permutation_to_dict(par_array)
            model = create_model(self.model_name, config=self.config, logger=self.logger)
            model._get_initial_state(hyper_params=hyper_params)

    def sample(self, nproc=None, overwrite=False, n_samples=None, burnin=None, **kwargs):
        """
        """
        if nproc is None:
            nproc = self.config["defaults"]["nproc"]
        if n_samples is None:
            n_samples = self.config["grid"]["n_samples"]
        if burnin is None:
            burnin = self.config["grid"]["burnin"]

        self._setup_datadir(overwrite)

        settings = {"n_permutations ": self.n_permutations,
                    "n_proc": nproc,
                    "n_samples": n_samples,
                    "burnin": burnin}
        self.logger.debug(f"Sampling model grid with settings: {settings}")

        func = partial(self._sample, n_samples=n_samples, burnin=burnin, **kwargs)
        with Pool(nproc) as pool:
            pool.map(func, np.arange(self.n_permutations))

        self.logger.debug("Finished sampling parameter grid.")

    def evaluate(self, nproc=None, save=True, **kwargs):
        """
        """
        if nproc is None:
            nproc = self.config["defaults"]["nproc"]
        self.logger.debug(f"Evaluating metrics using {nproc} processes.")

        fn = partial(self.evaluate_one, **kwargs)
        with Pool(nproc) as pool:
            result = pool.map(fn, np.arange(self.n_permutations))

        df = pd.concat(result, axis=1).T
        if save:
            self.logger.debug(f"Saving metrics to {self._metric_filename}.")
            df.to_csv(self._metric_filename)

        return df

    def summary_plot(self, index=None, metric=None, **kwargs):
        """
        """
        if index is None:
            index = self._get_best_index(metric=metric)
        df = self.load_sample(index, select=True)
        return fit_summary_plot(df=df, config=self.config, logger=self.logger, **kwargs)

    def slice_plot(self, df=None, x_key="rec_phys_alpha", z_key="uae_phys_k",
                   metric=None,  show=True):
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

    def load_best_sample(self, metric=None, **kwargs):
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
        return load_metrics(self.model_name, config=self.config)

    def identify_confident(self, metric=None, q=0.9, as_bool_array=False):
        """ Identify best models within confidence interval. """
        df = self.load_metrics()
        values = df[metric].values
        threshold = confidence_threshold(values, q=q)
        cond = values >= threshold
        if as_bool_array:
            return cond
        return df[cond].reset_index(drop=True)

    def load_confident_samples(self, **kwargs):
        """ Identify best models within confidence interval, returning a generator. """
        cond = self.identify_confident(as_bool_array=True, **kwargs)
        return (self.load_sample(i) for i in range(self.n_permutations) if cond[i])

    def get_best_metrics(self, metric="kstest_2d", **kwargs):
        """
        """
        df = self.load_metrics()
        index = self._get_best_index(df=df, metric=metric, **kwargs)
        return df.iloc[index]

    def get_best_hyper_parameters(self, **kwargs):
        """ Return the best fitting hyper parameters for a given model.
        Args:
            model_name (str): The name of the model.
        Returns:
            dict: The nested hyper parameter dictionary.
        """
        metrics = self.get_best_metrics(**kwargs)

        hyper_params = {}
        for quantity_name in self.quantity_names:
            hyper_params[quantity_name] = {}
            for parameter_name in self.parameter_names[quantity_name]:
                flattened_name = f"{quantity_name}_{parameter_name}"
                hyper_params[quantity_name][parameter_name] = metrics[flattened_name]

        return hyper_params

    def evaluate_one(self, index=None, metric=None, thinning=None, **kwargs):
        """
        """
        if index is None:
            index = self._get_best_index(metric=metric)

        # Load model data
        df = self.load_sample(index=index, select=False)

        if thinning is not None:
            df = df[::thinning].reset_index(drop=True)

        # Evaluate metrics
        result = pd.Series(self._evaluator.evaluate(df, **kwargs))

        par_array = self.permutations[index]
        permdict = self._permutation_to_dict(par_array)
        for quantity_name, parconf in permdict.items():
            for par_name, par_value in parconf.items():
                result[f"{quantity_name}_{par_name}"] = par_value

        return result

    def plot_2d_hist(self, xkey, ykey, metric=None, **kwargs):
        """
        """
        if metric is None:
            metric = self._default_metric
        df = self.load_metrics()
        return plot_2d_hist(df, xkey, ykey, metric=metric, **kwargs)

    def marginal_likelihood_histogram(self, key, metric=None, ax=None, show=True, **kwargs):
        """
        """
        if metric is None:
            metric = self._default_metric
            
        df = self.load_metrics()
        x = df[key].values
        z = df[metric].values

        cond = np.isfinite(z)
        x = x[cond]
        z = z[cond]

        # Plot marginal histogram
        if ax is None:
            fig, ax = plt.subplots()
        ax.hist(x, weights=z, **kwargs)

        if show:
            plt.show(block=False)

        return ax

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

    def _permutation_to_dict(self, par_array):
        """
        """
        i = 0
        result = {}
        for qname, parnames in self.parameter_names.items():
            result[qname] = {}
            for parname in parnames:
                result[qname][parname] = par_array[i]
                i += 1
        return result

    def _parameter_dict_to_list(self, param_dict):
        """ Order matters.
        """
        result = []
        for qname, parnames in self.parameter_names.items():
            for parname in parnames:
                result.append(param_dict[qname][parname])
        return result

    def _get_best_index(self, metric=None, df=None, func=np.nanargmax):
        """
        """
        if metric is None:
            metric = self._default_metric
        if df is None:
            df = self.load_metrics()
        return func(df[metric].values)

    def _sample(self, index, n_samples, burnin):
        """
        """
        # Create the model instance
        # Need to do this here so we don't run into pickle/multiprocessing problems
        model = create_model(self.model_name, config=self.config, logger=self.logger)

        # Package parameter permutation
        par_array = self.permutations[index]
        hyper_params = self._permutation_to_dict(par_array)

        # Get filename for this permutation
        filename = self._get_sample_filename(index)

        # Sample the model for this parameter permutation
        model.sample(hyper_params=hyper_params, filename=filename, n_samples=n_samples,
                     burnin=burnin)
        self.logger.debug(f"Finished grid index {index} of {self.n_permutations}.")
