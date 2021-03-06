""" Grid of model parameters """
import os
import shutil
import tempfile
import itertools
from contextlib import suppress
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from udgsizes.base import UdgSizesBase
from udgsizes.utils.library import load_module
from udgsizes.obs.sample import load_sample
from udgsizes.model.utils import create_model, get_model_config
from udgsizes.fitting.metrics import MetricEvaluator
from udgsizes.utils.selection import GR_MIN, GR_MAX
from udgsizes.utils.stats.confidence import confidence_threshold
from udgsizes.fitting.utils.plotting import fit_summary_plot, plot_2d_hist, threshold_plot
from udgsizes.utils.stats.likelihood import unlog_likelihood


class ParameterGrid(UdgSizesBase):
    """ N-dimensional nested parameter grid.
    """
    _default_metric = "posterior_kde_3d"  # TODO: Change to "posterior"

    def __init__(self, model_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        model_class = get_model_config(model_name, config=self.config)["type"]

        # Setup the directory to store sampled model data
        self.directory = os.path.join(self.config["directories"]["data"], "models", "grid",
                                      self.model_name)
        self._metrics_filename = os.path.join(self.directory, "metrics.csv")

        self._grid_config = self.config["grid"][model_class]
        self.quantity_names = list(self._grid_config["parameters"].keys())

        # Create the metric evaluator object
        self._evaluator = MetricEvaluator(config=self.config, logger=self.logger)

        # Configure the model parameters
        self.parameter_names = {}
        for quantity_name, quantity_config in self._grid_config["parameters"].items():
            self.parameter_names[quantity_name] = []
            for parameter_name, parameter_config in quantity_config.items():
                self.parameter_names[quantity_name].append(parameter_name)

        # Identify parameter permutations in grid
        self.permutations = self._get_permuations()
        self.n_permutations = len(self.permutations)

        self.logger.debug(f"Created ParameterGrid with {self.n_permutations} "
                          "parameter permutations.")

        # Seems to help with slow IO bottleneck
        self._copy_before_read = self.config.get("copy_before_read", False)

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

    def evaluate_one(self, index=None, df=None, metric=None, thinning=None, **kwargs):
        """
        """
        if index is None:
            index = self._get_best_index(metric=metric)

        # Load model data
        if df is None:
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

    def evaluate(self, nproc=None, save=True, filename=None, **kwargs):
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
            filename = self._metrics_filename if filename is None else filename
            self.logger.debug(f"Saving metrics to {filename}.")
            df.to_csv(filename)

        return df

    def load_best_sample(self, metric=None, **kwargs):
        """
        """
        index_best = self._get_best_index(metric=metric)
        return self.load_sample(index=index_best, **kwargs)

    def load_sample(self, index, select=True):
        """
        """
        filename = self._get_sample_filename(index)

        # Seems to help with slow reads as first copies to SSD
        if self._copy_before_read:
            with tempfile.NamedTemporyFile() as tf:
                shutil.copy(filename, tf.name)
                df = pd.read_csv(tf.name)
        else:
            df = pd.read_csv(filename)

        # TODO: Move to model
        colour = df["colour_obs"].values
        df["selected_colour"] = (colour >= GR_MIN) & (colour < GR_MAX)
        df["selected"] = (df["selected_jig"].values * df["selected_colour"].values).astype("bool")

        if select:
            cond = df["selected"].values.astype("bool")
            df = df[cond].reset_index(drop=True)

        return df

    def load_metrics(self, filename=None):
        """
        """
        if filename is None:
            filename = self._metrics_filename

        df = pd.read_csv(filename)
        df["grid_idx"] = np.arange(df.shape[0])

        # TODO: Move to metrics
        for col in df.columns:
            if col.startswith("log_likelihood"):
                new_col = col[4:]
                df[new_col] = unlog_likelihood(df[col].values)

        # TODO: Move to metrics
        keys = "kstest_colour_obs", "kstest_rec_obs_jig", "kstest_uae_obs_jig"
        values = [df[k].values for k in keys]
        df["kstest_min"] = np.min(values, axis=0)

        with suppress(KeyError):
            keys = "kstest_colour_obs", "kstest_2d"
            values = [df[k].values for k in keys]
            df["kstest_min_2d"] = np.min(values, axis=0)
            df["likelihood_ks"] = df["kstest_min_2d"] * df["kstest_colour_obs"]

        # Calculate prior
        prior = np.ones(df.shape[0])

        for quantity_name in self.quantity_names:
            for parameter_name in self.parameter_names[quantity_name]:
                try:
                    prior_config = self._grid_config["priors"][quantity_name][parameter_name]
                except KeyError:
                    continue
                flattened_name = f"{quantity_name}_{parameter_name}"

                func_name = prior_config["func"]
                self.logger.debug(f"Applying prior to {flattened_name}: {func_name}")

                func = load_module(func_name)
                with suppress(KeyError):
                    func = partial(func, **prior_config["pars"])

                prior *= func(df[flattened_name].values)

        df["prior"] = prior
        df["posterior"] = df["likelihood"] * df["prior"]
        df["posterior_gauss_3d"] = df["likelihood_gauss_3d"] * df["prior"]
        df["posterior_kde_3d"] = df["likelihood_kde_3d"] * df["prior"]
        with suppress(KeyError):
            df["posterior_ks"] = df["likelihood_ks"] * df["prior"]

        return df

    def load_faux_metrics(self):
        """ Load metrics evaluated using faux observed samples. """
        dir = os.path.join(self.directory, "faux")
        basenames = [f for f in os.listdir(dir) if f.startswith("metrics")]

        self.logger.info(f"Loading {len(basenames)} faux metric sets.")

        dfs = []
        for basename in basenames:
            df = self.load_metrics(filename=os.path.join(dir, basename))
            dfs.append(df)

        return dfs

    def load_confident_samples(self, **kwargs):
        """ Identify best models within confidence interval, returning a generator. """
        cond = self._identify_confident(**kwargs)
        return (self.load_sample(i) for i in range(self.n_permutations) if cond[i])

    def load_confident_metrics(self, **kwargs):
        """ Identify best models within confidence interval, returning a generator. """
        cond = self._identify_confident(**kwargs)
        return self.load_metrics()[cond]

    def get_best_metrics(self, df=None, metric=None, **kwargs):
        """
        """
        if df is None:
            df = self.load_metrics()
        index = self._get_best_index(df=df, metric=metric, **kwargs)
        return df.iloc[index]

    def get_best_hyper_parameters(self, flatten=False, **kwargs):
        """ Return the best fitting hyper parameters for a given model.
        Args:
            model_name (str): The name of the model.
            flatten (bool, optional): If True, return a flattened dict. Default False.
        Returns:
            dict: The nested hyper parameter dictionary.
        """
        metrics = self.get_best_metrics(**kwargs)

        hyper_params = {}
        for quantity_name in self.quantity_names:

            if not flatten:
                hyper_params[quantity_name] = {}

            for parameter_name in self.parameter_names[quantity_name]:
                flattened_name = f"{quantity_name}_{parameter_name}"

                if flatten:
                    hyper_params[flattened_name] = metrics[flattened_name]
                else:
                    hyper_params[quantity_name][parameter_name] = metrics[flattened_name]

        return hyper_params

    def parameter_stats(self, key, metric=None):
        """ Return mean and std for a given parameter """

        if metric is None:
            metric = self._default_metric

        df = self.load_metrics()

        weights = df[metric].values
        values = df[key].values

        mean = np.average(values, weights=weights)
        std = np.sqrt(np.average((values-mean)**2, weights=weights))

        return mean, std

    def make_faux_observations(self, df=None, dfo=None, makeplots=False, **kwargs):
        """ Get sample of mock observations from the model sample
        Args:
            df (pd.DataFrame): Model samples from the best-fitting model.
        Returns:
            pd.DataFrame: Faux observations extracted from model samples.
        """
        if df is None:
            df = self.load_best_sample(**kwargs)
        if dfo is None:
            dfo = load_sample(select=True)

        # Choose a random sample of the same size as observations
        indices = np.random.randint(0, df.shape[0], dfo.shape[0])

        # Map the model keys into observation keys
        dff = pd.DataFrame()
        for key, obskey in self.config["obskeys"].items():
            dff[obskey] = df[key].values[indices]

        if makeplots:
            self.summary_plot(dfo=dff)

        return dff

    # Plotting

    def plot_2d_hist(self, xkey, ykey, metric=None, plot_indices=False, df=None, **kwargs):
        """
        """
        if metric is None:
            metric = self._default_metric

        if df is None:
            df = self.load_metrics()

        metrics = df[metric].values

        ax = plot_2d_hist(df[xkey].values, df[ykey].values, metrics, **kwargs)
        ax.set_xlabel(xkey)
        ax.set_ylabel(ykey)

        if plot_indices:
            self.plot_indices(ax=ax, xkey=xkey, ykey=ykey)

        return ax

    def plot_indices(self, xkey, ykey, fontsize=8, color="b", ax=None, **kwargs):
        """
        """
        if ax is None:
            fig, ax = plt.subplots()

        # Extract coordinates
        # TODO: Remove dependence on metrics.csv
        df = self.load_metrics()
        xx = df[xkey].values
        yy = df[ykey].values

        for i, (x, y) in enumerate(zip(xx, yy)):
            ax.text(x, y, f"{i}", fontsize=fontsize, color=color, va="center", ha="center",
                    **kwargs)

        return ax

    def marginal_likelihood_histogram(self, key, metric=None, ax=None, show=True, **kwargs):
        """
        """
        if metric is None:
            metric = self._default_metric

        df = self.load_metrics()
        x = df[key].values
        z = df[metric].values
        zprior = z * df["prior"].values

        cond = np.isfinite(z)
        x = x[cond]
        z = z[cond]

        # Plot marginal histogram
        if ax is None:
            fig, ax = plt.subplots()
        ax.hist(x, weights=z, histtype="step", density=True, **kwargs)
        ax.hist(x, weights=zprior, histtype="step", density=True, **kwargs)

        if show:
            plt.show(block=False)

        return ax

    def threshold_plot(self, xkey, ykey, metric=None, plot_indices=False, **kwargs):
        """
        """
        if metric is None:
            metric = self._default_metric

        df = self.load_metrics()
        metrics = df[metric].values

        ax = threshold_plot(df[xkey].values, df[ykey].values, metrics, xlabel=xkey, ylabel=ykey,
                            **kwargs)

        return ax

    def summary_plot(self, index=None, metric=None, **kwargs):
        """
        """
        if index is None:
            index = self._get_best_index(metric=metric)
        df = self.load_sample(index, select=True)
        return fit_summary_plot(df=df, config=self.config, logger=self.logger, **kwargs)

    # Private methods

    def _setup_datadir(self, overwrite):
        """
        """
        if os.path.isdir(self.directory):
            if not overwrite:
                raise FileExistsError(f"Grid directory already exists: {self.directory}."
                                      " Pass overwrite=True to overwrite.")
            else:
                self.logger.warning(f"Removing existing grid data: {self.directory}.")
                shutil.rmtree(self.directory)
        os.makedirs(self.directory)

    def _get_sample_filename(self, permutation_number):
        """
        """
        basename = f"sample_{permutation_number}.csv"
        return os.path.join(self.directory, basename)

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

    def _get_best_index(self, metric=None, df=None, kstest_min=None, func=np.nanargmax):
        """
        """
        if metric is None:
            metric = self._default_metric

        if df is None:
            df = self.load_metrics()
        indices = np.arange(df.shape[0])

        if kstest_min is not None:
            self.logger.debug(f"Applying kstest_min>{kstest_min:.2f}")
            cond = df["kstest_min"].values >= kstest_min
            df = df[cond].reset_index(drop=True)
            indices = indices[cond]

        values = df[metric].values
        index = func(values)

        self.logger.debug(f"Best index: {indices[index]}")

        return indices[index]

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

    def _identify_confident(self, metric=None, q=0.9):
        """ Identify best models within confidence interval. """
        if metric is None:
            metric = self._default_metric

        df = self.load_metrics()

        values = df[metric].values
        threshold = confidence_threshold(values, q=q)
        cond = values >= threshold

        self.logger.debug(f"Identified {cond.sum()} models for q={q:.2f}.")

        return cond
