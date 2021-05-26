import itertools
import random
from contextlib import suppress
from functools import partial

import numpy as np
from scipy.interpolate import interp1d

from udgsizes.base import UdgSizesBase
from udgsizes.utils.library import load_module
from udgsizes.model.kcorrector import EmpiricalKCorrector
from udgsizes.obs.recovery import load_recovery_efficiency
from udgsizes.model.samplers.mcmc import Sampler
from udgsizes.model.jiggle import Jiggler
from udgsizes.obs.index_colour import load_classifier
from udgsizes.utils.mstar import EmpiricalSBCalculator
from udgsizes.model.utils import get_model_config

PARS_SKIP = ("index", "colour_rest", "colour_rest_offset")
# PARS_SKIP = ("colour_rest", "colour_rest_offset")


# Define this here so the model is pickleable
def _redshift_func(z, interp):
    return interp(z)


class ModelBase(UdgSizesBase):
    """ The purpose of the EmpiricalModel class is to sample re, uae, z values. It is *not* to
    sample the fitting paramters of the individual likelihood terms (e.g. size power law).
    """
    _par_order = None

    def __init__(self, model_name, use_interpolated_redshift=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cosmo = self.config["cosmology"]

        self.model_name = model_name

        self.model_config = get_model_config(model_name, config=self.config)
        self._pop_name = self.model_config["pop_name"]

        # Add model functions
        self._par_configs = {}
        self._likelihood_funcs = {}
        for par_name, par_config in self.model_config["variables"].items():

            self._par_configs[par_name] = par_config

            if par_name in PARS_SKIP:  # Likelihood functions not configurable yet
                continue

            module_name = f"udgsizes.model.components.{par_config['func']}"
            func = load_module(module_name)

            # Add fixed function parameters
            with suppress(KeyError):
                func = partial(func, **par_config['pars'])

            # Explicitly add cosmology if required
            if par_config.get("cosmo", False):
                self.logger.debug(f"Using cosmology for '{par_name}' variable.")
                func = partial(func, cosmo=self.cosmo)

            self._likelihood_funcs[par_name] = func

        self._colour_classifier = load_classifier(config=self.config)
        self._colour_index_likelihood = self._colour_classifier.vars["blue"].pdf

        self._kcorrector = EmpiricalKCorrector(config=self.config, logger=self.logger)
        self._sb_calculator = EmpiricalSBCalculator(config=self.config, logger=self.logger)

        # Load the recovery efficiency function
        self._recovery_efficiency = load_recovery_efficiency(config=self.config)

        # Prepare the sampler
        self._sampler = Sampler(par_names=self._par_order, config=self.config, logger=self.logger)

        # Create a jiggler object
        self._jiggler = Jiggler(config=self.config, logger=self.logger)

        # Resample the redshift for speed up
        if use_interpolated_redshift:
            self._use_interpolated_redshift()

    # Properties

    @property
    def n_parameters(self):
        return len(self._par_order)

    # Public methods

    def get_uae_phys(self, logmstar, rec, redshift, colour_rest):
        """ Override method to use empirical fit to ML ratio.
        """
        return self._sb_calculator.calculate_uae_phys(logmstar=logmstar, rec=rec, redshift=redshift,
                                                      colour_rest=colour_rest)

    def get_kcorr_r(self, colour_rest, redshift):
        """
        """
        return self._kcorrector.calculate_kr(colour_rest, redshift=redshift)

    def get_kcorr_gr(self, colour_rest, redshift):
        """
        """
        return self._kcorrector.calculate_kgr(colour_rest, redshift=redshift)

    # Model likelihood

    def _log_likelihood_redshift(self, redshift):
        """ Calculate the contribution to the likelihood from the redshift.
        """
        if redshift <= self._get_par_config("redshift", "min"):
            return -np.inf
        if redshift > self._get_par_config("redshift", "max"):
            return -np.inf
        return np.log(self._likelihood_funcs["redshift"](redshift))

    # Private methods

    def _get_par_config(self, par_name, par_type):
        """ Convenience function to get parameter config.
        Args:
            par_name (str): The parameter name.
            par_type (str): The parameter type.
        """
        return self._par_configs[par_name][par_type]

    def _get_initial_state(self, hyper_params, n_retries=3, retry_index=0):
        """ Search for a set of valid initial states that result in a finite LL.
        Returns:
            list: A list of initial parameters.
        """
        self.logger.info("Determining initial state before sampling.")
        inival_matrix = []
        for par_name in self._par_order:

            # Get dict of starting value ranges
            inival_dict = self._get_par_config(par_name, "initial")

            # Append array of possible starting values
            try:
                inivals = [float(inival_dict)]  # The case where a single value is specified
            except TypeError:
                maxval = inival_dict["max"] + inival_dict["step"]
                inivals = np.arange(inival_dict["min"], maxval, inival_dict["step"])
            inival_matrix.append(inivals)

        # Get random permutations of potential initial states
        perms = list(itertools.product(*inival_matrix))
        random.shuffle(perms)

        # Loop over inival permutations to find a finite set
        inivals = []
        for inival_perm in perms:

            # Check if we have finished
            if len(inivals) == self._sampler.n_walkers:
                break

            # Check if any of the values are already in the existing states
            if len(inivals) != 0:
                should_skip = False
                for i in range(self.n_parameters):
                    if all([_[i] == inival_perm[i] for _ in inivals]):
                        should_skip = True
                        break
                if should_skip:
                    continue

            # Check if the state returns a finite likelihood
            if np.isfinite(self._log_likelihood(inival_perm, hyper_params=hyper_params)):
                inivals.append(inival_perm)
                continue

        if len(inivals) != self._sampler.n_walkers:
            raise RuntimeError(f"Unable to determine initial state for {self} with hyper params:"
                               f" {hyper_params}.")

        if not self._sampler.validate_inital_state(inivals):
            if retry_index >= n_retries:
                raise RuntimeError("Invalid initial state after max retries reached for hyper"
                                   f" params: {hyper_params}")

            # Try again!
            self.logger.warning("Initial state invalid. Retrying.")
            return self._get_initial_state(hyper_params, retry_index=retry_index + 1)

        state_dict = [f"{a}:{b}" for a, b in zip(self._par_order, inivals)]
        self.logger.debug(f"Successfully determined initial state: {state_dict}")

        return inivals

    def _use_interpolated_redshift(self, n_samples=500):
        """ Interpolate the redshift function to make sampling faster.
        Args:
            n_samples (int): The number of redshift samples to use for interpolation.
        """
        self.logger.debug("Using interpolated redshift function.")

        zmin = self._get_par_config("redshift", "min")
        zmax = self._get_par_config("redshift", "max")

        zz = np.linspace(zmin, zmax, n_samples)
        yy = self._likelihood_funcs["redshift"](zz)

        interp = interp1d(zz, yy)

        self._likelihood_funcs["redshift"] = partial(_redshift_func, interp=interp)

    def _project_sample(self, *args, **kwargs):
        """ Project physical units to observable quantities. """
        raise NotImplementedError
