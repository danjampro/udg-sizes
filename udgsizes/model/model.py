import itertools
from contextlib import suppress
from functools import partial

import numpy as np
from scipy.interpolate import interp1d

from udgsizes.base import UdgSizesBase
from udgsizes.utils.library import load_module
from udgsizes.utils.cosmology import kpc_to_arcsec
from udgsizes.utils.dimming import SBDimming, Reddening
from udgsizes.obs.recovery import load_recovery_efficiency
from udgsizes.model.samplers.mcmc import Sampler
from udgsizes.model.jiggle import Jiggler
from udgsizes.obs.index_colour import load_classifier
from udgsizes.utils.mstar import SbCalculator
from udgsizes.model.utils import get_model_config


# Define this here so the model is pickleable
def _redshift_func(z, interp):
    return interp(z)


class Model(UdgSizesBase):
    """ The purpose of the EmpiricalModel class is to sample re, uae, z values. It is *not* to
    sample the fitting paramters of the individual likelihood terms (e.g. size power law).
    """
    _par_order = None

    def __init__(self, model_name, use_interpolated_redshift=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cosmo = self.config["cosmology"]

        self.model_name = model_name

        self.model_config = get_model_config(model_name, config=self.config)

        # Add model functions
        self._par_configs = {}
        self._likelihood_funcs = {}
        for par_name, par_config in self.model_config["variables"].items():

            self._par_configs[par_name] = par_config

            if par_name in ("index", "colour_rest"):  # Likelihood functions not configurable yet
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

        # Load the recovery efficiency function
        self._recovery_efficiency = load_recovery_efficiency(config=self.config)

        #
        self._colour_classifier = load_classifier(config=self.config)
        self._colour_index_likelihood = self._colour_classifier.vars["blue"].pdf

        # Create a SB dimmer object
        self._pop_name = self.model_config["pop_name"]
        self._dimming = SBDimming(self._pop_name, config=self.config, logger=self.logger)
        self._redenning = Reddening(self._pop_name, config=self.config, logger=self.logger)

        self._sb_calculator = SbCalculator(population_name=self._pop_name, cosmo=self.cosmo,
                                           mlratio=self.model_config["mlratio"])

        # Prepare the sampler
        self._sampler = Sampler(par_names=self._par_order, config=self.config, logger=self.logger)

        # Create a jiggler object
        self._jiggler = Jiggler(config=self.config, logger=self.logger)

        # Resample the redshift for speed up
        if use_interpolated_redshift:
            self._use_interpolated_redshift()

    def sample(self, n_samples, hyper_params, filename=None, **kwargs):
        """ Sample the model, returning a pd.DataFrame containing the posterior distribution.
        """
        raise NotImplementedError

    def _log_likelihood(self, state, rec_params, uae_params):
        """ The log-likelihood for the full model.
        """
        raise NotImplementedError

    def _log_likelihood_redshift(self, redshift):
        """ Calculate the contribution to the likelihood from the redshift.
        """
        if redshift <= self._get_par_config("redshift", "min"):
            return -np.inf
        if redshift > self._get_par_config("redshift", "max"):
            return -np.inf
        return np.log(self._likelihood_funcs["redshift"](redshift))

    def _log_likelihood_rec_phys(self, rec_phys, *args, **kwargs):
        """ Calculate the contribution to the likelihood from the physical size.
        """
        if rec_phys <= 0:
            return -np.inf
        return np.log(self._likelihood_funcs["rec_phys"](rec_phys, *args, **kwargs))

    def _log_likelihood_recovery(self, *args, **kwargs):
        """ Calculate the contribution to the likelihood from the recovery efficiency.
        """
        raise NotImplementedError

    def _log_likelihood_index_colour(self, logmstar, index, colour_rest, redshift):
        """
        """
        colour_proj = colour_rest + self._redenning(redshift)
        if colour_proj > 1:
            return -np.inf
        if colour_proj < 0:
            return -np.inf

        # TODO: Streamline
        _index = np.array([index])
        _colour_proj = np.array([colour_proj])

        # Return zero-likelihood if the sample does not satisfy selection criteria
        if not self._colour_classifier.predict(_index, colours=_colour_proj, which="blue")[0]:
            return -np.inf

        return np.log(self._colour_index_likelihood([index, colour_rest]))

    def _get_par_config(self, par_name, par_type):
        """ Convenience function to get parameter config.
        """
        return self._par_configs[par_name][par_type]

    def _get_initial_state(self, hyper_params):
        """ Search for a set of initial parameters that provide a finite LL.
        Returns:
            list: A list of initial parameters.
        """
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
            np.random.shuffle(inivals)
            inival_matrix.append(inivals)

        # Loop over inival permutations to find a finite set
        inivals = None
        for inival_perm in itertools.product(*inival_matrix):
            if np.isfinite(self._log_likelihood(inival_perm, hyper_params=hyper_params)):
                inivals = inival_perm
                break
        if inivals is None:
            raise RuntimeError(f"No finite set of initial values for {self} with hyper params:"
                               f" {hyper_params}.")

        _initial_state = [f"{a}:{b}" for a, b in zip(self._par_order, inivals)]
        self.logger.debug(f"Initial state: {_initial_state}")

        return inivals

    def _use_interpolated_redshift(self, n_samples=500):
        """ Interpolate the redshift function to make sampling faster.
        """
        self.logger.debug("Using interpolated redshift function.")
        zmin = self._get_par_config("redshift", "min")
        zmax = self._get_par_config("redshift", "max")
        zz = np.linspace(zmin, zmax, n_samples)
        yy = self._likelihood_funcs["redshift"](zz)
        interp = interp1d(zz, yy)

        self._likelihood_funcs["redshift"] = partial(_redshift_func, interp=interp)

    def _project_sample(self, df):
        """ Project physical units to observable quantities """
        redshift = df['redshift'].values
        self.logger.debug("Projecting sample to observable quantities.")
        # Save time
        if "rec_obs" not in df.columns:
            df['rec_obs'] = kpc_to_arcsec(df['rec_phys'], redshift=redshift, cosmo=self.cosmo)
        df['uae_obs'] = df['uae_phys'] + self._dimming(redshift=redshift)
        df['colour_obs'] = df['colour_rest'] + self._redenning(redshift=redshift)
        return df
