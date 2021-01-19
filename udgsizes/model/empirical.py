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


# Define this here so the model is pickleable
def _redshift_func(z, interp):
    return interp(z)


class Model(UdgSizesBase):
    """ The purpose of the EmpiricalModel class is to sample re, uae, z values. It is *not* to
    sample the fitting paramters of the individual likelihood terms (e.g. size power law).
    """

    _par_order = "rec_phys", "uae_phys", "redshift", "index", "colour_rest"

    def __init__(self, model_name, use_interpolated_redshift=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cosmo = self.config["cosmology"]

        self.model_name = model_name
        model_config = self.config["models"][self.model_name]

        # Add model functions
        self._par_configs = {}
        self._likelihood_funcs = {}
        for par_name, par_config in model_config["variables"].items():

            self._par_configs[par_name] = par_config

            if par_name in ("index", "colour_rest"):  # Likelihood functions not configurable yet
                continue

            module_name = f"udgsizes.model.components.{par_config['func']}"
            func = load_module(module_name)

            # Add fixed function parameters
            with suppress(KeyError):
                func = partial(func, **par_config['pars'])

            # Explicitly add cosmology for redshift function
            if par_name == "redshift":
                func = partial(func, cosmo=self.cosmo)

            self._likelihood_funcs[par_name] = func

        # Load the recovery efficiency function
        self._recovery_efficiency = load_recovery_efficiency(config=self.config)

        #
        self._colour_classifier = load_classifier(config=self.config)
        self._colour_index_likelihood = self._colour_classifier.vars["blue"].pdf

        # Create a SB dimmer object
        self._pop_name = model_config["pop_name"]
        self._dimming = SBDimming(self._pop_name, config=self.config, logger=self.logger)
        self._redenning = Reddening(self._pop_name, config=self.config, logger=self.logger)

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
        initial_state = self._get_initial_state()
        log_likelihood = partial(self._log_likelihood, rec_params=hyper_params['rec_phys'],
                                 uae_params=hyper_params['uae_phys'])
        df = self._sampler.sample(func=log_likelihood, n_samples=n_samples,
                                  initial_state=initial_state, **kwargs)
        # Project to observable quantities
        df = self._project_sample(df)

        # Jiggle results
        if self._jiggler is not None:
            df['uae_obs_jig'], df['rec_obs_jig'], df["selected_jig"] = self._jiggler.jiggle(
                uae=df['uae_obs'].values, rec=df['rec_obs'].values)

        # Save to file if filename is given
        if filename is not None:
            self.logger.debug(f"Saving model samples to: {filename}.")
            df.to_csv(filename)

        return df

    def _log_likelihood(self, state, rec_params, uae_params):
        """ The log-likelihood for the full model.
        """
        rec_phys, uae_phys, redshift, index, colour = state
        return (self._log_likelihood_recovery(rec_phys, uae_phys, redshift)
                + self._log_likelihood_rec_phys(rec_phys, *rec_params)
                + self._log_likelihood_uae_phys(uae_phys, *uae_params)
                + self._log_likelihood_index_colour(index, colour, redshift)
                + self._log_likelihood_redshift(redshift))

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

    def _log_likelihood_uae_phys(self, uae_phys, *args, **kwargs):
        """ Calculate the contribution to the likelihood from the surface brightness.
        """
        return np.log(self._likelihood_funcs["uae_phys"](uae_phys, *args, **kwargs))

    def _log_likelihood_recovery(self, rec_phys, uae_phys, redshift):
        """ Calculate the contribution to the likelihood from the recovery efficiency.
        """
        rec_obs = kpc_to_arcsec(rec_phys, redshift=redshift, cosmo=self.cosmo)
        uae_obs = uae_phys + self._dimming(redshift=redshift)
        return np.log(self._recovery_efficiency(uae_obs, rec_obs))

    def _log_likelihood_index_colour(self, index, colour_rest, redshift):
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
        if not self._colour_classifier.predict(_index, colours=_colour_proj, which="blue")[0]:
            return -np.inf
        return np.log(self._colour_index_likelihood([index, colour_rest]))

    def _get_par_config(self, par_name, par_type):
        """ Convenience function to get parameter config.
        """
        return self._par_configs[par_name][par_type]

    def _get_initial_state(self):
        """ Retrieve the intial state as a list.
        """
        initial_state = list()
        for par_name in self._par_order:
            initial_state.append(self._get_par_config(par_name, "initial"))
        _initial_state = [f"{a}:{b}" for a, b in zip(self._par_order, initial_state)]
        self.logger.debug(f"Initial state: {_initial_state}")
        return initial_state

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
        df['rec_obs'] = kpc_to_arcsec(df['rec_phys'], redshift=redshift, cosmo=self.cosmo)
        df['uae_obs'] = df['uae_phys'] + self._dimming(redshift=redshift)
        df['colour_obs'] = df['colour_rest'] + self._redenning(redshift=redshift)
        return df
