from functools import partial
import numpy as np

from udgsizes.model.model import Model
from udgsizes.utils.mstar import SbCalculator
from udgsizes.utils.cosmology import kpc_to_arcsec


class SmfSizeModel(Model):

    _par_order = "rec_phys", "logmstar", "redshift", "index", "colour_rest"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sb_calculator = SbCalculator(population_name=self._pop_name, cosmo=self.cosmo,
                                           mlratio=self.model_config["mlratio"])

    def sample(self, n_samples, hyper_params, filename=None, **kwargs):
        """ Sample the model, returning a pd.DataFrame containing the posterior distribution.
        """
        initial_state = self._get_initial_state()
        log_likelihood = partial(self._log_likelihood, rec_params=hyper_params['rec_phys'],
                                 smf_params=hyper_params['logmstar'])
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

    def _log_likelihood(self, state, rec_params, smf_params):
        """ The log-likelihood for the full model.
        """
        rec_phys, logmstar, redshift, index, colour = state
        return (self._log_likelihood_recovery(rec_phys, logmstar, redshift)
                + self._log_likelihood_rec_phys(rec_phys, *rec_params)
                + self._log_likelihood_logmstar(logmstar, *smf_params)
                + self._log_likelihood_index_colour(index, colour, redshift)
                + self._log_likelihood_redshift(redshift))

    def _log_likelihood_logmstar(self, logmstar, *args, **kwargs):
        """ Calculate the contribution to the likelihood from the stellar mass.
        """
        # return np.log(self._likelihood_funcs["logmstar"](logmstar, *args, **kwargs))
        return np.log(self._likelihood_funcs["logmstar"](logmstar))  # Already logged?

    def _log_likelihood_recovery(self, rec_phys, logmstar, redshift):
        """ Calculate the contribution to the likelihood from the recovery efficiency.
        """
        rec_obs = kpc_to_arcsec(rec_phys, redshift=redshift, cosmo=self.cosmo)
        uae_obs = self._sb_calculator.calculate_uae(logmstar, rec_obs, redshift)
        return np.log(self._recovery_efficiency(uae_obs, rec_obs))

    def _get_par_config(self, par_name, par_type):
        """ Convenience function to get parameter config.
        """
        return self._par_configs[par_name][par_type]

    def _project_sample(self, df):
        """ Project physical units to observable quantities """
        redshift = df["redshift"].values
        df["rec_obs"] = kpc_to_arcsec(df['rec_phys'], redshift=redshift, cosmo=self.cosmo)
        rec = df["rec_obs"].values
        logmstar = df["logmstar"].values
        df["uae_phys"] = self._sb_calculator.calculate_uae(logmstar, rec, redshift)
        return super()._project_sample(df=df)

    def _get_uae_phys(self):
        pass
