from functools import partial
import numpy as np

from udgsizes.model.model import Model
from udgsizes.utils.cosmology import kpc_to_arcsec


class SbSizeModel(Model):
    """ The purpose of the EmpiricalModel class is to sample re, uae, z values. It is *not* to
    sample the fitting paramters of the individual likelihood terms (e.g. size power law).
    """
    _par_order = "rec_phys", "uae_phys", "redshift", "index", "colour_rest"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        # Add stellar mass
        df["logmstar"] = self._sb_calculator.logmstar_from_uae_phys(
            uae_phys=df["uae_phys"], rec=df["rec_obs"], redshift=df["redshift"])

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
                + self._log_likelihood_rec_phys(rec_phys, **rec_params)
                + self._log_likelihood_uae_phys(uae_phys, **uae_params)
                + self._log_likelihood_index_colour(index, colour, redshift)
                + self._log_likelihood_redshift(redshift))

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
