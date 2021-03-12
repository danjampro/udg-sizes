from functools import partial
import numpy as np

from udgsizes.model.model import Model
from udgsizes.utils.cosmology import kpc_to_arcsec
from udgsizes.utils import shen

COLOUR_OFFSET = 0.035


def apply_rec_offset(rec_phys_mean, rec_phys_offset):
    """
    """
    return np.exp(np.log(rec_phys_mean) + rec_phys_offset)


class SmfDwarfModel(Model):

    _par_order = "rec_phys_offset", "logmstar", "redshift", "index", "colour_rest"

    def __init__(self, ignore_recov=False, *args, **kwargs):
        self._ignore_recov = ignore_recov
        super().__init__(*args, **kwargs)

    def sample(self, n_samples, hyper_params, filename=None, **kwargs):
        """ Sample the model, returning a pd.DataFrame containing the posterior distribution.
        """
        initial_state = np.array([self._get_initial_state(
            hyper_params=hyper_params) for _ in range(self._sampler.n_walkers)])

        log_likelihood = partial(self._log_likelihood, hyper_params=hyper_params)

        df = self._sampler.sample(func=log_likelihood, n_samples=n_samples,
                                  initial_state=initial_state, **kwargs)

        # Project to observable quantities
        df = self._project_sample(df, alpha=hyper_params["rec_phys_offset"]["alpha"])

        # Jiggle results
        if self._jiggler is not None:
            df['uae_obs_jig'], df['rec_obs_jig'], df["selected_jig"] = self._jiggler.jiggle(
                uae=df['uae_obs'].values, rec=df['rec_obs'].values)

        # Save to file if filename is given
        if filename is not None:
            self.logger.debug(f"Saving model samples to: {filename}.")
            df.to_csv(filename)

        return df

    def calculate_uae_phys(self, logmstar, rec, redshift):
        return self._sb_calculator.calculate_uae_phys(logmstar, rec, redshift)

    def _log_likelihood(self, state, hyper_params):
        """ The log-likelihood for the full model.
        """
        rec_phys_offset, logmstar, redshift, index, colour = state

        rec_phys_mean = self._mean_rec_phys(logmstar, **hyper_params['rec_phys_offset'])
        rec_phys = apply_rec_offset(rec_phys_mean, rec_phys_offset)

        ll = (self._log_likelihood_rec_phys_offset(rec_phys_offset, logmstar=logmstar)
              + self._log_likelihood_logmstar(logmstar, **hyper_params['logmstar'])
              + self._log_likelihood_index_colour(logmstar, colour, index, redshift)
              + self._log_likelihood_redshift(redshift))

        if not np.isfinite(ll):
            return -np.inf

        if not self._ignore_recov:
            ll += self._log_likelihood_recovery(rec_phys, logmstar, redshift)

        if not np.isfinite(ll):
            return -np.inf

        return ll

    def _log_likelihood_rec_phys_offset(self, rec_phys_offset, logmstar):
        """
        """
        return np.log(self._likelihood_funcs["rec_phys_offset"](rec_phys_offset, logmstar=logmstar))

    def _log_likelihood_logmstar(self, logmstar, *args, **kwargs):
        """ Calculate the contribution to the likelihood from the stellar mass. """
        return np.log(self._likelihood_funcs["logmstar"](logmstar, *args, **kwargs))

    def _log_likelihood_recovery(self, rec_phys, logmstar, redshift):
        """ Calculate the contribution to the likelihood from the recovery efficiency.
        """
        # Calculate projected quantities
        rec_obs = kpc_to_arcsec(rec_phys, redshift=redshift, cosmo=self.cosmo)
        uae_obs = self._sb_calculator.calculate_uae(logmstar, rec_obs, redshift)

        # Return recovery fraction
        return np.log(self._recovery_efficiency(uae_obs, rec_obs))

    def _log_likelihood_index_colour(self, logmstar, colour_rest, index, redshift):
        """
        """
        colour_proj = colour_rest + self._redenning(redshift)

        # Apply late-type galaxy selection criteria
        if index >= 2.5:
            return -np.inf

        # TODO: Streamline
        _index = np.array([index])
        _colour_proj = np.array([colour_proj])

        # Return zero-likelihood if the sample does not satisfy selection criteria
        if not self._colour_classifier.predict(_index, colours=_colour_proj, which="blue")[0]:
            return -np.inf

        # TODO: Offset rest colour

        return np.log(self._colour_index_likelihood([index, colour_rest + COLOUR_OFFSET]))

    def _get_par_config(self, par_name, par_type):
        """ Convenience function to get parameter config. """
        return self._par_configs[par_name][par_type]

    def _project_sample(self, df, alpha):
        """ Project physical units to observable quantities.
        Args:
            df (pd.DataFrame): The sample catalogue.
            alpha (float): The effective radius power law.
        Returns:
            pd.DataFrame: The catalouge with projected quantities.
        """
        redshift = df["redshift"].values

        rec_phys_mean = np.array([
            self._mean_rec_phys(lm, alpha=alpha) for lm in df["logmstar"].values])
        rec_phys_offset = df["rec_phys_offset"].values

        df["rec_phys"] = apply_rec_offset(rec_phys_mean, rec_phys_offset)
        df["rec_obs"] = kpc_to_arcsec(df['rec_phys'], redshift=redshift, cosmo=self.cosmo)

        rec = df["rec_obs"].values
        logmstar = df["logmstar"].values

        # Identify dwarfs
        df["is_dwarf"] = df["logmstar"].values < 9

        # Identify UDGs
        df["uae_phys"] = [self.calculate_uae_phys(
            logmstar[_], rec=rec[_], redshift=redshift[_]) for _ in range(rec.size)]
        df["is_udg"] = (df["rec_phys"].values > 1.5) & (df["uae_phys"].values > 24)

        # Apply k-corrections
        df['uae_obs'] = df['uae_phys'] + self._dimming(redshift=redshift)
        df['colour_obs'] = df['colour_rest'] + self._redenning(redshift=redshift)

        return df

    def _mean_rec_phys(self, logmstar, alpha, logmstar_kink=9):
        """ Return the mean circularised effective radius for this stellar mass. """
        if logmstar > logmstar_kink:
            return shen.logmstar_to_mean_rec(logmstar)
        else:
            # Calculate the normalisation term
            gamma = shen.GAMMA * (10 ** logmstar_kink) ** (shen.ALPHA - alpha)
            # Return power law
            return gamma * (10 ** logmstar) ** alpha
