from functools import partial
from multiprocessing import Pool
import numpy as np

from scipy.integrate import tplquad

from udgsizes.model.model import Model
from udgsizes.utils.cosmology import kpc_to_arcsec
from udgsizes.utils import shen
from udgsizes.utils.selection import GR_MAX, GR_MIN
from udgsizes.model.colour import EmpiricalColourModel
# from udgsizes.model.kcorrector import DummyKcorrector as KCorrector
from udgsizes.model.kcorrector import InvEmpiricalKCorrector as KCorrector
# from udgsizes.model.kcorrector import EmpiricalKCorrector as KCorrector


def apply_rec_offset(rec_phys_mean, rec_phys_offset):
    """
    """
    return np.exp(np.log(rec_phys_mean) + rec_phys_offset)


class SmfDwarfModel(Model):

    _par_order = "rec_phys_offset", "logmstar", "redshift", "colour_rest_offset"

    def __init__(self, ignore_recov=False, *args, **kwargs):
        self._ignore_recov = ignore_recov
        super().__init__(*args, **kwargs)

        self._kcorrector = KCorrector(config=self.config, logger=self.logger)

        self._colour_model = EmpiricalColourModel(config=self.config, logger=self.logger)

    def sample(self, n_samples, hyper_params, filename=None, **kwargs):
        """ Sample the model, returning a pd.DataFrame containing the posterior distribution.
        """
        initial_state = self._get_initial_state(hyper_params=hyper_params)

        log_likelihood = partial(self._log_likelihood, hyper_params=hyper_params)

        df = self._sampler.sample(func=log_likelihood, n_samples=n_samples,
                                  initial_state=initial_state, **kwargs)

        # Project to observable quantities
        df = self._project_sample(df, alpha=hyper_params["rec_phys_offset"]["alpha"])

        # Jiggle results
        if self._jiggler is not None:
            self.logger.debug("Jiggling samples...")
            df['uae_obs_jig'], df['rec_obs_jig'], df["selected_jig"] = self._jiggler.jiggle(
                uae=df['uae_obs'].values, rec=df['rec_obs'].values)

        # Save to file if filename is given
        if filename is not None:
            self.logger.debug(f"Saving model samples to: {filename}.")
            df.to_csv(filename)

        return df

    # Model likelihood

    def _log_likelihood(self, state, hyper_params):
        """ The log-likelihood for the full model.
        """
        rec_phys_offset, logmstar, redshift, colour_rest_offset = state

        rec_phys_mean = self._mean_rec_phys(logmstar, **hyper_params['rec_phys_offset'])
        rec_phys = apply_rec_offset(rec_phys_mean, rec_phys_offset)

        colour_rest_mean = self._colour_model.get_mean_colour_rest(logmstar)
        colour_rest = colour_rest_mean + colour_rest_offset

        ll = (self._log_likelihood_rec_phys_offset(rec_phys_offset, logmstar=logmstar)
              + self._log_likelihood_logmstar(logmstar, **hyper_params['logmstar'])
              + self._log_likelihood_colour_rest_offset(colour_rest_offset)
              + self._log_likelihood_redshift(redshift))

        if not np.isfinite(ll):
            return -np.inf

        if not self._ignore_recov:
            with np.errstate(divide='ignore'):  # Silence warnings
                ll += self._log_likelihood_recovery(logmstar, rec_phys, redshift, colour_rest)

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

    def _log_likelihood_colour_rest_offset(self, colour_rest_offset):
        """
        """
        return np.log(self._colour_model.offset_pdf(colour_rest_offset))

    def _log_likelihood_recovery(self, logmstar, rec_phys, redshift, colour_rest):
        """ Calculate the contribution to the likelihood from the recovery efficiency.
        """
        # Apply colour selection function
        colour_obs = colour_rest + self.get_kcorr_gr(colour_rest, redshift)
        if colour_obs >= GR_MAX:
            return -np.inf
        if colour_obs < GR_MIN:
            return -np.inf

        rec_obs = self.kpc_to_arcsec(rec_phys, redshift=redshift)

        uae_phys = self.get_uae_phys(logmstar, rec_obs, redshift=redshift, colour_rest=colour_rest)
        uae_obs = uae_phys + self.get_kcorr_r(colour_rest, redshift=redshift)

        return np.log(self._recovery_efficiency(uae_obs, rec_obs))

    # Private methods

    def _project_sample(self, df, alpha):
        """ Project physical units to observable quantities.
        Args:
            df (pd.DataFrame): The sample catalogue.
            alpha (float): The effective radius power law.
        Returns:
            pd.DataFrame: The catalouge with projected quantities.
        """
        redshift = df["redshift"].values
        logmstar = df["logmstar"].values

        colour_rest_av = np.array([self._colour_model.get_mean_colour_rest(_) for _ in logmstar])
        colour_rest = colour_rest_av + df["colour_rest_offset"].values
        df["colour_rest"] = colour_rest

        self.logger.debug("Projecting structural parameters...")
        rec_phys_mean = np.array([
            self._mean_rec_phys(lm, alpha=alpha) for lm in df["logmstar"].values])
        rec_phys_offset = df["rec_phys_offset"].values

        rec_phys = apply_rec_offset(rec_phys_mean, rec_phys_offset)
        df["rec_phys"] = rec_phys

        rec_obs = kpc_to_arcsec(df['rec_phys'], redshift=redshift, cosmo=self.cosmo)
        df["rec_obs"] = rec_obs

        uae_phys = np.array([self.get_uae_phys(
            logmstar[_], rec=rec_obs[_], redshift=redshift[_], colour_rest=colour_rest[_]
            ) for _ in range(df.shape[0])])
        df["uae_phys"] = uae_phys

        # Identify dwarfs and UDGs
        df["is_dwarf"] = logmstar < 9
        df["is_udg"] = (rec_phys > 1.5) & (uae_phys > 24)

        # Apply the k-corrections
        self.logger.debug("Applying k-corrections...")

        kr = [self.get_kcorr_r(a, b) for (a, b) in zip(colour_rest, redshift)]
        df['uae_obs'] = df['uae_phys'] + np.array(kr)

        kgr = [self.get_kcorr_gr(a, b) for (a, b) in zip(colour_rest, redshift)]
        df['colour_obs'] = df['colour_rest'] + np.array(kgr)

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

    def integrand(self, logmstar, rec_phys_offset, colour_rest_offset, redshift, hyper_params):
        """
        """
        rec_phys_mean = self._mean_rec_phys(logmstar, **hyper_params['rec_phys_offset'])
        rec_phys = apply_rec_offset(rec_phys_mean, rec_phys_offset)
        colour_rest_mean = self._colour_model.get_mean_colour_rest(logmstar)
        colour_rest = colour_rest_mean + colour_rest_offset

        # print(logmstar, rec_phys, colour_rest, redshift)

        a = self._log_likelihood_logmstar(logmstar=logmstar, **hyper_params['logmstar'])
        b = self._log_likelihood_redshift(redshift)
        c = self._log_likelihood_rec_phys_offset(rec_phys_offset, logmstar=logmstar)
        d = self._log_likelihood_colour_rest_offset(colour_rest_offset)
        e = self._log_likelihood_recovery(logmstar, rec_phys, redshift, colour_rest)

        result = sum([np.exp(_) for _ in [a, b, c, d, e]])
        if not np.isfinite(result):
            return 0
        return result

    def marginal_logmstar(self, logmstar_min, logmstar_max, n_samples, hyper_params, nproc=1,
                          **kwargs):
        """
        """
        logmstar = np.linspace(logmstar_min, logmstar_max, n_samples)
        func = partial(self._marginal_logmstar, hyper_params=hyper_params, **kwargs)

        if nproc == 1:
            result = np.array([func(lm) for lm in logmstar])
        else:
            with Pool(nproc) as pool:
                result = np.array(pool.map(func, logmstar))

        return logmstar, result

    def _marginal_logmstar(self, logmstar, hyper_params, epsabs=0, epsrel=1E-3):
        """
        """
        lims_rec_phys_offset = -3., 3.
        lims_colour_rest_offset = -1., 1.
        lims_redshift = 0.0001, 0.5

        def integrand(rec_phys_offset, colour_rest_offset, redshift):
            return self.integrand(logmstar, rec_phys_offset, colour_rest_offset, redshift=redshift,
                                  hyper_params=hyper_params)

        # For some reason the limits go the other way around
        lims = [*lims_rec_phys_offset, *lims_colour_rest_offset, *lims_redshift][::-1]

        integral = tplquad(integrand, *lims, epsabs=epsabs, epsrel=epsrel)[0]

        return integral
