from functools import partial
from multiprocessing import Pool
import numpy as np

from scipy.integrate import tplquad

from udgsizes.model.model import ModelBase
from udgsizes.utils.cosmology import kpc_to_arcsec
from udgsizes.utils import shen
from udgsizes.utils.selection import GR_MAX, GR_MIN
from udgsizes.model.colour import EmpiricalColourModel
# from udgsizes.model.kcorrector import EmpiricalKCorrector as KCorrector
from udgsizes.model.kcorrector import GamaKCorrector as KCorrector


class Model(ModelBase):

    _par_order = "rec_phys_offset", "logmstar", "redshift", "colour_rest_offset"

    def __init__(self, ignore_recov=False, *args, **kwargs):
        self._ignore_recov = ignore_recov
        super().__init__(*args, **kwargs)

        # Specify the k-corrector
        self._kcorrector = KCorrector(config=self.config, logger=self.logger)

        # Specify the colour model
        self._colour_model = EmpiricalColourModel(config=self.config, logger=self.logger)

        # Model hyper parameters
        self._logmstar_kink = self.model_config.get("logmstar_kink", 9)
        self.logger.debug(f"Kink mass: {self._logmstar_kink}")

    def sample(self, n_samples, hyper_params, filename=None, **kwargs):
        """ Sample the model, returning the model samples.
        Args:
            n_samples (int): The number of samples.
            hyper_params (dict): The set of model hyper params.
            filename (str, optional): The filename in which to save model samples.
        Returns:
            pd.DataFrame: The model samples.
        """
        # Get the initial state for each walker
        initial_state = self._get_initial_state(hyper_params=hyper_params)

        # Define the likelihood function
        log_likelihood = partial(self._log_likelihood, hyper_params=hyper_params)

        # Do the sampling
        df = self._sampler.sample(func=log_likelihood, n_samples=n_samples,
                                  initial_state=initial_state, **kwargs)

        # Project to observable quantities
        df = self._project_sample(df, alpha=hyper_params["rec_phys_offset"]["alpha"])

        # Jiggle results
        if self._jiggler is not None:
            self.logger.debug("Jiggling samples...")
            df['uae_obs_jig'], df['rec_obs_jig'], df["selected_jig"] = self._jiggler.jiggle(
                uae=df['uae_obs'].values, rec=df['rec_obs'].values)

        # Save samples to file
        if filename is not None:
            self.logger.debug(f"Saving model samples to: {filename}.")
            df.to_csv(filename)

        return df

    # Model likelihood

    def _log_likelihood(self, state, hyper_params):
        """ Calculate the model log-likelihood.
        Args:
            state (tuple): The state tuple.
            hyper_params (dict): The model hyper parameters.
        Returns:
            float: The log-likelihood.
        """
        # Unpack the state
        rec_phys_offset, logmstar, redshift, colour_rest_offset = state

        # Calculate the physical effective radius
        rec_phys_mean = self._mean_rec_phys(logmstar, **hyper_params['rec_phys_offset'])
        rec_phys = shen.apply_rec_offset(rec_phys_mean, rec_phys_offset)

        # Calculate the rest-frame colour
        colour_rest_mean = self._colour_model.get_mean_colour_rest(logmstar)
        colour_rest = colour_rest_mean + colour_rest_offset

        # Calculate the log-likelihood
        ll = (self._log_likelihood_rec_phys_offset(rec_phys_offset, logmstar=logmstar)
              + self._log_likelihood_logmstar(logmstar, **hyper_params['logmstar'])
              + self._log_likelihood_colour_rest_offset(colour_rest_offset)
              + self._log_likelihood_redshift(redshift))

        if not np.isfinite(ll):
            return -np.inf

        # Apply the recovery fraction to the LL
        if not self._ignore_recov:
            with np.errstate(divide='ignore'):  # Silence warnings
                ll += self._log_likelihood_recovery(logmstar, rec_phys, redshift, colour_rest)

        if not np.isfinite(ll):
            return -np.inf

        return ll

    def _log_likelihood_rec_phys_offset(self, rec_phys_offset, logmstar):
        """ Calculate the log-likelihood term for the physical effective radius.
        Args:
            rec_phys_offset (float): The size offset from the mean relation at this stellar mass.
            logmstar (float): The log10 stellar mass.
        Returns:
            float: The log-likelihood.
        """
        return np.log(self._likelihood_funcs["rec_phys_offset"](rec_phys_offset, logmstar=logmstar))

    def _log_likelihood_logmstar(self, logmstar, *args, **kwargs):
        """ Calculate the log-likelihood term for the stellar mass.
        Args:
            logmstar (float): The log10 stellar mass.
            *args, **kwargs: Parsed to configured likelihood function.
        Returns:
            float: The log-likelihood.
        """
        return np.log(self._likelihood_funcs["logmstar"](logmstar, *args, **kwargs))

    def _log_likelihood_colour_rest_offset(self, colour_rest_offset):
        """ Calculate the log-likelihood term for the rest frame colour.
        Args:
            colour_rest_offset (float): The offset from the mean colour trend in mags.
        Returns:
            float: The log-likelihood.
        """
        return np.log(self._colour_model.offset_pdf(colour_rest_offset))

    def _log_likelihood_recovery(self, logmstar, rec_phys, redshift, colour_rest):
        """ Calculate the log-likelihood term for the recovery fraction.
        Args:
            logmstar (float): The log10 stellar mass.
            rec_phys (float): The physical circularised effective radius.
            redshift (float): The redshift.
            colour_rest (float): The rest frame colour.
        Returns:
            float: The log-likelihood.
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

        rec_phys = shen.apply_rec_offset(rec_phys_mean, rec_phys_offset)
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

    def _mean_rec_phys(self, logmstar, alpha):
        """ Return the mean circularised effective radius for this stellar mass. """
        if logmstar > self._logmstar_kink:
            return shen.logmstar_to_mean_rec(logmstar)
        else:
            # Calculate the normalisation term
            gamma = shen.GAMMA * (10 ** self._logmstar_kink) ** (shen.ALPHA - alpha)
            # Return power law
            return gamma * (10 ** logmstar) ** alpha

    def integrand(self, logmstar, rec_phys_offset, colour_rest_offset, redshift, hyper_params):
        """
        """
        rec_phys_mean = self._mean_rec_phys(logmstar, **hyper_params['rec_phys_offset'])
        rec_phys = shen.apply_rec_offset(rec_phys_mean, rec_phys_offset)
        colour_rest_mean = self._colour_model.get_mean_colour_rest(logmstar)
        colour_rest = colour_rest_mean + colour_rest_offset

        # print(logmstar, rec_phys, colour_rest, redshift)

        a = self._log_likelihood_logmstar(logmstar=logmstar, **hyper_params['logmstar'])
        b = self._log_likelihood_redshift(redshift)
        c = self._log_likelihood_rec_phys_offset(rec_phys_offset, logmstar=logmstar)
        d = self._log_likelihood_colour_rest_offset(colour_rest_offset)
        e = self._log_likelihood_recovery(logmstar, rec_phys, redshift, colour_rest)

        result = np.exp(sum([a, b, c, d, e]))
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


class UDGModel(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _log_likelihood(self, state, hyper_params):
        """ The log-likelihood for the full model.
        """
        rec_phys_offset, logmstar, redshift, colour_rest_offset = state

        rec_phys_mean = self._mean_rec_phys(logmstar, **hyper_params['rec_phys_offset'])
        rec_phys = shen.apply_rec_offset(rec_phys_mean, rec_phys_offset)

        # Require UDG sizes
        if rec_phys < 1.5:
            return -np.inf

        return super()._log_likelihood(state, hyper_params)
