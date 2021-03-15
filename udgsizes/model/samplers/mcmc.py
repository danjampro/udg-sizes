import pandas as pd

import emcee

from udgsizes.base import UdgSizesBase


class Sampler(UdgSizesBase):

    def __init__(self, par_names=None, n_walkers=100, *args, **kwargs):
        self.par_names = par_names
        self.n_walkers = int(n_walkers)
        super().__init__(*args, **kwargs)

    def sample(self, func, initial_state, burnin=500, n_samples=5000, pool=None, get_sampler=False):
        """
        """
        self.logger.debug(f"Starting sampling with initial sate: {initial_state}.")
        sampler = emcee.EnsembleSampler(self.n_walkers, len(initial_state[0]), func, pool=pool)

        # Perform the burnin, updating initial state
        self.logger.debug(f"Performing burn-in for {burnin} iterations.")
        sampler.run_mcmc(initial_state, burnin, skip_initial_state_check=False, tune=True)
        sampler.reset()

        # Perform the main sampling
        self.logger.debug(f"Performing sampling for {n_samples} iterations.")
        sampler.run_mcmc(None, nsteps=n_samples, skip_initial_state_check=False)
        samples = sampler.get_chain(flat=True)

        self.logger.debug(f"Sampling complete. Mean acceptance fraction:"
                          f" {sampler.acceptance_fraction.mean():.2f}")

        # Turn into DataFrame
        if self.par_names is not None:
            df = pd.DataFrame()
            for i, par_name in enumerate(self.par_names):
                df[par_name] = samples[:, i]
            samples = df

        if get_sampler:
            return samples, sampler

        return samples

    def validate_inital_state(self, initial_state):
        """ Validate the initial state.
        See https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py.
        Args:
            initial_state (list): The initial state.
        Returns:
            bool: True if valid, else False.
        """
        state = emcee.state.State(initial_state, copy=True)
        return emcee.ensemble.walkers_independent(state.coords)
