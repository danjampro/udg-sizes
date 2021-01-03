import numpy as np
import pandas as pd

import emcee

from udgsizes.base import UdgSizesBase


class Sampler(UdgSizesBase):

    def __init__(self, par_names=None, *args, **kwargs):
        self.par_names = par_names
        super().__init__(*args, **kwargs)

    def sample(self, func, initial_state, n_walkers=10, burnin=500, n_samples=5000,
               pool=None, peturb_factor=0.01):
        """
        """
        self.logger.debug(f"Starting sampling with initial sate: {initial_state}.")
        sampler = emcee.EnsembleSampler(n_walkers, len(initial_state), func, pool=pool)

        # Peturb the initial state (required)
        _initial_state = np.array([initial_state for _ in range(n_walkers)])
        _initial_state += np.random.normal(0, peturb_factor*_initial_state)

        # Perform the burnin, updating initial state
        self.logger.debug(f"Performing burn-in for {burnin} iterations.")
        sampler.run_mcmc(_initial_state, burnin, skip_initial_state_check=True, tune=True)
        sampler.reset()

        # Perform the main sampling
        self.logger.debug(f"Performing sampling for {n_samples} iterations.")
        sampler.run_mcmc(None, nsteps=n_samples)
        samples = sampler.get_chain(flat=True)

        self.logger.debug(f"Sampling complete. Mean acceptance fraction:"
                          f" {sampler.acceptance_fraction.mean():.2f}")

        # Turn into DataFrame
        if self.par_names is not None:
            df = pd.DataFrame()
            for i, par_name in enumerate(self.par_names):
                df[par_name] = samples[:, i]
            return df
        return samples
