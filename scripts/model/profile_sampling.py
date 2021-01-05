import os
import cProfile

from udgsizes.core import get_config
from udgsizes.model.empirical import Model

if __name__ == "__main__":

    alpha = 3.5
    k = 1
    n_samples = 2000
    burnin = 500

    config = get_config()
    filename = os.path.join(config['directories']['data'], 'profile_sampling.prof')

    model = Model("blue")
    params = {"rec_phys": [alpha], "uae_phys": [k]}

    def func():  # Not sure if this is necessary
        model.sample(n_samples=n_samples, hyper_params=params, burnin=burnin)

    cProfile.run("func()", filename=filename)

    # snakeviz profile_sampling.prof
