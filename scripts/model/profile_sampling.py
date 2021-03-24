import os
import cProfile

from udgsizes.core import get_config
from udgsizes.model.utils import create_model


if __name__ == "__main__":

    n_samples = 400
    burnin = 100
    model_name = "blue_sedgwick_shen"

    config = get_config()
    filename = os.path.join(config['directories']['data'], 'profile_sampling.prof')

    model = create_model(model_name)

    hyper_params = {"rec_phys_offset": {"alpha": 0.4}, "logmstar": {"a": -1.45}}

    def func():  # Not sure if this is necessary
        model.sample(n_samples=n_samples, hyper_params=hyper_params, burnin=burnin)

    cProfile.run("func()", filename=filename)

    # snakeviz profile_sampling.prof
