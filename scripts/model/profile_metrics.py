import os
import cProfile

from udgsizes.core import get_config
from udgsizes.model.empirical import Model
from udgsizes.fitting.metrics import MetricEvaluator


if __name__ == "__main__":

    alpha = 3.5
    k = 1
    n_samples = 10000
    burnin = 1000

    model = Model("blue")
    params = {"rec_phys": [alpha], "uae_phys": [k]}

    config = get_config()
    filename = os.path.join(config['directories']['data'], 'profile_metrics.prof')

    df = model.sample(n_samples=n_samples, hyper_params=params, burnin=burnin)
    m = MetricEvaluator()

    def func():  # Not sure if this is necessary
        m.evaluate(df)

    cProfile.run("func()", filename=filename)
