from multiprocessing import Pool
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from udgsizes.model.empirical import Model
from udgsizes.utils.selection import select_samples
# from udgsizes.fitting.metrics import MetricEvaluator


def plot_hist(samples, key, bins=50):
    """
    """
    low = np.inf
    high = -np.inf
    for df in samples:
        cond = select_samples(uae=df["uae_obs_jig"], rec=df["rec_obs_jig"])
        values = df[key].values[cond]
        low = min(min(values), low)
        high = max(max(values), high)

    plt.figure()
    for df in samples:
        cond = select_samples(uae=df["uae_obs_jig"], rec=df["rec_obs_jig"])
        values = df[key].values[cond]
        plt.hist(values, range=(low, high), bins=bins, density=True, histtype="step")

    return


def func(i, n_samples, hyper_params, burnin):
    """
    """
    np.random.seed()
    return model.sample(n_samples=n_samples, hyper_params=params, burnin=burnin)


if __name__ == "__main__":

    alpha = 3.5
    k = 1
    n_samples = 50000
    burnin = 5000
    n_its = 5

    model = Model("blue")
    params = {"rec_phys": [alpha], "uae_phys": [k]}

    fn = partial(func, n_samples=n_samples, hyper_params=params, burnin=burnin)
    with Pool(5) as pool:
        samples = pool.map(fn, range(n_its))

    plot_hist(samples, key="uae_phys")
    plot_hist(samples, key="rec_phys")
    plt.show(block=False)
