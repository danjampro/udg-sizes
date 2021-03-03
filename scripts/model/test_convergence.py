from multiprocessing import Pool
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from udgsizes.model.smf_size import SmfSizeModel as Model
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
        plt.hist(values, range=(low, high), bins=bins, density=True, histtype="step",
                 label=f"N samples = {df.shape[0]}")

    plt.legend(loc="best")


def func(n_samples, hyper_params, burnin, model):
    """
    """
    np.random.seed()
    return model.sample(n_samples=n_samples, hyper_params=params, burnin=burnin)


if __name__ == "__main__":

    alpha = -0.5
    k = 1
    n_samples = 5000
    burnin = 1000
    n_its = 5

    # n_sample_range = np.linspace(0, n_samples, n_its+1)[1:].astype('int')
    n_sample_range = [5000, 10000, 20000, 40000]

    model = Model("blue_baldry")
    params = {"rec_phys": {"alpha": k}, "logmstar":  {"a": alpha}}

    fn = partial(func, hyper_params=params, burnin=burnin, model=model)
    samples = [fn(n) for n in n_sample_range]
    """
    with Pool(5) as pool:
        samples = pool.map(fn, n_sample_range)
    """

    plot_hist(samples, key="uae_phys")
    plot_hist(samples, key="rec_phys")
    plt.show(block=False)
