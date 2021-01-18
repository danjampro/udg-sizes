import random
import time
import numpy as np
import matplotlib.pyplot as plt

from udgsizes.model.empirical import Model
from udgsizes.fitting.metrics import MetricEvaluator


if __name__ == "__main__":

    alpha = 3.5
    k = 1
    n_samples = 10000
    burnin = 1000
    n_walkers = 10
    ns = [10000, 20000, 40000, 80000, 100000]
    refn = 1000000

    model = Model("blue")
    params = {"rec_phys": [alpha], "uae_phys": [k]}

    df = model.sample(n_samples=n_samples, hyper_params=params, burnin=burnin)
    m = MetricEvaluator()

    times = []
    for n in ns:
        indices = random.sample(range(n_samples * n_walkers), n)

        t0 = time.time()
        dfr = m.evaluate(df.iloc[indices])
        t = time.time() - t0

        times.append(t)

    coeff = np.polyfit(ns, times, deg=2)

    plt.figure()
    plt.plot(ns, times, 'kx')
    xx = np.linspace(min(ns), max(ns), 100)
    yy = np.polyval(coeff, xx)
    plt.plot(xx, yy, 'b--')
    plt.show(block=False)

    reft = np.polyval(coeff, refn)
    print(f"Extrapolated time for {refn} samples: {reft:.0f}s")
