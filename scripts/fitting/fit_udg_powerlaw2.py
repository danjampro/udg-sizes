import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from udgsizes.fitting.grid import ParameterGrid
from udgsizes.obs.recovery import load_recovery_efficiency

MODEL_NAME = "blue_sedgwick_shen_0.35"


def func(x, a, k):
    return a * x ** (-k)


if __name__ == "__main__":

    # Get best fitting hyper parameters
    grid = ParameterGrid(MODEL_NAME)
    df = grid.load_best_sample()

    # Identify UDGs
    cond = df["is_udg"].values == 1
    df = df[cond].reset_index(drop=True)

    # Get sizes
    rec_phys = df["rec_phys"].values

    # Get weights
    recov = load_recovery_efficiency()
    weights = 1. / recov(uae=df["uae_obs"], rec=df["rec_obs"])

    # Make histogram
    hist, edges = np.histogram(rec_phys, weights=weights, bins=20)
    centres = 0.5 * (edges[1:] + edges[:-1])

    # Do fit
    popt, pcov = curve_fit(func, ydata=hist, xdata=centres)
    powerlaw = -popt[1]

    fig, ax = plt.subplots()
    ax.plot(centres, hist, "ko")
    xx = np.linspace(rec_phys.min(), rec_phys.max(), 100)
    ax.plot(xx, func(xx, *popt), "b--")
    plt.show(block=False)

    print(f"Power law slope: {powerlaw:.2f}")
