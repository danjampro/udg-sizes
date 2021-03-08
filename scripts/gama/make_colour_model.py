"""
Model
"""
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from udgsizes.core import get_config
from udgsizes.obs.sample import load_gama_masses
from udgsizes.utils import colour as utils

CONFIG = get_config()
XLABEL = "g-r"
HISTBINS = (20, 20)


def make_1d_kde(values):
    """
    """
    return gaussian_kde(values)


def make_2d_kde(x, y):
    """
    """
    values = np.vstack([x.ravel(), y.ravel()])

    print(values.shape)
    return gaussian_kde(values)


def evaluate_2d_kde(x, y, kde):
    """
    """
    assert x.shape == y.shape
    values = np.vstack([x.reshape(-1), y.reshape(-1)])
    return kde(values).reshape(x.shape)


if __name__ == "__main__":

    selection_config = CONFIG["colour_model"]["selection"]
    df = load_gama_masses(config=CONFIG, **selection_config)
    logmstar = df["logmstar"].values
    colour = df["gr"].values
    n = df["n"].values

    bins = utils.get_logmstar_bins(config=CONFIG)
    idxs = utils.get_logmstar_bin_indices(logmstar, bins=bins)
    binwidth = bins[1] - bins[0]

    cc = np.linspace(colour.min(), colour.max(), 100)
    crange = colour.min(), colour.max()
    nrange = n.min(), n.max()

    fig = plt.figure(figsize=(bins.size * 3, 6))

    for idx in range(bins.size):

        cond = idxs == idx
        kde = make_2d_kde(colour[cond], n[cond])

        ax0 = plt.subplot(2, bins.size, idx + 1)
        ax0.hist2d(colour[cond], n[cond], range=(crange, nrange), density=True, bins=HISTBINS)

        ax1 = plt.subplot(2, bins.size, bins.size + idx + 1)
        xx, yy = np.meshgrid(np.linspace(crange[0], crange[1], 100),
                             np.linspace(nrange[0], nrange[1], 100))
        kde_values = evaluate_2d_kde(xx, yy, kde)
        ax1.imshow(kde_values, extent=(*crange, *nrange), origin="lower")

        """
        kde = make_kde(colour[cond])
        ax.hist(colour[cond], density=True, histtype="step", color="k")
        ax.plot(cc, kde(cc), "b--")

        lmass = bins[idx] + 0.5 * binwidth
        ax.set_title(f"M*={lmass:.2f}")
        ax.set_xlabel(YLABEL)
        if idx == 0:
            ax.set_ylabel("PDF")
            ax.set_title(f"M*<{bins[1]:.1f}")
        """

        lmass = bins[idx] + 0.5 * binwidth
        ax0.set_title(f"M*={lmass:.2f}")
        ax0.set_xlabel(XLABEL)

        if idx == 0:
            ax0.set_ylabel("n")
            ax0.set_title(f"M*<{bins[1]:.1f}")

    plt.tight_layout()

    fig = plt.figure(figsize=(bins.size * 3, 3))
    ratios = df["logmstar_absmag_r"].values
    for idx in range(bins.size):
        ax = plt.subplot(1, bins.size, idx+1)
        cond = idxs == idx
        p = np.polyfit(colour[cond], ratios[cond], 1)
        ax.plot(colour[cond], ratios[cond], "k+", markersize=1, alpha=0.5)
        ax.plot(cc, np.polyval(p, cc), "b--")

        ax.set_xlim(-0.2, 1.1)
        ax.set_ylim(-0.55, -0.43)

    plt.show(block=False)
