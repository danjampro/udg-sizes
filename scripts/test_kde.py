import numpy as np
import matplotlib.pyplot as plt

from udgsizes.fitting.grid import ParameterGrid
from udgsizes.utils.stats.kde import TransformedGaussianPDF, TransformedKDE
from udgsizes.obs.sample import load_sample

MAKEPLOTS = False


def plot_hist_2d(pdf, dfo, k1, k2, ko1, ko2, ax=None):
    """
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.hist2d(pdf.values[k1], pdf.values[k2], cmap="binary", bins=30)

    x = pdf.rescale_observations(dfo, k1)
    y = pdf.rescale_observations(dfo, k2)

    ax.plot(x, y, "bo", markersize=3, alpha=0.4)
    ax.set_xlabel(k1)
    ax.set_ylabel(k2)

    plt.show(block=False)


if __name__ == "__main__":

    metric = "log_likelihood_kde_3d"
    # metric = "posterior"
    # metric = "posterior_ks"

    grid = ParameterGrid("blue_sedgwick_shen_final")
    df = grid.load_best_sample(metric=metric)

    dfo = load_sample()

    cond = df["selected_jig"].values == 1
    df = df[cond].reset_index(drop=True)

    # pdf = TransformedGaussianPDF(df, makeplots=MAKEPLOTS)
    pdf = TransformedKDE(df, makeplots=MAKEPLOTS)

    pvals = pdf.evaluate(dfo)
    print(np.log(pvals).sum())

    # if MAKEPLOTS:
    # pdf.summary_plot(dfo=dfo)

    # """
    plt.figure(figsize=(12, 4))

    ax = plt.subplot(1, 3, 1)
    plot_hist_2d(pdf, dfo, "uae_obs_jig", "rec_obs_jig", "mueff_av", "rec_arcsec", ax=ax)

    ax = plt.subplot(1, 3, 2)
    plot_hist_2d(pdf, dfo, "uae_obs_jig", "colour_obs", "mueff_av", "g_r", ax=ax)

    ax = plt.subplot(1, 3, 3)
    plot_hist_2d(pdf, dfo, "rec_obs_jig", "colour_obs", "rec_arcsec", "g_r", ax=ax)
    # """

    plt.tight_layout()
