import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from udgsizes.fitting.grid import ParameterGrid
from udgsizes.utils import shen
from udgsizes.model.sm_size import Model

from udgsizes.obs.sample import load_gama_masses, load_leisman_udgs

LOGMSTAR_KINK = 9

logmstar_min = 6.5
logmstar_max = 11.5

logrec_min = -1
logrec_max = 1.2

BINS = 40
SIGMA_FACTOR = 1
FONTSIZE = 16

MODEL_NAME = "blue_sedgwick_shen_final"

SAVEFIG = True


def calculate_rec(logmstar, alpha, model):
    func = model._mean_rec_phys
    return np.array([func(_, alpha=alpha) for _ in logmstar])


def plot_gama(ax=None, hist=False, contour=True, n_contours=5):
    """
    """
    if ax is None:
        fig, ax = plt.subplots()

    dfg = load_gama_masses()

    x = dfg["logmstar"].values
    y = np.log10(dfg["rec_phys"].values)

    rng = (logmstar_min, logmstar_max), (logrec_min, logrec_max)

    if hist:
        ax.hist2d(x, y, cmap="binary", bins=BINS, norm=mpl.colors.LogNorm(), range=rng)

    if contour:
        h, xe, ye = np.histogram2d(x, y, bins=BINS, range=rng)
        xc = 0.5 * (xe[1:] + xe[:-1])
        yc = 0.5 * (ye[1:] + ye[:-1])
        yy, xx = np.meshgrid(yc, xc)

        h = h.astype("float")
        h[h == 0] = np.nan

        ax.contourf(xx, yy, h, n_contours, linewidths=1, cmap='binary', vmin=0)

    return ax


def plot_leisman(ax=None, axrat=0.7):
    """
    """
    if ax is None:
        fig, ax = plt.subplots()

    dfl = load_leisman_udgs()
    x = dfl["logmstar"].values

    rec_phys = dfl["re_phys"].values * np.sqrt(axrat)
    y = np.log10(rec_phys)

    ax.plot(x, y, "bo", markersize=2.5, label="UDG, Leisman et al. (2017, adapted)")


def plot_lange(ax, a=27.72E-3, b=0.21, q=0.7):
    """ https://arxiv.org/pdf/1411.6355.pdf """
    logmstar_ext = np.linspace(logmstar_min, LOGMSTAR_KINK, 20)

    re = a * (10 ** logmstar_ext) ** b
    rec = re * np.sqrt(q)

    ax.plot(logmstar_ext, np.log10(rec), 'k--', label="Lange et al. (2014, adapted)")

    return ax


def plot_shen(ax):
    """
    """
    logmstar = np.linspace(logmstar_min, logmstar_max, 100)
    logmstar_shen = np.linspace(LOGMSTAR_KINK, logmstar_max, 20)
    logmstar_ext = np.linspace(logmstar_min, LOGMSTAR_KINK, 20)

    yy = shen.logmstar_to_mean_rec(logmstar_shen)
    ax.plot(logmstar_shen, np.log10(yy), "r-", label="Shen et al. (2003)")
    yy = np.log10(shen.logmstar_to_mean_rec(logmstar_ext))
    ax.plot(logmstar_ext, yy, "r--")

    sigma = shen.logmstar_sigma(logmstar)
    yy0 = shen.logmstar_to_mean_rec(logmstar)
    yy1 = np.log10(shen.apply_rec_offset(yy0, -sigma))
    yy2 = np.log10(shen.apply_rec_offset(yy0, sigma))
    ax.fill_between(logmstar, yy1, yy2, color="r", alpha=0.2, linewidth=0.0)


def plot_new(ax):
    """
    """
    logmstar_ext = np.linspace(logmstar_min, LOGMSTAR_KINK, 20)

    rec_mins = calculate_rec(logmstar_ext, alpha=alpha_mean + SIGMA_FACTOR * alpha_std, model=model)
    rec_maxs = calculate_rec(logmstar_ext, alpha=alpha_mean - SIGMA_FACTOR * alpha_std, model=model)
    ax.fill_between(logmstar_ext, np.log10(rec_mins), np.log10(rec_maxs), color="dodgerblue",
                    alpha=0.4, label="Prole (2021, this study)", linewidth=0.0)

    sigma = shen.logmstar_sigma(logmstar_ext)
    yy1 = np.log10(shen.apply_rec_offset(rec_mins, -sigma))
    yy2 = np.log10(shen.apply_rec_offset(rec_maxs, sigma))
    ax.fill_between(logmstar_ext, yy1, yy2, color="dodgerblue", alpha=0.2, linewidth=0.0)


if __name__ == "__main__":

    grid = ParameterGrid(MODEL_NAME)
    dfg = load_gama_masses()
    model = Model(model_name=MODEL_NAME)

    alpha_mean, alpha_std = grid.parameter_stats("rec_phys_offset_alpha")

    # fig, ax = plt.subplots(figsize=(9, 6))
    fig, ax = plt.subplots(figsize=(12, 6))

    plot_gama(ax)

    # Plot shen result
    plot_shen(ax)

    # Plot new result
    plot_new(ax)

    # Plot Leisman UDGs
    plot_leisman(ax)

    plot_lange(ax)

    # Format axes
    ax.set_xlim((logmstar_min, logmstar_max))
    ax.set_ylim((logrec_min, logrec_max))

    ax.set_xlabel(r"$\log_{10}\ \mathrm{M_{*} / M_{\odot}}$", fontsize=FONTSIZE)
    ax.set_ylabel(r"$\log_{10}\ \mathrm{\hat{r}_{e}\ [kpc]}$", fontsize=FONTSIZE)

    plt.legend(loc="lower right", fontsize=FONTSIZE-3)

    if SAVEFIG:
        filename = os.path.join(grid.config["directories"]["images"], f"mstar-re-{MODEL_NAME}.png")
        plt.savefig(filename, dpi=150, bbox_inches="tight")

    plt.show(block=False)
