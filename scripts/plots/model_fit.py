import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from udgsizes.core import get_config
from udgsizes.obs.sample import load_sample
from udgsizes.fitting.grid import ParameterGrid

FIGSIZE = 14, 5.6
BINS = 15
BINS1D = 10
# RANGE = (24, 27), (np.log10(3), np.log10(10))
RANGE = (24, 27), (3, 7)
NORM = None  # mpl.colors.LogNorm()
CMAP = "binary"


def get_confidence_intervals(grid, keys, range, bins, density=True):
    """
    """
    xx = {}
    yy = {k: list() for k in keys}
    dfs = grid.load_confident_samples()
    for df in dfs:
        for key in keys:
            data = df[key].values
            h, e = np.histogram(data, range=range[key], bins=bins, density=density)
            yy[key] = h
            xx[key] = 0.5 * (e[1:] + e[:-1])
    return xx, yy


if __name__ == "__main__":

    model_name = "blue_final"
    metric = "poisson_likelihood_2d"

    config = get_config()
    image_dir = os.path.join(config["directories"]["data"], "images")
    image_filename = os.path.join(image_dir, f"model_fit_{model_name}.png")

    grid = ParameterGrid(model_name)
    dfm = grid.load_best_sample(metric=metric)
    dfo = load_sample()

    keys = "rec_obs_jig", "uae_obs_jig"
    range = {k: r for k, r in zip(keys, RANGE)}
    confs = get_confidence_intervals(grid, keys, range=range, bins=BINS1D)

    fig = plt.figure(figsize=FIGSIZE)
    """
    spec = gridspec.GridSpec(ncols=7, nrows=2, figure=fig, hspace=0.35, wspace=0.4)
    ax0 = fig.add_subplot(spec[:, 0:2])
    ax1 = fig.add_subplot(spec[:, 2:4])
    ax2 = fig.add_subplot(spec[0, 4:7])
    ax3 = fig.add_subplot(spec[1, 4:7])
    """

    spec = gridspec.GridSpec(ncols=10, nrows=2, figure=fig, hspace=0.35, wspace=0.4)
    ax1 = fig.add_subplot(spec[:, 0:4])
    ax2 = fig.add_subplot(spec[0, 4:10])
    ax3 = fig.add_subplot(spec[1, 4:10])

    """
    # 2D observation histogram
    xkey = "mueff_av"
    ykey = "rec_arcsec"
    x = dfo[xkey].values
    y = dfo[ykey].values
    # y = np.log10(dfo[ykey].values)
    ax0.hist2d(x, y, range=RANGE, bins=BINS, norm=NORM, density=True, cmap=CMAP)
    """

    # 2D model histogram
    xkey = "uae_obs_jig"
    ykey = "rec_obs_jig"
    x = dfm[xkey].values
    y = dfm[ykey].values
    # y = np.log10(dfm[ykey].values)
    ax1.hist2d(x, y, range=RANGE, bins=BINS, norm=NORM, density=True, cmap=CMAP)

    xkey = "mueff_av"
    ykey = "rec_arcsec"
    x = dfo[xkey].values
    y = dfo[ykey].values
    ax1.plot(x, y, "o", markersize=1, color="royalblue")

    # uae fit
    xmkey = "uae_obs_jig"
    xokey = "mueff_av"
    xm = dfm[xmkey].values
    xo = dfo[xokey].values
    ax2.hist(xm, range=RANGE[0], bins=BINS1D, density=True, color="k", histtype="step")
    h, e = np.histogram(xo, range=RANGE[0], bins=BINS1D)
    c = 0.5*(e[1:] + e[:-1])
    err = np.sqrt(h)
    norm = h.sum() * (e[1]-e[0])
    ax2.errorbar(c, h/norm, yerr=err/norm, elinewidth=1, markersize=3, color="b", linestyle=None,
                 linewidth=0, marker="o")

    # rec fit
    xmkey = "rec_obs_jig"
    xokey = "rec_arcsec"
    xm = dfm[xmkey].values
    xo = dfo[xokey].values
    ax3.hist(xm, range=RANGE[1], bins=BINS1D, density=True, color="k", histtype="step")
    h, e = np.histogram(xo, range=RANGE[1], bins=BINS1D)
    c = 0.5*(e[1:] + e[:-1])
    err = np.sqrt(h)
    norm = h.sum() * (e[1]-e[0])
    ax3.errorbar(c, h/norm, yerr=err/norm, elinewidth=1, markersize=3, color="b", linestyle=None,
                 linewidth=0, marker="o")

    plt.show(block=False)
