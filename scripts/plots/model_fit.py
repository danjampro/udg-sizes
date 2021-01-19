import os
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from udgsizes.core import get_config
from udgsizes.obs.sample import load_sample
from udgsizes.fitting.grid import ParameterGrid

FIGSIZE = 14, 5.2
BINS = 15
BINS1D = 30
BINS1D_OBS = 20
# RANGE = (24, 27), (np.log10(3), np.log10(10))
RANGE = (24, 27), (3, 7)
NORM = None  # mpl.colors.LogNorm()
CMAP = "binary"
LOGSCALE = False
FONTSIZE = 14
LEGEND_FONTSIZE = 12
Q = 0.95
SAVE = True


def get_confidence_intervals(dfs, dfbest, keys, range, bins, density=True):
    """ Not pretty and it takes a while.
    """
    xx = {}
    yy = {k: list() for k in keys}
    yybest = {}
    for df in dfs:
        for key in keys:
            data = df[key].values
            h, e = np.histogram(data, range=range[key], bins=bins, density=density)
            yy[key].append(h)
            xx[key] = 0.5 * (e[1:] + e[:-1])

    for key in keys:
        data = dfbest[key].values
        h, e = np.histogram(data, range=range[key], bins=bins, density=density)
        yybest[key] = h

    # Now collapse into min / max
    result = {}
    for key in keys:
        result[key] = {}
        values = np.vstack(yy[key])
        result[key]["ymin"] = values.min(axis=0)
        result[key]["ymax"] = values.max(axis=0)
        assert result[key]["ymin"].size == bins
        result[key]["x"] = xx[key]
        result[key]["ybest"] = yybest[key]

    return result


if __name__ == "__main__":

    model_name = "blue_final"
    metric = "poisson_likelihood_2d"
    confs = None

    config = get_config()
    image_dir = os.path.join(config["directories"]["data"], "images")
    image_filename = os.path.join(image_dir, f"model_fit_{model_name}.png")

    grid = ParameterGrid(model_name)
    dfm = grid.load_best_sample(metric=metric)
    dfo = load_sample()

    keys = "uae_obs_jig", "rec_obs_jig"
    range = {k: r for k, r in zip(keys, RANGE)}
    dfs = grid.load_confident_samples(q=Q)
    dfbest = grid.load_best_sample()
    confs = get_confidence_intervals(dfs, dfbest, keys, range=range, bins=BINS1D)

    fig = plt.figure(figsize=FIGSIZE)

    spec = gridspec.GridSpec(ncols=10, nrows=2, figure=fig, hspace=0.35, wspace=1.6)
    ax1 = fig.add_subplot(spec[:, 0:4])
    ax2 = fig.add_subplot(spec[0, 4:10])
    ax3 = fig.add_subplot(spec[1, 4:10])

    # 2D model histogram
    xkey = "uae_obs_jig"
    ykey = "rec_obs_jig"
    x = dfm[xkey].values
    y = dfm[ykey].values
    # y = np.log10(dfm[ykey].values)
    ax1.hist2d(x, y, range=RANGE, bins=BINS, norm=NORM, density=True, cmap=CMAP,
               label="best-fitting model")

    xkey = "mueff_av"
    ykey = "rec_arcsec"
    x = dfo[xkey].values
    y = dfo[ykey].values
    ax1.plot(x, y, "o", markersize=1, color="royalblue", label="observed")

    ax1.legend(loc="upper right", fontsize=LEGEND_FONTSIZE)
    ax1.set_xlabel(r"$\bar{\mu}_{e}\ [\mathrm{mag\ arcsec^{-2}}]$", fontsize=FONTSIZE)
    ax1.set_ylabel(r"$\bar{r}_{e}\ [\mathrm{arcsec}]$", fontsize=FONTSIZE)

    # uae fit
    xmkey = "uae_obs_jig"
    xokey = "mueff_av"
    xm = dfm[xmkey].values
    xo = dfo[xokey].values
    # ax2.hist(xm, range=RANGE[0], bins=BINS1D, density=True, color="k", histtype="step")
    h, e = np.histogram(xo, range=RANGE[0], bins=BINS1D_OBS)
    c = 0.5*(e[1:] + e[:-1])
    err = np.sqrt(h)
    norm = h.sum() * (e[1]-e[0])
    ax2.errorbar(c, h/norm, yerr=err/norm, elinewidth=1, markersize=3, color="b", linestyle=None,
                 linewidth=0, marker="o")
    if confs:
        conf = confs[xmkey]
        ax2.fill_between(conf["x"], conf["ymin"], conf["ymax"], color="b", alpha=0.2)
        ax2.plot(conf["x"], conf["ybest"], "r--", linewidth=1.5)
    ax2.set_xlabel(r"$\bar{\mu}_{e}\ [\mathrm{mag\ arcsec^{-2}}]$", fontsize=FONTSIZE)
    ax2.set_ylabel("PDF", fontsize=FONTSIZE)
    ax2.set_xlim(*RANGE[0])

    # rec fit
    xmkey = "rec_obs_jig"
    xokey = "rec_arcsec"
    xm = dfm[xmkey].values
    xo = dfo[xokey].values
    # ax3.hist(xm, range=RANGE[1], bins=BINS1D, density=True, color="k", histtype="step")
    h, e = np.histogram(xo, range=RANGE[1], bins=BINS1D_OBS)
    c = 0.5*(e[1:] + e[:-1])
    err = np.sqrt(h)
    norm = h.sum() * (e[1]-e[0])
    ax3.errorbar(c, h/norm, yerr=err/norm, elinewidth=1, markersize=3, color="b", linestyle=None,
                 linewidth=0, marker="o", label="observed")
    if confs:
        conf = confs[xmkey]
        ax3.fill_between(conf["x"], conf["ymin"], conf["ymax"], color="b", alpha=0.2,
                         label="95% confidence interval")
        ax3.plot(conf["x"], conf["ybest"], "r--", linewidth=1.5, label="best fit")
    ax3.set_xlabel(r"$\bar{r}_{e}\ [\mathrm{arcsec}]$", fontsize=FONTSIZE)
    ax3.set_ylabel("PDF", fontsize=FONTSIZE)
    ax3.set_xlim(*RANGE[1])

    ax3.legend(loc="upper right", fontsize=LEGEND_FONTSIZE)

    if LOGSCALE:
        ax2.set_yscale("log")
        ax3.set_yscale("log")
    plt.show(block=False)

    if SAVE:
        plt.savefig(image_filename, dpi=150, bbox_inches="tight")
