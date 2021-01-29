import os
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from udgsizes.core import get_config
from udgsizes.obs.sample import load_sample
# from udgsizes.fitting.grid import ParameterGrid
from udgsizes.fitting.grid import InterpolatedGrid
from udgsizes.utils.selection import parameter_ranges

FIGSIZE = 14, 4.75
BINS = 20
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


def get_confidence_intervals(interpgrid, xkey="uae_obs_jig", ykey="rec_obs_jig"):
    """ Not pretty and it takes a while.
    """
    model_best = interpgrid.get_best_model()
    models_conf = interpgrid.get_confident_models()

    prange = parameter_ranges['uae'], parameter_ranges['rec']
    bins = interpgrid._bins
    _, xx = np.histogram([], range=(prange[0][0], prange[0][1]), bins=bins)
    _, yy = np.histogram([], range=(prange[1][0], prange[1][1]), bins=bins)

    best_x = model_best.sum(axis=0)
    best_y = model_best.sum(axis=1)

    min_x = models_conf.min(axis=0)
    max_x = models_conf.max(axis=0)
    min_y = models_conf.min(axis=1)
    max_y = models_conf.max(axis=1)

    result = {}
    result[xkey] = {"ymin": min_x, "ymax": max_x, "ybest": best_x, "x": xx}
    result[ykey] = {"ymin": min_y, "ymax": max_y, "ybest": best_y, "x": yy}

    return result


if __name__ == "__main__":

    model_name = "blue_baldry_final"
    metric = "poisson_likelihood_2d"
    confs = None

    # Load observations
    dfo = load_sample()

    # Load model grid
    grid = InterpolatedGrid(model_name)
    confs = get_confidence_intervals(grid)

    assert False

    fig = plt.figure(figsize=FIGSIZE)

    spec = gridspec.GridSpec(ncols=10, nrows=2, figure=fig, hspace=0.35, wspace=1.6)
    ax1 = fig.add_subplot(spec[:, :4])
    ax2 = fig.add_subplot(spec[0, 4:])
    ax3 = fig.add_subplot(spec[1, 4:])

    # 2D model histogram
    xkey = "uae_obs_jig"
    ykey = "rec_obs_jig"
    model_best = grid.get_best_model()
    extent = (parameter_ranges[0][0], parameter_ranges[0][1],
              parameter_ranges[1][0], parameter_ranges[1][1])
    ax1.imshow(model_best, origin="lower", extent=extent, cmap=CMAP, label="best-fitting model")

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
    xo = dfo[xokey].values
    h, e = np.histogram(xo, range=RANGE[0], bins=BINS1D_OBS)
    c = 0.5*(e[1:] + e[:-1])
    err = np.sqrt(h)
    norm = h.sum() * (e[1]-e[0])
    ax2.errorbar(c, h/norm, yerr=err/norm, elinewidth=1, markersize=3, color="k", linestyle=None,
                 linewidth=0, marker="o", zorder=10)
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
    xo = dfo[xokey].values
    h, e = np.histogram(xo, range=RANGE[1], bins=BINS1D_OBS)
    c = 0.5*(e[1:] + e[:-1])
    err = np.sqrt(h)
    norm = h.sum() * (e[1]-e[0])
    ax3.errorbar(c, h/norm, yerr=err/norm, elinewidth=1, markersize=3, color="k", linestyle=None,
                 linewidth=0, marker="o", label="observed", zorder=10)
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
        config = get_config()
        image_dir = os.path.join(config["directories"]["data"], "images")
        image_filename = os.path.join(image_dir, f"model_fit_{model_name}.png")
        plt.savefig(image_filename, dpi=150, bbox_inches="tight")
