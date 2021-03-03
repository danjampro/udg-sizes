import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from udgsizes.core import get_config
from udgsizes.obs.sample import load_sample
from udgsizes.fitting.interpgrid import InterpolatedGrid
from udgsizes.utils.selection import parameter_ranges

FIGSIZE = 12, 4
BINS = 20
BINS1D = 30
BINS1D_OBS = 10
RANGE = (24, 27), (3, 10)
NORM = None
CMAP = "binary"
LOGSCALE = False
FONTSIZE = 14
LEGEND_FONTSIZE = 10.5
Q = 0.95
SAVE = True


def get_confidence_intervals(interpgrid, xkey="uae_obs_jig", ykey="rec_obs_jig", q=0.95):
    """ Not pretty and it takes a while.
    """
    model_best = interpgrid.get_best_model(transpose=True)
    models_conf = np.stack(interpgrid.get_confident_models(transpose=True, q=q))

    prange = parameter_ranges['uae'], parameter_ranges['rec']
    bins = interpgrid._bins
    _, xe = np.histogram([], range=(prange[0][0], prange[0][1]), bins=bins)
    _, ye = np.histogram([], range=(prange[1][0], prange[1][1]), bins=bins)
    xc = 0.5 * (xe[1:] + xe[:-1])
    yc = 0.5 * (ye[1:] + ye[:-1])

    best_x = model_best.sum(axis=0)
    best_x /= best_x.sum() * (xe[1]-xe[0])
    best_y = model_best.sum(axis=1)
    best_y /= best_y.sum() * (ye[1]-ye[0])

    min_x = np.ones_like(best_x) * np.inf
    max_x = np.ones_like(best_x) * -np.inf
    min_y = np.ones_like(best_y) * np.inf
    max_y = np.ones_like(best_y) * -np.inf

    for model in models_conf:
        xm = model.sum(axis=0)
        xm /= model.sum() * (xe[1]-xe[0])
        ym = model.sum(axis=1)
        ym /= model.sum() * (ye[1]-ye[0])
        min_x[:] = np.minimum(min_x, xm)
        max_x[:] = np.maximum(max_x, xm)
        min_y[:] = np.minimum(min_y, ym)
        max_y[:] = np.maximum(max_y, ym)

    result = {}
    result[xkey] = {"ymin": min_x, "ymax": max_x, "ybest": best_x, "x": xc}
    result[ykey] = {"ymin": min_y, "ymax": max_y, "ybest": best_y, "x": yc}

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

    """
    fig = plt.figure(figsize=FIGSIZE)

    spec = gridspec.GridSpec(ncols=10, nrows=2, figure=fig, hspace=0.5, wspace=0.5)
    ax1 = fig.add_subplot(spec[:, :5])
    ax2 = fig.add_subplot(spec[0, 5:])
    ax3 = fig.add_subplot(spec[1, 5:])
    """
    fig = plt.figure(figsize=(8, 7))
    spec = GridSpec(ncols=4, nrows=4, figure=fig, hspace=0.1, wspace=0.1)
    ax1 = fig.add_subplot(spec[1:4, 0:3])
    ax2 = fig.add_subplot(spec[0, 0:3])
    ax3 = fig.add_subplot(spec[1:4, 3])

    # 2D model histogram
    xkey = "uae_obs_jig"
    ykey = "rec_obs_jig"
    model_best = grid.get_best_model(transpose=True)
    extent = (parameter_ranges["uae"][0], parameter_ranges["uae"][1],
              parameter_ranges["rec"][0], parameter_ranges["rec"][1])
    ax1.imshow(model_best, origin="lower", extent=extent, cmap=CMAP, label="best-fitting model")
    ax1.set_aspect(0.37)
    ax1.set_xlim(*RANGE[0])
    ax1.set_ylim(*RANGE[1])

    xkey = "mueff_av"
    ykey = "rec_arcsec"
    x = dfo[xkey].values
    y = dfo[ykey].values
    ax1.plot(x, y, "o", markersize=1, color="royalblue", label="Observed")

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
                 linewidth=0, marker="o", zorder=10, label="Observed")
    if confs:
        conf = confs[xmkey]
        ax2.fill_between(conf["x"], conf["ymin"], conf["ymax"], color="b", alpha=0.2,
                         label="95% confidence interval")
        ax2.plot(conf["x"], conf["ybest"], "r--", linewidth=1.5, label="best fit")
    ax2.set_ylabel("PDF", fontsize=FONTSIZE)
    ax2.set_xlim(*RANGE[0])

    ax2.legend(loc="upper right", fontsize=LEGEND_FONTSIZE)

    # rec fit
    xmkey = "rec_obs_jig"
    xokey = "rec_arcsec"
    xo = dfo[xokey].values
    h, e = np.histogram(xo, range=RANGE[1], bins=BINS1D_OBS)
    c = 0.5*(e[1:] + e[:-1])
    err = np.sqrt(h)
    norm = h.sum() * (e[1]-e[0])
    ax3.errorbar(h/norm, c, xerr=err/norm, elinewidth=1, markersize=3, color="k", linestyle=None,
                 linewidth=0, marker="o", label="Observed", zorder=10)
    if confs:
        conf = confs[xmkey]
        ax3.fill_betweenx(conf["x"], conf["ymin"], conf["ymax"], color="b", alpha=0.2,
                          label="95% confidence interval")
        ax3.plot(conf["ybest"], conf["x"], "r--", linewidth=1.5, label="best fit")
    ax3.set_xlabel("PDF", fontsize=FONTSIZE)
    ax3.xaxis.set_label_position("top")
    ax3.xaxis.tick_top()
    ax3.set_ylim(*RANGE[1])

    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)

    if LOGSCALE:
        ax2.set_yscale("log")
        ax3.set_yscale("log")
    plt.show(block=False)

    if SAVE:
        config = get_config()
        image_dir = os.path.join(config["directories"]["data"], "images")
        image_filename = os.path.join(image_dir, f"model_fit_{model_name}.png")
        plt.savefig(image_filename, dpi=150, bbox_inches="tight")
