import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from udgsizes.core import get_config
from udgsizes.fitting.grid import ParameterGrid
from udgsizes.fitting.utils.plotting import threshold_plot


def marginal_likelihood_plot(ax, values, weights, range=None, bins=10, fontsize=15, legend=False,
                             orientation="vertical", label_axes=False, weights_no_prior=None):
    """
    """
    if range is None:
        range = values.min(), values.max()

    weights[~np.isfinite(weights)] = 0

    histkwargs = {"range": range, "bins": bins, "orientation": orientation, "density": True}

    ax.hist(values, weights=weights, histtype="step", color="k", **histkwargs)
    ax.hist(values, weights=weights, alpha=0.1, color="k", label="Normal Prior", **histkwargs)

    if weights_no_prior is not None:
        ax.hist(values, weights=weights_no_prior, histtype="step", color="dodgerblue",
                label="No Prior", **histkwargs)

    cond = (values >= range[0]) & (values < range[1])
    values = values[cond]
    weights = weights[cond]

    mean = np.average(values, weights=weights)
    variance = np.average((values-mean)**2, weights=weights)
    std = np.sqrt(variance)

    if orientation == "vertical":
        ylim = ax.get_ylim()
        ax.plot([mean-std, mean-std], ylim, **CONFLINEKWARGS)
        ax.plot([mean+std, mean+std], ylim, **CONFLINEKWARGS)
        ax.set_ylim(ylim)
    else:
        xlim = ax.get_xlim()
        ax.plot(xlim, [mean-std, mean-std], **CONFLINEKWARGS)
        ax.plot(xlim, [mean+std, mean+std], **CONFLINEKWARGS)
        ax.set_xlim(xlim)

    if orientation == "vertical":
        ax.set_ylabel("PDF", fontsize=fontsize)
    else:
        ax.set_xlabel("PDF", fontsize=fontsize)
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()

    if legend:
        ax.legend(loc="best")

    return mean, std


if __name__ == "__main__":

    SAVEFIG = True
    CONFLINEKWARGS = {"linewidth": 1.3, "color": "springgreen", "linestyle": "--"}
    FONTSIZE = 14

    model_name = "blue_sedgwick_shen_final"
    xkey = "rec_phys_offset_alpha"
    ykey = "logmstar_a"
    zkey_noprior = "likelihood_kde_3d"
    zkey = "posterior_kde_3d"
    xlabel = r"$\beta$"
    ylabel = r"$\alpha$"

    grid = ParameterGrid(model_name)
    df = grid.load_metrics()
    metrics = grid.get_best_metrics(metric=zkey)

    x = df[xkey].values
    y = df[ykey].values
    znoprior = df[zkey_noprior].values
    z = df[zkey].values

    # xrange = x.min(), x.max()
    xrange = x.min(), x.max()
    yrange = y.min(), y.max()

    fig = plt.figure(figsize=(7, 7))
    spec = GridSpec(ncols=10, nrows=10, figure=fig)
    ax0 = fig.add_subplot(spec[3:10, 0:7])
    ax1 = fig.add_subplot(spec[0:3, 0:7])
    ax2 = fig.add_subplot(spec[3:10, 7:10])

    # Likelihood heatmap
    threshold_plot(x=x, y=y, z=z, ax=ax0, xrange=xrange, yrange=yrange, smooth=0.5)
    # grid.plot_2d_hist(xkey, ykey, metric=zkey, xrange=xrange, yrange=yrange,  ax=ax0, smooth=0.5)

    # Marginal likelihood hists
    m1, s1 = marginal_likelihood_plot(ax1, x, z, range=xrange, weights_no_prior=znoprior)
    m2, s2 = marginal_likelihood_plot(ax2, y, z, range=yrange, orientation="horizontal",
                                      weights_no_prior=znoprior)
    print(m1, s1)
    print(m2, s2)

    # Legend and axes labels
    # ax0.legend(loc="lower left", fontsize=FONTSIZE-3)
    ax0.set_xlabel(xlabel, fontsize=FONTSIZE+2)
    ax0.set_ylabel(ylabel, fontsize=FONTSIZE+2)

    xlim = ax0.get_xlim()
    ylim = ax0.get_ylim()
    ax0.plot([m1-s1, m1-s1], ylim, **CONFLINEKWARGS)
    ax0.plot([m1+s1, m1+s1], ylim, **CONFLINEKWARGS)
    ax0.plot(xlim, [m2-s2, m2-s2], **CONFLINEKWARGS)
    ax0.plot(xlim, [m2+s2, m2+s2], **CONFLINEKWARGS)
    ax0.set_xlim(xlim)
    ax0.set_ylim(ylim)

    # Plot best fit model
    # ax0.plot(metrics[xkey], metrics[ykey], "ro", markersize=5, label="Best fit")

    # Turn off tick labels on marginals
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)

    # Consistent axis ranges
    ax1.set_xlim(ax0.get_xlim())
    ax2.set_ylim(ax0.get_ylim())

    bbox_to_anchor = (1.005, 0.9)
    ax1.legend(fontsize=FONTSIZE-2.5, bbox_to_anchor=bbox_to_anchor)

    if SAVEFIG:
        config = get_config()
        image_dir = os.path.join(config["directories"]["data"], "images")
        image_filename = os.path.join(image_dir, f"corner_plot_{model_name}.png")
        plt.savefig(image_filename, dpi=150, bbox_inches="tight")

    plt.show(block=False)
