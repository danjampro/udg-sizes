import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from udgsizes.core import get_config
from udgsizes.fitting.grid import load_metrics
from udgsizes.fitting.utils.plotting import likelihood_threshold_plot, plot_ext


def marginal_likelihood_plot(ax, df, key, range, bins=10, fontsize=15, legend=False,
                             orientation="vertical", label_axes=False):
    """
    """
    values = df[key].values
    weights = np.exp(df["poisson_likelihood_2d"].values + 1000)
    weights[~np.isfinite(weights)] = 0

    ax.hist(values, range=range, bins=bins, weights=weights, density=True, histtype="step",
            color="k", label="marginal PDF", orientation=orientation)
    ax.hist(values, range=range, bins=bins, weights=weights, density=True, alpha=0.1,
            color="k", label="marginal PDF", orientation=orientation)

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

    if key == "rec_phys_alpha":
        plot_ext(ax)

    if label_axes:
        if key == "rec_phys_alpha":
            ax.set_xlabel(r"$\alpha$", fontsize=fontsize)
        else:
            ax.set_xlabel("$k$", fontsize=fontsize)
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

    SAVE = False
    CONFLINEKWARGS = {"linewidth": 1.3, "color": "springgreen", "linestyle": "--"}
    FONTSIZE = 14

    model_name = "blue_baldry_final"
    xkey = "logmstar_a"
    ykey = "rec_phys_alpha"
    zkey = "poisson_likelihood_2d"
    xlabel = r"$\alpha$"
    ylabel = r"$\kappa$"

    xrange = (-1., 0.5)
    yrange = (3, 7)

    df = load_metrics(model_name)

    x = np.random.rand(50)
    y = np.random.rand(50)

    fig = plt.figure(figsize=(7, 7))
    spec = GridSpec(ncols=4, nrows=4, figure=fig)
    ax0 = fig.add_subplot(spec[1:4, 0:3])
    ax1 = fig.add_subplot(spec[0, 0:3])
    ax2 = fig.add_subplot(spec[1:4, 3])

    # Likelihood heatmap
    likelihood_threshold_plot(ax=ax0, df=df, xkey=xkey, ykey=ykey, xrange=xrange, yrange=yrange)

    # Marginal likelihood hists
    m1, s1 = marginal_likelihood_plot(ax1, df, xkey, range=xrange)
    m2, s2 = marginal_likelihood_plot(ax2, df, ykey, range=yrange, orientation="horizontal")

    # External parameters
    plot_ext(ax0)

    # Legend and axes labels
    ax0.legend(loc="lower left", fontsize=FONTSIZE-3)
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

    # Turn off tick labels on marginals
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)

    # Consistent axis ranges
    ax1.set_xlim(ax0.get_xlim())
    ax2.set_ylim(ax0.get_ylim())

    if SAVE:
        config = get_config()
        image_dir = os.path.join(config["directories"]["data"], "images")
        image_filename = os.path.join(image_dir, f"model_likelihood_{model_name}.png")
        plt.savefig(image_filename, dpi=150, bbox_inches="tight")

    plt.show(block=False)
