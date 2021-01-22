import os
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage.filters import gaussian_filter

from udgsizes.core import get_config
from udgsizes.fitting.grid import ParameterGrid

SAVE = False
CONFLINEKWARGS = {"linewidth": 1.3, "color": "springgreen", "linestyle": "--"}


def likelihood_quantile(values, q):
    """
    """
    values_sorted = values.copy().reshape(-1)
    values_sorted.sort()
    values_sorted = values_sorted[::-1]
    csum = np.cumsum(values_sorted)
    total = csum[-1]
    idx = np.argmin(abs(csum - q*total))
    return values_sorted[idx]


def plot_ext(ax, alpha=0.2):
    ax.axhspan(4.4-0.19, 4.4+0.19, color="r", alpha=alpha,
               label="van der Burg +16 (clusters), Amorisco +16")
    ax.axhspan(3.71-0.33, 3.71+0.33, color="b", alpha=alpha,
               label="van der Burg +17 (groups)")


def contour_plot(ax, df, smooth=False):

    xkey = "rec_phys_alpha"
    ykey = "uae_phys_k"
    zkey = "poisson_likelihood_2d"

    x = df[xkey].values
    y = df[ykey].values
    z = df[zkey].values
    extent = (y.min(), y.max(), x.min(), x.max())

    nx = np.unique(x).size
    ny = np.unique(y).size

    zz = z.reshape(nx, ny)
    zzexp = np.exp(zz+1000)

    if smooth:
        zzexp = gaussian_filter(zzexp, 0.5)

    levels = (likelihood_quantile(zzexp, 0.999999426696856),
              likelihood_quantile(zzexp, 0.997),
              likelihood_quantile(zzexp, 0.95),
              likelihood_quantile(zzexp, 0.68))
    colors = "k"
    contour_labels = r"$5\sigma$", r"$3\sigma$", r"$2\sigma$", r"$1\sigma$"

    cs = ax.contour(zzexp, linewidths=0.8, colors=colors, extent=extent, levels=levels)
    fmt = {}
    for l, s in zip(cs.levels, contour_labels):
        fmt[l] = s
    ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=10)

    xrange = 0, 1.2
    yrange = 2.5, 5.2
    fontsize = 15

    ax.plot(xrange, [4.4, 4.4], 'b--', label="van der Burg +16 (clusters), Amorisco +16")
    ax.plot(xrange, [3.71, 3.71], 'r--', label="van der Burg +17 (groups)")
    ax.axhspan(4.4-0.19, 4.4+0.19, color="b", alpha=0.2)
    ax.axhspan(3.71-0.33, 3.71+0.33, color="r", alpha=0.2)

    ax.set_xlim(*xrange)
    ax.set_ylim(2.5, 5.2)
    ax.set_aspect((xrange[1]-xrange[0])/(yrange[1]-yrange[0]))
    ax.set_xlabel("$k$", fontsize=fontsize)
    ax.set_ylabel(r"$\alpha$", fontsize=fontsize)
    ax.legend(loc="best")


def contour_plot_gauss(ax, df, fontsize=15):

    xkey = "rec_phys_alpha"
    ykey = "uae_phys_k"
    zkey = "poisson_likelihood_2d"

    x = df[xkey].values
    y = df[ykey].values
    z = np.exp(df[zkey].values + 1000)
    extent = (y.min(), y.max(), x.min(), x.max())

    nx = np.unique(x).size
    ny = np.unique(y).size

    mean = (np.average(x, weights=z), np.average(y, weights=z))
    cov = np.cov(x, y, aweights=z)
    norm = stats.multivariate_normal(mean=mean, cov=cov)

    zz = norm.pdf(np.vstack([x, y]).T).reshape(nx, ny)

    levels = (likelihood_quantile(zz, 0.999999426696856),
              likelihood_quantile(zz, 0.997),
              likelihood_quantile(zz, 0.95),
              likelihood_quantile(zz, 0.68))
    colors = "k"
    contour_labels = r"$5\sigma$", r"$3\sigma$", r"$2\sigma$", r"$1\sigma$"

    cs = ax.contour(zz, linewidths=0.8, colors=colors, extent=extent, levels=levels)
    fmt = {}
    for l, s in zip(cs.levels, contour_labels):
        fmt[l] = s
    ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=10)

    xrange = 0, 1.2
    yrange = 2.5, 5.2

    ax.plot(xrange, [4.4, 4.4], 'b--')
    ax.plot(xrange, [3.71, 3.71], 'r--')
    ax.axhspan(4.4-0.19, 4.4+0.19, color="b", alpha=0.2,
               label="vdB+16 (clusters), Amorisco+16")
    ax.axhspan(3.71-0.33, 3.71+0.33, color="r", alpha=0.2,
               label="vdB+17 (groups)")

    ax.set_xlim(*xrange)
    ax.set_ylim(2.5, 5.2)
    ax.set_aspect((xrange[1]-xrange[0])/(yrange[1]-yrange[0]))
    ax.set_xlabel("$k$", fontsize=fontsize)
    ax.set_ylabel(r"$\alpha$", fontsize=fontsize)
    ax.legend(loc="best")


def contour_plot_threshold(ax, df, legend=True, fontsize=15):

    xkey = "rec_phys_alpha"
    ykey = "uae_phys_k"
    zkey = "poisson_likelihood_2d"

    x = df[xkey].values
    y = df[ykey].values
    z = df[zkey].values
    extent = (y.min(), y.max(), x.min(), x.max())

    nx = np.unique(x).size
    ny = np.unique(y).size

    zz = z.reshape(nx, ny)
    zzexp = np.exp(zz+1000)

    levels = (likelihood_quantile(zzexp, 0.999999426696856),
              likelihood_quantile(zzexp, 0.999936657516334),
              likelihood_quantile(zzexp, 0.997),
              likelihood_quantile(zzexp, 0.95),
              likelihood_quantile(zzexp, 0.68))

    labels = r">$5\sigma$", r"$5\sigma$", r"$4\sigma$", r"$3\sigma$", r"$2\sigma$",  r"$1\sigma$"

    tmap = np.zeros_like(zz)
    for level in levels:
        tmap[zzexp >= level] += 1

    ax.imshow(tmap, origin="lower", cmap="binary", extent=extent, aspect="auto")

    xrange = 0, 1.2
    yrange = 2.4, 5.2

    # ax.plot(xrange, [4.4, 4.4], 'r--')
    # ax.plot(xrange, [3.71, 3.71], 'b--')
    plot_ext(ax)

    bounds = np.array([0, 1, 2, 3, 4, 5, 6])
    cmap = plt.cm.binary
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    axb = inset_axes(ax, width="55%", height="3%", loc=1, borderpad=0.8)
    ticks = bounds+0.5
    cb = mpl.colorbar.ColorbarBase(axb, cmap=cmap, norm=norm, spacing='proportional', ticks=ticks,
                                   boundaries=bounds, format='%1i', orientation='horizontal')
    cb.ax.set_xticklabels(labels, fontsize=fontsize-4)

    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    ax.set_aspect((xrange[1]-xrange[0])/(yrange[1]-yrange[0]))
    ax.set_xlabel("$k$", fontsize=fontsize)
    ax.set_ylabel(r"$\alpha$", fontsize=fontsize)
    if legend:
        ax.legend(loc="lower left", frameon=False, fontsize=fontsize-4)


def marginal_likelihood_plot(ax, df, key, range, bins=10, fontsize=15, legend=False,
                             orientation="vertical", label_axes=False):
    """
    """
    values = df[key].values
    weights = np.exp(df["poisson_likelihood_2d"].values + 1000)
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

    model_name = "blue_final"
    xkey = "rec_phys_alpha"
    ykey = "uae_phys_k"
    zkey = "poisson_likelihood_2d"

    config = get_config()
    image_dir = os.path.join(config["directories"]["data"], "images")
    image_filename = os.path.join(image_dir, f"model_likelihood_{model_name}.png")

    grid = ParameterGrid(model_name)
    df = grid.load_metrics()

    x = np.random.rand(50)
    y = np.random.rand(50)

    fig = plt.figure(figsize=(7, 7))
    spec = GridSpec(ncols=4, nrows=4, figure=fig)
    ax0 = fig.add_subplot(spec[1:4, 0:3])
    ax1 = fig.add_subplot(spec[0, 0:3])
    ax2 = fig.add_subplot(spec[1:4, 3])

    contour_plot_threshold(ax0, df)
    m1, s1 = marginal_likelihood_plot(ax1, df, "uae_phys_k", range=(0.25, 1))
    m2, s2 = marginal_likelihood_plot(ax2, df, "rec_phys_alpha", range=(3, 4.5),
                                      orientation="horizontal")

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
        plt.savefig(image_filename, dpi=150, bbox_inches="tight")

    plt.show(block=False)
