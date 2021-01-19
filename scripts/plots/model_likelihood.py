import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from udgsizes.core import get_config
from udgsizes.fitting.grid import ParameterGrid


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


def contour_plot_threshold(ax, df, legend=True):

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
              likelihood_quantile(zzexp, 0.997),
              likelihood_quantile(zzexp, 0.95),
              likelihood_quantile(zzexp, 0.68))

    labels = r"$5\sigma$", r"$3\sigma$", r"$2\sigma$", r"$1\sigma$"

    tmap = np.zeros_like(zz)
    for level in levels:
        tmap[zzexp >= level] += 1

    ax.imshow(tmap, origin="lower", cmap="binary", extent=extent, aspect="auto")

    xrange = 0, 1.2
    yrange = 2.5, 5.8
    fontsize = 15

    ax.plot(xrange, [4.4, 4.4], 'b--')
    ax.plot(xrange, [3.71, 3.71], 'r--')
    ax.axhspan(4.4-0.19, 4.4+0.19, color="b", alpha=0.2,
               label="van der Burg +16 (clusters),\nAmorisco +16")
    ax.axhspan(3.71-0.33, 3.71+0.33, color="r", alpha=0.2,
               label="van der Burg +17 (groups)")

    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    ax.set_aspect((xrange[1]-xrange[0])/(yrange[1]-yrange[0]))
    ax.set_xlabel("$k$", fontsize=fontsize)
    ax.set_ylabel(r"$\alpha$", fontsize=fontsize)
    if legend:
        ax.legend(loc="best", frameon=False)


def marginal_likelihood_plot(ax, df, key, range, bins=10, fontsize=15, legend=True):
    """
    """
    values = df[key].values
    weights = np.exp(df["poisson_likelihood_2d"].values + 1000)
    ax.hist(values, range=range, bins=bins, weights=weights, density=True, histtype="step",
            color="k", label="marginal PDF")

    cond = (values >= range[0]) & (values < range[1])
    values = values[cond]
    weights = weights[cond]

    mean = np.average(values, weights=weights)
    variance = np.average((values-mean)**2, weights=weights)
    std = np.sqrt(variance)

    ylim = ax.get_ylim()
    ax.plot([mean-std, mean-std], ylim, 'k--', linewidth=1.3)
    ax.plot([mean+std, mean+std], ylim, 'k--', linewidth=1.3)
    ax.set_ylim(ylim)

    if key == "rec_phys_alpha":
        ax.axvspan(4.4-0.19, 4.4+0.19, color="b", alpha=0.2)
        ax.axvspan(3.71-0.33, 3.71+0.33, color="r", alpha=0.2)
        ax.set_xlabel(r"$\alpha$", fontsize=fontsize)
    else:
        ax.set_xlabel("$k$", fontsize=fontsize)

    if legend:
        ax.legend(loc="best")


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

    fig = plt.figure(figsize=(12, 4))

    #spec = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
    #ax0 = fig.add_subplot(spec[:, 0])
    #ax1 = fig.add_subplot(spec[:, 1])
    #ax2 = fig.add_subplot(spec[:, 2])

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4))

    #contour_plot_gauss(ax0, df)
    contour_plot_threshold(ax0, df)
    marginal_likelihood_plot(ax1, df, "rec_phys_alpha", range=(3, 4.5))
    marginal_likelihood_plot(ax2, df, "uae_phys_k", range=(0.25, 1))

    for ax in (ax0, ax1, ax2):
        asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
        ax.set_aspect(asp)

    plt.show(block=False)
    plt.savefig(image_filename, dpi=150, bbox_inches="tight")
