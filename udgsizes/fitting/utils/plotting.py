import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from udgsizes.obs.sample import load_sample
from udgsizes.utils.stats.confidence import confidence_threshold


def fit_summary_plot(df, dfo=None, show=True, bins=15, select=True, **kwargs):

    if dfo is None:
        dfo = load_sample(select=select, **kwargs)
    if select:
        df = df[df["selected_jig"] == 1].reset_index(drop=True)

    fig = plt.figure(figsize=(8, 4))

    ax0 = plt.subplot(2, 1, 1)
    histkwargs = dict(density=True, histtype="step")
    rng = (min(dfo['mueff_av'].min(), df['uae_obs_jig'].min()),
           max(dfo['mueff_av'].max(), df['uae_obs_jig'].max()))
    ax0.hist(dfo['mueff_av'].values, color="k", range=rng, bins=bins, label="obs", **histkwargs)
    ax0.hist(df['uae_obs_jig'].values, color="b", range=rng, bins=bins, label="model",
             **histkwargs)
    ax0.legend(loc="best")
    ax0.set_xlabel("uae")

    ax1 = plt.subplot(2, 1, 2)
    rng = (min(dfo['rec_arcsec'].min(), df['rec_obs_jig'].min()),
           max(dfo['rec_arcsec'].max(), df['rec_obs_jig'].max()))
    ax1.hist(dfo['rec_arcsec'].values, color="k", range=rng, bins=bins, **histkwargs,
             label="obs")
    ax1.hist(df['rec_obs_jig'].values, color="b", range=rng, bins=bins, **histkwargs,
             label="model")
    ax1.legend(loc="best")
    ax1.set_xlabel("rec")

    plt.tight_layout()
    if show:
        plt.show(block=False)

    return fig


def likelihood_threshold_plot_2d(df, xkey, ykey, metric="poisson_likelihood_2d", ax=None,
                                 legend=True, xrange=None, yrange=None, fontsize=15, **kwargs):

    if ax is None:
        fig, ax = plt.subplots(**kwargs)

    x = df[xkey].values
    y = df[ykey].values
    z = df[metric].values
    extent = (y.min(), y.max(), x.min(), x.max())

    nx = np.unique(x).size
    ny = np.unique(y).size

    zz = z.reshape(nx, ny)
    zzexp = np.exp(zz+1000)

    # Identify the thresholds and make thresholded image
    levels = (confidence_threshold(zzexp, 0.999999426696856),
              confidence_threshold(zzexp, 0.999936657516334),
              confidence_threshold(zzexp, 0.997),
              confidence_threshold(zzexp, 0.95),
              confidence_threshold(zzexp, 0.68))
    labels = r">$5\sigma$", r"$5\sigma$", r"$4\sigma$", r"$3\sigma$", r"$2\sigma$",  r"$1\sigma$"
    tmap = np.zeros_like(zz)
    for level in levels:
        tmap[zzexp >= level] += 1

    # Display the thresholded image
    ax.imshow(tmap, origin="lower", cmap="binary", extent=extent, aspect="auto")

    # Add the inset discrete threshold colourbar
    bounds = np.array([0, 1, 2, 3, 4, 5, 6])
    cmap = plt.cm.binary
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    axb = inset_axes(ax, width="55%", height="3%", loc=1, borderpad=0.8)
    ticks = bounds+0.5
    cb = mpl.colorbar.ColorbarBase(axb, cmap=cmap, norm=norm, spacing='proportional', ticks=ticks,
                                   boundaries=bounds, format='%1i', orientation='horizontal')
    cb.ax.set_xticklabels(labels, fontsize=fontsize-4)

    # Format axes
    if xrange is not None:
        ax.set_xlim(*xrange)
    if yrange is not None:
        ax.set_ylim(*yrange)
    if (xrange is not None) and (yrange is not None):
        ax.set_aspect((xrange[1]-xrange[0])/(yrange[1]-yrange[0]))
    ax.set_xlabel(ykey, fontsize=fontsize)
    ax.set_ylabel(xkey, fontsize=fontsize)

    if legend:
        ax.legend(loc="lower left", frameon=False, fontsize=fontsize-4)
