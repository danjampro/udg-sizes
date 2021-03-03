import numpy as np
from scipy.ndimage.filters import gaussian_filter

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from udgsizes.core import get_config
from udgsizes.utils.library import load_module
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


def likelihood_threshold_plot(df, xkey, ykey, metric="poisson_likelihood_2d", ax=None,
                              legend=True, xrange=None, yrange=None, fontsize=15, show=True,
                              **kwargs):
    """
    """
    if ax is None:
        fig, ax = plt.subplots(**kwargs)

    x = df[xkey].values
    y = df[ykey].values
    z = df[metric].values
    extent = (x.min(), x.max(), y.min(), y.max())  # l r b t

    nx = np.unique(x).size
    ny = np.unique(y).size

    zz = z.reshape(nx, ny).T
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
    ax.set_xlabel(xkey, fontsize=fontsize)
    ax.set_ylabel(ykey, fontsize=fontsize)

    if legend:
        ax.legend(loc="lower left", frameon=False, fontsize=fontsize-4)
    if show:
        plt.show(block=False)
    return ax


def contour_plot(df, xkey, ykey, metric="poisson_likelihood_2d", ax=None, xrange=None,
                 yrange=None, fontsize=15, show=True, smooth=True, color="k", label_contours=True,
                 label=None, **kwargs):
    """
    """
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
              confidence_threshold(zzexp, 0.997),
              confidence_threshold(zzexp, 0.68))
    labels = r"$5\sigma$", r"$3\sigma$", r"$1\sigma$"

    if smooth:
        zzexp = gaussian_filter(zzexp, 0.5)

    cs = ax.contour(zzexp, linewidths=0.8, colors=color, extent=extent, levels=levels)
    fmt = {}
    for l, s in zip(cs.levels, labels):
        fmt[l] = s
    if label_contours:
        ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=10)

    # Label axis
    if label is not None:
        cs.collections[0].set_label(label)

    # Format axes
    if xrange is not None:
        ax.set_xlim(*xrange)
    if yrange is not None:
        ax.set_ylim(*yrange)
    if (xrange is not None) and (yrange is not None):
        ax.set_aspect((xrange[1]-xrange[0])/(yrange[1]-yrange[0]))
    ax.set_xlabel(ykey, fontsize=fontsize)
    ax.set_ylabel(xkey, fontsize=fontsize)

    if show:
        plt.show(block=False)

    return ax


def smf_plot(pbest, prange=None, which="schechter_baldry", pref=[-1.45], range=(4, 12), logy=True,
             nsamples=100, ax=None, show=True, config=None, pfixed_ref=[0.00071, 10.72],
             pfixed=None, fitxmax=9, linewidth=1.5, color="b", plot_ref=True, **kwargs):
    """
    """
    if config is None:
        config = get_config()
        cosmo = config["cosmology"]

    if pfixed is None:
        pfixed = pfixed_ref

    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    func = load_module(f"udgsizes.model.components.mstar.{which}")

    xx = np.linspace(range[0], range[1], nsamples)
    is_fit = xx < fitxmax

    if plot_ref:
        ax.plot(xx, func(xx, *pref, *pfixed_ref, cosmo=cosmo), 'k-', linewidth=linewidth)
    ax.plot(xx[is_fit], func(xx[is_fit], *pbest, *pfixed, cosmo=cosmo), '--', linewidth=linewidth,
            color=color)

    if prange is not None:
        mins = np.ones(is_fit.sum()) * np.inf
        maxs = -mins.copy()
        for ps in prange:
            ys = func(xx[is_fit], *ps, *pfixed, cosmo=cosmo)
            mins[:] = np.minimum(mins, ys)
            maxs[:] = np.maximum(maxs, ys)
        ax.fill_between(x=xx[is_fit], y1=mins, y2=maxs, alpha=0.2, color=color,
                        linewidth=linewidth)

    if logy:
        ax.set_yscale("log")

    if show:
        plt.show(block=False)

    return ax


def plot_ext(ax, alpha=0.2, labels=True):
    """
    """
    label = "van der Burg +16 (clusters), Amorisco +16" if labels else None
    ax.axhspan(4.4-0.19, 4.4+0.19, color="r", alpha=alpha,
               label=label)
    label = "van der Burg +17 (groups)" if labels else None
    ax.axhspan(3.71-0.33, 3.71+0.33, color="b", alpha=alpha,
               label=label)
