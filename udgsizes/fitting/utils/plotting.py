from functools import partial

import numpy as np
from scipy.stats import kstest
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

    fig = plt.figure(figsize=(6, 6))

    ax0 = plt.subplot(3, 1, 1)
    histkwargs = dict(density=True, histtype="step")
    rng = (min(dfo['mueff_av'].min(), df['uae_obs_jig'].min()),
           max(dfo['mueff_av'].max(), df['uae_obs_jig'].max()))
    ax0.hist(dfo['mueff_av'].values, color="k", range=rng, bins=bins, label="obs", **histkwargs)
    ax0.hist(df['uae_obs_jig'].values, color="b", range=rng, bins=bins, label="model",
             **histkwargs)
    ks = kstest(dfo['mueff_av'].values, df['uae_obs_jig'].values)[1]
    ax0.legend(loc="best")
    ax0.set_xlabel(f"uae (KS pval={ks:.2f})")

    ax1 = plt.subplot(3, 1, 2)
    rng = (min(dfo['rec_arcsec'].min(), df['rec_obs_jig'].min()),
           max(dfo['rec_arcsec'].max(), df['rec_obs_jig'].max()))
    ax1.hist(dfo['rec_arcsec'].values, color="k", range=rng, bins=bins, **histkwargs,
             label="obs")
    ax1.hist(df['rec_obs_jig'].values, color="b", range=rng, bins=bins, **histkwargs,
             label="model")
    ks = kstest(dfo['rec_arcsec'].values, df['rec_obs_jig'].values)[1]
    ax1.legend(loc="best")
    ax1.set_xlabel(f"rec (KS pval={ks:.2f})")

    ax1 = plt.subplot(3, 1, 3)
    rng = (min(dfo['g_r'].min(), df['colour_obs'].min()),
           max(dfo['g_r'].max(), df['colour_obs'].max()))
    ax1.hist(dfo['g_r'].values, color="k", range=rng, bins=bins, **histkwargs, label="obs")
    ax1.hist(df['colour_obs'].values, color="b", range=rng, bins=bins, **histkwargs, label="model")
    ks = kstest(dfo['g_r'].values, df['colour_obs'].values)[1]
    ax1.legend(loc="best")
    ax1.set_xlabel(f"g-r (KS pval={ks:.2f})")

    plt.tight_layout()
    if show:
        plt.show(block=False)

    return fig


def plot_2d_hist(x, y, z, ax=None, xrange=None, yrange=None, show=True, smooth=None, contour=False):
    """
    """
    if ax is None:
        fig, ax = plt.subplots()

    xsorted = np.unique(x)
    ysorted = np.unique(y)
    dx2 = (xsorted[1] - xsorted[0]) / 2
    dy2 = (ysorted[1] - ysorted[0]) / 2
    extent = (x.min() - dx2, x.max() + dx2, y.min() - dy2, y.max() + dy2)  # l r b t

    nx = np.unique(x).size
    ny = np.unique(y).size
    zz = z.reshape(nx, ny).T

    if smooth:
        zz = gaussian_filter(zz, smooth)

    # Display the thresholded image
    ax.imshow(zz, origin="lower", cmap="binary", extent=extent, aspect="auto")

    if contour:
        levels = [confidence_threshold(zz, 0.997),
                  confidence_threshold(zz, 0.95),
                  confidence_threshold(zz, 0.68)]
        ax.contour(zz, colors="b", linewidths=1, zorder=10, levels=levels, extent=extent)

    # Format axes
    if xrange is not None:
        ax.set_xlim(*xrange)
    if yrange is not None:
        ax.set_ylim(*yrange)
    if (xrange is not None) and (yrange is not None):
        ax.set_aspect((xrange[1]-xrange[0])/(yrange[1]-yrange[0]))

    if show:
        plt.show(block=False)
    return ax


def threshold_plot(x, y, z, ax=None, legend=True, xrange=None, yrange=None, fontsize=15, show=True,
                   xlabel=None, ylabel=None, smooth=None, **kwargs):
    """
    """
    if ax is None:
        fig, ax = plt.subplots(**kwargs)

    extent = (x.min(), x.max(), y.min(), y.max())  # l r b t

    nx = np.unique(x).size
    ny = np.unique(y).size

    zz = z.reshape(nx, ny).T

    if smooth is not None:
        zz = gaussian_filter(zz, smooth)

    # Identify the thresholds and make thresholded image
    levels = (confidence_threshold(zz, 0.999999426696856),
              confidence_threshold(zz, 0.999936657516334),
              confidence_threshold(zz, 0.997),
              confidence_threshold(zz, 0.95),
              confidence_threshold(zz, 0.68))
    labels = r">$5\sigma$", r"$5\sigma$", r"$4\sigma$", r"$3\sigma$", r"$2\sigma$",  r"$1\sigma$"

    levels = list(levels)[1:]
    labels = list(labels)[1:]
    labels[0] = ">" + labels[1]

    tmap = np.zeros_like(zz)
    for level in levels:
        tmap[zz >= level] += 1

    # Display the thresholded image
    ax.imshow(tmap, origin="lower", cmap="binary", extent=extent, aspect="auto")

    # Add the inset discrete threshold colourbar
    # bounds = np.array([0, 1, 2, 3, 4, 5, 6])
    bounds = np.arange(len(levels) + 2)
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
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    if legend:
        ax.legend(loc="lower left", frameon=False, fontsize=fontsize-4)
    if show:
        plt.show(block=False)
    return ax


def smf_plot(pbest, prange=None, which="schechter_baldry", pref=[-1.45], range=(4, 12), logy=True,
             nsamples=100, ax=None, show=True, config=None, pfixed_ref=[0.00071, 10.72],
             pfixed=None, fitxmax=15, linewidth=1.5, color="b", plot_ref=False, **kwargs):
    """
    """
    if config is None:
        config = get_config()
        cosmo = config["cosmology"]

    if pfixed is None:
        pfixed = pfixed_ref

    if ax is None:
        fig, ax = plt.subplots()
    func = load_module(f"udgsizes.model.components.mstar.{which}")
    try:
        func = partial(func, min=0)
    except Exception:
        pass

    xx = np.linspace(range[0], range[1], nsamples)
    is_fit = xx < fitxmax

    if plot_ref:
        yyref = [func(_, *pref, *pfixed_ref, cosmo=cosmo) for _ in xx]
        ax.plot(xx, yyref, 'k-', linewidth=linewidth)

    yy = [func(_, *pbest, *pfixed, cosmo=cosmo) for _ in xx[is_fit]]
    ax.plot(xx[is_fit], yy, '--', linewidth=linewidth, color=color, **kwargs)

    if prange is not None:
        mins = np.ones(is_fit.sum()) * np.inf
        maxs = -mins.copy()
        for ps in prange:
            ys = [func(_, *ps, *pfixed, cosmo=cosmo) for _ in xx[is_fit]]
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


def finite_model_plot(df, xkey, ykey, metric="poisson_likelihood_2d"):
    """ Make a plot to show which permuations have finite metric values.
    """
    x = df[xkey].values
    y = df[ykey].values
    z = df[metric].values

    cond = np.isfinite(z)

    fig, ax = plt.subplots()
    ax.plot(x[cond], y[cond], "b+")
    ax.plot(x[~cond], y[~cond], "r+")

    ax.set_xlabel(xkey)
    ax.set_ylabel(ykey)
    plt.title(metric)

    for i, (x, y) in enumerate(zip(x, y)):
        ax.text(x, y, f"{i}", fontsize=9, color="k")

    plt.show(block=False)

    return ax
