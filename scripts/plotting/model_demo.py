import os
from contextlib import suppress
import numpy as np
import matplotlib.pyplot as plt

from udgsizes.core import get_config
from udgsizes.model.empirical import Model
from udgsizes.obs.sample import load_sample

KEYS = ("rec_phys", "uae_phys", "redshift", "rec_obs_jig", "uae_obs_jig")
# COLOURS = ("red", "blue", "orange", "darkviolet", "deepskyblue", "chocolate")
COLOURS = ("crimson", "darkorange", "royalblue", "darkviolet", "deepskyblue", "tomato")
LOGY = ("rec_phys", "uae_phys", "redshift", "rec_obs_jig", "uae_obs_jig")
PLOTOBS = {"rec_obs_jig": "rec_arcsec",
           "uae_obs_jig": "mueff_av"}

XLABELS = {"rec_phys": r"$\hat{r}_{e}\ \mathrm{[kpc]}$",
           "uae_phys": r"$\hat{\mu}_{e}\ \mathrm{[mag\ arcsec^{-2}]}$",
           "redshift": r"$z$",
           "rec_obs_jig": r"$\bar{r}_{e}\ \mathrm{[arcsec]}$",
           "uae_obs_jig": r"$\bar{\mu}_{e}\ \mathrm{[mag\ arcsec^{-2}]}$"}


def sample_model(model, alpha, k, alphas, ks, burnin, n_samples):
    """
    """
    dfs_fixed_k = []
    for _alpha in alphas:
        params = {"uae_phys": [k], "rec_phys": [_alpha]}
        df = model.sample(burnin=burnin, n_samples=n_samples, hyper_params=params)
        df = df[df["selected_jig"].values == 1].reset_index(drop=True)
        dfs_fixed_k.append(df)

    dfs_fixed_alpha = []
    for _k in ks:
        params = {"uae_phys": [_k], "rec_phys": [alpha]}
        df = model.sample(burnin=burnin, n_samples=n_samples, hyper_params=params)
        df = df[df["selected_jig"].values == 1].reset_index(drop=True)
        dfs_fixed_alpha.append(df)

    return dfs_fixed_k, dfs_fixed_alpha


def plot_observations(ax, key, range, bins=10, marker="o", color="k", markersize=3, linewidth=0):
    """ TODO: Move to utils.
    """
    df = load_sample(select=True)
    values = df[key].values
    h, e = np.histogram(values, range=range, bins=bins)
    c = 0.5*(e[1:] + e[:-1])
    err = np.sqrt(h)
    norm = h.sum() * (e[1]-e[0])
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.errorbar(c, h/norm, yerr=err/norm, elinewidth=1, markersize=markersize, color=color,
                linewidth=linewidth, marker=marker, label="observed", zorder=10)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax


def make_plot(alpha, k, alphas, ks, dfs_fixed_k, dfs_fixed_alpha, keys=KEYS, colors=COLOURS,
              bins=20, figsize=(14, 5), fontsize=12):
    """
    """
    n_keys = len(keys)
    histkwargs = {"bins": bins, "histtype": "step", "density": True}

    plt.figure(figsize=figsize)

    idx = 1
    for key in keys:
        ax = plt.subplot(2, n_keys, idx)
        if key in LOGY:
            ax.set_yscale("log")
        values = [df[key].values for df in dfs_fixed_k]
        vmin = min([_.min() for _ in values])
        vmax = max([_.max() for _ in values])
        for i, v in enumerate(values):
            if key == keys[-1]:
                print(alphas[i], k)
                label = rf"$\alpha={alphas[i]:.1f},k={k:.1f}$"
            else:
                label = None
            color = colors[i]
            ax.hist(v, range=(vmin, vmax), color=color, label=label, **histkwargs)
        for i, v in enumerate(values):
            color = colors[i]
            plot_quantile(ax, v, color)
        if idx == 1:
            ax.set_ylabel("PDF", fontsize=fontsize)
        idx += 1
        with suppress(KeyError):
            kk = PLOTOBS[key]
            plot_observations(ax, kk, range=(vmin, vmax))
        if (key == keys[-1]):  # or key == "uae_obs_jig":
            ax.legend(loc="lower left", fontsize=fontsize-3)
        ax.set_xlim(vmin, vmax)
        ax.tick_params(axis='y', which='major', labelsize=fontsize-4)
        ax.tick_params(axis='y', which='minor', labelsize=fontsize-4)

    for key in keys:
        ax = plt.subplot(2, n_keys, idx)
        if key in LOGY:
            ax.set_yscale("log")
        values = [df[key].values for df in dfs_fixed_alpha]
        vmin = min([_.min() for _ in values])
        vmax = max([_.max() for _ in values])
        for i, v in enumerate(values):
            if key == keys[-1]:
                label = rf"$\alpha={alpha:.1f},k={ks[i]:.1f}$"
            else:
                label = None
            color = colors[len(dfs_fixed_k)+i]
            ax.hist(v, range=(vmin, vmax), color=color, label=label, **histkwargs)
        for i, v in enumerate(values):
            color = colors[len(dfs_fixed_k)+i]
            plot_quantile(ax, v, color)
        ax.set_xlabel(XLABELS[key], fontsize=fontsize)
        if idx == n_keys+1:
            ax.set_ylabel("PDF", fontsize=fontsize)
        idx += 1
        with suppress(KeyError):
            kk = PLOTOBS[key]
            plot_observations(ax, kk, range=(vmin, vmax))
        if key == keys[-1]:
            ax.legend(loc="lower left", fontsize=fontsize-3)
        ax.set_xlim(vmin, vmax)
        ax.tick_params(axis='y', which='major', labelsize=fontsize-4)
        ax.tick_params(axis='y', which='minor', labelsize=fontsize-4)

    # plt.tight_layout()


def plot_quantile(ax, values, color, q=0.9, linestyle="--", linewidth=0.9):
    """
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    v = np.quantile(values, q)
    ax.plot([v, v], ylim, linestyle=linestyle, linewidth=linewidth, color=color)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax


if __name__ == "__main__":

    model_name = "blue_trunc"
    burnin = 5000
    n_samples = 100000
    save = True

    k = 0.5
    # alphas = [3, 4, 5]
    alphas = [4, 6, 7]

    alpha = 4
    ks = [0, 0.5, 1]

    model = Model(model_name)
    dfs_fixed_k, dfs_fixed_alpha = sample_model(model=model, k=k, alpha=alpha, alphas=alphas,
                                                ks=ks, burnin=burnin, n_samples=n_samples)

    make_plot(alpha=alpha, k=k, alphas=alphas, ks=ks, dfs_fixed_k=dfs_fixed_k,
              dfs_fixed_alpha=dfs_fixed_alpha)
    plt.show(block=False)

    config = get_config()
    imagedir = os.path.join(config["directories"]["data"], "images")
    os.makedirs(imagedir, exist_ok=True)
    filename = os.path.join(imagedir, f"model_demo_{model_name}.png")

    if save:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
