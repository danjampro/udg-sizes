import os
import numpy as np
import matplotlib.pyplot as plt

from udgsizes.fitting.grid import ParameterGrid
from udgsizes.obs.sample import load_sample
from udgsizes.utils.selection import GR_MIN, GR_MAX

MODEL_NAME = "blue_sedgwick_shen_final"
SAVEFIG = True
FONTSIZE = 14
BINS_OBS = 10
BINS_MODEL = 30
METRIC = "posterior_kde_3d"
PAR_NAMES = "logmstar", "redshift", "rec_phys", "uae_obs_jig", "rec_obs_jig", "colour_obs"

LABELS = {"logmstar": r"$\log_{10}\ \mathrm{M_{*} / M_{\odot}}$",
          "rec_phys": r"$\mathrm{\hat{r}_{e}\ [kpc]}$",
          "redshift": r"$z$",
          "uae_obs_jig": r"$\mathrm{\bar{\mu}_{e}\ [mag\ arcsec^{-2}]}$",
          "rec_obs_jig": r"$\mathrm{\bar{r}_{e}\ [arcsec]}$",
          "colour_obs": r"$(g-r)$"}

RANGES = {"logmstar": (5, 10.5),
          "redshift": (0, 0.3),
          "rec_phys": (0, 20),
          "uae_obs_jig": (24, 28),
          "rec_obs_jig": (3, 10),
          "colour_obs": (GR_MIN, GR_MAX)}

OBSKEYS = {"uae_obs_jig": "mueff_av",
           "rec_obs_jig": "rec_arcsec",
           "colour_obs": "g_r"}

KSTEST_KEYS = {"uae_obs_jig": "kstest_uae_obs_jig",
               "rec_obs_jig": "kstest_rec_obs_jig",
               "colour_obs": "kstest_colour_obs"}


def get_best_samples(grid, q=0.9):
    """
    """
    return grid.load_confident_samples(q=q, metric=METRIC)


def plot_best_samples(dfs, ax_dict, q=0.9):
    """
    """
    mins = {_: np.ones(BINS_MODEL) * np.inf for _ in PAR_NAMES}
    maxs = {_: np.ones(BINS_MODEL) * -np.inf for _ in PAR_NAMES}
    centres = {}

    for df in dfs:

        # Apply selection
        df = df[df["selected"].values == 1].reset_index(drop=True)

        for key in PAR_NAMES:
            values = df[key].values
            hist, edges = np.histogram(values, range=RANGES[key], bins=BINS_MODEL, density=True)
            centres[key] = 0.5 * (edges[1:] + edges[:-1])

            mins[key] = np.minimum(mins[key], hist)
            maxs[key] = np.maximum(maxs[key], hist)

    for key, ax in ax_dict.items():
        ax.fill_between(centres[key], mins[key], maxs[key], color="k", alpha=0.3, linewidth=0.0)

    return ax_dict


def plot_best_sample(grid, ax_dict, linewidth=1.5, color="k"):
    """
    """
    df = grid.load_best_sample(select=True, metric=METRIC)

    for key, ax in ax_dict.items():

        values = df[key].values
        hist, edges = np.histogram(values, range=RANGES[key], bins=BINS_MODEL, density=True)
        centres = 0.5 * (edges[1:] + edges[:-1])

        ax.plot(centres, hist, "-", linewidth=linewidth, color=color)


def plot_observations(grid, ax_dict, color="b"):
    """
    """
    dfo = load_sample(select=True)

    # Find the good-fitting model with the best ks-statistics
    dfm = grid.load_confident_metrics()
    metrics = dfm.iloc[np.argmax(dfm["kstest_min"])]

    for key, ax in ax_dict.items():
        if key in OBSKEYS:

            ax = ax_dict[key]

            values = dfo[OBSKEYS[key]].values
            rng = RANGES[key]

            hist_norm, edges = np.histogram(values, range=rng, bins=BINS_OBS, density=True)
            centres = 0.5 * (edges[1:] + edges[:-1])

            hist, _ = np.histogram(values, range=rng, bins=BINS_OBS, density=False)
            yerr = np.sqrt(hist) * hist_norm.max() / hist.max()

            ax.errorbar(centres, hist_norm, yerr=yerr, color=color, linewidth=0, linestyle=None,
                        elinewidth=1.5, marker="o", markersize=3, zorder=10)

            pval = metrics[KSTEST_KEYS[key]]
            ax.text(0.22, 0.75, r"$p_{KS}=$" + rf"{pval:.2f}", transform=ax.transAxes,
                    fontsize=FONTSIZE-1, color="k")


if __name__ == "__main__":

    grid = ParameterGrid(MODEL_NAME)

    fig = plt.figure(figsize=(5, 7.5))

    ax_dict = {}
    for i, key in enumerate(PAR_NAMES):
        ax_dict[key] = plt.subplot(3, 2, i+1)

        # Axes labels and tick formatting
        if i % 2 == 0:
            ax_dict[key].set_ylabel("PDF", fontsize=FONTSIZE)
            ax_dict[key].set_xlabel(LABELS[key], fontsize=FONTSIZE)

        ax_dict[key].set_xlabel(LABELS[key], fontsize=FONTSIZE)
        ax_dict[key].axes.yaxis.set_ticklabels([])

    dfs = get_best_samples(grid)
    plot_best_samples(dfs, ax_dict)

    plot_best_sample(grid, ax_dict)

    plot_observations(grid, ax_dict)

    # Additional formatting
    plt.tight_layout()

    if SAVEFIG:
        filename = os.path.join(grid.config["directories"]["images"],
                                f"model-summary-{MODEL_NAME}.png")
        plt.savefig(filename, dpi=150, bbox_inches="tight")

    plt.show(block=False)
