import os
import numpy as np
import matplotlib.pyplot as plt
from udgsizes.fitting.grid import ParameterGrid
from udgsizes.obs.sample import load_sample

MODEL_NAME = "blue_sedgwick_shen_final"
SAVEFIG = False
FIGHEIGHT = 2
FONTSIZE = 14
BINS = 10
BINS_MODEL = 20
PAR_NAMES = "logmstar", "redshift", "rec_phys", "uae_obs_jig", "rec_obs_jig", "colour_obs"
HISTKWARGS = {"density": True, "bins": BINS}

LABELS = {"logmstar": r"$\log_{10}\ \mathrm{M_{*} / M_{\odot}}$",
          "rec_phys": r"$\mathrm{\hat{r}_{e}\ [kpc]}$",
          "redshift": r"$z$",
          "uae_obs_jig": r"$\mathrm{\bar{\mu}_{e}\ [mag arcsec^{-2}]}$",
          "rec_obs_jig": r"$\mathrm{\bar{r}_{e}\ [kpc]}$",
          "colour_obs": r"$(g-r)$"}

RANGES = {"logmstar": (5, 10.5),
          "redshift": (0, 0.3),
          "rec_phys": (0, 20),
          "uae_obs_jig": (24, 28),
          "rec_obs_jig": (3, 10),
          "colour_obs": (-0.1, 0.4)}

OBSKEYS = {"uae_obs_jig": "mueff_av",
           "rec_obs_jig": "rec_arcsec",
           "colour_obs": "g_r"}


# TODO: Add KS values to graphs
# TODO: Make into grid method / plotting utils


def plot_all_samples(grid, ax_dict, alpha_max=0.2, alpha_min=0, metric="likelihood", linewidth=1):
    """
    """
    dfm = grid.load_metrics()

    # Scale transparency with likelihood
    alphas = alpha_max * dfm[metric].values / dfm[metric].max()

    for i in range(dfm.shape[0]):

        if alphas[i] <= alpha_min:
            continue

        # Apply selection
        df = grid.load_sample(i)
        df = df[df["selected_jig"].values == 1].reset_index(drop=True)

        for key, ax in ax_dict.items():

            values = df[key].values
            hist, edges = np.histogram(values, range=RANGES[key], bins=BINS_MODEL, density=True)
            centres = 0.5 * (edges[1:] + edges[:-1])

            ax.plot(centres, hist, "-", alpha=alphas[i], linewidth=linewidth, color="k")

    return ax_dict


def plot_best_sample(grid, ax_dict, metric="likelihood", linewidth=1, color="deepskyblue"):
    """
    """
    df = grid.load_best_sample(select=True)
    df = df[df["selected_jig"].values == 1].reset_index(drop=True)

    for key, ax in ax_dict.items():

        values = df[key].values
        hist, edges = np.histogram(values, range=RANGES[key], bins=BINS_MODEL, density=True)
        centres = 0.5 * (edges[1:] + edges[:-1])

        ax.plot(centres, hist, "--", linewidth=linewidth, color=color)


if __name__ == "__main__":

    grid = ParameterGrid(MODEL_NAME)

    fig = plt.figure(figsize=(FIGHEIGHT * len(PAR_NAMES), FIGHEIGHT * 1.2))

    ax_dict = {}
    for i, key in enumerate(PAR_NAMES):
        ax_dict[key] = plt.subplot(1, len(PAR_NAMES), i+1)

        if i == 0:
            ax_dict[key].set_ylabel("PDF", fontsize=FONTSIZE)
            ax_dict[key].set_xlabel(LABELS[key], fontsize=FONTSIZE)

    plot_all_samples(grid, ax_dict)

    plot_best_sample(grid, ax_dict)

    plt.show(block=False)



    """
    df_list = grid.load_confident_samples(apply_prior=True)
    df = df[df["selected_jig"].values == 1].reset_index(drop=True)

    dfo = load_sample(select=True)

    fig = plt.figure(figsize=(FIGHEIGHT * len(PAR_NAMES), FIGHEIGHT * 1.2))

    for i, par_name in enumerate(PAR_NAMES):

        ax = plt.subplot(1, len(PAR_NAMES), i+1)
        values = df[par_name].values
        ax.hist(values, color="k", alpha=0.4, **HISTKWARGS)

        if par_name in OBSKEYS:
            values_obs = dfo[OBSKEYS[par_name]].values
            rng = values.min(), values.max()
            ax.hist(values_obs, range=rng, color="b", alpha=0.4, **HISTKWARGS)

        if i == 0:
            ax.set_ylabel("PDF", fontsize=FONTSIZE)
        ax.set_xlabel(LABELS[par_name], fontsize=FONTSIZE)

        ax.axes.yaxis.set_ticklabels([])

    plt.tight_layout()
    if SAVEFIG:
        filename = os.path.join(grid.config["directories"]["images"],
                                f"model-summary-{MODEL_NAME}.png")
        plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show(block=False)
    """
