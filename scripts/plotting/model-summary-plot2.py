import numpy as np
import matplotlib.pyplot as plt
from udgsizes.fitting.grid import ParameterGrid
from udgsizes.obs.sample import load_sample

# MODEL_NAME = "blue_sedgwick_shen"
MODEL_NAME = "blue_sedgwick_shen_final"
SAVEFIG = False
FIGHEIGHT = 2
FONTSIZE = 14
BINS_OBS = 10
BINS_MODEL = 30
PAR_NAMES = "logmstar", "redshift", "rec_phys", "uae_obs_jig", "rec_obs_jig", "colour_obs"

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


"""
def plot_all_samples(grid, ax_dict, alpha_max=0.2, alpha_min=0, metric="likelihood", linewidth=1):

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
"""


def plot_best_samples(grid, ax_dict, metric="likelihood", q=0.9):
    """
    """
    mins = {_: np.ones(BINS_MODEL) * np.inf for _ in PAR_NAMES}
    maxs = {_: np.ones(BINS_MODEL) * -np.inf for _ in PAR_NAMES}
    centres = {}

    for df in grid.load_best_samples(q=q, metric=metric):

        # Apply selection
        df = df[df["selected_jig"].values == 1].reset_index(drop=True)

        for key in PAR_NAMES:
            values = df[key].values
            hist, edges = np.histogram(values, range=RANGES[key], bins=BINS_MODEL, density=True)
            centres[key] = 0.5 * (edges[1:] + edges[:-1])

            mins[key] = np.minimum(mins[key], hist)
            maxs[key] = np.maximum(maxs[key], hist)

    for key, ax in ax_dict.items():
        ax.fill_between(centres[key], mins[key], maxs[key], color="k", alpha=0.3, linewidth=0.0)

        print(maxs[key])

    return ax_dict


def plot_best_sample(grid, ax_dict, metric="likelihood", linewidth=1.5, color="k"):
    """
    """
    df = grid.load_best_sample(select=True)
    df = df[df["selected_jig"].values == 1].reset_index(drop=True)

    for key, ax in ax_dict.items():

        values = df[key].values
        hist, edges = np.histogram(values, range=RANGES[key], bins=BINS_MODEL, density=True)
        centres = 0.5 * (edges[1:] + edges[:-1])

        ax.plot(centres, hist, "-", linewidth=linewidth, color=color)


def plot_observations(ax_dict, color="b"):
    """
    """
    dfo = load_sample(select=True)

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


if __name__ == "__main__":
    """
    from udgsizes.core import get_config
    config = get_config()
    model_type = "udgsizes.model.smf_dwarf.SmfDwarfModel"
    config["grid"][model_type]["parameters"]["rec_phys_offset"]["alpha"]["max"] = 0.7
    config["grid"][model_type]["parameters"]["rec_phys_offset"]["alpha"]["step"] = 0.05
    config["grid"][model_type]["parameters"]["logmstar"]["a"]["min"] = -1.50
    config["grid"][model_type]["parameters"]["logmstar"]["a"]["max"] = -1.35
    config["grid"][model_type]["parameters"]["logmstar"]["a"]["step"] = 0.05
    grid = ParameterGrid(MODEL_NAME, config=config)
    """

    grid = ParameterGrid(MODEL_NAME)

    # fig = plt.figure(figsize=(FIGHEIGHT * len(PAR_NAMES), FIGHEIGHT * 1.2))
    fig = plt.figure(figsize=(5, 7.5))

    ax_dict = {}
    for i, key in enumerate(PAR_NAMES):
        ax_dict[key] = plt.subplot(3, 2, i+1)

        # Axes labels and tick formatting
        if i == 0:
            ax_dict[key].set_ylabel("PDF", fontsize=FONTSIZE)
            ax_dict[key].set_xlabel(LABELS[key], fontsize=FONTSIZE)

        ax_dict[key].set_xlabel(LABELS[key], fontsize=FONTSIZE)
        ax_dict[key].axes.yaxis.set_ticklabels([])

    plot_best_samples(grid, ax_dict)

    plot_best_sample(grid, ax_dict)

    plot_observations(ax_dict)

    # Additional formatting
    plt.tight_layout()

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
