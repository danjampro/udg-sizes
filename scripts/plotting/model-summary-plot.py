import os
import matplotlib.pyplot as plt
from udgsizes.fitting.grid import ParameterGrid
from udgsizes.obs.sample import load_sample

MODEL_NAME = "blue_sedgwick_shen_0.35"
SAVEFIG = False
FIGHEIGHT = 2
FONTSIZE = 14
BINS = 10
PAR_NAMES = "logmstar", "redshift", "rec_phys", "uae_obs_jig", "rec_obs_jig", "colour_obs"
HISTKWARGS = {"density": True, "bins": BINS}

LABELS = {"logmstar": r"$\log_{10}\ \mathrm{M_{*} / M_{\odot}}$",
          "rec_phys": r"$\mathrm{\hat{r}_{e}\ [kpc]}$",
          "redshift": r"$z$",
          "uae_obs_jig": r"$\mathrm{\bar{\mu}_{e}\ [mag arcsec^{-2}]}$",
          "rec_obs_jig": r"$\mathrm{\bar{r}_{e}\ [kpc]}$",
          "colour_obs": r"$(g-r)$"}

OBSKEYS = {"uae_obs_jig": "mueff_av",
           "rec_obs_jig": "rec_arcsec",
           "colour_obs": "g_r"}


# TODO: Add KS values to graphs
# TODO: Make into grid method / plotting utils

if __name__ == "__main__":

    grid = ParameterGrid(MODEL_NAME)
    df = grid.load_best_sample(apply_prior=True, select=True)
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
