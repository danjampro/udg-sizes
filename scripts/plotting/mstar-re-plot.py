import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from udgsizes.fitting.grid import ParameterGrid
from udgsizes.utils.shen import logmstar_to_mean_rec
from udgsizes.model.smf_dwarf import SmfDwarfModel as Model

from udgsizes.obs.sample import load_gama_masses

LOGMSTAR_KINK = 9

logmstar_min = 6
logmstar_max = 12

logrec_min = -1
logrec_max = 1.2

BINS = 40
SIGMA_FACTOR = 3
FONTSIZE = 14

MODEL_NAME = "blue_sedgwick_shen_final"

SAVEFIG = False


def calculate_rec(logmstar, alpha):
    func = Model(model_name=MODEL_NAME)._mean_rec_phys
    return np.array([func(_, alpha=alpha) for _ in logmstar])


if __name__ == "__main__":

    grid = ParameterGrid(MODEL_NAME)
    dfg = load_gama_masses()

    alpha_mean, alpha_std = grid.parameter_stats("rec_phys_offset_alpha")

    logmstar_shen = np.linspace(LOGMSTAR_KINK, logmstar_max, 20)
    logmstar_ext = np.linspace(logmstar_min, LOGMSTAR_KINK, 20)

    fig, ax = plt.subplots()

    ax.hist2d(dfg["logmstar"].values, np.log10(dfg["rec_phys"].values), cmap="binary",
              range=((logmstar_min, logmstar_max), (logrec_min, logrec_max)), bins=BINS,
              norm=mpl.colors.LogNorm())

    ax.plot(logmstar_shen, np.log10(logmstar_to_mean_rec(logmstar_shen)), "r-",
            label="Shen et al. (2003)")
    ax.plot(logmstar_ext, np.log10(logmstar_to_mean_rec(logmstar_ext)), "r--")

    rec_mins = calculate_rec(logmstar_ext, alpha=alpha_mean + SIGMA_FACTOR * alpha_std)
    rec_maxs = calculate_rec(logmstar_ext, alpha=alpha_mean - SIGMA_FACTOR * alpha_std)
    ax.fill_between(logmstar_ext, np.log10(rec_mins), np.log10(rec_maxs), color="dodgerblue",
                    alpha=0.4, label="Prole (2021), this study")

    ax.set_xlim((logmstar_min, logmstar_max))
    ax.set_ylim((logrec_min, logrec_max))

    ax.set_xlabel(r"$\log_{10}\ \mathrm{M_{*} / M_{\odot}}$", fontsize=FONTSIZE)
    ax.set_ylabel(r"$\log_{10}\ \mathrm{\hat{r}_{e}\ [kpc]}$", fontsize=FONTSIZE)

    plt.legend(loc="best", fontsize=FONTSIZE-2)

    if SAVEFIG:
        filename = os.path.join(grid.config["directories"]["images"], f"mstar-re-{MODEL_NAME}.png")
        plt.savefig(filename, dpi=150, bbox_inches="tight")

    plt.show(block=False)
