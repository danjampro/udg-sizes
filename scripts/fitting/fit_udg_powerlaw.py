import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal

from udgsizes.model.utils import create_model
from udgsizes.fitting.grid import ParameterGrid

MODEL_NAME = "blue_sedgwick_shen_0.35"
UDG_MODEL_NAME = "blue_sedgwick_shen_udg"
SAVEFIG = True


def func(x, m, c):
    return m * x + c


def udg_powerlaw(rec_phys, m, c):
    log_rec = np.log(rec_phys)
    y = np.exp(func(log_rec, m=m, c=c))
    return y


def pretty_plot(rec_phys, popt, pcov, bins=10, nsamples=500, fontsize=15, filename=None):
    """
    """
    y, edges = np.histogram(np.log10(rec_phys), bins=bins, density=True)  # Log10 bins
    centres = 0.5 * (edges[1:] + edges[:-1])

    fig, ax = plt.subplots()

    var = multivariate_normal(popt, pcov)
    xx = np.linspace(rec_phys.min(), rec_phys.max(), 10)

    for i in range(nsamples):
        m, c = var.rvs()
        yy = udg_powerlaw(xx, m, c)
        ax.plot(np.log10(xx), yy, "-", alpha=0.01, linewidth=2, color="0.1")

    ax.plot(centres, y, "bo", fillstyle="none", markeredgewidth=1.5, label="Model UDG",
            markersize=5, zorder=20)
    # ax.plot(centres, y, "ko", markeredgecolor="b", label="Model UDG")

    yya = udg_powerlaw(xx, -2.71, popt[1])
    ax.plot(np.log10(xx), yya, "--", color="dodgerblue", linewidth=1.5, label="vdB+17 (groups)")

    yyb = udg_powerlaw(xx, -3.40, popt[1])
    ax.plot(np.log10(xx), yyb, "--", color="orchid", linewidth=1.5, label="vdB+16 (clusters)")

    yybest = udg_powerlaw(xx, *popt)
    ax.plot(np.log10(xx), yybest, "--", color="r", label="Best Fit")

    ax.set_yscale("log")

    ax.legend(loc="upper right", fontsize=fontsize-3)
    ax.set_xlabel(r"$\log_{10} \hat{r}_{e}\ \mathrm{[kpc]}$", fontsize=fontsize)
    ax.set_ylabel(r"$\mathrm{PDF}$", fontsize=fontsize)

    if filename is not None:
        plt.savefig(filename, dpi=150, bbox_inches="tight")

    plt.show(block=False)


if __name__ == "__main__":

    n_samples = 10000
    burnin = 1000

    # Get best fitting hyper parameters
    grid = ParameterGrid(MODEL_NAME)

    # Get best fitting hyper parameters
    hyper_params = grid.get_best_hyper_parameters()

    # Sample the model with no recovery efficiency
    model = create_model(UDG_MODEL_NAME, ignore_recov=True)
    df = model.sample(burnin=burnin, n_samples=n_samples, hyper_params=hyper_params)

    # Identify UDGs
    cond = df["is_udg"].values == 1
    df = df[cond].reset_index(drop=True)

    # Get sizes
    x = df["rec_phys"].values

    # Make histogram
    y, edges = np.histogram(np.log10(x), bins=10, density=True)  # Log10 bins
    centres = 0.5 * (edges[1:] + edges[:-1])

    logx = np.log(10**centres)
    logy = np.log(y)

    # Do fit
    popt, pcov = curve_fit(func, ydata=logy, xdata=logx)
    power = popt[0]

    fig, ax = plt.subplots()
    ax.plot(centres, np.log10(np.exp(logy)), "ko")

    logxx = centres  # Log10 bins
    yy = np.exp(func(logx, *popt))
    ax.plot(logxx, np.log10(yy), "b-")

    yyref = ((10 ** logxx) ** power)
    normref = yy[0] / yyref[0]
    ax.plot(logxx, np.log10(yyref * normref), "r--")  # Should overlap with blue line

    plt.show(block=False)
    print(f"Power law slope: {power:.2f}")

    filename = None
    if SAVEFIG:
        filename = os.path.join(grid.config["directories"]["images"],
                                f"udg-powerlaw-{MODEL_NAME}.png")
    pretty_plot(x, popt, pcov, filename=filename)
