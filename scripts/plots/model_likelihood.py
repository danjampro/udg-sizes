import os
import numpy as np
import matplotlib.pyplot as plt

from udgsizes.core import get_config
from udgsizes.fitting.grid import ParameterGrid


def likelihood_quantile(values, q):
    """
    """
    values_sorted = values.copy().reshape(-1)
    values_sorted.sort()
    values_sorted = values_sorted[::-1]
    csum = np.cumsum(values_sorted)
    total = csum[-1]
    idx = np.argmin(abs(csum - q*total))
    return values_sorted[idx]


if __name__ == "__main__":

    model_name = "blue_final"
    xkey = "rec_phys_alpha"
    ykey = "uae_phys_k"
    zkey = "poisson_likelihood_2d"

    config = get_config()
    image_dir = os.path.join(config["directories"]["data"], "images")
    image_filename = os.path.join(image_dir, f"model_likelihood_{model_name}.png")

    grid = ParameterGrid(model_name)

    df = grid.load_metrics()
    x = df[xkey].values
    y = df[ykey].values
    z = df[zkey].values
    extent = (y.min(), y.max(), x.min(), x.max())

    nx = np.unique(x).size
    ny = np.unique(y).size

    xx = x.reshape(nx, ny)
    yy = y.reshape(nx, ny)
    zz = z.reshape(nx, ny)
    zzexp = np.exp(zz+1000)

    levels = likelihood_quantile(zzexp, 0.95), likelihood_quantile(zzexp, 0.68)

    plt.figure()

    ax = plt.subplot()
    ax.imshow(zz, extent=extent, cmap="binary", origin='lower', vmin=-1.15E+3)
    ax.contour(zzexp, linewidths=0.8, colors=["w", "deepskyblue"], extent=extent, levels=levels)

    xrange = 0, 1.2
    fontsize = 15

    ax.plot(xrange, [4.4, 4.4], 'b--', label="van der Burg +16 (clusters), Amorisco +16")
    ax.plot(xrange, [3.7, 3.7], 'r--', label="van der Burg +17 (groups)")

    ax.set_xlim(*xrange)
    ax.set_ylim(2.5, 5)

    ax.set_aspect(1.2/2.5)

    ax.set_xlabel("$k$", fontsize=fontsize)
    ax.set_ylabel(r"$\alpha$", fontsize=fontsize)

    ax.legend(loc="best")

    plt.show(block=False)

    plt.savefig(image_filename, dpi=150, bbox_inches="tight")
