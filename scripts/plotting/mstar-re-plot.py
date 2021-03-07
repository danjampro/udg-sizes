import numpy as np
import matplotlib.pyplot as plt

from udgsizes.model.smf import SmfModel
from udgsizes.utils.shen import logmstar_to_mean_rec


# ALPHAS = [0.1, 0.14, 0.18]
ALPHAS = [0.1, 0.14, 0.25, 0.3]


if __name__ == "__main__":

    model = SmfModel(model_name="blue_sedgwick_shen")
    logmstar_shen = np.linspace(9, 11, 20)

    logmstar = np.linspace(5, 9, 20)

    fig, ax = plt.subplots()
    ax.plot(logmstar_shen, np.log10(logmstar_to_mean_rec(logmstar_shen)), "k-")

    for alpha in ALPHAS:
        rec = [model._mean_rec_phys(_, alpha=alpha) for _ in logmstar]
        ax.plot(logmstar, np.log10(rec), "k--")

    plt.show(block=False)
