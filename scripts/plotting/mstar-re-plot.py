import numpy as np
import matplotlib.pyplot as plt

from udgsizes.model.smf_dwarf import SmfDwarfModel as Model
from udgsizes.utils.shen import logmstar_to_mean_rec


# ALPHAS = [0.1, 0.14, 0.18]
ALPHAS = [0.1, 0.14, 0.25, 0.3, 0.5]
LOGMSTAR_KINK = 9


if __name__ == "__main__":

    model = Model(model_name="blue_sedgwick_shen")
    logmstar_shen = np.linspace(LOGMSTAR_KINK, 11, 20)

    logmstar = np.linspace(5, LOGMSTAR_KINK, 20)

    fig, ax = plt.subplots()
    ax.plot(logmstar_shen, np.log10(logmstar_to_mean_rec(logmstar_shen)), "k-")

    for alpha in ALPHAS:
        rec = [model._mean_rec_phys(_, alpha=alpha) for _ in logmstar]
        ax.plot(logmstar, np.log10(rec), "k--")

    plt.show(block=False)
