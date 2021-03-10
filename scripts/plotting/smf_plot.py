import os
import matplotlib.pyplot as plt

from udgsizes.core import get_config
from udgsizes.fitting.utils.plotting import smf_plot

SAVE = False

if __name__ == "__main__":

    config = get_config()
    image_dir = os.path.join(config["directories"]["data"], "images")
    image_filename = os.path.join(image_dir, f"smf_plot.png")

    fig, ax = plt.subplots()
    # smf_plot([1], pfixed=[0.00071, 7.5], prange=[[-0.5], [100]], ax=ax)
    # smf_plot([-0.3], color="r", prange=[[-0.85], [-0.125]], ax=ax, plot_ref=False)

    smf_plot([-1.45], pfixed=[0.00071, 10.72], ax=ax, color="b", linestyle="-")
    smf_plot([-1.41], pfixed=[0.00132, 10.54], ax=ax, color="r", linestyle="-")

    smf_plot([-1.25], pfixed=[0.00071, 10.72], ax=ax, color="k", linestyle="--")

    # smf_plot([-1.35], pfixed=[0.00071, 10.72], ax=ax, color="b", linestyle="--")
    # smf_plot([-1.45], pfixed=[0.00132, 10.54], ax=ax, color="r", linestyle="--")


    if SAVE:
        plt.savefig(image_filename, dpi=150, bbox_inches="tight")
