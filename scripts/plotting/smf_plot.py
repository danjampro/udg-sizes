import os
import matplotlib.pyplot as plt

from udgsizes.core import get_config
from udgsizes.fitting.utils.plotting import smf_plot

SAVE = True

if __name__ == "__main__":

    config = get_config()
    image_dir = os.path.join(config["directories"]["data"], "images")
    image_filename = os.path.join(image_dir, f"smf_plot.png")

    fig, ax = plt.subplots()
    smf_plot([1], pfixed=[0.00071, 7.5], prange=[[0], [5]], ax=ax)
    smf_plot([-0.3], color="r", prange=[[-0.5], [-0.1]], ax=ax, plot_ref=False)

    if SAVE:
        plt.savefig(image_filename, dpi=150, bbox_inches="tight")
