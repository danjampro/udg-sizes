"""
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

from udgsizes.core import get_config
from udgsizes.fitting.utils.plotting import contour_plot

SAVE = True

if __name__ == "__main__":

    ax = None
    colors = ["orangered", "royalblue", "k"]
    xkey = "rec_phys_alpha"
    ykey = "logmstar_a"
    model_names = ["blue_baldry_dwarf", "blue_baldry_trunc", "blue_baldry"]

    for model_name, color in zip(model_names, colors):
        df = pd.read_csv(f"/Users/danjampro/Desktop/{model_name}.csv")
        ax = contour_plot(df, xkey, ykey, color=color, ax=ax, show=False)

    plt.show(block=False)

    if SAVE:
        config = get_config()
        image_dir = os.path.join(config["directories"]["data"], "images")
        image_filename = os.path.join(image_dir, f"contour_plot.png")
        plt.savefig(image_filename, dpi=150, bbox_inches="tight")
