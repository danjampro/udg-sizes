"""
"""
import os
import matplotlib.pyplot as plt

from udgsizes.core import get_config
from udgsizes.fitting.grid import load_metrics
from udgsizes.fitting.utils.plotting import contour_plot
from udgsizes.fitting.utils.plotting import plot_ext

SAVE = True
FONTSIZE = 14
MODEL_LABELS = {"blue_baldry_final": "Model 1",
                "blue_baldry_trunc_final": "Model 2",
                "blue_baldry_dwarf_final": "Model 3"}

if __name__ == "__main__":

    ax = None
    colors = ["k", "darkblue", "r"]
    xkey = "rec_phys_alpha"
    ykey = "logmstar_a"
    xlabel = r"$\alpha$"
    ylabel = r"$\kappa$"
    model_names = ["blue_baldry_dwarf_final", "blue_baldry_trunc_final", "blue_baldry_final"]

    for model_name, color in zip(model_names, colors):
        df = load_metrics(model_name)
        ax = contour_plot(df, xkey, ykey, color=color, ax=ax, show=False, label_contours=True,
                          label=MODEL_LABELS[model_name])

    plot_ext(ax=ax, labels=False)

    ax.legend(loc="upper right", fontsize=FONTSIZE-1)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE+2)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE+2)

    plt.show(block=False)

    if SAVE:
        config = get_config()
        image_dir = os.path.join(config["directories"]["data"], "images")
        image_filename = os.path.join(image_dir, f"contour_plot.png")
        plt.savefig(image_filename, dpi=150, bbox_inches="tight")
