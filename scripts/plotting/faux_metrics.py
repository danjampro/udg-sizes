import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from udgsizes.fitting.grid import ParameterGrid


def plot_par(name, values, filename):
    """
    """
    fig, ax = plt.subplots()
    ax.hist(values, density=False)
    ax.set_xlabel(name)

    title = f"{name}: {values.mean():.2f}±{values.std():.2f}"
    ax.set_title(title)

    if filename is not None:
        plt.savefig(filename, dpi=150, bbox_inches="tight")

    return ax


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="The model name.")
    parsed_args = parser.parse_args()
    model_name = parsed_args.model_name

    grid = ParameterGrid(model_name)
    image_dir = grid.config["directories"]["images"]

    # Load metrics evaluated from faux observations
    dfms = grid.load_faux_metrics()

    # Get best hyper parameters for each faux sample
    df = []
    for dfm in dfms:
        best_pars = grid.get_best_hyper_parameters(df=dfm, flatten=True, kstest_min=None)
        df.append(best_pars)

    df = pd.DataFrame(df)

    # Plot hist for each parameter
    for parname in df.columns:
        filename = os.path.join(image_dir, f"faux_{parname}_{model_name}.png")

        values = df[parname].values
        plot_par(parname, values, filename=filename)

    plt.show(block=False)
