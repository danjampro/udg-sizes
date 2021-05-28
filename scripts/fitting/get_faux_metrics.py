"""
Calculate metrics using model samples generated from best-fit model in place of real observations.
"""
import os
import numpy as np
import pandas as pd

from udgsizes.core import get_config
from udgsizes.obs.sample import load_sample
from udgsizes.fitting.grid import ParameterGrid

CONFIG = get_config()
MODEL_NAME = "blue_sedgwick_shen_final"
METRICS_IGNORE = ["kstest_2d"]
NITERS = 100
NPROC = 4


def get_faux_observations(df, dfo):
    """ Get sample of mock observations from the model sample
    Args:
        df (pd.DataFrame): Model samples from the best-fitting model.
    Returns:
        pd.DataFrame: Faux observations extracted from model samples.
    """
    # Choose a random sample of the same size as observations
    indices = np.random.randint(0, df.shape[0], dfo.shape[0])

    # Map the model keys into observation keys
    dff = pd.DataFrame()
    for key, obskey in CONFIG["obskeys"].items():
        dff[obskey] = df[key].values[indices]

    return dff


if __name__ == "__main__":

    # Load observations
    dfo = load_sample()

    # Load best sample
    grid = ParameterGrid(MODEL_NAME)
    df = grid.load_best_sample()

    # Make directory for output
    directory = os.path.join(grid.directory, "faux")
    os.makedirs(directory, exist_ok=True)

    # Calculate metrics
    for i in range(NITERS):

        dff = get_faux_observations(df, dfo)

        filename = os.path.join(directory, f"metrics_{i}.csv")

        grid.evaluate(dfo=dff, filename=filename, nproc=NPROC, metrics_ignore=METRICS_IGNORE)
