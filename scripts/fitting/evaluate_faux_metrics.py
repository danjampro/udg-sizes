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
MODEL_NAME = "blue_sedgwick_shen_0.35"
METRICS_IGNORE = ["kstest_2d"]
NITERS = 100
NPROC = 4


if __name__ == "__main__":

    # Load best sample
    grid = ParameterGrid(MODEL_NAME)

    # Make directory for output
    directory = os.path.join(grid.directory, "faux")
    os.makedirs(directory, exist_ok=True)

    # Calculate metrics
    for i in range(NITERS):

        grid.logger.debug(f"Iteration {i+1} of {NITERS}.")

        dff = grid.make_faux_observations()

        filename = os.path.join(directory, f"metrics_{i}.csv")

        grid.evaluate(dfo=dff, filename=filename, nproc=NPROC, metrics_ignore=METRICS_IGNORE)
