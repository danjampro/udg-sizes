"""
Calculate metrics using model samples generated from best-fit model in place of real observations.
"""
import argparse
import os

from udgsizes.core import get_config
from udgsizes.fitting.grid import ParameterGrid

CONFIG = get_config()
METRICS_IGNORE = ["kstest_2d"]
NITERS = 100
NPROC = 4


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_arg("model_name", type="str", help="The model name.")

    parsed_args = parser.parse_args()
    model_name = parsed_args.model_name

    # Load best sample
    grid = ParameterGrid(model_name)

    # Make directory for output
    directory = os.path.join(grid.directory, "faux")
    os.makedirs(directory, exist_ok=True)

    # Calculate metrics
    for i in range(NITERS):

        grid.logger.debug(f"Iteration {i+1} of {NITERS}.")

        dff = grid.make_faux_observations()

        filename = os.path.join(directory, f"metrics_{i}.csv")

        grid.evaluate(dfo=dff, filename=filename, nproc=NPROC, metrics_ignore=METRICS_IGNORE)
