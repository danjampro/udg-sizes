import argparse
from contextlib import suppress
import numpy as np

from udgsizes.fitting.grid import ParameterGrid
from udgsizes.fitting.interpgrid import InterpolatedGrid

METRICS_IGNORE = ["kstest_2d"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("--interpolate", action="store_true")

    args = parser.parse_args()
    model_name = args.model_name

    if args.interpolate:
        grid = InterpolatedGrid(model_name)
    else:
        grid = ParameterGrid(model_name)

    with np.errstate(divide='ignore'):  # Silence warnings
        grid.sample(overwrite=True)

    with suppress(AttributeError):
        grid.interpolate()

    grid.evaluate(metrics_ignore=METRICS_IGNORE)
