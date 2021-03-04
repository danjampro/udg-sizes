""" Script to update a single metric with optional thinning. """
import argparse
from udgsizes.fitting.grid import ParameterGrid

KEY = "kstest_2d"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("--thinning", type=int, default=None)

    args = parser.parse_args()
    model_name = args.model_name
    thinning = args.thinning

    grid = ParameterGrid(model_name)
    values = []
    for i in range(99):
        values.append(grid.evaluate_one(index=i, thinning=100)[KEY])

    df = grid.load_metrics()
    df[KEY] = values
    df.to_csv(grid._metric_filename)
