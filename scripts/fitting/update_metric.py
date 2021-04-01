""" Script to update a single metric with optional thinning. """
import argparse
from udgsizes.fitting.grid import ParameterGrid

KEYS_IGNORE = ["kstest_2d"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("metric", type=str)
    parser.add_argument("--thinning", type=int, default=None)

    args = parser.parse_args()
    model_name = args.model_name
    metric_name = args.metric
    thinning = args.thinning

    grid = ParameterGrid(model_name)

    KEYS_IGNORE.extend([k for k in grid._evaluator._metric_names if k != metric_name])
    keys_ignore = [k for k in KEYS_IGNORE if k != metric_name]

    values = []
    for i in range(grid.n_permutations):
        values.append(grid.evaluate_one(index=i, thinning=thinning,
                                        metrics_ignore=keys_ignore)[metric_name])

    df = grid.load_metrics()
    df[metric_name] = values
    df.to_csv(grid._metric_filename)
