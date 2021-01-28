import numpy as np

# from udgsizes.fitting.grid import ParameterGrid
from udgsizes.fitting.interpgrid import InterpolatedGrid

if __name__ == "__main__":

    # TODO: Move to command line arg
    # model_names = "blue_baldry", "blue_baldry_dwarf", "blue_baldry_trunc"
    model_names = ["blue_baldry_highml"]
    metrics_ignore = ["kstest_2d"]  # Takes too long for whole grid

    for model_name in model_names:

        p = InterpolatedGrid(model_name)

        with np.errstate(divide='ignore'):  # Silence annoying warnings
            p.sample(overwrite=True)
        p.evaluate(metrics_ignore=metrics_ignore)
