import numpy as np

from udgsizes.fitting.utils import plotting
from udgsizes.fitting.grid import ParameterGrid

if __name__ == "__main__":

    model_name = "blue_baldry"  # TODO: Move to command line arg
    makeplots = False
    metrics_ignore = ["kstest_2d"]  # Takes too long for whole grid

    p = ParameterGrid(model_name)

    with np.seterr(divide='ignore'):  # Silence annoying warnings
        p.sample(overwrite=True)
    dfm = p.evaluate(metrics_ignore=metrics_ignore)

    if makeplots:
        p.summary_plot()
        plotting.likelihood_threshold_plot_2d(df=dfm, xkey="rec_phys_alpha", ykey="logmstar_a")
