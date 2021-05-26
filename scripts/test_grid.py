import matplotlib.pyplot as plt

from udgsizes.core import get_config
from udgsizes.fitting.grid import ParameterGrid

MAKEPLOTS = True
CHECK_INITIAL_VALUES = True
SAMPLE = False

if __name__ == "__main__":

    model_name = "blue_sedgwick_shen_highkink"
    model_type = "udgsizes.model.sm_size.Model"

    config = get_config()
    config["grid"][model_type]["parameters"]["rec_phys_offset"]["alpha"]["max"] = 0.6
    config["grid"][model_type]["parameters"]["rec_phys_offset"]["alpha"]["step"] = 0.05
    config["grid"][model_type]["parameters"]["logmstar"]["a"]["min"] = -1.50
    config["grid"][model_type]["parameters"]["logmstar"]["a"]["max"] = -1.45
    config["grid"][model_type]["parameters"]["logmstar"]["a"]["step"] = 0.05

    metrics_ignore = ["kstest_2d"]  # Takes too long for whole grid
    n_samples = 500
    burnin = 250

    grid = ParameterGrid(model_name, config=config)

    if CHECK_INITIAL_VALUES:
        grid.check_initial_values()

    if SAMPLE:
        grid.sample(overwrite=True, n_samples=n_samples, burnin=burnin)

    grid.evaluate(metrics_ignore=metrics_ignore)
    dfm = grid.load_metrics()

    if MAKEPLOTS:
        grid.summary_plot()
        plt.show(block=False)
