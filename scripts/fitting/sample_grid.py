from udgsizes.fitting.grid import ParameterGrid

if __name__ == "__main__":

    pop_name = "blue_trunc"  # TODO: Move to command line arg
    makeplots = True
    metrics_ignore = ["kstest_2d"]  # Takes too long for whole grid

    p = ParameterGrid(pop_name)
    p.sample(overwrite=True)
    p.evaluate(metrics_ignore=metrics_ignore)

    if makeplots:
        p.summary_plot()
