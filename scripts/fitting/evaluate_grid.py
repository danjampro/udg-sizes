""" Code to evaluate a grid, replacing whatever metrics have already been calculated """

from udgsizes.fitting.grid import ParameterGrid

if __name__ == "__main__":

    pop_name = "blue"
    metrics_ignore = ["_kstest_2d"]
    nproc = 4

    p = ParameterGrid(pop_name)
    p.evaluate(metrics_ignore=metrics_ignore, nproc=nproc)
