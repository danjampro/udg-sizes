from udgsizes.fitting.grid import ParameterGrid

if __name__ == "__main__":

    pop_name = "blue"
    makeplots = True

    p = ParameterGrid(pop_name)
    p.sample(overwrite=True)
    p.evaluate()

    if makeplots:
        p.summary_plot()
