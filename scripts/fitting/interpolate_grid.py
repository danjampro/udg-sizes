""" This script is to update old models that did not save their interps. """
from multiprocessing import Pool
from udgsizes.fitting.interpgrid import InterpolatedGrid


def func(model_name):
    print(f"Interpolating {model_name}")
    p = InterpolatedGrid(model_name)
    p._interpolate(save=True)


if __name__ == "__main__":
    model_names = "blue_baldry", "blue_baldry_dwarf", "blue_baldry_trunc", "blue_baldry_highml"
    with Pool(4) as pool:
        pool.map(func, model_names)
