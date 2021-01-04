""" Make the pickled recovery efficiency object. We need to avoid module dependency on the
original code used by Prole+20, so load the pickled object here (only works if the old code is
installed) and then re-pickle without the dependency. This script should only be run once."""
import os
import dill as pickle

from udgsizes.core import get_config


def repickle(filename_in, filename_out):
    """
    """
    with open(filename_in, "rb") as f:
        obj = pickle.load(f)
    with open(filename_out, "wb") as f:
        pickle.dump(obj, f)


if __name__ == "__main__":

    config = get_config()
    filename = os.path.join(config["directories"]["data"], "input", "injections", "recoveff.pkl")

    filename_in = filename
    filename_out = filename_in
    repickle(filename_in, filename_out)
