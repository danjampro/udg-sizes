import os
import dill as pickle

from udgsizes.core import get_config, get_logger


def load_recovery_efficiency(config=None, logger=None):
    """
    """
    if config is None:
        config = get_config()
    if logger is None:
        logger = get_logger()
    dirname = os.path.join(config["directories"]["data"], "input", "injections")
    filename = config["injections"]["receff_filename"]
    logger.debug(f"Loading interpolated recover efficiency: {filename}.")
    with open(os.path.join(dirname, filename), "rb") as f:
        interp = pickle.load(f)
    return interp
