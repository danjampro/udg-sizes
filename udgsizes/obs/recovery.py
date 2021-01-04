import os
import dill as pickle
from functools import partial
import numpy as np

from udgsizes.core import get_config, get_logger


def recovery_efficiency(uae, rec, interp):
    """
    """
    result = interp(uae, rec)
    if isinstance(result, np.ndarray):
        result[result < 0] = 0
        result[result > 1] = 1
        result[~np.isfinite(result)] = 0
    else:
        result = min(max(result, 0), 1)
        if not np.isfinite(result):
            result = 0
    return result


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

    fn = partial(recovery_efficiency, interp=interp)

    return fn
