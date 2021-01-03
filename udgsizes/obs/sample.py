import os
import pandas as pd

from udgsizes.core import get_config, get_logger
from udgsizes.utils.selection import select_samples


def load_sample(config=None, logger=None, select=False):
    """
    """
    if config is None:
        config = get_config()
    if logger is None:
        logger = get_logger()
    filename = os.path.join(config['directories']['data'], 'input', 'lsbgs_public.csv')
    df = pd.read_csv(filename)

    if select:
        cond = select_samples(uae=df['mueff_av'].values, rec=df['rec_arcsec'].values)
        df = df[cond].reset_index(drop=True)

    logger.debug(f"Loaded {df.shape[0]} LSBGs from file.")
    return df


def load_injections(config=None, logger=None):
    """
    """
    if config is None:
        config = get_config()
    if logger is None:
        logger = get_logger()
    path = os.path.join(config['directories']['data'], 'input', 'injections')
    filename = os.path.join(path, config['injections']['catalogue_filename'])
    df = pd.read_csv(filename)
    logger.debug(f"Loaded {df.shape[0]} injections from file.")
    return df
