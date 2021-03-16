import os
import numpy as np
import pandas as pd

from udgsizes.core import get_config, get_logger
from udgsizes.utils.selection import select_samples


def load_sample(config=None, logger=None, select=True):
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
        cond &= df["is_red"].values == 0
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


def load_gama_specobj(config=None, logger=None):
    """
    """
    if config is None:
        config = get_config()
    if logger is None:
        logger = get_logger()
    filename = os.path.join(config['directories']['data'], 'input', 'gama_specobj.csv')
    df = pd.read_csv(filename)
    df["ra"] = df["RA"]
    df["dec"] = df["DEC"]
    logger.debug(f"Loaded {df.shape[0]} GAMA objects.")
    return df


def load_gama_masses(config, logmstar_min=6, logmstar_max=13, z_max=0.1, gi_max=None,
                     ur_max=None, gr_max=None, lambdar=True, n_max=None):
    """
    http://www.gama-survey.org/dr3/schema/dmu.php?id=9
    """

    # Load catalogue from file
    lstr = "_lambdar" if lambdar else ""
    filename = os.path.join(config['directories']['data'], 'input', f'gama_masses{lstr}.csv')
    dfg = pd.read_csv(filename)

    fluxscale = dfg["fluxscale"].values
    h = config["cosmology"].h

    # Translate columns
    df = pd.DataFrame()

    # log10 M*,total = logmstar + log10(fluxscale) - 2 log10(h/0.7)
    df["logmstar"] = dfg["logmstar"] + np.log10(fluxscale) - 2 * np.log10(h / 0.7)

    # M_X,total = absmag_X - 2.5 log10(fluxscale) + 5 log10(h/0.7)
    df["absmag_r"] = dfg["absmag_r"] - 2.5 * np.log10(fluxscale) + 5 * np.log10(h / 0.7)

    df["gi"] = dfg["gminusi"].values
    df["ur"] = dfg["uminusr"].values
    df["gr"] = dfg["gminusi"] + dfg["absmag_i"] - dfg["absmag_r"]
    df["redshift"] = dfg["Z"]
    df["logmstar_absmag_r"] = df["logmstar"] / df["absmag_r"]

    df["n"] = dfg["GALINDEX_r"]

    df["kcorr_g"] = dfg["KCORR_G"]
    df["kcorr_r"] = dfg["KCORR_R"]
    df["kcorr_i"] = dfg["KCORR_I"]

    # Apply selections
    logmstar = df["logmstar"].values
    cond = (logmstar >= logmstar_min) & (logmstar < logmstar_max)
    cond &= (df["redshift"].values < z_max)
    if gi_max is not None:
        cond &= (df["gi"].values < gi_max)
    if ur_max is not None:
        cond &= (df["ur"].values < ur_max)
    if gr_max is not None:
        cond &= (df["gr"].values < gr_max)
    if n_max is not None:
        cond &= (df["n"].values < n_max)
    df = df[cond].reset_index(drop=True)

    return df
