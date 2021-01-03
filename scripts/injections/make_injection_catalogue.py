""" For this project, we need a catalogue of injections that were selected according to the
criteria of Prole+20 (i.e. the same criteria used to create the observed catalogue). We start from
a catalogue of raw detected injections and apply the selection criteria. The selection parameters
are fixed and this script should only be run once to produce injections/selected.csv. """
import os
import pandas as pd

from redudgs.hsc.selection import select
from udgsizes.core import get_config


def select_injections(df):
    """ The settings are fixed from Prole+20.
    """
    cond = select(df, ml=True, colour=False, unique=False, starmask=False, presel=True)
    return cond


if __name__ == "__main__":

    config = get_config()
    injectdir = os.path.join(config["directories"]["data"], "input", "injections")

    # Load detected injection catalogue
    input_filename = os.path.join(injectdir, "detected.csv")
    dfi = pd.read_csv(input_filename)

    # Apply selection
    cond = select_injections(dfi)
    dfo = dfi[cond].reset_index(drop=True)

    # Save result
    output_filename = os.path.join(injectdir, "selected.csv")
    dfo.to_csv(output_filename)
