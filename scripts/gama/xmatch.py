import os

from udgsizes.core import get_config, get_logger
from udgsizes.obs.sample import load_sample, load_gama_specobj
from udgsizes.utils import xmatch

if __name__ == "__main__":

    logger = get_logger()
    radius = 3. / 3600

    df = load_sample(select=False)
    dfg = load_gama_specobj()

    dfm = xmatch.match_dataframe(df, dfg, radius=radius)
    logger.info(f"Matched {dfm.shape[0]} sources.")

    datadir = get_config()["directories"]["data"]
    dfm.to_csv(os.path.join(datadir, "input", "lsbgs_gama_xmatch.csv"))
