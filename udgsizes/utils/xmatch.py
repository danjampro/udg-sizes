import numpy as np
from scipy.spatial import cKDTree as KDTree


def match(x1, y1, x2, y2, radius):
    """
    """
    # Build the trees
    p1 = np.array([*zip(x1, y1)])
    t1 = KDTree(p1)
    p2 = np.array([*zip(x2, y2)])
    t2 = KDTree(p2)

    # Do the matching
    m = t1.query_ball_tree(t2, radius)

    return m


def match_dataframe(df1, df2, k1='ra', k2='dec', radius=5./3600):
    """ First matching
    """
    x1, y1 = df1[k1].values, df1[k2].values
    x2, y2 = df2[k1].values, df2[k2].values

    matches = match(x1, y1, x2, y2, radius=radius)

    df1_match = df1.iloc[[_ for _, m in enumerate(matches) if len(m) > 0]].reset_index()

    idxs = [m[0] for m in matches if len(m) > 0]
    df2_match = df2.iloc[idxs].copy().reset_index()

    for key in df2_match.columns:
        df1_match[key] = df2_match[key].values

    return df1_match
