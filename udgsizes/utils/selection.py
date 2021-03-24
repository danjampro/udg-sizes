GR_MAX = 0.42
GR_MIN = 0  # P21

parameter_ranges = {'uae': [24, 27],    # These are fixed
                    'rec': [3, 10]}


def select_samples(uae, rec):
    """ Apply the basic selection criteria of Prole+21. Parameters are kept fixed.
    # TODO: Include colour selection.
    """
    cond = uae > parameter_ranges['uae'][0]
    cond &= uae <= parameter_ranges['uae'][1]
    cond &= rec > parameter_ranges['rec'][0]
    cond &= rec <= parameter_ranges['rec'][1]
    return cond
