parameter_ranges = {'uae': [24, 27],    # These are fixed
                    'rec': [3, 10]}


def select_samples(uae, rec):
    """ Apply the basic selection criteria of Prole+20. Parameters are kept fixed.
    """
    cond = uae > parameter_ranges['uae'][0]
    cond &= uae <= parameter_ranges['uae'][1]
    cond &= rec > parameter_ranges['rec'][0]
    cond &= rec <= parameter_ranges['rec'][1]
    return cond
