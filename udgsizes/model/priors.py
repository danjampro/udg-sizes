from scipy import stats


def gaussian(values, mu, sigma):
    """
    """
    return stats.norm(loc=mu, scale=sigma).pdf(values)
