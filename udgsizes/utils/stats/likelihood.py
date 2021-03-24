from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import simps


def unlog_likelihood(log_values):
    """
    """
    return np.exp(log_values - np.nanmax(log_values))


def unnormalised_gaussian_pdf(value, sigma):
    """
    """
    return np.exp(-0.5 * (value / sigma)**2)


def gaussian(x, a, mu, sigma):
    """
    """
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma**2))


def fit_colour_gaussian(colour, mu0=0.4, sigma0=0.1, bins=20, makeplots=False):
    """
    """
    hist, edges = np.histogram(colour, bins=bins)
    centres = 0.5 * (edges[1:] + edges[:-1])

    p0 = [hist.max(), mu0, sigma0]
    popt, pcov = curve_fit(gaussian, xdata=centres, ydata=hist, p0=p0)

    xx = np.linspace(colour.min(), colour.max(), 100)
    yy = gaussian(xx, *popt)
    norm = simps(yy, xx)

    func = partial(gaussian, a=popt[0]/norm, mu=popt[1], sigma=popt[2])

    if makeplots:
        fig, ax = plt.subplots()
        ax.hist(colour, bins=bins, density=True)
        ax.plot(xx, func(xx))
        plt.show(block=False)

    return func
