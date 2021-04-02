import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def scale_to_gaussian(df, key, offset=1E-32, factor=1, xmin=None, xmax=None, makeplots=False,
                      lmbda=None, **kwargs):
    """
    """
    values = factor * df[key].values.copy()

    if xmin is None:
        xmin = values.min()
    if xmax is None:
        xmax = values.max()

    values = (values - xmin) / (xmax - xmin) + offset

    result = stats.boxcox(values, lmbda=lmbda, **kwargs)
    if lmbda is None:
        result, lam = result
    else:
        lam = lmbda

    if makeplots:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.hist(df[key])
        plt.subplot(1, 2, 2)
        plt.hist(result)
        plt.tight_layout()
        plt.show(block=False)

    return result, lam, xmin, xmax


class TransformedGaussianPDF():

    keys = ("uae_obs_jig", "rec_obs_jig", "colour_obs")

    keys_obs = {"uae_obs_jig": "mueff_av",
                "rec_obs_jig": "rec_arcsec",
                "colour_obs": "g_r"}

    factors = {"colour_obs": -1}

    def __init__(self, df, **kwargs):
        """
        """
        self.values = {}
        self.lambdas = {}
        self.mins = {}
        self.maxs = {}

        for k in self.keys:
            factor = self.factors.get(k, 1)
            self.values[k], self.lambdas[k], self.mins[k], self.maxs[k] = scale_to_gaussian(
                df, key=k, factor=factor, **kwargs)

        points = np.vstack([self.values[k] for k in self.keys])

        self.means = points.mean(axis=1)
        self.cov = np.cov(points)
        self.var = stats.multivariate_normal(mean=self.means, cov=self.cov)

    def evaluate(self, df, **kwargs):
        """
        """
        values = {}
        for k in self.keys:
            ko = self.keys_obs[k]
            factor = self.factors.get(k, 1)
            values[k], _, _, _ = scale_to_gaussian(df, key=ko, factor=factor, xmin=self.mins[k],
                                                   xmax=self.maxs[k], lmbda=self.lambdas[k],
                                                   **kwargs)
        points = np.vstack([values[k] for k in self.keys])
        return self.var.pdf(points.T)

    def summary_plot(self):
        """
        """
        plt.figure(figsize=(4 * len(self.keys), 4))

        for i, key in enumerate(self.keys):
            ax = plt.subplot(1, len(self.keys), i + 1)

            values = self.values[key]
            ax.hist(values, density=True)

            xx = np.linspace(values.min(), values.max(), 100)
            pdf = stats.norm(loc=self.means[i], scale=np.sqrt(self.cov[i][i])).pdf
            ax.plot(xx, pdf(xx), "r--")

            ax.set_title(key)

        plt.tight_layout()
        plt.show(block=False)


class RescaledKde3D():

    keys = ("uae_obs_jig", "rec_obs_jig", "colour_obs")

    keys_obs = {"uae_obs_jig": "mueff_av",
                "rec_obs_jig": "rec_arcsec",
                "colour_obs": "g_r"}

    factors = {"colour_obs": -1}

    def __init__(self, df, **kwargs):
        """
        """
        self.values = {}
        self.lambdas = {}
        self.mins = {}
        self.maxs = {}

        for k in self.keys:
            factor = self.factors.get(k, 1)
            self.values[k], self.lambdas[k], self.mins[k], self.maxs[k] = scale_to_gaussian(
                df, key=k, factor=factor, **kwargs)

        points = np.vstack([self.values[k] for k in self.keys])

        self.kde = stats.gaussian_kde(points)

    def evaluate(self, df, **kwargs):
        """
        """
        values = {}
        for k in self.keys:
            ko = self.keys_obs[k]

            factor = self.factors.get(k, 1)
            values[k], _, _, _ = scale_to_gaussian(df, key=ko, factor=factor, xmin=self.mins[k],
                                                   xmax=self.maxs[k], lmbda=self.lambdas[k],
                                                   **kwargs)
        points = np.vstack([values[k] for k in self.keys])

        return self.kde(points)

    def summary_plot(self):
        """ # Doesn't work...
        """
        from mayavi import mlab

        # Create a regular 3D grid with 50 points in each dimension
        xmin, ymin, zmin = self.mins.values()
        xmax, ymax, zmax = self.maxs.values()
        xi, yi, zi = np.mgrid[xmin:xmax:20j, ymin:ymax:20j, zmin:zmax:20j]

        # Evaluate the KDE on a regular grid...
        coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
        density = self.kde(coords).reshape(xi.shape)

        # Visualize the density estimate as isosurfaces
        mlab.contour3d(xi, yi, zi, density, opacity=0.5)
        mlab.axes()
        mlab.show()
