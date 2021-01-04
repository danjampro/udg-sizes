""" Code identical to that used in Prole+20. """
import os
import dill as pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans

from udgsizes.core import get_config


class Classifier():

    def __init__(self, makeplots=False, save=False, **kwargs):
        self.vars = {}

    def normalise(self, indices, colours, store=False):
        '''

        '''
        data = np.vstack([indices, colours]).T.copy()

        if store:
            self.mu1 = np.median(data[:, 0])
            self.sig1 = data[:, 0].std()
            self.mu2 = np.median(data[:, 1])
            self.sig2 = data[:, 1].std()

        data[:, 0] = (data[:, 0]-self.mu1) / self.sig1
        data[:, 1] = (data[:, 1]-self.mu2) / self.sig2

        return data

    def get_init(self):
        '''
        Get initial guess for cluster centres.
        '''
        return np.random.normal(0, 0.1, (2, 2))

    def fit(self, indices, colours, filename=None, init='k-means++', makeplots=False):
        '''

        '''
        # Get data & normalise
        cond = np.isfinite(indices*colours)
        data = self.normalise(indices[cond], colours[cond], store=True)

        # Get initial values
        if init is None:
            init = self.get_init()

        # Fit
        self.kmeans = KMeans(n_clusters=2, init=init, max_iter=1000, random_state=0).fit(data)

        # Store indices
        self.idx_red = np.argmax(self.kmeans.cluster_centers_[:, 1])
        self.idx_blue = 0 if self.idx_red == 1 else 1

        # Fit normal distributions to clusters
        self._fit_norms(indices, colours)

        if filename is not None:
            self._save(filename)

        if makeplots:
            indices_rand, colours_rand = self.vars["blue"].rvs(500).T
            _summary_plot(indices, colours, red=self.predict(indices, colours, which="red"),
                          indices_rand=indices_rand, colours_rand=colours_rand)

    def predict(self, indices, colours, which='blue', keep_nans=False):
        '''

        '''
        if which == 'red':
            idx = self.idx_red
        elif which == 'blue':
            idx = self.idx_blue

        if keep_nans:
            result = np.ones(indices.size, dtype='bool')
        else:
            result = np.zeros(indices.size, dtype='bool')

        # Normalise data and remove non-finite values
        data = self.normalise(indices, colours)
        cond = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])

        # Do the predictions, ignoring nans
        result[cond] = self.kmeans.predict(data[cond]) == idx

        return result

    def _fit_norms(self, indices, colours, nits=3, rejfrac=0.1):
        """
        """
        for which in ("red", "blue"):
            cond = self.predict(indices, colours, which=which)
            x = indices[cond]
            y = colours[cond]

            means = x.mean(), y.mean()
            cov = np.cov(x, y)
            var = stats.multivariate_normal(mean=means, cov=cov)
            cond_keep = np.ones_like(x, dtype="bool")
            for i in range(nits):

                p = var.pdf(np.vstack([x, y]).T)
                pmin = np.quantile(p, rejfrac)
                cond_keep &= p > pmin

                means = x[cond_keep].mean(), y[cond_keep].mean()
                cov = np.cov(x[cond_keep], y[cond_keep])
                var = stats.multivariate_normal(mean=means, cov=cov)

            self.vars[which] = var

    def _save(self, filename):
        '''

        '''
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


def load_classifier(config=None):
    fname = get_classifier_filename(config=config)
    with open(fname, 'rb') as f:
        C = pickle.load(f)
    return C


def get_classifier_filename(config=None):
    if config is None:
        config = get_config()
    return os.path.join(config["directories"]["data"], "index_colour.pkl")


def _summary_plot(indices, colours, red, indices_rand, colours_rand):

    histkwargs = {'histtype': 'step', 'bins': 15}
    fontsize = 16
    cb = 'royalblue'
    cr = 'crimson'
    xlabel = 'Sersic index'
    ylabel = 'Colour'

    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(nrows=40, ncols=2, width_ratios=[3, 4])

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.plot(indices[red], colours[red], 'o', markersize=1.5, color=cr)
    ax0.plot(indices[~red], colours[~red], 'o', markersize=1.5, color=cb)
    ax0.set_xlabel(xlabel, fontsize=fontsize)
    ax0.set_ylabel(ylabel, fontsize=fontsize)

    ax0.plot(indices_rand, colours_rand, 'k+', markersize=1, alpha=0.5)

    ax1 = fig.add_subplot(gs[0:17, 1])
    ax1.hist(colours, range=ax0.get_ylim(), color='k', **histkwargs)
    ax1.hist(colours[red], range=ax0.get_ylim(), color=cr, **histkwargs)
    ax1.hist(colours[~red], range=ax0.get_ylim(), color=cb, **histkwargs)
    ax1.set_xlabel(ylabel, fontsize=fontsize)
    ax1.set_ylabel('Number', fontsize=fontsize)

    ax2 = fig.add_subplot(gs[23:, 1])
    ax2.hist(indices, range=ax0.get_xlim(), color='k', **histkwargs)
    ax2.hist(indices[red], range=ax0.get_xlim(), color=cr, **histkwargs)
    ax2.hist(indices[~red], range=ax0.get_xlim(), color=cb, **histkwargs)
    ax2.set_xlabel(xlabel, fontsize=fontsize)
    ax2.set_ylabel('Number', fontsize=fontsize)
