import numpy as np
from scipy.stats import binned_statistic_2d

from udgsizes.base import UdgSizesBase
from udgsizes.obs.sample import load_injections
from udgsizes.utils.selection import select_samples


class Jiggler(UdgSizesBase):
    """ Bin selected injections in 2D (uae_true, rec_true) and assign correlated errors to model
    samples by matching them with a random injection in their nearest bin. The relative errors
    in uae and rec are applied to make the jiggled values.
    """
    _keys = "uae", "rec"

    def __init__(self, n_bins=6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_bins = n_bins

        # Load the injections catalogue
        self._injections = load_injections(config=self.config, logger=self.logger)
        self._range = [[self._injections[k].min(), self._injections[k].max()
                        ] for k in ('uae_true', 'rec_true')]

        # Bin injections in 2D
        self._injection_indices = self._digitise(uae=self._injections['uae_true'].values,
                                                 rec=self._injections['rec_true'].values)

    def jiggle(self, uae, rec, maxits=30):
        """
        """
        cond = np.zeros_like(uae, dtype="bool")
        uae_jig = np.empty(uae.shape, dtype=uae.dtype)
        rec_jig = np.empty(rec.shape, dtype=rec.dtype)

        self.logger.debug(f"Jiggling {uae.size} uae/rec pairs.")
        for i in range(maxits):
            uae_jig[~cond], rec_jig[~cond] = self._jiggle(uae[~cond], rec[~cond])
            cond[~cond] = select_samples(uae=uae_jig[~cond], rec=rec_jig[~cond])
            if cond.all():
                break
            if i == maxits - 1:
                self.logger.warning(f"{cond.size-cond.sum()}/{cond.size} samples do not meet"
                                    " selection criteria.")
        self.logger.debug(f"Finished jiggling after {i+1}/{maxits} iterations.")
        return uae_jig, rec_jig

    def _jiggle(self, uae, rec):
        """
        """
        bin_indices = self._digitise(uae, rec)

        # Match samples with random injections in same bin
        match_indices = np.zeros_like(uae, dtype="int")
        arange = np.arange(0, self._injections.shape[0])
        for bin_index in np.unique(bin_indices):

            in_bin = bin_indices == bin_index
            inj_in_bin = self._injection_indices == bin_index

            n_in_bin = in_bin.sum()
            n_inj_in_bin = inj_in_bin.sum()

            _match_indices = np.random.randint(0, n_inj_in_bin, n_in_bin)
            match_indices[in_bin] = arange[inj_in_bin][_match_indices]

        # Calculate relative errors
        rel_err_uae = (self._injections["uae_fit"] / self._injections["uae_true"]).values
        rel_err_rec = (self._injections["rec_fit"] / self._injections["rec_true"]).values

        # Apply relative errors
        uae_jiggle = rel_err_uae[match_indices] * uae
        rec_jiggle = rel_err_rec[match_indices] * rec

        return uae_jiggle, rec_jiggle

    def _digitise(self, uae, rec):
        """
        """
        uae = uae.copy()
        rec = rec.copy()

        # Truncate at range boundaries
        uae[uae > self._injections['uae_fit'].max()] = self._injections['uae_fit'].max()
        uae[uae < self._injections['uae_fit'].min()] = self._injections['uae_fit'].min()
        rec[rec > self._injections['rec_fit'].max()] = self._injections['rec_fit'].max()
        rec[rec < self._injections['rec_fit'].min()] = self._injections['rec_fit'].min()

        # Get bin indices
        _, _, _, bin_indices = binned_statistic_2d(uae, rec, values=rec, range=self._range,
                                                   bins=self._n_bins)
        return bin_indices
