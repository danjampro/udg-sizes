from udgsizes.utils import shen
from udgsizes.utils.stats.likelihood import unnormalised_gaussian_pdf


def power(rec_phys, alpha, min=0):
    if rec_phys < min:
        return 0
    return rec_phys ** -alpha


def power_trunc(rec_phys, alpha, r_trunc, min=0):
    if rec_phys < min:
        return 0
    if rec_phys < r_trunc:
        return r_trunc ** -alpha
    return rec_phys ** -alpha


def gaussian_offset_shen(offset, logmstar):
    mstar = 10 ** logmstar
    norm = 1 + (mstar / shen.M0) ** 2
    sigma = shen.SIGMA_2 + (shen.SIGMA_1 - shen.SIGMA_2) / norm
    return unnormalised_gaussian_pdf(offset, sigma)
