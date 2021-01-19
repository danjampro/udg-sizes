"""
"""


def power(rec_phys, alpha):
    return rec_phys ** -alpha


def power_trunc(rec_phys, alpha, r_trunc):
    if rec_phys < r_trunc:
        return r_trunc ** -alpha
    return rec_phys ** -alpha
