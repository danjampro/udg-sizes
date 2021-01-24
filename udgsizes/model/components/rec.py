"""
"""


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
