""" How significant is the difference in alpha value between our results and vdB+16? """

import numpy as np

if __name__ == "__main__":

    a1 = 3.749
    da1 = 0.1837  # TODO, calculate from model

    a2 = 4.4
    da2 = 0.19
    n_samples = 1000000

    x1 = np.random.normal(a1, da1, n_samples)
    x2 = np.random.normal(a2, da2, n_samples)

    cond = x2 > x1
    frac = cond.mean()

    print(f"Fraction of x2>x1: {frac}")  # 99.3%!
