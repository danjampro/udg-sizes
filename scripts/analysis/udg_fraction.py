import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from udgsizes.model.components.rec import power, power_trunc


def power_law(r, alpha, rzero=0):
    result = power(r, alpha=alpha)
    result[r <= rzero] = 0
    return result


def power_law_t(r, alpha, rtrunc, rzero=0):
    result = np.array([power_trunc(_, alpha=alpha, r_trunc=rtrunc) for _ in r])
    result[r <= rzero] = 0
    return result


if __name__ == "__main__":

    rtrunc = 1.0
    rudg = 1.5
    rzero = 0.0  # Kpc, based on local group dwarfs

    alphas = np.linspace(0, 7, 200)
    fudg = np.zeros_like(alphas)
    fudg_t = np.zeros_like(alphas)

    for i, alpha in enumerate(alphas):

        xx = np.linspace(0, 20, 1000)
        yy = power_law(xx, alpha=alpha, rzero=rzero)
        total = simps(yy, xx)
        xx = np.linspace(rudg, 20, 1000)
        yy = power_law(xx, alpha=alpha, rzero=rzero)
        total_udg = simps(yy, xx)
        fudg[i] = total_udg/total

        xx = np.linspace(0, 20, 1000)
        yy = power_law_t(xx, alpha, rtrunc, rzero=rzero)
        total = simps(yy, xx)
        xx = np.linspace(rudg, 20, 1000)
        yy = power_law_t(xx, alpha, rtrunc, rzero=rzero)
        total_udg = simps(yy, xx)
        fudg_t[i] = total_udg/total

    plt.figure()
    plt.plot(alphas, fudg, 'k-')
    plt.plot(alphas, fudg_t, 'b-')

    rzero = 0.1  # Kpc, based on local group dwarfs
    for i, alpha in enumerate(alphas):

        xx = np.linspace(0, 20, 1000)
        yy = power_law(xx, alpha=alpha, rzero=rzero)
        total = simps(yy, xx)
        xx = np.linspace(rudg, 20, 1000)
        yy = power_law(xx, alpha=alpha, rzero=rzero)
        total_udg = simps(yy, xx)
        fudg[i] = total_udg/total

        xx = np.linspace(0, 20, 1000)
        yy = power_law_t(xx, alpha, rtrunc, rzero=rzero)
        total = simps(yy, xx)
        xx = np.linspace(rudg, 20, 1000)
        yy = power_law_t(xx, alpha, rtrunc, rzero=rzero)
        total_udg = simps(yy, xx)
        fudg_t[i] = total_udg/total

    plt.plot(alphas, fudg, 'k--')
    plt.plot(alphas, fudg_t, 'b--')

    plt.show(block=False)
