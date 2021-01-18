import numpy as np
import matplotlib.pyplot as plt
from udgsizes.fitting.grid import ParameterGrid
#from scipy.ndimage.filters import sobel
from scipy.ndimage.morphology import binary_dilation


def likelihood_quantile(values, q):
    """
    """
    values_sorted = values.copy().reshape(-1)
    values_sorted.sort()
    values_sorted = values_sorted[::-1]
    csum = np.cumsum(values_sorted)
    total = csum[-1]
    idx = np.argmin(abs(csum - q*total))
    print(csum.size, values.size, values_sorted.size, idx)
    print(values_sorted.shape)
    return values_sorted[idx]


if __name__ == "__main__":

    model_name = "blue_final"
    xkey = "rec_phys_alpha"
    ykey = "uae_phys_k"
    zkey = "poisson_likelihood_2d"

    grid = ParameterGrid(model_name)

    df = grid.load_metrics()
    x = df[xkey].values
    y = df[ykey].values
    z = df[zkey].values
    extent = (y.min(), y.max(), x.min(), x.max())

    nx = np.unique(x).size
    ny = np.unique(y).size

    xx = x.reshape(nx, ny)
    yy = y.reshape(nx, ny)
    zz = z.reshape(nx, ny)

    zzexp = np.exp(zz+1000)

    from scipy.ndimage.filters import gaussian_filter
    #zzexp = gaussian_filter(zzexp, 0.5)

    #zz10 = zzexp <= -np.quantile(-zzexp, 0.1)
    #zz50 = zzexp >= np.quantile(-zzexp, 0.5)
    #zz10 = zzexp  >= likelihood_quantile(zzexp, 0.1)
    zz68 = zzexp  >= likelihood_quantile(zzexp, 0.68)
    #zz68 = binary_dilation(zz68, structure=np.ones((3,3)))
    zz90 = zzexp  >= likelihood_quantile(zzexp, 0.90)
    #zz90 = binary_dilation(zz90, structure=np.ones((3,3)))
    zz95 = zzexp  >= likelihood_quantile(zzexp, 0.95)
    #zz95 = binary_dilation(zz90, structure=np.ones((3,3)))
    #ss90 = abs(sobel(zz90.astype("float")))
    #ss90[ss90 == 0] = np.nan
    #ss90[zz90] = np.nan
    #ss90[np.isfinite(ss90)] = 1
    #zz95 = zzexp  >= likelihood_quantile(zzexp, 0.95)
    levels = likelihood_quantile(zzexp, 0.95), likelihood_quantile(zzexp, 0.68)

    plt.figure()

    ax = plt.subplot()
    # plt.imshow(zz, extent=extent, cmap="viridis", origin='lower', vmin=-1.12E+3)
    ax.imshow(zz, extent=extent, cmap="binary", origin='lower', vmin=-1.15E+3)
    #plt.imshow(np.arcsinh(zzexp), extent=extent, cmap="viridis", origin='lower')
    #plt.imshow(zzexp, extent=extent, cmap="binary", origin='lower', vmax=1E-43)
    #plt.imshow(ss90, extent=extent, cmap="Reds", origin='lower')
    # plt.contour(zz10, linewidths=1, colors="b", extent=extent)
    # plt.contour(zz50, linewidths=1, colors="b", extent=extent)
    #plt.contour(zz68, linewidths=1, colors="b", extent=extent, levels=[1])
    #plt.contour(zz90, linewidths=1, colors="b", extent=extent, levels=[1])
    #plt.contour(zz95, linewidths=1, colors="b", extent=extent, levels=[1])
    ax.contour(zzexp, linewidths=0.8, colors=["w", "deepskyblue"], extent=extent, levels=levels)
    # plt.contour(zz95, linewidths=1, colors="b", extent=extent)

    xrange = 0, 1.2
    fontsize = 15

    ax.plot(xrange, [4.4, 4.4], 'b--', label="van der Burg +16 (clusters), Amorisco +16")
    ax.plot(xrange, [3.7, 3.7], 'r--', label="van der Burg +17 (groups)")

    ax.set_xlim(*xrange)
    ax.set_ylim(2.5, 5)

    ax.set_aspect(1.2/2.5)

    ax.set_xlabel("$k$", fontsize=fontsize)
    ax.set_ylabel(r"$\alpha$", fontsize=fontsize)

    ax.legend(loc="best")

    plt.show(block=False)
