""" Calulcate the observed dwarf / UDG fraction with uncertainties. """
import numpy as np

from udgsizes.fitting.grid import ParameterGrid

if __name__ == "__main__":

    model_name = "blue_sedgwick_shen_final"
    metric_name = "posterior_kde_3d"

    grid = ParameterGrid(model_name)
    df = grid.load_confident_metrics(metric=metric_name, q=0.9)

    weights = df[metric_name].values

    dwarf_fracs = df["n_dwarf"].values / df["n_selected"].values
    dwarf_frac_av = np.average(dwarf_fracs, weights=weights)
    dwarf_frac_std = np.sqrt(np.average((dwarf_fracs - dwarf_frac_av) ** 2, weights=weights))

    udg_fracs = df["n_udg"].values / df["n_selected"].values
    udg_frac_av = np.average(udg_fracs, weights=weights)
    udg_frac_std = np.sqrt(np.average((udg_fracs - udg_frac_av) ** 2, weights=weights))

    print(f"Dwarf fraction: {dwarf_frac_av:.2f} ± {dwarf_frac_std:.2f}")
    print(f"UDG fraction: {udg_frac_av:.2f} ± {udg_frac_std:.2f}")
