from udgsizes.fitting.grid import ParameterGrid
from udgsizes.utils.stats.kde import RescaledKde3D

if __name__ == "__main__":

    grid = ParameterGrid("blue_sedgwick_shen")
    df = grid.load_best_sample()

    kde = RescaledKde3D(df, makeplots=True)

    kde.summary_plot()
