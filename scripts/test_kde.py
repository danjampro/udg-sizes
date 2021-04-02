from udgsizes.fitting.grid import ParameterGrid
from udgsizes.utils.stats.kde import TransformedGaussianPDF

if __name__ == "__main__":

    grid = ParameterGrid("blue_sedgwick_shen")
    df = grid.load_best_sample()

    cond = df["selected_jig"].values == 1
    df = df[cond].reset_index(drop=True)

    pdf = TransformedGaussianPDF(df, makeplots=True)

    pdf.summary_plot()
