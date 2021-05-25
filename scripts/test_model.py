import matplotlib.pyplot as plt

from udgsizes.model.utils import create_model
from udgsizes.fitting.utils.plotting import fit_summary_plot

if __name__ == "__main__":

    n_samples = 500
    burnin = 250
    ignore_recov = False

    model_name = "blue_sedgwick_shen_highkink"

    model = create_model(model_name, ignore_recov=ignore_recov)
    hyper_params = {"rec_phys_offset": {"alpha": 0.4}, "logmstar": {"a": -1.45}}
    df = model.sample(burnin=burnin, n_samples=n_samples, hyper_params=hyper_params)

    if not ignore_recov:
        cond = df["selected_jig"] == 1
        df = df[cond].reset_index(drop=True)
        fit_summary_plot(df=df)

    plt.show(block=False)
