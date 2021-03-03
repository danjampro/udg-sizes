import matplotlib.pyplot as plt
import powerlaw

from udgsizes.model.utils import create_model

REC_PHYS_MIN = 1.5


if __name__ == "__main__":

    n_samples = 2000
    burnin = 500

    model_name = "blue_baldry_shen_udg"
    hyper_params = {"rec_phys_offset": {"alpha": 0.3}, "logmstar": {"a": -1.45}}

    model = create_model(model_name, ignore_recov=True)

    df = model.sample(burnin=burnin, n_samples=n_samples, hyper_params=hyper_params)

    rec_phys = df["rec_phys"].values[df["is_udg"].values == 1]

    fit_result = powerlaw.Fit(rec_phys, xmin=REC_PHYS_MIN)


    fig, ax = plt.subplots()
    ax.hist(rec_phys, density=True)
    fit_result.power_law.plot_pdf(color='b', linestyle='--', ax=plt.gca())

    print(f"Power law slope: -{fit_result.power_law.alpha:.2f}")
