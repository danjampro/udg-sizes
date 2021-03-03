import matplotlib.pyplot as plt
import powerlaw

from udgsizes.model.utils import create_model
from udgsizes.fitting.grid import ParameterGrid

REC_PHYS_MIN = 1.5
MODEL_NAME = "blue_baldry_shen_final"
UDG_MODEL_NAME = "blue_baldry_shen_udg"


if __name__ == "__main__":

    n_samples = 10000
    burnin = 1000

    # Get best fitting hyper parameters
    hyper_params = ParameterGrid(MODEL_NAME).get_best_hyper_parameters()

    # Sample the model with no recovery efficiency
    model = create_model(UDG_MODEL_NAME, ignore_recov=True)
    df = model.sample(burnin=burnin, n_samples=n_samples, hyper_params=hyper_params)

    # Fit power law for physical size distribution of UDGs
    rec_phys = df["rec_phys"].values[df["is_udg"].values == 1]
    fit_result = powerlaw.Fit(rec_phys, xmin=REC_PHYS_MIN)

    fig, ax = plt.subplots()
    ax.hist(rec_phys, density=True)
    fit_result.power_law.plot_pdf(color='b', linestyle='--', ax=plt.gca())
    plt.show(block=False)

    print(f"Power law slope: -{fit_result.power_law.alpha:.2f}")
