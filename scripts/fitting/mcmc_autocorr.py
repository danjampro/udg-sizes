from functools import partial

from udgsizes.model.utils import create_model

if __name__ == "__main__":

    model_name = "blue_sedgwick_shen"
    hyper_params = {"rec_phys_offset": {"alpha": 0.45}, "logmstar": {"a": -1.45}}
    n_samples = 10000
    burnin = 500

    model = create_model(model_name)

    initial_state = model._get_initial_state(hyper_params=hyper_params)

    log_likelihood = partial(model._log_likelihood, hyper_params=hyper_params)

    df, sampler = model._sampler.sample(func=log_likelihood, n_samples=n_samples, burnin=burnin,
                                        initial_state=initial_state, get_sampler=True)

    tau = sampler.get_autocorr_time()
