from udgsizes.model.sm_size import Model as SmSizeModel
from udgsizes.model.colour import EmpiricalColourModel


class Model(SmSizeModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._colour_model = None

    def sample(self, n_samples, hyper_params, filename=None, **kwargs):
        """
        """
        # Override the colour model using model hyper params
        self._colour_model = EmpiricalColourModel(config=self.config, logger=self.logger,
                                                  **hyper_params["colour_model"])
        # Sample using base class method
        return super().sample(n_samples=n_samples, hyper_params=hyper_params, filename=filename,
                              **kwargs)
