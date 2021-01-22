from udgsizes.core import get_config
from udgsizes.utils.library import load_module


def create_model(model_name, config=None, *args, **kwargs):
    """
    """
    if config is None:
        config = get_config()
    model_class_name = config["models"][model_name]["type"]
    return load_module(model_class_name)(model_name=model_name, *args, **kwargs)
