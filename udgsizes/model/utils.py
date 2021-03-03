from contextlib import suppress
from udgsizes.core import get_config, update_config
from udgsizes.utils.library import load_module


def get_model_config(model_name, config):
    """
    """
    if model_name.endswith("_final"):
        model_name = model_name[:-6]

    model_config = config["models"][model_name].copy()

    with suppress(KeyError):
        model_name_base = model_config["base"]
        model_config_base = get_model_config(model_name_base, config=config)
        model_config = update_config(model_config_base, model_config)

    return model_config


def create_model(model_name, config=None, *args, **kwargs):
    """
    """
    if config is None:
        config = get_config()
    model_class_name = get_model_config(model_name, config=config)["type"]
    return load_module(model_class_name)(model_name=model_name, config=config, *args, **kwargs)
