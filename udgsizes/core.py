import os
import sys
import logging
import yaml
from contextlib import suppress
from collections import abc
import astropy.units as u

from udgsizes.utils.library import load_module

# TODO: Load HSC-SSP config under hsc_ssp key

ROOTDIR_ENV_NAME = "UDGSIZES_HOME"


def get_config(config_dir=None, ignore_local=False, testing=False, parse=True, name="default"):
    """

    """
    if config_dir is None:
        try:
            config_dir = os.path.join(os.environ[ROOTDIR_ENV_NAME], "config")
        except KeyError:
            raise RuntimeError(f"Unable to determine config directory: {ROOTDIR_ENV_NAME}"
                               " environment variable not set and config_dir is None.")
    config = _load_yaml(os.path.join(config_dir, f"{name}.yml"))
    # Update the config with local version
    if not ignore_local:
        with suppress(FileNotFoundError):
            config_local = _load_yaml(os.path.join(config_dir, f"{name}_local.yml"))
            config = _update_config(config, config_local)
    # Update the config with testing values
    if testing:
        with suppress(FileNotFoundError):
            config_test = _load_yaml(os.path.join(config_dir, "testing.yml"))
            config = _update_config(config, config_test)
    # Parse config
    if parse:
        config = _parse_config(config)
    return config


def get_logger():
    """

    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _load_yaml(filename):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _update_config(d, u):
    """Recursively update nested dictionary d with u."""
    for k, v in u.items():
        if isinstance(v, abc.Mapping):
            d[k] = _update_config(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def _parse_quantities(d):
    for k, v in d.items():
        if isinstance(v, abc.Mapping):
            _parse_quantities(v)
        elif k.endswith("_quantity"):
            d[k] = u.Quantity(v)
    return d


def _parse_directories(d):
    """ Recursively parse directories, expanding environment variables. """
    for k, v in d.items():
        if isinstance(v, abc.Mapping):
            _parse_directories(v)
        else:
            d[k] = os.path.expandvars(v)
    return d


def _parse_cosmology(config):
    cosmo_config = config["cosmology"].copy()
    cosmo_class = cosmo_config.pop("class")
    return load_module(cosmo_class)(**cosmo_config)


def _parse_config(config):
    config["cosmology"] = _parse_cosmology(config)
    config["directories"] = _parse_directories(config["directories"])
    _parse_quantities(config)
    return config
