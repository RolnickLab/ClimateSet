from typing import Union, List
from omegaconf import DictConfig


def get_name_for_hydra_config_class(config: DictConfig) -> str:
    if "name" in config and config.get("name") is not None:
        return config.get("name")
    elif "_target_" in config:
        return config._target_.split(".")[-1]
    return "$"


def get_detailed_name(config) -> str:  # TODO: adjust that to our liking
    """This is a prefix for naming the runs for a more agreeable logging."""
    s = config.get("name", "")

    s += get_name_for_hydra_config_class(config.model) + "_"

    if config.model.get("dropout") is not None:
        if config.model.get("dropout") > 0:
            s += f"{config.model.get('dropout')}dout_"

    if config.model.get("activation_fucniton") is not None:
        s += config.model.get("activation_function") + "_"
    s += get_name_for_hydra_config_class(config.model.optimizer) + "_"
    # s += get_name_for_hydra_config_class(config.model.scheduler) + '_'

    s += f"{config.datamodule.get('batch_size')}bs_"
    s += f"{config.model.optimizer.get('lr')}lr_"
    if config.model.optimizer.get("weight_decay") > 0:
        s += f"{config.model.optimizer.get('weight_decay')}wd_"
    s += f"{config.get('seed')}seed"

    return s.replace("None", "")


def get_group_name(config) -> str:
    s = get_name_for_hydra_config_class(config.model)
    s = (
        s.lower()
        .replace("net", "")
        .replace("_", "")
        .replace("causalpaca", "")
        .replace("with", "+")
        .upper()
    )

    # if config.normalizer.get('spatial_normalization_in') and config.normalizer.get('spatial_normalization_out'):
    #    s += '+spatialNormed'
    # elif config.normalizer.get('spatial_normalization_in'):
    #    s += '+spatialInNormed'
    # elif config.normalizer.get('spatial_normalization_out'):
    #    s += '+spatialOutNormed'

    return s
