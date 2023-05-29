
import logging
from omegaconf import DictConfig, OmegaConf

from typing import Union, Sequence, List, Dict, Optional, Callable
from pytorch_lightning.utilities import rank_zero_only
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset
import numpy as np

from emulator.src.core.losses import RMSELoss, NRMSELoss_ClimateBench, NRMSELoss_g_ClimateBench, NRMSELoss_s_ClimateBench, LLweighted_MSELoss_Climax, LLweighted_RMSELoss_Climax, LLWeighted_RMSELoss_WheatherBench

def to_DictConfig(obj: Optional[Union[List, Dict]]):
    if isinstance(obj, DictConfig):
        return obj

    if isinstance(obj, list):
        try:
            dict_config = OmegaConf.from_dotlist(obj)
        except ValueError as e:
            dict_config = OmegaConf.create(obj)

    elif isinstance(obj, dict):
        dict_config = OmegaConf.create(obj)

    else:
        dict_config = OmegaConf.create()  # empty

    return dict_config


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def get_activation_function(name: str, functional: bool = False, num: int = 1):
    name = name.lower().strip()

    def get_functional(s: str) -> Optional[Callable]:
        return {"softmax": F.softmax, "relu": F.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid,
                "identity": nn.Identity(),
                None: None, 'swish': F.silu, 'silu': F.silu, 'elu': F.elu, 'gelu': F.gelu, 'prelu': nn.PReLU(),
                }[s]

    def get_nn(s: str) -> Optional[Callable]:
        return {"softmax": nn.Softmax(dim=1), "relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(),
                "identity": nn.Identity(), 'silu': nn.SiLU(), 'elu': nn.ELU(), 'prelu': nn.PReLU(),
                'swish': nn.SiLU(), 'gelu': nn.GELU(),
                }[s]

    if num == 1:
        return get_functional(name) if functional else get_nn(name)
    else:
        return [get_nn(name) for _ in range(num)]


def get_loss_function(name, reduction='mean'): #TODO: include further paremeters
    name = name.lower().strip().replace('-', '_')
    if name in ['l1', 'mae', "mean_absolute_error"]:
        loss = nn.L1Loss(reduction=reduction)
    elif name in ['l2', 'mse', "mean_squared_error"]:
        # TODO: clarify with time dimension
        loss = nn.MSELoss(reduction=reduction)
    elif name in ['rmse', "root_mean_squared_error"]:
        loss = RMSELoss(reduction=reduction)
    elif name in ['nrmse_g_cb', "weighted_nrmse_global", "weighted_normalized_root_mean_squared_error_global", "climate_bench_nrmse_global"]:
        loss = NRMSELoss_g_ClimateBench()
    elif name in ['nrmse_s_cb', "weighted_nrmse_spatial", "weighted_normalized_root_mean_squared_error_spatial", "climate_bench_nrmse_spatial"]:
        loss = NRMSELoss_s_ClimateBench()
    elif name in ['nrmse_cb', "weighted_nrmse", "weighted_normalized_root_mean_squared_error", "climate_bench_nrmse"]:
        loss = NRMSELoss_ClimateBench()
    elif name in ['llrmse_wb', "longitude_latitude_weighted_root_mean_squared_error_wheather_ench", "wheather_bench_lon_lat_rmse" ]:
        loss = LLWeighted_RMSELoss_WheatherBench()
    elif name in ['llrmse_cx', "longitude_latitude_weighted_root_mean_squared_error_climax", "climax_lon_lat_rmse",]:
        loss = LLweighted_RMSELoss_Climax()
    elif name in ['llmse_cx', "longitude_latitude_weighted_mean_squared_error_climax", "climax_lon_lat_mse",]:
        loss = LLweighted_MSELoss_Climax()
    

    elif name in ['smoothl1', 'smooth']:
        loss = nn.SmoothL1Loss(reduction=reduction)
    else:
        raise ValueError(f'Unknown loss function {name}')
    return loss


def get_trainable_params(model):
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    return trainable_params


def no_op(*args, **kwargs):
    pass



def random_split(dataset, lengths,
                 generator=default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]

def diff_max_min(x,dim):
    return torch.max(x,dim=dim)-torch.min(x,dim=dim)

def diff_max_min_np(x,dim):
    return np.max(x,axis=dim)-np.min(x,axis=dim)


def weighted_global_mean(input, weights):
    # weitghs * input summed over lon lat / lon+lat    
    return np.mean(input*weights, axis=(-1,-2))


def get_epoch_ckpt_or_last(ckpt_files: List[str], epoch: int = None):
    if epoch is None:
        if "last.ckpt" in ckpt_files:
            model_ckpt_filename = "last.ckpt"
        else:
            ckpt_epochs = [int(name.replace("epoch", "")[:3]) for name in ckpt_files]
            # Use checkpoint with latest epoch if epoch is not specified
            max_epoch = max(ckpt_epochs)
            model_ckpt_filename = [
                name for name in ckpt_files if str(max_epoch) in name
            ][0]
        logging.warning(
            f"Multiple ckpt files exist: {ckpt_files}. Using latest epoch: {model_ckpt_filename}"
        )
    else:
        # Use checkpoint with specified epoch
        model_ckpt_filename = [name for name in ckpt_files if str(epoch) in name]
        if len(model_ckpt_filename) == 0:
            raise ValueError(
                f"There is no ckpt file for epoch={epoch}. Try one of the ones in {ckpt_files}!"
            )
        model_ckpt_filename = model_ckpt_filename[0]
    return model_ckpt_filename

if __name__=="__main__":
    print("hallo")