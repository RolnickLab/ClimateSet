import torch
import numpy as np

from emulator.src.utils.utils import get_logger 

log = get_logger()

def diff_max_min(x, dim):
    return torch.max(x, dim=dim) - torch.min(x, dim=dim)


def diff_max_min_np(x, dim):
    return np.max(x, axis=dim) - np.min(x, axis=dim)

# weighting to account for decreasing grid-cell area towards pole
def get_latitude_weights_np(lat_size: int) -> np.ndarray:
    """ Returns latitude weights for a given number of data points along the latitude axis (y.shape[-2]).
    The weights are 1 at the equator and decrease towards the poles. 
    Parameters:
        lat_size (int): How many latitude data points should be considered
    Returns:
        np.ndarray: Latitude weights
    """
    lats = np.linspace(-89.75, 89.75, lat_size)
    weights = np.cos((np.pi * lats) / 180)
    weights = np.expand_dims(weights, axis=-1)
    return weights

def weighted_global_mean_np(input: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # weights * input summed over lon lat / lon+lat
    return np.mean(input * weights, axis=(-2, -1)) # axis order doesn't matter

def check_lat_lon_np(pred: np.ndarray, y: np.ndarray):
    """ Functions that checks if latitude and longitude is behaving as expected.
    Parameters:
        pred (np.ndarray): Predictions
        y (np.ndarray): Targets
    """
    if len(pred.shape) > 4:
        log.warning("Tensors handed to loss function have more than 4 dimensions. Check if added channels need to be treated differently.")

    # Expected shape: [4, 12, 96, 144] -> [batch, time, latitude, longitude]
    if (pred.shape[-1] == 1) or (y[-1].shape == 1):
        raise ValueError("Loss function: Last dimension (values/channels) must be squeezed away")
    
    if (pred.shape[-1] < pred.shape[-2]) or (y.shape[-1] < y.shape[-2]):
        raise ValueError("There are more latitude than longitude grid cells. Check if you swapped longitude and latitude.")

