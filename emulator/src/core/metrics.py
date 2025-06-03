import torch
import numpy as np
from emulator.src.utils.log import get_logger
from emulator.src.utils.metric_utils import weighted_global_mean_np, get_latitude_weights_np, check_lat_lon_np

log = get_logger()

# all functions assuming input dimensions (batch_size / N, time, lon, lat)

def MSE(preds: np.ndarray, y: np.ndarray):
    return np.mean((preds - y) ** 2)

def RMSE(preds: np.ndarray, y: np.ndarray):
    return np.sqrt(MSE(preds, y))

def NRMSE_s_ClimateBench(preds: np.ndarray, y: np.ndarray):
    """
    Spatial normalized weighted RMSE taken from Climate Bench.
    Weighting to account for decreasing grid size towards the pole.
    """
    check_lat_lon_np(preds, y)
    weights = get_latitude_weights_np(y.shape[-2])

    # nrmse = sqrt((weights * (pred_mean_b_t - y_mean_b_t)**2)_mean_s) / ((weights*y)_mean_s)_mean_b_t
    nrmse_s = np.sqrt(
        weighted_global_mean_np(
            (preds.mean(axis=(0, 1)) - y.mean(axis=(0, 1))) ** 2, weights
        )
    ) / weighted_global_mean_np(y, weights).mean(axis=(0, 1))

    return nrmse_s

def NRMSE_g_ClimateBench(preds: np.ndarray, y: np.ndarray):
    """
    Spatial normalized weighted RMSE taken from Climate Bench.
    Weigting to account for decreasing grid size towards the pole.
    """
    check_lat_lon_np(preds, y)
    weights = get_latitude_weights_np(y.shape[-2])

    nrmse_g = (
        np.sqrt(
            (
                (
                    weighted_global_mean_np(preds, weights)
                    - weighted_global_mean_np(y, weights)
                ) ** 2
            ).mean(axis=(0, 1))
        )
        / weighted_global_mean_np(y, weights).mean(axis=(0, 1))
    )

    return nrmse_g

def NRMSE_ClimateBench(preds: np.ndarray, y: np.ndarray, alpha: int = 5):
    """
    Combination of global weighted and spatially weighted nrmse.
    """
    check_lat_lon_np(preds, y)
    nrmseg = NRMSE_g_ClimateBench(preds, y)
    nrmses = NRMSE_s_ClimateBench(preds, y)
    nrmse = nrmses + alpha * nrmseg
    return nrmse

def LLWeighted_RMSE_WeatherBench(preds: np.ndarray, y: np.ndarray):
    """
    Weigthed RMSE taken from Weather Bench.
    Weighting to account for decreasing grid sizes towards the pole.

    rmse = mean over forecasts and time of np.sqrt( mean over lon lat L(lat_j)*)MSE(preds, y)
    
    original code does:     weights = cos(latitude)/cos(latitude).mean()
    --> we are not doing that
    """
    check_lat_lon_np(preds, y)
    weights = get_latitude_weights_np(y.shape[-2])
    rmse = np.mean(np.sqrt(np.mean(weights * ((preds - y) ** 2), axis=(-2, -1))))
    return rmse


def LLweighted_MSE_Climax(
    preds: np.ndarray, y: np.ndarray, mask=None
):
    """
    Latitude weighted mean squared error taken from ClimaX.
    Allows to weight the  by the cosine of the latitude to account for gridding differences at equator vs. poles.
    Applied per variable.
    If given a mask, normalized by sum of that.
    """
    check_lat_lon_np(preds, y)
    if mask is not None:
        raise NotImplementedError("Masking is not supported in the metric functions anymore.")
        
    weights = get_latitude_weights_np(y.shape[-2])
    error = (((preds - y) ** 2) * weights).mean()

    return error


def LLweighted_RMSE_Climax(
    preds: np.ndarray, y: np.ndarray, mask=None
):
    """
    Latitude weighted root mean squared error taken from ClimaX.
    Allows to weight the  by the cosine of the latitude to account for gridding differences at equator vs. poles.
    Applied per variable.
    If given a mask, normalized by sum of that.
    """
    check_lat_lon_np(preds, y)
    if mask is not None:
        raise NotImplementedError("Masking is not supported in the metric functions anymore.")
    weights = get_latitude_weights_np(y.shape[-2])

    # if mask is not None:
    #     error = (((preds - y) ** 2) * weights * mask).sum() / mask.sum()

    # rmse for each month and each batch
    error = np.sqrt(np.mean(((preds - y) ** 2) * weights, axis=(-1, -2)))
    # mean over all months and batch samples
    error = error.mean()

    return error


if __name__ == "__main__":
    batch_size = 16
    out_time = 10
    lat = 96
    lon = 144
    # try to get the same rand matrices 
    dummy = torch.rand(size=(batch_size, out_time, lat, lon))#.cuda()
    targets = torch.rand(size=(batch_size, out_time, lat, lon))
    # and convert them to numpy arrays
    dummy = dummy.numpy()
    targets = targets.numpy()

    # dummy = np.random.randn(batch_size, out_time, lat, lon)
    # targets = np.random.randn(batch_size, out_time, lat, lon)  # .cuda()

    # targets = np.ones(shape=(batch_size, out_time, lat, lon))
    # dummy = targets + 0.1

    #reduction = "mean"
    mse = MSE(dummy, targets)
    rmse = RMSE(dummy, targets)

    nrmse_s = NRMSE_s_ClimateBench(dummy, targets)
    nrmse_g = NRMSE_g_ClimateBench(dummy, targets)
    nrmse = NRMSE_ClimateBench(dummy, targets)

    llrmse_wb = LLWeighted_RMSE_WeatherBench(dummy, targets)

    llmse_cx = LLweighted_MSE_Climax(dummy, targets)
    llrmse_cx = LLweighted_RMSE_Climax(dummy, targets)

    loss = mse
    print("MSE loss", loss, loss.shape)

    loss = rmse
    print("RMSE loss", loss, loss.shape)

    loss = nrmse_s
    print("CB nrmse s metric", loss, loss.shape)

    loss = nrmse_g
    print("CB nrmse g metric", loss, loss.shape)

    loss = nrmse
    print("CB nrmse metric", loss, loss.shape)

    loss = llrmse_wb
    print("WB rmse metric", loss, loss.shape)

    loss = llmse_cx
    print("CX mse metric", loss, loss.shape)

    loss = llrmse_cx
    print("CX rmse metric", loss, loss.shape)
