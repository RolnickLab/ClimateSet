import numpy as np
from emulator.src.utils.utils import get_logger, weighted_global_mean

log = get_logger()

# all functions assuming input dimensions (batch_size / N, time, lon, lat)


def MSE(preds: np.ndarray, y: np.ndarray):
    return np.mean((preds - y) ** 2)


# CHECKED and adapted
def RMSE(preds: np.ndarray, y: np.ndarray):
    return np.sqrt(MSE(preds, y))

# CHCKED and adapted
def NRMSE_s_ClimateBench(preds: np.ndarray, y: np.ndarray, deg2rad: bool = True):
    """
    Spatial normalized weighted RMSE taken from Climate Bench.
    Weighting to account for decreasing grid size towards the pole.
    """

    # weighting to account for decreasing grid-cell area towards pole
    # latitude weights
    lat_size = y.shape[-1]
    lats = np.linspace(-89.75, 89.75, lat_size)
    if deg2rad:
        weights = np.cos((np.pi * lats) / 180)
    else:
        weights = np.cos(lats)

    # nrmse = sqrt((weights * (pred_mean_b_t - y_mean_b_t)**2)_mean_s) / ((weights*y)_mean_s)_mean_b_t
    nrmse_s = np.sqrt(
        weighted_global_mean(
            (preds.mean(axis=(0, 1)) - y.mean(axis=(0, 1))) ** 2, weights
        )
    ) / weighted_global_mean(y, weights).mean(axis=(0, 1))

    return nrmse_s

# CHECKED and adapted
def NRMSE_g_ClimateBench(preds: np.ndarray, y: np.ndarray, deg2rad: bool = True):
    """
    Spatial normalized weighted RMSE taken from Climate Bench.
    Weigting to account for decreasing grid size towards the pole.
    """
    # latitude weighting to account for decreasing grid-cell area towards pole
    lat_size = y.shape[-1]
    lats = np.linspace(-89.75, 89.75, lat_size)
    if deg2rad:
        weights = np.cos((np.pi * lats) / 180)
    else:
        weights = np.cos(lats)

    # denom is not alowed to be zero!
    if np.any(preds == 0):
        log.warning("Predictions contain zero-values, adding epsilon to metric.")
        preds[preds == 0] += 1e-6

    under_sqrt = (
        (
            weighted_global_mean(preds, weights)
            - weighted_global_mean(y, weights)
        ) ** 2
    ).mean(axis=(0, 1))
    if np.isnan(under_sqrt).sum() > 0:
        log.info("under sqrt is nan")
        raise ValueError("NRMSE_g got nan under sqrt")
    nrmse_g = (
        np.sqrt(
            (
                (
                    weighted_global_mean(preds, weights)
                    - weighted_global_mean(y, weights)
                ) ** 2
            ).mean(axis=(0, 1))
        )
        / weighted_global_mean(y, weights).mean(axis=(0, 1))
    )

    return nrmse_g

# CHECKED
def NRMSE_ClimateBench(preds: np.ndarray, y: np.ndarray, alpha: int = 5):
    """
    Combination of global weighted and spatially weighted nrmse.
    """

    nrmseg = NRMSE_g_ClimateBench(preds, y)
    nrmses = NRMSE_s_ClimateBench(preds, y)
    nrmse = nrmses + alpha * nrmseg
    return nrmse

# CONTINUE HERE
def LLWeighted_RMSE_WheatherBench(preds: np.ndarray, y: np.ndarray):
    """
    Weigthed RMSE taken from Weather Bench.
    Weighting to account for decreasing grid sizes towards the pole.

    rmse = mean over forecasts and time of np.sqrt( mean over lon lat L(lat_j)*)MSE(preds, y)
    weights = cos(latitude)/cos(latitude).mean()
    """
    lat_size = y.shape[-1]
    lats = np.linspace(-90, 90, lat_size)
    

    weights = (np.cos(lats) / np.cos(lats)).mean()

    rmse = np.sqrt(np.mean(weights * ((preds - y) ** 2), axis=(-1, -2))).mean()

    return rmse


def LLweighted_MSE_Climax(
    preds: np.ndarray, y: np.ndarray, deg2rad: bool = True, mask=None
):
    """
    Latitude weighted mean squared error taken from ClimaX.
    Allows to weight the  by the cosine of the latitude to account for gridding differences at equator vs. poles.
    Applied per variable.
    If given a mask, normalized by sum of that.

    """

    # lattitude weights
    lat_size = y.shape[-1]
    lats = np.linspace(-90, 90, lat_size)
    if deg2rad:
        weights = np.cos((np.pi * lats) / 180)
    else:
        weights = np.cos(lats)

    # they normalize the weights first
    weights = weights / weights.mean()

    if mask is not None:
        error = (((preds - y) ** 2) * weights * mask).sum() / mask.sum()
    else:
        error = (((preds - y) ** 2) * weights).mean()

    return error


def LLweighted_RMSE_Climax(
    preds: np.ndarray, y: np.ndarray, deg2rad: bool = True, mask=None
):
    """
    Latitude weighted root mean squared error taken from ClimaX.
    Allows to weight the  by the cosine of the latitude to account for gridding differences at equator vs. poles.
    Applied per variable.
    If given a mask, normalized by sum of that.
    """
    # latitude weights
    lat_size = y.shape[-1]
    lats = np.linspace(-90, 90, lat_size)
    if deg2rad:
        weights = np.cos((np.pi * lats) / 180)
    else:
        weights = np.cos(lats)

    # they normalize the weights first
    weights = weights / weights.mean()

    if mask is not None:
        error = (((preds - y) ** 2) * weights * mask).sum() / mask.sum()
    else:
        error = (((preds - y) ** 2) * weights).mean()

    error = np.sqrt(error)

    return error


if __name__ == "__main__":
    batch_size = 16
    out_time = 10
    lat = 96
    lon = 144
    # dummy = np.random.randn(batch_size, out_time, lat, lon)
    # targets = np.random.randn(batch_size, out_time, lat, lon)  # .cuda()

    targets = np.ones(shape=(batch_size, out_time, lat, lon))
    dummy = targets + 0.1

    #reduction = "mean"
    mse = MSE(dummy, targets)
    rmse = RMSE(dummy, targets)

    nrmse_g = NRMSE_g_ClimateBench(dummy, targets)
    nrmse_s = NRMSE_s_ClimateBench(dummy, targets)
    nrmse = NRMSE_ClimateBench(dummy, targets)

    llrmse_wb = LLWeighted_RMSE_WheatherBench(dummy, targets)

    llmse_cx = LLweighted_MSE_Climax(dummy, targets)
    llrmse_cx = LLweighted_RMSE_Climax(dummy, targets)

    loss = mse
    print("MSE loss", loss, loss.shape)

    loss = rmse
    print("RMSE loss", loss, loss.shape)

    loss = nrmse_g
    print("CB nrmse g metric", loss, loss.shape)

    loss = nrmse_s
    print("CB nrmse s metric", loss, loss.shape)

    loss = nrmse
    print("CB nrmse metric", loss, loss.shape)

    exit(0)

    loss = llrmse_wb
    print("WB rmse metric", loss, loss.shape)

    loss = llmse_cx
    print("CX mse metric", loss, loss.shape)

    loss = llrmse_cx
    print("CX nmse metric", loss, loss.shape)
