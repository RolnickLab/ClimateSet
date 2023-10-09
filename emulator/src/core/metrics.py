import numpy as np
from emulator.src.utils.utils import get_logger, weighted_global_mean

log = get_logger()

# all functions assuming input dimensions (batch_size / N, time, lon, lat)


def MSE(preds: np.ndarray, y: np.ndarray):
    return np.mean((preds - y) ** 2)


def RMSE(preds: np.ndarray, y: np.ndarray):
    return np.mean(np.sqrt(MSE(preds, y)))


def NRMSE_s_ClimateBench(preds: np.ndarray, y: np.ndarray, deg2rad: bool = True):
    """
    Spatial normalized weighted RMSE taken from Climate Bench.
    Weigting to account for decreasing grid size towards the pole.
    """

    # weighting to account for decreasing grid-cell area towards pole
    # lattitude weights
    if deg2rad:
        weights = np.cos((np.pi * np.arange(y.shape[-1])) / 180)
    else:
        weights = np.cos(np.arange(y.shape[-1]))

    # nrmses = sqrt((weights * (x_mean_t -y_mean_n_t)**2))_mean_s / ((weights*y)_mean_s)_mean_t_n
    nrmse_s = np.sqrt(
        weighted_global_mean(
            (preds.mean(axis=(0, 1)) - y.mean(axis=(0, 1))) ** 2, weights
        )
    ) / weighted_global_mean(y, weights).mean(axis=(0, 1))

    return nrmse_s


def NRMSE_g_ClimateBench(preds: np.ndarray, y: np.ndarray, deg2rad: bool = True):
    """
    Spatial normalized weighted RMSE taken from Climate Bench.
    Weigting to account for decreasing grid size towards the pole.
    """
    # weighting to account for decreasing grid-cell area towards pole
    # lattitude weights
    if deg2rad:
        weights = np.cos((np.pi * np.arange(y.shape[-1])) / 180)
    else:
        weights = np.cos(np.arange(y.shape[-1]))

    denom = weighted_global_mean(y, weights).mean(axis=(0, 1))

    # denom is not alowed to be zero!
    if np.any(preds == 0):
        log.warn("predictions contains zeros!! adding epsilon")
        preds[preds == 0] += 1e-6

    under_sqrt = (
        (
            weighted_global_mean(preds.mean(axis=0), weights)
            - weighted_global_mean(y.mean(axis=0), weights)
        )
        ** 2
    ).mean(axis=0)
    if np.isnan(under_sqrt).sum() > 0:
        log.info("under sqrt is nan")
    nrmse_g = (
        np.sqrt(
            (
                weighted_global_mean(preds.mean(axis=0), weights)
                - weighted_global_mean(y.mean(axis=0), weights) ** 2
            ).mean(axis=(0))
        )
        / denom
    )

    return nrmse_g


def NRMSE_ClimateBench(preds: np.ndarray, y: np.ndarray, alpha: int = 5):
    """
    Combination of global weighted and spatially weighted nrmse.
    """

    nrmseg = NRMSE_g_ClimateBench(preds, y)
    nrmses = NRMSE_s_ClimateBench(preds, y)
    nrmse = nrmses + alpha * nrmseg
    return nrmse


def LLWeighted_RMSE_WheatherBench(preds: np.ndarray, y: np.ndarray):
    """
    Weigthed RMSE taken from Wheather Bench.
    Weighting to account for decreasing grid sizes towards the pole.

    rmse = mean over forecasts and time of np.sqrt( mean over lon lat L(lat_j)*)MSE(preds, y)
    weights = cos(latitude)/cos(latitude).mean()
    """

    weights = (np.cos(np.arange(y.shape[-1])) / np.cos(np.arange(y.shape[-1]))).mean()

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
    if deg2rad:
        weights = np.cos((np.pi * np.arange(y.shape[-1])) / 180)
    else:
        weights = np.cos(np.arange(y.shape[-1]))

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

    # lattitude weights
    if deg2rad:
        weights = np.cos((np.pi * np.arange(y.shape[-1])) / 180)
    else:
        weights = np.cos(np.arange(y.shape[-1]))

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
    lon = 32
    lat = 64
    dummy = np.random.randn(batch_size, out_time, lon, lat)

    targets = np.random.randn(batch_size, out_time, lon, lat)  # .cuda()

    reduction = "mean"
    mse = MSE(dummy, targets)
    # rmse=RMSE(reduction=reduction)

    nrmse_g = NRMSE_g_ClimateBench(dummy, targets)
    nrmse_s = NRMSE_s_ClimateBench(dummy, targets)
    nrmse = NRMSE_ClimateBench(dummy, targets)

    llrmse_wb = LLWeighted_RMSE_WheatherBench(dummy, targets)

    llmse_cx = LLweighted_MSE_Climax(dummy, targets)
    llrmse_cx = LLweighted_RMSE_Climax(dummy, targets)

    loss = nrmse_g
    print("CB nrmse g loss", loss, loss.shape)

    loss = nrmse_s
    print("CB nrmse s loss", loss, loss.shape)

    loss = nrmse
    print("CB nrmse loss", loss, loss.shape)

    loss = llrmse_wb
    print("WB rmse loss", loss, loss.shape)

    loss = llmse_cx
    print("CX mse loss", loss, loss.shape)

    loss = llrmse_cx
    print("CX nmse loss", loss, loss.shape)
