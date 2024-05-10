from typing import Dict
import numpy as np
from emulator.src.core.metrics import MSE

from emulator.src.utils.utils import get_logger
from emulator.src.core.metrics import (
    RMSE,
    NRMSE_s_ClimateBench,
    NRMSE_g_ClimateBench,
    NRMSE_ClimateBench,
    LLWeighted_RMSE_WheatherBench,
    LLweighted_MSE_Climax,
    LLweighted_RMSE_Climax,
)

log = get_logger(__name__)


def evaluate_preds(Ytrue: np.ndarray, preds: np.ndarray):
    # compute all stats for evaluation

    # get rid of empty var dimension
    preds = np.squeeze(preds)
    Ytrue = np.squeeze(Ytrue)

    mse = MSE(preds, Ytrue)
    rmse = RMSE(preds, Ytrue)
    #nrmse_g_climate_bench = NRMSE_g_ClimateBench(preds, Ytrue)
    #nrmse_s_climate_bench = NRMSE_s_ClimateBench(preds, Ytrue)
    #nrmse_climate_bench = NRMSE_ClimateBench(preds, Ytrue)
    llrmse_wheather_bench = LLWeighted_RMSE_WheatherBench(preds, Ytrue)
    llmse_climax = LLweighted_MSE_Climax(preds, Ytrue)
    llrmse_climax = LLweighted_RMSE_Climax(preds, Ytrue)

    stats = {
        "mse": mse,
        "rmse": rmse,
        #"nrmse_s_climate_bench": nrmse_s_climate_bench,
        #"nrmse_g_climate_bench": nrmse_g_climate_bench,
        #"nrmse_climate_bench": nrmse_climate_bench,
        "llrmse_wheather_bench": llrmse_wheather_bench,
        "llmse_climax": llmse_climax,
        "llrmse_climax": llrmse_climax,
    }

    return stats


def evaluate_per_target_variable(
    Ytrue: dict,
    preds: dict,
    data_split: str = None,
) -> Dict[str, float]:
    stats = dict()
    if not isinstance(Ytrue, dict):
        log.warning(
            f" Expected a dictionary var_name->Tensor/nd_array, but got {type(Ytrue)} for Ytrue!"
        )
        return stats

    var_stats = [
        evaluate_preds(Ytrue[var_name], preds[var_name]) for var_name in Ytrue.keys()
    ]

    # aggregator for mean over vars
    for m in var_stats[0].keys():
        stats[f"{data_split}/{m}"] = 0

    for var_name, var_stat in zip(Ytrue.keys(), var_stats):
        for metric_name, metric_stat in var_stat.items():
            # pre-append the variable's name to its specific performance on the returned metrics dict

            stats[f"{data_split}/{var_name}/{metric_name}"] = metric_stat
            stats[f"{data_split}/{metric_name}"] += metric_stat

            # if we ever include pressure levels as well...
        """
        num_height_levels = #extract num heights
        for lvl in range(0, num_height_levels):
            stats_lvl = evaluate_preds(Ytrue[var_name][:, lvl], preds[var_name][:, lvl])
            for metric_name, metric_stat in stats_lvl.items():
                stats[f"levelwise/{data_split}/{var_name}_level{lvl}/{metric_name}"] = metric_stat

            if lvl == num_height_levels - 1:
                for metric_name, metric_stat in stats_lvl.items():
                    stats[f"{data_split}/{var_name}_surface/{metric_name}"] = metric_stat
            if lvl == 0:
                for metric_name, metric_stat in stats_lvl.items():
                    stats[f"{data_split}/{var_name}_toa/{metric_name}"] = metric_stat
        """

    # mean over vars
    for m in var_stats[0].keys():
        stats[f"{data_split}/{metric_name}"] /= len(Ytrue.keys())

    stats = {
        **stats,
    }
    return stats
