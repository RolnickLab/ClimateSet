"""
This code is adapted from the original codebase that can be found at: https://github.com/RolnickLab/climart
Adaptations include the insertion of functions and classes, altering functions and classes, inserting comments and other changes.

"""

import json
import logging
import os
import pathlib
from os.path import isdir, isfile
from typing import Union, Callable, List, Optional, Sequence

import wandb
import pandas as pd
from omegaconf import OmegaConf, DictConfig

from emulator.src.utils.config_utils import (
    print_config,
    get_config_from_hydra_compose_overrides,
)
from emulator.src.utils.utils import (
    get_logger,
    get_epoch_ckpt_or_last,
)

DF_MAPPING = Callable[[pd.DataFrame], pd.DataFrame]
log = get_logger(__name__)


def get_wandb_ckpt_name(run_path: str, epoch: Optional[int] = None) -> str:
    """
    Get the wandb ckpt name for a given run_path and epoch.
    Args:
        run_path: PROJECT/group/RUN_ID
        epoch: If a int, the ckpt name will be the one for that epoch, otherwise the latest epoch ckpt will be returned.

    Returns:
        The wandb ckpt file-name, that can be used as follows to restore the checkpoint locally:
           >>> ckpt_name = get_wandb_ckpt_name(run_path, epoch)
           >>> wandb.restore(ckpt_name, run_path=run_path, replace=True, root=os.getcwd())
    """
    run_api = wandb.Api(timeout=77).run(run_path)
    if "best_model_filepath" in run_api.summary and epoch is None:
        best_model_path = run_api.summary["best_model_filepath"]
    else:
        ckpt_files = [f.name for f in run_api.files() if f.name.endswith(".ckpt")]
        if len(ckpt_files) == 0:
            raise ValueError(
                f"Wandb run {run_path} has no checkpoint files (.ckpt) saved in the cloud!"
            )
        elif len(ckpt_files) >= 2:
            best_model_path = get_epoch_ckpt_or_last(ckpt_files, epoch)
        else:
            best_model_path = ckpt_files[0]
    return best_model_path


def restore_model_from_wandb_cloud(run_path: str, **kwargs) -> str:
    """
    Restore the model from the wandb cloud to local file-system.
    Args:
        run_path: PROJECT/group/RUN_ID

    Returns:
        The ckpt filename that can be used to reload the model locally.
    """
    best_model_path = get_wandb_ckpt_name(run_path, **kwargs)
    best_model_path_fname = best_model_path.split("/")[-1]
    best_model_path = wandb.restore(
        best_model_path_fname, run_path=run_path, replace=True, root=os.getcwd()
    ).name
    wandb_id = run_path.split("/")[-1]
    os.rename(best_model_path, f"{wandb_id}-{best_model_path_fname}")
    best_model_path = f"{wandb_id}-{best_model_path_fname}"
    return best_model_path


def load_hydra_config_from_wandb(
    run_path: str,
    base_config: Optional[DictConfig] = None,
    override_config: Optional[DictConfig] = None,
    override_key_value: List[str] = None,
    create_with_compose: bool = False,
    try_local_recovery: bool = False,
) -> DictConfig:
    """
    Args:
        run_path (str): the wandb group/PROJECT/ID (e.g. ID=2r0l33yc) corresponding to the config to-be-reloaded
        base_config (DictConfig): the base config to be override by the wandb config (e.g. the default config).
                This is (only) useful when the wandb config does not contain all the keys of the base config.
        override_config (DictConfig): each of its keys will override the corresponding entry loaded from wandb
        override_key_value: each element is expected to have a "=" in it, like datamodule.num_workers=8
    """
    if override_config is not None and override_key_value is not None:
        log.warning(f"Both override_config and override_key_value are not None! ")
    run = wandb.Api(timeout=77).run(run_path)
    override_key_value = override_key_value or []
    if isinstance(override_key_value, dict):
        override_key_value = [f"{k}={v}" for k, v in override_key_value.items()]
    if not isinstance(override_key_value, list):
        raise ValueError(
            f"override_key_value must be a list of strings, but has type {type(override_key_value)}"
        )
    # copy overrides to new list
    overrides = list(override_key_value.copy())
    # Download from wandb cloud
    wandb_restore_kwargs = dict(run_path=run_path, replace=True, root=os.getcwd())
    try:
        wandb.restore("hydra_config.yaml", **wandb_restore_kwargs)
        kwargs = dict(config_path="../../", config_name="hydra_config.yaml")
    except ValueError:  # hydra_config has not been saved to wandb :(
        overrides += json.load(
            wandb.restore("wandb-metadata.json", **wandb_restore_kwargs)
        )["args"]
        kwargs = dict()
        if len(overrides) == 0:
            raise ValueError(
                "wandb-metadata.json had no args, are you sure this is correct?"
            )
            # also wandb-metadata.json is unexpected (was likely overwritten)
    overrides += [
        f"logger.wandb.id={run.id}",
        #f"logger.wandb.entity={run.entity}",
        f"logger.wandb.project={run.project}",
        f"logger.wandb.tags={run.tags}",
        f"logger.wandb.group={run.group}",
    ]
    if create_with_compose:
        config = get_config_from_hydra_compose_overrides(overrides, **kwargs)
        OmegaConf.set_struct(config, False)
    else:
        config = OmegaConf.load("hydra_config.yaml")
        overrides = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.unsafe_merge(config, overrides)

    if (base_config is not None) & (base_config != []):
        # merge base config with wandb config

        config = OmegaConf.update(base_config, config)
    if override_config is not None:
        # override config with override_config (which needs to be the second argument of OmegaConf.merge)
        config = OmegaConf.unsafe_merge(
            config, override_config
        )  # unsafe_merge since override_config is not needed

    os.remove("hydra_config.yaml") if os.path.exists("hydra_config.yaml") else None
    os.remove("../../hydra_config.yaml") if os.path.exists(
        "../../hydra_config.yaml"
    ) else None

    if run.id != config.logger.wandb.id and run.id in config.logger.wandb.name:
        config.logger.wandb.id = run.id
    assert (
        config.logger.wandb.id == run.id
    ), f"{config.logger.wandb.id} != {run.id}. \nFull Hydra config: {config}"
    return config

def reload_checkpoint_from_wandb(
    run_id: str,
    group: str = "causalpaca", 
    project: str = "emulator", 
    epoch: Union[str, int] = None,
    override_key_value: Union[Sequence[str], dict] = None,
    local_checkpoint_path: str = None,
    try_local_recovery: bool = False,
    manual_backup: bool = False,
    **reload_kwargs,
) -> dict:
    """
    Reload model checkpoint based on only the Wandb run ID

    Args:
        run_id (str): the wandb run ID (e.g. 2r0l33yc) corresponding to the model to-be-reloaded
        group (str): the wandb group corresponding to the model to-be-reloaded
        project (str): the project group corresponding to the model to-be-reloaded
        epoch (str or int): If 'best', the reloaded model will be the best one stored, if 'last' the latest one stored),
                             if an int, the reloaded model will be the one save at that epoch (if it was saved, otherwise an error is thrown)
        override_key_value: If a dict, every k, v pair is used to override the reloaded (hydra) config,
                            e.g. pass {datamodule.num_workers: 8} to change the corresponding flag in config.
                            If a sequence, each element is expected to have a "=" in it, like datamodule.num_workers=8
        local_checkpoint_path (str): If not None, the path to the local checkpoint to be reloaded.
    """
    from emulator.src.utils.interface import reload_model_from_config_and_ckpt

    run_path = f"{group}/{project}/{run_id}"
    config = load_hydra_config_from_wandb(
        run_path, override_key_value=override_key_value, manual_backup=manual_backup
    )

    if local_checkpoint_path is not None:
        best_model_fname = best_model_path = local_checkpoint_path
    else:
        best_model_path = get_wandb_ckpt_name(run_path, epoch=epoch)
        best_model_fname = best_model_path.split("/")[
            -1
        ]  # in case the file contains local dir structure
        # IMPORTANT ARGS replace=True: see https://github.com/wandb/client/issues/3247
        wandb_restore_kwargs = dict(run_path=run_path, replace=True, root=os.getcwd())
        wandb.restore(
            best_model_fname, **wandb_restore_kwargs
        )  # download from the cloud

    assert config.logger.wandb.id == run_id, f"{config.logger.wandb.id} != {run_id}"

    try:
        reloaded_model_data = reload_model_from_config_and_ckpt(
            config, best_model_fname, **reload_kwargs
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"You have probably changed the model code, making it incompatible with older model "
            f"versions. Tried to reload the model ckpt for run.id={run_id} from {best_model_path}.\n"
            f"config.model={config.model}\n{e}"
        )
    os.remove(best_model_fname) if os.path.exists(
        best_model_fname
    ) else None  # delete the downloaded ckpt
    return {**reloaded_model_data, "config": config}



def reupload_run_history(run):
    """
    This function can be called when for weird reasons your logged metrics do not appear in run.summary.
    All metrics for each epoch (assumes that a key epoch=i for each epoch i was logged jointly with the metrics),
    will be reuploaded to the wandb run summary.
    """
    summary = {}
    for row in run.scan_history():
        if "epoch" not in row.keys() or any(["gradients/" in k for k in row.keys()]):
            continue
        summary.update(row)
    run.summary.update(summary)


#####################################################################
#
# Pre-filtering of wandb runs
#
def has_finished(run):
    return run.state == "finished"


def has_final_metric(run) -> bool:
    return "test/rsuc/rmse" in run.summary.keys()


def has_keys(keys: Union[str, List[str]]) -> Callable:
    if isinstance(keys, str):
        keys = [keys]
    return lambda run: all(
        [(k in run.summary.keys() or k in run.config.keys()) for k in keys]
    )


def has_max_metric_value(
    metric: str = "test/rsuc/rmse", max_metric_value: float = 1.0
) -> Callable:
    return lambda run: run.summary[metric] <= max_metric_value


def has_tags(tags: Union[str, List[str]]) -> Callable:
    if isinstance(tags, str):
        tags = [tags]
    return lambda run: all([tag in run.tags for tag in tags])


def hasnt_tags(tags: Union[str, List[str]]) -> Callable:
    if isinstance(tags, str):
        tags = [tags]
    return lambda run: all([tag not in run.tags for tag in tags])


def hyperparams_list_api(**hyperparams) -> dict:
    return [
        {
            f"config.{hyperparam.replace('.', '/')}": value
            for hyperparam, value in hyperparams.items()
        }
    ]


def has_hyperparam_values(**hyperparams) -> Callable:
    return lambda run: all(
        hyperparam in run.config and run.config[hyperparam] == value
        for hyperparam, value in hyperparams.items()
    )


def larger_than(**kwargs) -> Callable:
    return lambda run: all(
        hasattr(run.config, hyperparam) and value > run.config[hyperparam]
        for hyperparam, value in kwargs.items()
    )


def lower_than(**kwargs) -> Callable:
    return lambda run: all(
        hasattr(run.config, hyperparam) and value < run.config[hyperparam]
        for hyperparam, value in kwargs.items()
    )


def df_larger_than(**kwargs) -> DF_MAPPING:
    def f(df) -> pd.DataFrame:
        for k, v in kwargs.items():
            df = df.loc[getattr(df, k) > v]
        return df

    return f


def df_lower_than(**kwargs) -> DF_MAPPING:
    def f(df) -> pd.DataFrame:
        for k, v in kwargs.items():
            df = df.loc[getattr(df, k) < v]
        return df

    return f


str_to_run_pre_filter = {
    "has_finished": has_finished,
    "has_final_metric": has_final_metric,
}


#####################################################################
#
# Post-filtering of wandb runs (usually when you need to compare runs)
#
def topk_runs(
    k: int = 5, metric: str = "best_val/NRMSE_sd", lower_is_better: bool = True
) -> DF_MAPPING:
    if lower_is_better:
        return lambda df: df.nsmallest(k, metric)
    else:
        return lambda df: df.nlargest(k, metric)


def topk_run_of_each_model_type(
    k: int = 1, metric: str = "best_val/NRMSE_sd", lower_is_better: bool = True
) -> DF_MAPPING:
    topk_filter = topk_runs(k, metric, lower_is_better)

    def topk_runs_per_model(df: pd.DataFrame) -> pd.DataFrame:
        models = df.model.unique()
        dfs = []
        for model in models:
            dfs += [topk_filter(df[df.model == model])]
        return pd.concat(dfs)

    return topk_runs_per_model


def non_unique_cols_dropper(df: pd.DataFrame) -> pd.DataFrame:
    nunique = df.nunique()
    cols_to_drop = nunique[nunique == 1].index
    df = df.drop(cols_to_drop, axis=1)
    return df


def groupby(
    df: pd.DataFrame,
    group_by: str = "seed",
    metrics: List[str] = "best_val/NRMSE_sd",
    keep_columns: List[str] = "model/name",
) -> pd.DataFrame:
    """

    Returns:
        A dataframe grouped by `group_by` with columns
        `metric`/mean and `metric`/std for each metric passed in `metrics` and all columns in `keep_columns` remain intact.
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    if isinstance(keep_columns, str):
        keep_columns = [keep_columns]

    grouped_df = df.groupby([group_by], as_index=False)
    agg_metrics = {m: ["mean", "std"] for m in metrics}
    agg_remain_intact = {c: "first" for c in keep_columns}
  
    stats = grouped_df.agg({**agg_metrics, **agg_remain_intact})
    stats.columns = [
        (f"{c[0]}/{c[1]}" if c[1] in ["mean", "std"] else c[0]) for c in stats.columns
    ]
    for m in metrics:
        stats[f"{m}/std"].fillna(value=0, inplace=True)

    return stats


str_to_run_post_filter = {
    **{f"top{k}": topk_runs(k=k) for k in range(1, 21)},
    "best_per_model": topk_run_of_each_model_type(k=1),
    **{f"top{k}_per_model": topk_run_of_each_model_type(k=k) for k in range(1, 6)},
    "unique_columns": non_unique_cols_dropper,
}


def get_wandb_filters_dict_list_from_list(filters_list) -> dict:
    if filters_list is None:
        filters_list = []
    elif not isinstance(filters_list, list):
        filters_list: List[Union[Callable, str]] = [filters_list]
    filters_wandb = []  # dict()
    for f in filters_list:
        if isinstance(f, str):
            f = str_to_run_pre_filter[f.lower()]
        filters_wandb.append(f)
    return filters_wandb


def get_best_model_config(
    metric: str = "best_val/NRMSE_sd",
    mode: str = "min",
    filters: Union[str, List[Union[Callable, str]]] = "has_finished",
    group: str = "causalpaca", 
    project: str = "emulator", 
    wandb_api=None,
    topk: int = 1,
) -> dict:
    filters_wandb = get_wandb_filters_dict_list_from_list(filters)
    api = wandb_api or wandb.Api(timeout=77)
    # Project is specified by <group/project-name>
    pm = "+" if mode == "min" else "-"
    print(filters_wandb)
    filters_wandb = {"$and": [filters_wandb]}
    runs = api.runs(
        f"{group}/{project}",
        filters=filters_wandb,
        order=f"{pm}summary_metrics.{metric}",
    )
    return {"id": runs[:k].id, **runs[:k].config}


def get_run_ids_for_hyperparams(
    hyperparams: dict,
    group: str = "causalpaca", 
    project: str = "emulator", 
    wandb_api=None,
) -> List[str]:
    runs = filter_wandb_runs(
        hyperparams, group=group, project=project, wandb_api=wandb_api
    )
    run_ids = [run.id for run in runs]
    return run_ids


def filter_wandb_runs(
    hyperparam_filter: dict = None,
    filter_functions: Sequence[Callable] = None,
    order="-created_at",
    group: str = "causalpaca",  # ecc-mila7",
    project: str = "emulator",  # ClimART',
    wandb_api=None,
    verbose: bool = True,
):
    """
    Args:
        hyperparam_filter: a dict str -> value, e.g. {'model/name': 'mlp', 'datamodule/exp_type': 'pristine'}
        filter_functions: A set of callable functions that take a wandb run and return a boolean (True/False) so that
                            any run with one or more return values being False is discarded/filtered out
    """
    hyperparam_filter = hyperparam_filter or dict()
    filter_functions = filter_functions or []
    api = wandb_api or wandb.Api(timeout=100)
    filter_wandb_api, filters_post = dict(), dict()
    for k, v in hyperparam_filter.items():
        if any(tpl in k for tpl in ["datamodule", "normalizer"]):
            filter_wandb_api[k] = v
        else:
            filters_post[k.replace(".", "/")] = v  # wandb keys are / separated
    filter_wandb_api = hyperparams_list_api(**filter_wandb_api)
    filter_wandb_api = {"$and": filter_wandb_api}  # MongoDB query lang
    runs = api.runs(
        f"{group}/{project}", filters=filter_wandb_api, per_page=100, order=order
    )
    n_runs1 = len(runs)
    filters_post_func = has_hyperparam_values(**filters_post)
    runs = [
        run
        for run in runs
        if filters_post_func(run) and all(f(run) for f in filter_functions)
    ]
    if verbose:
        log.info(f"#Filtered runs = {len(runs)}, (wandb API filtered {n_runs1})")
    return runs


if __name__=="__main__":
    print("hello")