from typing import Union, List, Callable, Sequence
import pandas as pd
import numpy as np
import wandb
from pandas.api.types import is_numeric_dtype
import copy
import re
import os
DF_MAPPING = Callable[[pd.DataFrame], pd.DataFrame]


# SOME FILTER CONSTANTS
USERNAME='julia-kaltenborn' 
# filter for correct climax runs
CLIMAX_WARMUP_EPOCHS=5
# filter for other tags
FILTER_TAGS=["run2"]

ENTITY="causalpaca"
PROJECT="emulator"
TOP_K_METRIC='val/llmse_climax' # metric to evaluate best run



def hyperparams_list_api(**hyperparams) -> dict:
    return [{f"config.{hyperparam.replace('.', '/')}": value for hyperparam, value in hyperparams.items()}]

def has_hyperparam_values(**hyperparams) -> Callable:
    return lambda run: all(hyperparam in run.config and run.config[hyperparam] == value
                           for hyperparam, value in hyperparams.items())


def filter_wandb_runs(hyperparam_filter: dict = None,
                      filter_functions: Sequence[Callable] = None,
                      order='-created_at',
                      entity: str = ENTITY,
                      project: str = PROJECT,
                      wandb_api=None,
                      verbose: bool = True
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
        if any(tpl in k for tpl in ['datamodule', 'normalizer']):
            filter_wandb_api[k] = v
        else:
            filters_post[k.replace('.', '/')] = v  # wandb keys are / separated
    filter_wandb_api = hyperparams_list_api(**filter_wandb_api)
    filter_wandb_api = {"$and": filter_wandb_api}  # MongoDB query lang
    runs = api.runs(f"{entity}/{project}", filters=filter_wandb_api, per_page=100, order=order)
    n_runs1 = len(runs)
    filters_post_func = has_hyperparam_values(**filters_post)
    runs = [run for run in runs if filters_post_func(run) and all(f(run) for f in filter_functions)]
    if verbose:
        #log.info(f"#Filtered runs = {len(runs)}, (wandb API filtered {n_runs1})")
        print("#Filtered runs = {len(runs)}, (wandb API filtered {n_runs1})")
    return runs

def get_runs_df(
        get_metrics: bool = True,
        hyperparam_filter: dict = None,
        run_pre_filters: Union[str, List[Union[Callable, str]]] = 'has_finished',
        run_post_filters: Union[str, List[Union[DF_MAPPING, str]]] = None,
        verbose: int = 1,
        make_hashable_df: bool = False,
        **kwargs
) -> pd.DataFrame:
    """

        get_metrics:
        run_pre_filters:
        run_post_filters:
        verbose: 0, 1, or 2, where 0 = no output at all, 1 is a bit verbose
    """
    if isinstance(run_pre_filters, str):
        run_pre_filters = [run_pre_filters]
    run_pre_filters = [(f if callable(f) else str_to_run_pre_filter[f.lower()]) for f in run_pre_filters]

    if run_post_filters is None:
        run_post_filters = []
    elif not isinstance(run_post_filters, list):
        run_post_filters: List[Union[Callable, str]] = [run_post_filters]
    run_post_filters = [(f if callable(f) else str_to_run_post_filter[f.lower()]) for f in run_post_filters]

    # Project is specified by <entity/project-name>
    runs = filter_wandb_runs(hyperparam_filter, run_pre_filters, **kwargs)
    summary_list = []
    config_list = []
    group_list = []
    name_list = []
    tag_list = []
    id_list = []

    # addition
    model_list = []

    for i, run in enumerate(runs):
        if i % 50 == 0:
            print(f"Going after run {i}")
        # if i > 100: break
        # run.summary are the output key/values like accuracy.
        # We call ._json_dict to omit large files
        model=None
        if 'model/_target_' not in run.config.keys():
            if verbose >= 1:
                print(f"Run {run.name} not containing model/_target_run.config.")
                print("Trying to infer from tag.")
            if "unet" in run.tags:
                model="Unet"
            elif "conv_lstm" in run.tags:
                model="CNN_LSTM_ClimateBench"
            elif "climax" in run.tags:
                model="ClimaX"
            elif "climax_frozen" in run.tags:
                model="ClimaX_frozen"
            else:
                print("Cannot infer model from tag. Skipping run.")
        else:
            model=run.config['model/_target_'].split('.')[-1]
        
        if model is None:
            if verbose>=1:
                print("No model found")
            continue
        if verbose >= 1:
            print("Found model", model)
        model_list.append(model)

        id_list.append(str(run.id))
        tag_list.append(str(run.tags))
        
        
        if get_metrics:
            summary_list.append(run.summary._json_dict)
            # run.config is the input metrics.
            config={k: v for k, v in run.config.items() if k not in run.summary.keys()}
        else:
            config=run.config
        
        # refactor dicts datamodule already as single columns and not as 
        desired_dicts=["datamodule", "model", "optim", "trainer"]

        old_keys=copy.deepcopy(config).keys()
        for d in desired_dicts:
            if d not in old_keys:
                config[d]={}
        for k in old_keys:
            if len(k.split('/'))>1:
                if verbose >= 1:
                    print("found suspicious config key", k)
                if k.split('/')[0] in desired_dicts:
                    value=config[k]
                    config[k.split('/')[0]].update({k.split('/')[1]: value})
                    del config[k]
           
        config_list.append(config)
     
       
        # run.name is the name of the run.
        name_list.append(run.name)
        group_list.append(run.group)

    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({'name': name_list, 'id': id_list, 'tags': tag_list, 'model': model_list})
    group_df = pd.DataFrame({'group': group_list})
    all_df = pd.concat([name_df, config_df, summary_df, group_df], axis=1)

    cols = [c for c in all_df.columns if not c.startswith('gradients/') and c != 'graph_0']
    all_df = all_df[cols]
    if all_df.empty:
        raise ValueError('Empty DF!')
    for post_filter in run_post_filters:
        all_df = post_filter(all_df)
    if make_hashable_df:
        all_df = all_df.applymap(lambda x: tuple(x) if isinstance(x, list) else x)

    all_df = clean_hparams(all_df)
    return all_df

def clean_hparams(df: pd.DataFrame): #TODO: adapt to our case
    # Combine/unify columns of optim/scheduler which might be present in stored params more than once
    combine_cols = [col for col in df.columns if col.startswith('model/optim') or col.startswith('model/scheduler')]
    for col in combine_cols:
        new_col = col.replace('model/', '').replace('optimizer', 'optim')
        if not hasattr(df, new_col):
            continue
        getattr(df, new_col).fillna(getattr(df, col), inplace=True)
        # E.g.: all_df.Temp_Rating.fillna(all_df.Farheit, inplace=True)
        del df[col]

    """ deprectade

    if "model/loss_function" in df.columns:
        getattr(df, "model/loss_function").fillna("mean_squared_error", inplace=True)
    if "model/hr_loss_weighting" in df.columns:
        getattr(df, "model/hr_loss_weighting").fillna(False, inplace=True)
    if "model/hr_train_loss_scale" in df.columns:
        getattr(df, "model/hr_train_loss_scale").fillna("ksec", inplace=True)
    if "model/dont_predict_sw_incident_toa" in df.columns:
        getattr(df, "model/dont_predict_sw_incident_toa").fillna(False, inplace=True)
    if "model/scale_shortwave_flux_by_incident_toa" in df.columns:
        getattr(df, "model/scale_shortwave_flux_by_incident_toa").fillna(False, inplace=True)
    if "optim/name" in df.columns and 'optim/_target_' in df.columns:
        opt_name_df = getattr(df, 'optim/_target_').apply(lambda x: x if x != x else x.split('.')[-1].lower())
        getattr(df, "optim/name").fillna(opt_name_df, inplace=True)
        del df['optim/_target_']
    if 'optim/eps' in df.columns:
        getattr(df, 'optim/eps').fillna(1e-8, inplace=True)
    if "model/padding" in df.columns:
        getattr(df, "model/padding").fillna(0.0, inplace=True)
    if "model/dilation" in df.columns:
        df['model/dilation'] = df['model/dilation'].apply(
            lambda x: x if (x != x or isinstance(x, tuple)) else (x,))
    if "normalizer/input_normalization" in df.columns:
        getattr(df, "normalizer/input_normalization").fillna("z", inplace=True)
    if "normalizer/log_scaling" in df.columns:
        getattr(df, "normalizer/log_scaling").fillna(False, inplace=True)
   """
    return df



def topk_runs(k: int = 1,
              metric: str = TOP_K_METRIC,
              lower_is_better: bool = True) -> DF_MAPPING:
    if lower_is_better:
        return lambda df: df.nsmallest(k, metric)
    else:
        return lambda df: df.nlargest(k, metric)

def topk_run_of_each_model_type(k: int = 1,
                                metric: str = TOP_K_METRIC,
                                lower_is_better: bool = True) -> DF_MAPPING:
    topk_filter = topk_runs(k, metric, lower_is_better)

    def topk_runs_per_model(df: pd.DataFrame) -> pd.DataFrame:
       
        models = df.model.unique()
    
        dfs = []
        for model in models:
            #print("top k", topk_filter(df[df.model == model]))
            dfs += [topk_filter(df[df.model == model])]
        return pd.concat(dfs)

    return topk_runs_per_model

def topk_run_of_each_tag(k: int = 1,
                                metric: str = TOP_K_METRIC,
                                lower_is_better: bool = True) -> DF_MAPPING:
    topk_filter = topk_runs(k, metric, lower_is_better)

    def topk_runs_per_tag(df: pd.DataFrame) -> pd.DataFrame:
        
        tags = df.tags.unique()
        #print("unique tags", tags)
        dfs = []
        for t in tags:
            #print("top k", topk_filter(df[df.tags == t]))
            dfs += [topk_filter(df[df.tags == t])]
        return pd.concat(dfs)

    return topk_runs_per_tag


def non_unique_cols_dropper(df: pd.DataFrame) -> pd.DataFrame:
    nunique = df.nunique()
    cols_to_drop = nunique[nunique == 1].index
    df = df.drop(cols_to_drop, axis=1)
    return df

def has_finished(run):
    return run.state == "finished"

def not_test_tag(run):
    return 'test' not in run.tags

def filter_user(run, user=USERNAME):
    return run.user.username == user

def no_model_target(run):
    return 'model/_target_' in run.config.keys()


def filter_warmup_climax(run, num_epochs=CLIMAX_WARMUP_EPOCHS):
    model_name=run.config['model/_target_'].split('.')[-1]
    # if model is climax or climax frozen 
    # warmup epochs must match       
    ret = run.config['scheduler/warmup_epochs']==num_epochs if (model_name=='ClimaX') else True
    return ret

def has_tags(run, tags=FILTER_TAGS):
    #print(run.tags)
    #print([f in run.tags for f in FILTER_TAGS])
    return np.any([f in run.tags for f in FILTER_TAGS])


str_to_run_pre_filter = {
    'has_finished': has_finished,
    'not_test_tag': not_test_tag,
    'user': filter_user,
    'warmup_epochs_climax': filter_warmup_climax,
    'no_model_target': no_model_target,
    'has_tags': has_tags,
}


str_to_run_post_filter = {
    **{
        f"top{k}": topk_runs(k=k)
        for k in range(1, 21)
    },
    'best_per_model': topk_run_of_each_model_type(k=1),
    **{
        f'top{k}_per_model': topk_run_of_each_model_type(k=k)
        for k in range(1, 6)
    },
    'unique_columns': non_unique_cols_dropper,
    'best_per_unique_tag' : topk_run_of_each_tag(k=1),
}

def mean_str(col):
    if is_numeric_dtype(col):
        return col.mean()
    else:
        return col.unique()[0] if col.nunique() == 1 else np.NaN
def std_str(col):
    if is_numeric_dtype(col):
        return col.std()
    else:
        return col.unique()[0] if col.nunique() == 1 else np.NaN

def newest(path):
    try:
        files = os.listdir(path)
        paths = [os.path.join(path, basename) for basename in files]
        return max(paths, key=os.path.getctime)
    except:
        return "Missing checkpoint."
        
    


def get_best_run_per_unique_tag(verbose=0):

    #get df with all finished runs that are not tests
    #for each unique combination of tags, get best model run id...
    #filter 
    df = get_runs_df(run_pre_filters=['has_finished', 'not_test_tag', 'user', 'has_tags'], run_post_filters='best_per_unique_tag', verbose=verbose)
    column_list=['name', 'id', 'tags', 'val/llmse_climax','dirs/work_dir']
    df_best_per_tag=df[column_list]

    # edit checkpoints paths
    
    #df_best_per_tag['checkpoint_dir']=df_best_per_tag.apply(lambda x: f"{x['dirs/work_dir']}/runs/emulator/{x['id']}/checkpoints/", axis=1)
    df_best_per_tag['checkpoint_dir']=df_best_per_tag.apply(lambda x: f"{x['dirs/work_dir']}/runs/emulator/{x['id']}/checkpoints/", axis=1)
    #print(df_best_per_tag['checkpoint_dir'])
    df_best_per_tag['checkpoint_path']=df_best_per_tag.apply(lambda x: f"{newest(x['checkpoint_dir'])}", axis=1)
    print(df_best_per_tag['checkpoint_path'])
    new_column_list=['name', 'id', 'tags', 'val/llmse_climax', 'checkpoint_path']
    
    # TODO:
    # check datamodule/seq_len -> same for all? -> think we only ran seq-to-seq for now
    # what metrics to choose best model for? currently: val/llmse_climax
    # could also be val/llrmse_wheather_bench or test_metrics -> adjust in top_k function if wanted

    df_best_per_tag=df_best_per_tag[new_column_list]
    df_best_per_tag.to_csv("scores/best_per_tag_checkpoints.csv")


def get_averages(metrics= ['llrmse_climax'], experiment = 'finetuning_emulator', test_scenarios=['ssp245'], variables= ['pr', 'tas'], test_models=['NorESM2-LM']):
    
    df = get_runs_df(run_pre_filters=['has_finished','not_test_tag', 'user'])
    tags = df.tags.unique()
   
    metrics_list=['val/llrmse_climax']
    for m in metrics:
        for cm in test_models:
            for s in test_scenarios:
                for v in variables:
                    metrics_list.append(f'test/{s}_{cm}/{v}/{m}')
                metrics_list.append(f'test/{s}_{cm}/{m}') # avgs over all vars

    # filter for finetuning experiments
    tags = [t for t in tags if experiment in t]
    print("experiment", experiment)
    print(tags)
    attrs_list=['model']
    column_list= ['name', 'id', 'tags'] + attrs_list + metrics_list
    
    df_final=df[column_list].loc[df['tags'].isin(tags)]
    
    # group by uniqe tag, get mean for each metric
    # for each unique tag get mean of stats
    df_final_means=df_final.groupby(['tags'])[metrics_list].agg(mean_str).round(3)
    df_final_stds=df_final.groupby(['tags'])[metrics_list].agg(std_str).round(3)
    df_final_means.to_csv(f"scores/average_stats_{experiment}_mean.csv")
    df_final_stds.to_csv(f"scores/average_stats_{experiment}_std.csv")

    # TODO: for readability, aggregate colums
    """
    clean_metrics_colums= [f"test/{s}/{v}" for s in test_scenarios for v in variables]

    for s in test_scenarios:
        df_final_means=df_final_means.rename(columns=lambda x: re.sub(f'{s}_.*?/', f'{s}/', x))
        df_final_stds=df_final_stds.rename(columns=lambda x: re.sub(f'{s}_.*?/', f'{s}/', x))
    df_final_means.to_csv(f"scores/average_stats_{experiment}_mean.csv")
    df_final_stds.to_csv(f"scores/average_stats_{experiment}_std.csv")
    """
    

def get_single_averages(metrics= ['llrmse_climax','mse','rmse','llrmse_wheather_bench', 'nrmse_climate_bench'], test_scenarios=['ssp245'], variables= ['pr', 'tas'], eval_metric="llrmse_climax"):

    df = get_runs_df(run_pre_filters=['has_finished', 'not_test_tag', 'user'])#,'warmup_epochs_climax',  'no_model_target'])

    tags = df.tags.unique()
 
    tags_single= [t for t in tags if 'single_emulator' in t]
   
    # filter for single emulator experiments
    df_single=df.loc[df['tags'].isin(tags_single)]
    climate_models=[]
    for d in df_single['datamodule']:
       
        climate_models.append(d['train_models'])
   
    climate_models=np.unique(climate_models)
   
    # problem: each unique tag has a different test stat as it depends on the climate model used in training!
    metrics_list_single=['val/llrmse_climax']
    for m in metrics:
        for cm in climate_models:
            for s in test_scenarios:
                for v in variables:
                    metrics_list_single.append(f'test/{s}_{cm}/{v}/{m}')
                metrics_list_single.append(f'test/{s}_{cm}/{m}') # avgs over all vars
    # readability
    
    # for each unique tag get mean of stats
    df_single_final_means=df_single.groupby(['tags'])[metrics_list_single].agg(mean_str).round(3)
    df_single_final_stds=df_single.groupby(['tags'])[metrics_list_single].agg(std_str).round(3)

    df_single_final_means.to_csv("scores/average_stats_single_mean.csv")
    df_single_final_stds.to_csv("scores/average_stats_single_std.csv")

    """
    # now reorder and clean so we get nicer table
    # first column is climate model
    models=["unet", "conv_lstm", "climax", "climax_frozen"]
    columns_eval=["climate_model"]+[f"{m}/{v}" for m in models for v in variables]
    df_eval = pd.DataFrame(columns=columns_eval)
    
    print(df_single_final_means.index)
    for cm in climate_models:
        for m in models:
            new_indexes=df_single_final_means.loc[cm in df_single_final_means.index].loc[m in df_single_final_means.index]
            print(new_indexes)


    """
    """
    # TODO: for readability, aggregate colums
    clean_metrics_colums= [f"test/{s}/{v}" for s in test_scenarios for v in variables]
    for s in test_scenarios:
        print(s)
        df_single_final_means=df_single_final_means.rename(columns=lambda x: re.sub(f'{s}_.*?/', f'{s}/', x))
        df_single_final_stds=df_single_final_stds.rename(columns=lambda x: re.sub(f'{s}_.*?/', f'{s}/', x))
    df_single_final_means.to_csv("scores/average_stats_single_mean_clean.csv")
    df_single_final_stds.to_csv("scores/average_stats_single_std_clean.csv")
    """





if __name__=="__main__":

    metrics= ['llrmse_climax','mse','rmse','llrmse_wheather_bench']#, 'nrmse_climate_bench']
    variables= ['pr', 'tas']
    test_scenarios=['ssp245']
    finetuning_model=['NorESM2-LM']
    super_models=["MPI-ESM1-2-HR", "AWI-CM-1-1-MR", "NorESM2-LM", "FGOALS-f3-L", "EC-Earth3", "BCC-CSM2-MR"]
  
    #df_best_per_exp = get_best_run_per_unique_tag()
    get_single_averages(metrics=metrics, test_scenarios=test_scenarios, variables=variables)
    get_averages(experiment='finetuning_emulator', metrics=metrics, test_scenarios=test_scenarios, test_models=finetuning_model)
    get_averages(experiment='super_emulation', metrics=metrics, test_scenarios=test_scenarios, test_models=super_models)
  