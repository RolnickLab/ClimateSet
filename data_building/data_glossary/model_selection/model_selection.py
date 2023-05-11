#TODO check out NorESM why have we left out some of the scenarios
#TODO expand the list of models for the four new ones, even if they wont be selected
import pandas as pd
from pathlib import Path
from proposed_models import ESGF_MODELS

def select_models():
    """ Prints and saves a subset of CMIP6 models, based on a certain set
    of evaluation criteria.
    """
    # list the criteria
    mon_frequency = True
    min_resolution = 250
    scenarios = ["ssp126", "ssp245", "ssp370", "ssp585"]
    min_ensemble = 3
    allow_diff_res = True
    ensembles = {"ssp126": min_ensemble, "ssp245": min_ensemble, "ssp370": min_ensemble, "ssp585": min_ensemble}
    # list of models where only the resolution was changed (if atmo models have been changed -> diff model)
    # always take the first one
    diff_res = [("AWI-CM-1-1-MR", "AWI-CM-1-1-LR"), #AWI. first one has monthly data available
                ("CNRM-CM6-1", "CNRM-CM6-1-HR"), #CNRM. first one is LR and has more ensembles
                ("EC-Earth3-Veg-LR", "EC-Earth3-Veg"), #EC. HR is not available on the node
                ("MPI-ESM1-2-LR", "MPI-ESM1-2-HR"), #MPI. LR has much more runs
                ("NorESM2-LM", "NorESM2-MM"), #NorESM. LM so we dont need to aggregate
                ("HadGEM3-GC31-LL", "HadGEM3-GC31-MM"), #HadGEM3. LL has more scenarios
                ]

    # load the csv
    all_models = pd.read_csv("scenarioMIP_models.csv", sep=", ", header=0, engine="python")
    #print(all_models.columns)
    # for model in all_models.source_id:
    #     print(model)

    # filter for criterias
    # frequency, resolution, availability esgf
    sel_models = all_models[(all_models["mon_frequency"] == mon_frequency)
                            & (all_models["nominal_resolution_km"] <= min_resolution)
                            & (all_models["esgf_available"] == True)]

    # resolution 2.0, remove model if two are the same model with diff resolutions
    # if you find the first of the tuple, remove the second one from the list
    sel_models_ids = list(sel_models.source_id)
    if not allow_diff_res:
        for (keep_model, drop_model) in diff_res:
            if (keep_model in sel_models_ids) and (drop_model in sel_models_ids):
                sel_models = sel_models.loc[sel_models.source_id != drop_model]

    # scenarios
    def contains_all_scenarios(model_scenarios, desired_scenarios):
        return all([desired_scenario in model_scenarios for desired_scenario in desired_scenarios])
    contains_scenarios_mask = sel_models.apply(lambda row: contains_all_scenarios(row["scenarios"], scenarios), axis=1)
    sel_models = sel_models[contains_scenarios_mask]

    # print models. ensembles, node available
    print("\nModel Selection (monthly, esgf_available, scenarios, resolution), total {}:".format(len(sel_models)))
    print(sel_models[["source_id", "nominal_resolution_km", "scenarios", "num_ensemble_members", "data_node_available"]])

    # ensembles (in theory)
    # example
    def contains_min_ensembles(model_scenarios, model_ensembles, desired_scenario_ensembles):
        model_scenarios = model_scenarios.split(' ')
        model_ensembles = [int(num_ensemble) for num_ensemble in model_ensembles.split(' ')]
        model_scenario_ensembles = dict(zip(model_scenarios, model_ensembles))
        return all([model_scenario_ensembles[scenario] >= min_ensemble for scenario, min_ensemble in desired_scenario_ensembles.items()])

    contains_ensembles_mask = sel_models.apply(lambda row: contains_min_ensembles(row["scenarios"], row["num_ensemble_members"], ensembles), axis=1)
    sel_ens_models = sel_models[contains_ensembles_mask]

    # print models again.
    print("\nModel Selection (minimum ensemble members), total {}:".format(len(sel_ens_models)))
    print(sel_ens_models[["source_id", "nominal_resolution_km", "scenarios", "num_ensemble_members", "data_node_available"]])

    # add contains_min_ensembles to list
    sel_models["contains_min_{}_ensemble".format(min_ensemble)] = contains_ensembles_mask

    # store file: names of the models, resolution, scenarios, ensembles, node_availability
    filepath = Path("selected_scenarioMIPs.csv")
    sel_models[["source_id", "nominal_resolution_km", "scenarios", "num_ensemble_members", "data_node_available"]].to_csv(filepath)

def compare_model_list(proposed_models):
    """ Compare if a given list of models is contained in the csv provided by this repo.
    Prints differences.

    Args:
        proposed_models (list): List of models that should be contained in the csv provided here
    """
    # read csv
    all_models = pd.read_csv("scenarioMIP_models.csv", sep=", ", header=0, engine="python")
    actual_models = set(all_models.source_id)
    proposed_models = set(proposed_models)

    print("Proposing {} models.".format(len(proposed_models)))
    print("Currently having {} models.".format(len(actual_models)))

    print("Models that are currently not contained:")

    print(proposed_models - actual_models)


if __name__ == '__main__':
    # load list from proposed_models.py
    proposed_models = ESGF_MODELS
    compare_model_list(proposed_models)
    select_models()

# Describing the esgf look-up process:
# 1. choose model from CMIP6 list
# 2. Choose it (source-id) in the portal
# 3. Check if it has monthly data available (Frequency: mon)
# 4. Choose a variable (Variable: tas)
# 5. Check if it has several resolutions
# 6. Check how many ensembles there are for each scenario
# -> Each scenario is listed in Experiment ID. The number you find behind each ID -> number of ensembles (variant label)

# notes:
# - the nominal_resolutions in the list -> resolution available for monthly data!
#   there might be more resolutions available for different kind of data
# - I might have left out some of the scenarios (special covid ones etc., check with NorESM)
# - how i checked for missing models: check all ssp scenarios, choose monthly, choose tas,
#   and check out which models appear in the Source ID category
# - when comparing the two lists, the following differences came up:
#   {'BCC-ESM1', 'UKESM1-ice-LL', 'CESM2-FV2', 'GISS-E2-1-G-CC'}
#   checking those separately:
#   - BCC-ESM1 -> contains only SSP370 scenarios (include anyway)
#   - UKESM1-ice-LL -> only withism scenarios?? I think we can really drop this one
#   - CESM2-FV2 -> contains only ssp370 and ssp585 (include anyway)
#   - GISS-E2-1-G-CC -> contains ssp126, ssp245, ssp460 and ssp585-bgc - should be included (250km)
