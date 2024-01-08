# init script to establish first data basis
from pathlib import Path
from typing import List, Tuple, Dict

# from data_generation.generators.mother_data import *
from data_building.parameters.esm_params import MOTHER_PARAMS
from data_building.parameters.data_paths import RAW_DATA, PROCESSED_DATA, LOAD_DATA

# interacts with:
# mother_params,
# inits and executes downloader --> stored in raw
# preprocesses downloaded data [raw_preprocesser] --> stored in preprocessed
# first resolution processing [rew_preprocesser] --> stored in load

# no classes, just script

# ATTENTION
# during aggregation: take care of units and how they change (should be handled within raw_preprocesser)


def birth(
    models: List[str],
    scenarios: List[str],
    years: List[int],
    in_vars: List[str],
    out_vars: List[str],
    resolutions: Tuple[int, int, int, int],
    grid: str,
    aggregations: Dict[str, str],
    interpolations: Dict[str, str],
    raw_path: Path,
    res_path: Path,
    load_path: Path,
    **kwargs
):
    """Creates the initial climate data that is needed. Starts downloader,
    makes initial preprocessing and default resolution processing. Data can
    afterwards be loaded via Loader() or run.py. Prints if process was successful.

    Parameters:
        models (list<str>): String indicating which models should be used
        scenarios (list<str>): String indicaitng which scenarios should be considered
        years (list<int>): List of ints - the years for which the data is retrieved
        in_vars (list<str>): List of model input variables that should be downlaoded
        out_vars (list<str>): List of targets / output from models that should be downloaded
        resolutions (tuple<int>): temporal, vertical, longitude and latitude resolution
        grid (str): Which type of grid is used
        agreggations (dict<str, str>): all in_vars and out_vars must be contained
            in this dictionary. The value decodes which aggregation method should be used
            in case this variable must be aggregated.
        interpolations (dict<str, str>): Same as before, just interpolation
        raw_path (Path): Path where raw data is downloaded into
        res_path (Path): Path where the processed data is stored
        load_path (Path): Here, the data is ready for being loaded
    """
    # DOWNLOAD DATA
    print("Starting to download data ...")
    # init downloader
    # downloader = Downloader()
    # execute downloader
    # downloader.download_from_model()
    print("... data was downloaded successfully.")

    # PREPROCESS DATA
    print("Starting to process the data ...")
    # init preprocesser
    # TODO

    # execute preprocesser
    # (might be different funcs)

    print("... data was preprocessed successfully.")

    # CREATE RIGHT RESOLUTIONS
    print("Starting to get the right resolutions ...")
    # init resolution processer

    # do stuff

    print("... data is now available in the right resolutions.")
    print("\nClimate data is now available in {}.".format("/TO/DO/"))


# params (dict): Dictionary containing parameters (stored in mother_params.py)
#     such as resolutions, variables, grid type, years and aggregation type.
# paths (dict): where the data is stored: raw data, preprocessed data and
#     data that can be immediately loaded
if __name__ == "__main__":
    # load mother_params

    # and put all of this into birth()
    birth(
        **MOTHER_PARAMS, raw_path=RAW_DATA, res_path=PROCESSED_DATA, load_path=LOAD_DATA
    )
