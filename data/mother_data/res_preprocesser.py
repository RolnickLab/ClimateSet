# preprocesser, resolution is preprocessed
# holt sich daten von PREPROCESSED, passt resolution an, und steckt sie in LOAD
from typing import List, Tuple, Dict

# TODO MOVE THIS to utils/constants or sth like that
# indicating which aggregation and interpolation methods this class can handle
AGGREGATION_METHODS = ["mean", "median", "minimum", "max", "total"] # and more ...
INTERPOLATION_METHODS = ["repeat"] # only one for the beginning
RES_HIERARCHY = ["TODO"] # hierarchy of resolutions

# class
# params:
# data_source: which data should be processed for desired resolutions
# res_params: RESOLUTION parameters, aggregation params (tuple with variables)
# data_store: where end product is stored (in correct resolution)

# functions / tasks:

# - check resolutions available
# - derive which methods to use in order to get the same resolution for everything
# - aggregate and interpolate everything and store the results

# move out:
# - aggregation funcs in all dimensions
# - interpolation funcs in all dimensions

# OPEN
# TODO
# - scale for all models / ensembles / experiments
# - scale for all dimensions, not only temporal resolution
class ResProcesser:
    def __init__(self, resolutions: Tuple[int, int, int, int],
        aggregations: Dict[str, str], interpolations: Dict[str, str],
        source: Path, store: Path,
        models: List[str] = [], ensemble: List[str] = [],
        experiments: List[str] = []):
        """ Creates the desired resolutions of the data (same resolution for all).
        Parameters:
            resolutions (tuple): temporal, vertical, longitudinal and latitudinal desired resolution
            aggregations (dict<str, str>): mapping which aggregation should be used for which variable
            interpolations (dict<str, str>): mapping which interpolation should be used for which variable
            source (pathlib.Path): Where the data comes from (root before model dirs start)
            store (pathlib.Path): Where the resolution-processed data is stored (root before model dirs start)
            models (list<str>): Models for which the resolution is adapted. Default is all existing models.
            ensembles (list<str>): Ensemble members of models for which res is adapted. Default is all existing members.
            experiments (list<str>): Experiments/SSP secenarios for which res is adapted. Default is all existing experiments.
        """
        self.resolutions = resolutions
        self.aggre = aggregations
        self.inter = interpolations
        self.source = source
        self.store = store
        self.models = models
        self.ensembles = assembles
        self.experiments = experiments
        # put more stuff here if necessary

    # TODO scale this: for all models/assembles/scenarios/ (adapt efficiently)
    def create_resolutions(self, vars: List[str], res: str):
        """ Method that looks at a set of variables and creates the same resolutions
        for all of them.
        Assuming right now: All data has the same year-duration!
        Parameters:
            vars (list<str>]): Variables that should get the same resolution
            res (str): Which resolution all the vars should have
        """
        # check if the res already exists for all variables
        existance_list = self.check_existing_res(vars, res)

        # if all vars exist already: copy everything?

        # go through the vars-existance list
            # if exists:
                # copy resolution to LOAD_DATA
            # else:
                # pair with the right aggregation/interpolation method
                # create path where the new data will live: new_res_path (self.store + new_res + var)
                # apply agg/int method on variable & store results:
                # self.__apply_res_func__(func_str, var_path, new_res_path)

        raise NotImplementedError

    def __check_existing_res__(self, vars: List[str], res: str) -> List[int]:
        """ Checks if the resolution already exists for the variables.
        Parameters:
            vars (list<str>]): Variables that should get the same resolution
            res (str): Which resolution all the vars should have
        Returns: a list for the vars:
            0 := variable does not exist at all
            1 := variable exists only in lower res
            2 := variable exists only in higher res
        """
        # go into var directory and check if the res exists (and is not empty)
        # store if it exists and if it exists only in higher or lower resolution
        raise NotImplementedError

    def __apply_res_func__(self, func_str: str, var_path: Path, store_path: Path):
        """ Applies the right aggregation and interpolation function on a given
        variable (location). Stores it in new dir (self.store).
        Parameters:
            func_str (str): string of the aggregation/interpolation function
            var_path (pathlib.Path): where the lower/higher res var is stored right now
            store_path (pathlib.Path): where the new resolution variable will be stored
        """
        # create real aggregation / interpolation object (lives in utils)
        # give this during init the var_path and where to store the new data (self.store) [new_res_path]
        raise NotImplementedError
