# preprocesser, resolution is preprocessed
# TODO import python typing
# holt sich daten von PREPROCESSED, passt resolution an, und steckt sie in LOAD

# indicating which aggregation and interpolation methods this class can handle
AGGREGATION_METHODS = ["mean", "median", "minimum", "max", "total"] # and more ...
INTERPOLATION_METHODS = ["repeat"] # only one for the beginning

# class
# params:
# data_source: which data should be processed for desired resolutions
# res_params: RESOLUTION parameters, aggregation params (tuple with variables)
# data_store: where end product is stored (in correct resolution)

# functions / tasks:

# - check resolutions available
# - derive which methods to use in order to get the same resolution for everything
# - aggregate and interpolate everything and store the results

# maybe here, maybe move out:
# - aggregation funcs in all dimensions
# - interpolation funcs in all dimensions

# Attention:
# How to store different data resolutions and variables: FIRST Variables, then on second level resolutions

class ResProcesser:
    def __init__(self, resolutions: Tuple[int, int, int, int],
        aggregations: Dict[str, str], interpolations: Dict[str, str],
        source: Path, store: Path):
        """ Creates the desired resolutions of the data (same resolution for all).
        Parameters:
            resolutions (tuple): temporal, vertical, longitudinal and latitudinal desired resolution
            aggregations (dict<str, str>): mapping which aggregation should be used for which variable
            interpolations (dict<str, str>): mapping which interpolation should be used for which variable
            source (pathlib.Path): Where the data comes from
            store (pathlib.Path): Where the resolution-processed data is stored
        """
        self.resolutions = resolutions
        self.aggre = aggregations
        self.inter = interpolations
        self.source = source
        self.store = store
        # TODO ...

    def create_resolutions(self):
        """ Method that creates differente resolutions
        """
        # get data from self.source
        # TODO do stuff with it, replicate it
        # store data in self.store
        raise NotImplementedError
