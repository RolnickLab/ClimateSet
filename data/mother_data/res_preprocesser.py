# preprocesser, resolution is preprocessed

# holt sich daten von PREPROCESSED, passt resolution an, und steckt sie in LOAD

# indicating which aggregation and interpolation methods this class can handle
AGGREGATION_METHODS = ["mean", "median", "total"] # and more ...
INTERPOLATION_METHODS = [] # only one for the beginning

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
