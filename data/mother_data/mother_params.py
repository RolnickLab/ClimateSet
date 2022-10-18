### DOWNLOADER PARAMS ##########################################################

# these resolutions are stored in RESOLUTION
TEMP_RES = 0
VERT_RES = 0
LON_RES = 0
LAT_RES = 0

# resolution of the end-data-product
RESOLUTION = (TEMP_RES, VERT_RES, LON_RES, LAT_RES)

# list of years that are considered for the data
YEARS = [0]


""" distinction not necessary for the mother as we are first just providing data not designing the loader yet, and a lookup table to check where to downloda what from anyway
# variables used as input for the climate model
IN_VARS = []

# predicted / target variables of the climate model
OUT_VARS = []
# suggestion charlie
VARS = ["nan"]
# Julia: Birth has three steps: downloading, preprocessing, creating the different resolutions
# and we already need to distinct between in_vars and out_vars for that
"""
IN_VARS = ["nan"]
OUT_VARS = ["nan"]

# scenarios
SCENARIOS = ["nan"]

# model
MODELS = ["nan"]

# number of esemble members to be considered
NUM_ESEMBLE = 1

# which type of grid
GRID = "grid"

### RAW PROCESSER PARAMS #######################################################
# you will see after downloading

### RESOLUTION PROCESSER PARAMS ################################################

# THIS must be moved somewhere else, because it's not static
# tuple of "means" of preprocesser for each variable, e.g.
# [("CO2", "mean"), ["CH4", "median"]
CHOSEN_AGGREGATIONS = ["MeanAggregation", "MinAggregation", "MaxAggregation", "InstAggregation"]
# TODO communicate to other persons which data structure etc you use here
CHOSEN_INTERPOLATIONS = {"nan"}
# TODO create a fixed list for all vars: which aggregation and interpolation

### ALL PARAMS IN DICT #########################################################
MOTHER_PARAMS = {
    "models": MODELS,
    "scenarios": SCENARIOS,
    "years": YEARS,
    "in_vars": IN_VARS,
    "out_vars": OUT_VARS,
    "resolutions": RESOLUTION,
    "grid": GRID,
    "aggregations": CHOSEN_AGGREGATIONS,
    "interpolations": CHOSEN_INTERPOLATIONS,
}
