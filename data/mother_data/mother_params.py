### DOWNLOADER PARAMS ##########################################################

# resolution of the end-data-product
RESOLUTION = []

# these resolutions are stored in RESOLUTION
TEMP_RES
VERT_RES
LON_LAT_RES

# list of years that are considered for the data
YEARS = []

# variables used as input for the climate model
IN_VARS = []

# predicted / target variables of the climate model
OUT_VARS = []

# scenarios
SCENARIOS = []

# model
MODELS = []

# which type of grid
GRID = []

### RAW PROCESSER PARAMS #######################################################
# you will see after downloading

### RESOLUTION PROCESSER PARAMS ################################################


# THIS must be moved somewhere else, because it's not static
# tuple of "means" of preprocesser for each variable, e.g.
# [("CO2", "mean"), ["CH4", "median"]
CHOSEN_AGGREGATIONS = []
# TODO communicate to other persons which data structure etc you use here
CHOSEN_INTERPOLATIONS = []
