# Data constants


LON = 96
LAT = 144
NUM_LEVELS = 1
SEQ_LEN = 12
INPUT4MIPS_TEMP_RES = 'mon'
CMIP6_TEMP_RES = 'mon'
TEMP_RES ='mon'
INPUT4MIPS_NOM_RES = 'map_250_km'
CMIP6_NOM_RES = '250_km' #TODO: not allow different resolutions
SEQ_LEN_MAPPING ={
    'mon': 12
}

DATA_DIR = '/home/mila/v/venkatesh.ramesh/scratch/causal_data' # MILA cluster
# Model : (historical_obe_files, future_obe_files)
OPENBURNING_MODEL_MAPPING = {
    "other" : ("anthro-fires", "anthro-fires"),
    "CESM2-WACCM": ("no-fires", "no-fires" ),
    "CNRM-ESM2-1": ("anthro-fires", "anthro-fires"),
    "CMCC-ESM2": ("no-fires", "no-fires" ),
    "EC-Earth3-Veg": ("anthro-fires", "anthro-fires"),
    "EC-Earth3-Veg-LR": ("anthro-fires", "anthro-fires"),
    "MPI-ESM1-2-LR": ("anthro-fires", "anthro-fires"),
    "NorESM2-LM": ("no-fires", "no-fires" ),
    "NorESM2-MM": ("no-fires", "no-fires" ),
    "GFDL-ESM4": ("no-fires", "no-fires" ),
    "TaiESM1": ("anthro-fires", "all-fires"),
    "CESM2": ("anthro-fires", "all-fires"),
    "MRI-ESM-2.0": ("anthro-fires", "all-fires")
}

ORIGINAL_OPENBURNING_MODEL_MAPPING = {
    "other" : ("all-fires", "all-fires"),
    "CESM2-WACCM": ("no-fires", "no-fires" ),
    "CNRM-ESM2-1": ("anthro-fires", "anthro-fires"),
    "CMCC-ESM2": ("no-fires", "no-fires" ),
    "EC-Earth3-Veg": ("anthro-fires", "anthro-fires"),
    "EC-Earth3-Veg-LR": ("anthro-fires", "anthro-fires"),
    "MPI-ESM1-2-LR": ("anthro-fires", "anthro-fires"),
    "NorESM2-LM": ("no-fires", "no-fires" ),
    "NorESM2-MM": ("no-fires", "no-fires" ),
    "GFDL-ESM4": ("no-fires", "no-fires" ),
    "TaiESM1": ("anthro-fires", "all-fires"),
    "CESM2": ("anthro-fires", "all-fires"),
    "MRI-ESM-2.0": ("anthro-fires", "all-fires")
}
NO_OPENBURNING_VARS = ['CO2_sum']
