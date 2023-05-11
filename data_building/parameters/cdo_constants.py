SPAT_REMAPPING_ALGS = [
    # simple operator
    "remap",
    # specific operators
    "remapbil",
    "remapbic",
    "remapnn",
    "remapdis",
    "remapcon",
    "remapcon2",
    "remaplaf",
]

# creating remap weights
SPAT_REMAPPING_WEIGHTS = {
    "remapbil": "genbil",
    "remapbic": "genbic",
    "remapnn": "gennn",
    "remapdis": "gendis",
    "remapcon": "gencon",
    "remapcon2": "gencon2",
    "remaplaf": "genlaf",
}

TEMP_INTERPOLATION_ALGS = [
    "inttime",
    "intntime"
    "intyear",

]

TEMP_AGGREGATION_ALGS = [

]

# for later usage
LEVS_INTERPOLATION_ALGS = [
    "remapeta"
    "ml2pl",
    "ml2hl",
    "ap2pl",
    "gh2hl"
]

VAR3D_INTERPOLATION_ALGS = [
    "intlevel",
    "intlevel3d",
    "intlevelx3d"
]
