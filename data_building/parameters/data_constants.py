import pandas as pd

DATA_CSV = pd.read_csv("data_building/parameters/selected_scenariosMIPs.csv")


META_ENDINGS_PRC = [
    "_percentage_AGRI",
    "_percentage_BORF",
    "_percentage_DEFO",
    "_percentage_PEAT",
    "_percentage_SAVA",
    "_percentage_TEMF",
]
META_ENDINGS_SHAR = ["_openburning_share"]
