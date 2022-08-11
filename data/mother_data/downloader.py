
# TODO argparse with parameter where data should be stored

# Downloads data of two types:

# maybe one class "downloader"
# two subclasses for input / output

# Input of climate models (different source)
    # "really raw input": normal input variables
    # "model raw input": CO2 mass and other specific variables: assumptions within the model, depending on the SSP path
    # (different preprocessing, normalize (CO2 mass minus baseline), see ClimateBench)

# Predictions / output of climate models (different source)

# Storage:
# Raw-raw data can be deleted after preprocessing (preprocessing always with highest resolution)

# Resolution:
# highest possible resolution [??]

# class Downloader
# source: link whatever
# storage: where to store (data_paths)
# params: mother_params
# ATTENTION: download always highest resolution, res params are only used later on during res_preprocesser

# TODO: where to store sources links for data

# no returns, but communicates to user where stuff was stored and if sucessful
# hand over exceptions if there are any problems, or print commands
