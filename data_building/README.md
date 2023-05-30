# Overview


## Data Description
This folder contains useful information for understanding and checking the data. No scripts.


## Mother Data

### Birth
#todo

### Downloader
Class to handle the downloading and appropriate storing of the data.
Required are experiment and variable names when initializing. All other specifications have a default values that can be overwritten.

```python
from causalpaca.data_building.builders.downloader import Downloader
# define a list of variables to download, inut4mips and CMIP vars can be mixed
vars = ['tas', 'CO2_em_anthro']
# define a list of experiments for which the data should be downloaded

downloader = Downloader(experiments=experiments, vars=vars)

# CMIP data (target vars)
downloader.download_from_model(
            project="CMIP6",
            default_frequency="mon",
            default_version="latest",
            default_grid_label="gn")

# input4mips data (input vars)
downloader.download_raw_input(
        project="input4mips",
        institution_id='PNNL-JGCRI',
        default_frequency="mon",
        default_version="latest",
        default_grid_label="gn")

```


### Core-dataset params
#todo should contain the final default core-data setting, should be given to downloader in build_core_dataset.py

### Raw Preprocesser

### Res Prerocessor
