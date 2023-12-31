# ***ClimateSet*** - : A Large-Scale Climate Model Dataset for Machine Learning

## Official implementation for the data downloader

Abstract: *Climate models have been key for assessing the impact of climate change and simulating future climate scenarios depending on humanity’s socioeconomic choices.

The machine learning (ML) community has taken an increased interest in supporting climate scientists’ efforts on various tasks such as climate emulation, downscaling, and prediction tasks. Many of those tasks have been addressed on datasets created with single climate models. However, both the climate science and the ML communities have communicated that to address those tasks at scale, we need large, consistent, and ML-ready climate model datasets. Here, we introduce ClimateSet, a dataset containing the inputs and outputs of 36 climate models from the CMIP6 and Input4MIPs archives. In addition, we provide a modular dataset pipeline for retrieving and pre-processing additional climate models and scenarios. 

We showcase the potential of our dataset by using it as a benchmark for ML-based climate emulation. We gain new insights about the performance and generalization capabilities of the different ML models by analyzing them across different climate models. Furthermore, the dataset is used to train an ML model on all 36 climate models, i.e. not only one specific climate model but the entire CMIP6 archive is emulated. With this, we can quickly project new climate scenarios capturing the inter-model variability of climate models - similar to the “averaged climate scenarios” provided to policymakers. We believe ClimateSet will create the basis needed for the ML community to tackle climate model related tasks at scale.*

## Usage 
### Create an enviroment

To setup the environment for the downloader, we use python>=3.10. 

Follow the following steps to create the environment:

```bash
python -m venv env_downloader
source env_downloader/bin/activate
pip install -r requirements_downloader.txt
```

### Downloader

To ownload data, you can run the downloader module with a desired config specifying what climate models, experiments and variables you want to download data for.
You can also specify a list of ensemble members or a maximum number of ensemble members per climate model. 

The following parameters for the downloader are the default:

```python
    experiments: List[str] = [
            "historical",
            "ssp370",
            "hist-GHG",
            "piControl",
            "ssp434",
            "ssp126",
        ],  # sub-selection of ClimateBench default
    vars: List[str] = ["tas", "pr", "SO2", "BC"],
    data_dir: str = os.path.join(ROOT_DIR, "data"),
    max_ensemble_members: int = 10, # if -1 take all available models
    ensemble_members: List[str] = None # preferred ensemble members used, if None not considered
    overwrite=False,  # flag if files should be overwritten
    download_biomassburning=True,  # get biomassburning data for input4mips
    download_metafiles=True,  # get input4mips meta files
    plain_emission_vars=True, # specifies if plain variabsle for emissions data are given and rest is inferred or if variables are specified

```

To run the downloader, create a config in which you specify what models you want to download data for. If no model is given, we assume you only want to download "input4mips" data.
You can override any of the downloader kwargs in this onfig.
As an example, to download the filese needed to create the core dataset, see this [example downloader config](data_building/configs/downloader/core_dataset.yaml).
To run the downloader with this default example, excecute the following command:

 ```bash
 python -m data_building.builders.downloader --cfg data_building/configs/downloader/core_dataset.yaml
 ``` 

Feel free to change create new configs to change the variables and experiments being downloaded.

Per default, will expect plain variable names for the emissions variables for esier usage and will infer other variable names for building the full dataset. For example, passing on ```BC``` will dowlnoad data for ```BC_em_anthro```,  ```BC_em_AIR_anthro``` and ```BC_em_openburning```, and if biomassburning (```BC```) and percentage files if desired.
If you wish to change this behavior and be specific about what variables to download, pass on ```plain_emission_vars: False``` in your config.

Per default, the downloader will create two subfolders in your specified directory, one named ```raw``` containing unprocessed ```input4mips``` and ```CMIP6``` files and one named ```meta``` containing files concerning fire emission data and othter files needed to achieve consistent preprocessing emission data. 

Per default, the downloader will create a structure that already specifies most of the needed meta information of each file like nominal resolution, temporal resolution, experiment, source etc. Please do not change this structure if you wish to be using the preprocessing module out of the box.

#### Available Variables

To check what CMIP6 variables are available, you can refer to this (table)[data_building/data_glossary/mappings/variableid2tableid.csv] in our data glossary mapping long variable names to ids and units. Please use the ids to prompt the downloader.

We provide more detailed information on all variables available to our example model ```NorESM2-LM``` in our (data glossary)[data_building/data_glossary/] as well with a collection of (helpful links)[data_building/data_glossary/helpful_links.txt] to get you started.

## Development

This repository is currently under active development and you may encounter bugs with some functionality. 
Any feedback, extensions & suggestions are welcome!
