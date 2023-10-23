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

```

To run the downloader, create a config in which you specify what models you want to download data for. If no model is given, we assume you only want to download "input4mips" data.
You can override any of the downloader kwargs in this onfig.
For example, see this [example downloader config](data_building/configs/downloader/default_config.yaml).
To run the downloader with this default example, excecute the following command:

 ```bash
 python -m data_building.builders.downloader --cfg data_building/configs/downloader/default_config.yaml
 ``` 

Feel free to change create new configs to change the variables and experiments being downloaded.


## Development

This repository is currently under active development and you may encounter bugs with some functionality. 
Any feedback, extensions & suggestions are welcome!
