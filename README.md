# Causalpaca: Emulator Models

This branch contains the code for running the climate model emulation benchmark experiments on ClimateSet. 
Here we provide a quick documentation on installation, setup and a quickstart guide to reproduce our experiments.

## Getting started
### Getting the data
To download and store the preprocessed dataset ready for training locally, execute the following command:

```bash
bash download_climateset.sh
```

You should now see a newly created directory called "Climateset_DATA" containing inputs and targets. This folder will be referenced within the emulator pipeline.

### Setting up the environment
The environment set-up happens separately from the dataset set-up since a different set of packages is needed for running the emulators. 
For a minimal list of all packages needed see [requirements_minimal](minimal_requirements.txt).
To reproduce the environment in which most experiments were conducted, use the [requirements_all file](requirements_all.txt). 
Finally, setup the emulator module.

```python
python -m venv env_emulator
source env_emulator/bin/activate
pip install -r requirements_emulator.txt
pip install -e .
```

Needed Packages:
pytorch, pytorh lightning, wandb, dask, xarray, segmentation models pytorch


## Structure
![Visualization of the codebase](./diagram.svg)


The repository is inspired by [ClimART](https://github.com/RolnickLab/climart/tree/main) and PL+Hydra template implementation [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)


# Running a model
To run the model, edit the [main config](emulator/configs/main_config.yaml) to fit what you want to run. 
Executing the run.py script plain will use the main config. 

```python
python emulator/run.py # will run with configs/main_config.yml
```
To exectute one of the preset experiments or to run your own experiments you can create and pass on experiment configs:

```python
python emulator/run.py experiment=test # will run whatever is specified by the configs/experiment/test.yml file
```

You can make use of the [experiment template](emulator/configs/experiment/templatte.yaml).

## Logging locally
To run your model locally can either use no logger at all or tell the wandb logger to log your experiments offlien you will need to overwrite the default logger (wandb) and set it to offline:

Option 1: Setting loger to none
``` yaml
# In your experiment_name.yml
defaults:
  - override /logger: none.yaml
  ...
 ```
Option 2: Use wandb offline
``` yaml
# In your experiment_name.yml
logger:
    wandb:
        project: "YOUR_PROJECT"
        group: "YOUR_GROUP"
        offline: True
 ```

## Logging to wandb

To run with logging to wandb, you can simply use the wandb logger overwriting the project and group with your project and group and set offline to False (default).

``` yaml
# In your experiment_name.yml
logger:
    wandb:
        project: "YOUR_PROJECT"
        group: "YOUR_GROUP"
        offline: False #default
 ```
## How to add new models
You can add new models in `emulator/src/core/models`. Each model should inherit from the Basemodel class you can find in `basemodel.py`. Add a new config file for your model in `emulator/config/models/`.
