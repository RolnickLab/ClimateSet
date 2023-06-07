# Emulator Models

## Overview
This readme should help you to get started on running and implementing emulator models for causalpaca.

## Getting started
The environment set-up happens separately from the dataset set-up since a different set of packages is needed for running the emulators. 
For a minimal list of all packages needed see [requirements_minimal](requirements/minimal_requirements.txt).
To reproduce the environment in which most experiments were conducted, use the [requirements_all file](requirements/requirements_all.txt). 
Finally, setup the emulator module.

```python
python -m venv env_emulator
source env_emulator/bin/activate
pip install -r requirements/requirements_emulator.txt
pip install -e .
```

Needed Packages:
pytorch, pytorh lightning, wandb, dask, xarray, segmentation models pytorch


## Structure
![Visualization of the codebase](../diagram.svg)

├── configs
│   ├── experiments
│   ├── hparams_search
│   ├── local
│   ├── logger
│   ├── models
│   ├── optimizers
│   └── main_config.yaml
│
├── results
│   ├── figures
│   ├── outputs
│   ├── tables
│   └── tuning
│
├── src
│   ├── core
│   │   ├── callbacks
│   │   └── models
│   │       ├── baselines
│   │       ├── causal_emulator
│   │       ├── sota
│   │       └── basemodel.py
|   |
│   ├── losses.py
│   ├── metrics.py
│   └── optimizers.py
│
├── tests
├── LICENSE
├── README.md
├── requirements
    ├── requirements_minimal.txt
    └── equirements_all.txt
├── run.py
└── setup.py

The repository is inspired by the PL+Hydra template implementation [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)

# Running a model
To run the model, edit the [main config](configs/main_config.yaml) to fit what you want to run. 
Executing the run.py script plain will use the main config. 

```python
python run.py # will run with configs/main_config.yml
```
To exectute one of the preset experiments or to run your own experiments you can create and pass on experiment configs:

```python
python run.py experiment=test # will run whatever is specified by the configs/experiment/test.yml file
```

You can make use of the [experiment template](configs/experiment/templatte.yaml).

# Running a model
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
        group: "YOUR GROUP"
        offline: True
 ```

## Logging to wandb

To run with logging to wandb, you can simply use the wandb logger overwriting the project and group with your project and group and set offline to False (default).

``` yaml
# In your experiment_name.yml
logger:
    wandb:
        project: "YOUR_PROJECT"
        group: "YOUR GROUP"
        offline: False #default
 ```
## How to add new models
You can add new models in `src/core/models`. Each model should inherit from the Basemodel class you can find in `basemodel.py`. Add a new config file for your model in `config/models/`.

