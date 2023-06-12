# Causalpaca: Emulator Models

This branch contains the code for running the climate model emulation benchmark experiments on ClimateSet. 
Here we provide a quick documentation on installation, setup and a quickstart guide to reproduce our experiments.

## Getting started
### Getting the data
To download and store the preprocessed dataset ready for training locally, execute the following command:

```bash
bash download_climateset.sh
```

#### Note that this by default only downloads NorESM2-LM data. To download data for all climate models, please uncomment the line with the for loop.

You should now see a newly created directory called "Climateset_DATA" containing inputs and targets. This folder will be referenced within the emulator pipeline. 

### Setting up the environment

To setup the environment for causalpaca, we use python3.10. There are two separate requirements file for creating environments.

To create the environment used for training unet & convlstm models, use [requirements](requirements.txt) and climax related experiments, use [requirements_climax](requirements_climax.txt).


Follow the following steps to create the environment:

```python
python -m venv env_emulator
source env_emulator/bin/activate
pip install -r requirements.txt
pip install -e .
```

# Running a model

We provide some experiment configs in ```emulator/configs/experiment``` to recreate some of our models. Here are some example to recreate single emulator experimnts for NorESM2-LM.

```python
python emulator/run.py experiment=single_emulator/unet/NorESM2-LM_unet_tas+pr_run-01.yaml seed=3423
```

This will train the U-Net model on NorESM2-LM dataset. To change some of the parameters of the experiment, you can use hydra to override them. For eg. to run with different experiment seed:

```python
python emulator/run.py experiment=single_emulator/unet/NorESM2-LM_unet_tas+pr_run-01.yaml seed=22201
```

For running experiments with other models, here are some example commands:

```python
python emulator/run.py experiment=single_emulator/climax/NorESM2-LM_climax_tas+pr_run-01.yaml seed=3423
```

```python
python emulator/run.py experiment=single_emulator/climax_frozen/NorESM2-LM_climax_tas+pr_run-01.yaml seed=3423
```

```python
python emulator/run.py experiment=single_emulator/convlstm/NorESM2-LM_convlstm_tas+pr_run-01.yaml seed=3423
```

For climax & climax_frozen models, we will need to use a different requirements file to create another environment.

For the single-emulator experiments, we example templates for a couple of climate models for each ml model in ```configs/experiment/single_emulator```  and for fine-tuning experimnts, the configs can be found in ```configs/experiment/finetuning_emulator```.

For finetuning, we need to fill in ```pretrained_run_id``` and ```pretrained_ckpt_dir``` in the config files for resuming the experiments.

An example command for finetuning would look like this:
```python
python run.py experiment=finetuning_emulator/climax/NorESM2-LM_FGOALS-f3-L_climax_tas+pr_run-01.yaml seed=3423
```

## Logging locally
To run your model locally can either use no logger at all or tell the wandb logger to log your experiments offline you will need to overwrite the default logger (wandb) and set it to offline:

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

## Structure
![Visualization of the codebase](./diagram.svg)


The repository is inspired by [ClimART](https://github.com/RolnickLab/climart/tree/main) and PL+Hydra template implementation [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)