# Causalpaca: Emulator Models

This branch contains the code for running the climate model emulation benchmark experiments on the core ClimateSet data. 
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

To setup the environment for causalpaca, we use ```python>=3.10```. There are two separate requirements file for creating environments.

To create the environment used for training unet & convlstm models, use [requirements](requirements.txt) and climax related experiments, use [requirements_climax](requirements_climax.txt).


Follow the following steps to create the environment:

```python
python -m venv env_emulator
source env_emulator/bin/activate
pip install -r requirements.txt
pip install -e .
```

### For ClimaX: Download pretrained checkpoints

To work with ClimaX, you will need to download the pretrained checkpoints from the original release and place them in the correct folder. To do so, exectute the following command:

```bash
bash download_climax_checkpoints.sh
```

# Running a model

To run the model, edit the [main config](emulator/configs/main_config.yaml) to fit what you want to run. 
Executing the run.py script plain will use the main config. 

The [configs folder](emulator/configs/) serves as a blueprint, listing all the modules available. To get a better understanding of our codebases structure please refer to the section on [Structure](#structure) 

```python
python run.py # will run with configs/main_config.yml
```

To exectute one of the preset experiments or to run your own experiments you can create and pass on experiment configs:

```python
python run.py experiment=test # will run whatever is specified by the configs/experiment/test.yml file
```

You can make use of the [experiment template](emulator/configs/experiment/templatte.yaml).


## Reproducing experiments
We provide some experiment configs in ```emulator/configs/experiment``` to recreate some of our models.

We ran 3 different configurations of experiments:
- *single emulator*: A single ML-model - climate-model pairing.
- *finetuning emulator*: A single ML-model that was pretrained on one climate-model and fine-tuned on another.
- *super emulator*: A single ML-model that was trained on multiple climate-models.

Here are some example to recreate single emulator experimnts for NorESM2-LM.

```python
python emulator/run.py experiment=single_emulator/unet/NorESM2-LM_unet_tas+pr_run-01.yaml logger=none seed=3423
```

This will train the U-Net model on NorESM2-LM dataset. To change some of the parameters of the experiment, you can use hydra to override them. For eg. to run with different experiment seed:

```python
python emulator/run.py experiment=single_emulator/unet/NorESM2-LM_unet_tas+pr_run-01.yaml logger=none seed=22201
```

For running experiments with other models, here are some example commands:

```python
python emulator/run.py experiment=single_emulator/climax/NorESM2-LM_climax_tas+pr_run-01.yaml logger=none seed=3423
```

```python
python emulator/run.py experiment=single_emulator/climax_frozen/NorESM2-LM_climax_tas+pr_run-01.yaml logger=none seed=3423
```

```python
python emulator/run.py experiment=single_emulator/convlstm/NorESM2-LM_convlstm_tas+pr_run-01.yaml logger=none seed=3423
```

For climax & climax_frozen models, we will need to use a different requirements file to create another environment.

For the single-emulator experiments, we proide configs for each ml model in ```emulator/configs/experiment/single_emulator```  and for fine-tuning experimnts, the configs can be found in ```emulator/configs/experiment/finetuning_emulator```.

For finetuning, we need to fill in ```pretrained_run_id``` and ```pretrained_ckpt_dir``` in the config files for resuming the experiments.

An example command for finetuning would look like this:
```python
python run.py experiment=finetuning_emulator/climax/NorESM2-LM_FGOALS-f3-L_climax_tas+pr_run-01.yaml seed=3423 logger=none
```

For the superemulation experiments, we provide the configs of our experiments in ```emulator/configs/experiment/superemulator```. Nothe that here, dataloiding is changed slightly to the superemulator infrastructure and a decoder must be set.

An example command to run a superemulaton experiment would look like this:

```python
python run.py experiment=superemulator/superemulator_climax.yaml seed=3423 logger=none
```
## Reloading our trained models

We provide all our trained models from the experiments mentioned in the paper which are stored in ```pretrained_models```.
If you wish to load an existing models, choose an experiment configuration, meaning superemulation, single-emulation or fine-tuning and a desired machine learning model. For each combinaiton you will have a choice of experiments running with different seeds. In each folder, the exact information of what data and other parameter were used, see the ```hydra_config.yaml```.

Once you selected a model, decide whether you want to adjust or freeze the model weights and extract the run id and the path of the checkpoint you want to load.

For example, you choose to reload a ClimaX model from a single emulator experiment running on NorESM-LM data. Choose a respective folder, eg. ```/pretrained_models/single_emulator/ClimaX/NorESM2-LM_climax_run1_single_emulator_tas+pr/``` and pick on of the run ids e.g. ```0ltetwu3```.
In the respective folder you will find one or more checkpoints stored in a ```checkpoints``` folder. Choose one and copy that path location, e.g. ```pretrained_models/single_emulator/ClimaX/NorESM2-LM_climax_run1_single_emulator_tas+pr/0ltetwu3/checkpoints/epoch=49-step=2950.ckpt```.

To retrain this model (thus fine-tuning), pass on the following arguments and the experiment config to run, or alternativeley, create a new config setting these parameters:

```yaml
# In your experiment_name.yml
model:
  finetune: True # allow further training or freeze if set to False
  pretrained_run_id: "0ltetwu3" 
  pretrained_ckpt_dir: "pretrained_models/single_emulator/ClimaX/NorESM2-LM_climax_run1_single_emulator_tas+pr/0ltetwu3/checkpoints/epoch=49-step=2950.ckpt"
```

You can also override the parameters directly when running (but pay attention, strings containing equal signs need to be put in double quotes!):

```python
python emulator/run.py  experiment=single_emulator/climax/NorESM2-LM_climax_tas+pr_run-01 logger=none model.pretrained_run_id="0ltetwu3" model.pretrained_ckpt_dir='"pretrained_models/single_emulator/ClimaX/NorESM2-LM_climax_run1_single_emulator_tas+pr/0ltetwu3/checkpoints/epoch=49-step=2950.ckpt"' model.finetune=True
```

## Reloading your own pretrained checkpoints

Similar to the fine-tuning experiments you can load, fine-tune and test preexisting models by adjusting the following parameters in the respective experiment config:

``` yaml
# In your experiment_name.yml
model:
    finetune: True
    pretrained_run_id: "" #eg "3u0ys0d5"
    pretrained_ckpt_dir: "" # eg. "causalpaca/emulator/emulator/ne8oyt48/checkpoints/epoch=49-step=2950.ckpt"
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

The codebase is divided in two main parts, [configs](emulator/configs/) and [src](emulator/src/).
The *configs* folder provides parameterization for all the modules and experiments possible within the code provided in *src*.

Within *src*, *core* includes all the code for the training and testing pipeline, 
*data* loads and creates a custom dataset object from the core ClimateSet dataset and *datamodules* handles interfacing with the dataset for the different configurations.


![Visualization of the codebase](./diagram.svg)


The repository is inspired by [ClimART](https://github.com/RolnickLab/climart/tree/main) and PL+Hydra template implementation [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)