# Emulator Models

## Overview
This readme should help you to get started on running and implementing emulator models for causalpaca.

## Getting started
The environment set-up happens separately from the dataset set-up since a different set of
packages is needed for running the emulators.

```python
python -m venv env_emulator
source env_emulator/bin/activate
pip install -r requirements_emulator.txt
```
Bigger requirements:
pytorch, wandb

## Run a model locally

## Run experiments on wandb

## How to add new models
You can add new models in `src/core/models`. Each model should inherit from the Basemodel class you can find in `basemodel.py`. Add a new config file for your model in `config/models/`.

## Structure
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
├── requirements_emulator.txt
├── run.py
└── setup.py

## Notes
What else could be added:
- Docs: will be added on higher level
- Makefile: not sure
- notebooks: not needed
- references: would be nice
- tox.ini: not sure
