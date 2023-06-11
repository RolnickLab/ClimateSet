#!/bin/bash
#SBATCH --job-name=causalpaca_climax
#SBATCH --output=nohup.out
#SBATCH --error=error.out
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=48Gb
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --partition=main
#SBATCH -c 4

module add python/3.10
source $HOME/env_emulator/bin/activate

python run.py experiment=finetuning_emulator/climax/NorESM2-LM_AWI-CM-1-1-MR_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=finetuning_emulator/climax/NorESM2-LM_AWI-CM-1-1-MR_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=finetuning_emulator/climax/NorESM2-LM_AWI-CM-1-1-MR_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=finetuning_emulator/climax/NorESM2-LM_EC-Earth3_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=finetuning_emulator/climax/NorESM2-LM_EC-Earth3_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=finetuning_emulator/climax/NorESM2-LM_EC-Earth3_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=finetuning_emulator/climax/NorESM2-LM_MPI-ESM1-2-HR_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=finetuning_emulator/climax/NorESM2-LM_MPI-ESM1-2-HR_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=finetuning_emulator/climax/NorESM2-LM_MPI-ESM1-2-HR_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=finetuning_emulator/climax/NorESM2-LM_BCC-CSM2-MR_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=finetuning_emulator/climax/NorESM2-LM_BCC-CSM2-MR_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=finetuning_emulator/climax/NorESM2-LM_BCC-CSM2-MR_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=finetuning_emulator/climax/NorESM2-LM_FGOALS-f3-L_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=finetuning_emulator/climax/NorESM2-LM_FGOALS-f3-L_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=finetuning_emulator/climax/NorESM2-LM_FGOALS-f3-L_climax_tas+pr_run-01.yaml seed=7
