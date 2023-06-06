#!/bin/bash
#SBATCH --job-name=causalpaca_lstm
#SBATCH --output=nohup.out
#SBATCH --error=error.out
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --mem=48Gb
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --partition=main
#SBATCH -c 4

module add python/3.10
source $HOME/env_emulator/bin/activate

python run.py experiment=single_emulator/convlstm/AWI-CM-1-1-MR_convlstm_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/convlstm/AWI-CM-1-1-MR_convlstm_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/convlstm/AWI-CM-1-1-MR_convlstm_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/convlstm/FGOALS-f3-L_convlstm_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/convlstm/FGOALS-f3-L_convlstm_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/convlstm/FGOALS-f3-L_convlstm_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/convlstm/EC-Earth3_convlstm_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/convlstm/EC-Earth3_convlstm_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/convlstm/EC-Earth3_convlstm_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/convlstm/BCC-CSM2-MR_convlstm_tas_run-01.yaml seed=3423
python run.py experiment=single_emulator/convlstm/BCC-CSM2-MR_convlstm_tas_run-01.yaml seed=22201
python run.py experiment=single_emulator/convlstm/BCC-CSM2-MR_convlstm_tas_run-01.yaml seed=7

python run.py experiment=single_emulator/convlstm/NorESM2-LM_convlstm_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/convlstm/NorESM2-LM_convlstm_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/convlstm/NorESM2-LM_convlstm_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/convlstm/CESM2-WACCM_convlstm_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/convlstm/CESM2-WACCM_convlstm_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/convlstm/CESM2-WACCM_convlstm_tas+pr_run-01.yaml seed=7

