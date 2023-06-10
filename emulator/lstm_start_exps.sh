#!/bin/bash
#SBATCH --job-name=causalpaca_convlstm
#SBATCH --output=nohup.out
#SBATCH --error=error.out
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=48Gb
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --partition=long
#SBATCH -c 4

module add python/3.10
source $HOME/env_emulator/bin/activate

python run.py experiment=single_emulator/convlstm/TaiESM1_convlstm_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/convlstm/TaiESM1_convlstm_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/convlstm/TaiESM1_convlstm_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/convlstm/INM-CM5-0_convlstm_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/convlstm/INM-CM5-0_convlstm_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/convlstm/INM-CM5-0_convlstm_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/convlstm/CMCC-CM2-SR5_convlstm_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/convlstm/CMCC-CM2-SR5_convlstm_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/convlstm/CMCC-CM2-SR5_convlstm_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/convlstm/CAMS-CSM1-0_convlstm_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/convlstm/CAMS-CSM1-0_convlstm_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/convlstm/CAMS-CSM1-0_convlstm_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/convlstm/NorESM2-MM_convlstm_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/convlstm/NorESM2-MM_convlstm_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/convlstm/NorESM2-MM_convlstm_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/convlstm/MRI-ESM2-0_convlstm_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/convlstm/MRI-ESM2-0_convlstm_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/convlstm/MRI-ESM2-0_convlstm_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/convlstm/INM-CM4-8_convlstm_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/convlstm/INM-CM4-8_convlstm_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/convlstm/INM-CM4-8_convlstm_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/convlstm/GFDL-ESM4_convlstm_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/convlstm/GFDL-ESM4_convlstm_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/convlstm/GFDL-ESM4_convlstm_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/convlstm/EC-Earth3-Veg-LR_convlstm_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/convlstm/EC-Earth3-Veg-LR_convlstm_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/convlstm/EC-Earth3-Veg-LR_convlstm_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/convlstm/CNRM-CM6-1-HR_convlstm_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/convlstm/CNRM-CM6-1-HR_convlstm_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/convlstm/CNRM-CM6-1-HR_convlstm_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/convlstm/CMCC-ESM2_convlstm_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/convlstm/CMCC-ESM2_convlstm_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/convlstm/CMCC-ESM2_convlstm_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/convlstm/CAS-ESM2-0_convlstm_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/convlstm/CAS-ESM2-0_convlstm_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/convlstm/CAS-ESM2-0_convlstm_tas+pr_run-01.yaml seed=7
