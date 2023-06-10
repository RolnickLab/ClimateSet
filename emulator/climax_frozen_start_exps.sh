#!/bin/bash
#SBATCH --job-name=causalpaca_climax_frozen
#SBATCH --output=nohup.out
#SBATCH --error=error.out
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --mem=48Gb
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --partition=long
#SBATCH -c 4

module add python/3.10
source $HOME/env_emulator/bin/activate

python run.py experiment=single_emulator/climax_frozen/AWI-CM-1-1-MR_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax_frozen/AWI-CM-1-1-MR_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax_frozen/AWI-CM-1-1-MR_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax_frozen/FGOALS-f3-L_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax_frozen/FGOALS-f3-L_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax_frozen/FGOALS-f3-L_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax_frozen/EC-Earth3_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax_frozen/EC-Earth3_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax_frozen/EC-Earth3_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax_frozen/BCC-CSM2-MR_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax_frozen/BCC-CSM2-MR_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax_frozen/BCC-CSM2-MR_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax_frozen/NorESM2-LM_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax_frozen/NorESM2-LM_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax_frozen/NorESM2-LM_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax_frozen/MPI-ESM1-2-HR_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax_frozen/MPI-ESM1-2-HR_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax_frozen/MPI-ESM1-2-HR_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax_frozen/TaiESM1_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax_frozen/TaiESM1_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax_frozen/TaiESM1_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax_frozen/INM-CM5-0_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax_frozen/INM-CM5-0_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax_frozen/INM-CM5-0_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax_frozen/CMCC-CM2-SR5_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax_frozen/CMCC-CM2-SR5_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax_frozen/CMCC-CM2-SR5_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax_frozen/CAMS-CSM1-0_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax_frozen/CAMS-CSM1-0_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax_frozen/CAMS-CSM1-0_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax_frozen/NorESM2-MM_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax_frozen/NorESM2-MM_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax_frozen/NorESM2-MM_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax_frozen/MRI-ESM2-0_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax_frozen/MRI-ESM2-0_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax_frozen/MRI-ESM2-0_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax_frozen/INM-CM4-8_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax_frozen/INM-CM4-8_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax_frozen/INM-CM4-8_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax_frozen/GFDL-ESM4_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax_frozen/GFDL-ESM4_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax_frozen/GFDL-ESM4_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax_frozen/EC-Earth3-Veg-LR_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax_frozen/EC-Earth3-Veg-LR_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax_frozen/EC-Earth3-Veg-LR_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax_frozen/CNRM-CM6-1-HR_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax_frozen/CNRM-CM6-1-HR_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax_frozen/CNRM-CM6-1-HR_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax_frozen/CMCC-ESM2_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax_frozen/CMCC-ESM2_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax_frozen/CMCC-ESM2_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax_frozen/CAS-ESM2-0_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax_frozen/CAS-ESM2-0_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax_frozen/CAS-ESM2-0_climax_tas+pr_run-01.yaml seed=7

