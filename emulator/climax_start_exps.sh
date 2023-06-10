#!/bin/bash
#SBATCH --job-name=causalpaca_climax
#SBATCH --output=nohup.out
#SBATCH --error=error.out
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --mem=48Gb
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --partition=main
#SBATCH -c 4

module add python/3.10
source $HOME/env_emulator/bin/activate

python run.py experiment=single_emulator/climax/AWI-CM-1-1-MR_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax/AWI-CM-1-1-MR_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax/AWI-CM-1-1-MR_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax/FGOALS-f3-L_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax/FGOALS-f3-L_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax/FGOALS-f3-L_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax/EC-Earth3_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax/EC-Earth3_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax/EC-Earth3_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax/BCC-CSM2-MR_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax/BCC-CSM2-MR_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax/BCC-CSM2-MR_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax/NorESM2-LM_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax/NorESM2-LM_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax/NorESM2-LM_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax/MPI-ESM1-2-HR_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax/MPI-ESM1-2-HR_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax/MPI-ESM1-2-HR_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax/TaiESM1_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax/TaiESM1_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax/TaiESM1_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax/INM-CM5-0_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax/INM-CM5-0_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax/INM-CM5-0_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax/CMCC-CM2-SR5_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax/CMCC-CM2-SR5_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax/CMCC-CM2-SR5_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax/CAMS-CSM1-0_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax/CAMS-CSM1-0_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax/CAMS-CSM1-0_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax/NorESM2-MM_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax/NorESM2-MM_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax/NorESM2-MM_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax/MRI-ESM2-0_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax/MRI-ESM2-0_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax/MRI-ESM2-0_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax/INM-CM4-8_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax/INM-CM4-8_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax/INM-CM4-8_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax/GFDL-ESM4_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax/GFDL-ESM4_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax/GFDL-ESM4_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax/EC-Earth3-Veg-LR_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax/EC-Earth3-Veg-LR_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax/EC-Earth3-Veg-LR_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax/CNRM-CM6-1-HR_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax/CNRM-CM6-1-HR_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax/CNRM-CM6-1-HR_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax/CMCC-ESM2_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax/CMCC-ESM2_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax/CMCC-ESM2_climax_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/climax/CAS-ESM2-0_climax_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/climax/CAS-ESM2-0_climax_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/climax/CAS-ESM2-0_climax_tas+pr_run-01.yaml seed=7

