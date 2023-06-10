#!/bin/bash
#SBATCH --job-name=causalpaca_unet
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

#python run.py experiment=single_emulator/unet/TaiESM1_unet_tas+pr_run-01.yaml seed=3423
#python run.py experiment=single_emulator/unet/TaiESM1_unet_tas+pr_run-01.yaml seed=22201
#python run.py experiment=single_emulator/unet/TaiESM1_unet_tas+pr_run-01.yaml seed=7

#python run.py experiment=single_emulator/unet/INM-CM5-0_unet_tas+pr_run-01.yaml seed=3423
#python run.py experiment=single_emulator/unet/INM-CM5-0_unet_tas+pr_run-01.yaml seed=22201
#python run.py experiment=single_emulator/unet/INM-CM5-0_unet_tas+pr_run-01.yaml seed=7

#python run.py experiment=single_emulator/unet/CMCC-CM2-SR5_unet_tas+pr_run-01.yaml seed=3423
#python run.py experiment=single_emulator/unet/CMCC-CM2-SR5_unet_tas+pr_run-01.yaml seed=22201
#python run.py experiment=single_emulator/unet/CMCC-CM2-SR5_unet_tas+pr_run-01.yaml seed=7

#python run.py experiment=single_emulator/unet/CAMS-CSM1-0_unet_tas+pr_run-01.yaml seed=3423
#python run.py experiment=single_emulator/unet/CAMS-CSM1-0_unet_tas+pr_run-01.yaml seed=22201
#python run.py experiment=single_emulator/unet/CAMS-CSM1-0_unet_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/unet/NorESM2-MM_unet_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/unet/NorESM2-MM_unet_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/unet/NorESM2-MM_unet_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/unet/MRI-ESM2-0_unet_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/unet/MRI-ESM2-0_unet_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/unet/MRI-ESM2-0_unet_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/unet/INM-CM4-8_unet_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/unet/INM-CM4-8_unet_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/unet/INM-CM4-8_unet_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/unet/GFDL-ESM4_unet_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/unet/GFDL-ESM4_unet_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/unet/GFDL-ESM4_unet_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/unet/EC-Earth3-Veg-LR_unet_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/unet/EC-Earth3-Veg-LR_unet_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/unet/EC-Earth3-Veg-LR_unet_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/unet/CNRM-CM6-1-HR_unet_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/unet/CNRM-CM6-1-HR_unet_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/unet/CNRM-CM6-1-HR_unet_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/unet/CMCC-ESM2_unet_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/unet/CMCC-ESM2_unet_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/unet/CMCC-ESM2_unet_tas+pr_run-01.yaml seed=7

python run.py experiment=single_emulator/unet/CAS-ESM2-0_unet_tas+pr_run-01.yaml seed=3423
python run.py experiment=single_emulator/unet/CAS-ESM2-0_unet_tas+pr_run-01.yaml seed=22201
python run.py experiment=single_emulator/unet/CAS-ESM2-0_unet_tas+pr_run-01.yaml seed=7
