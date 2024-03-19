#!/bin/bash
#SBATCH --job-name=backup_ckpts
#SBATCH --output=ckpts.out
#SBATCH --error=error.out
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=16Gb
#SBATCH --partition=long
#SBATCH -c 4

module add python/3.10
source $HOME/env_emulator_climax/bin/activate

python /home/mila/c/charlotte.lange/scratch/neurips23/causalpaca/get_scores_save_models.py