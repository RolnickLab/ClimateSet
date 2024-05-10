#!/bin/bash
#SBATCH --job-name=upload_models
#SBATCH --output=upload_models.out
#SBATCH --error=error.out
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --mem=30Gb
#SBATCH --partition=long
#SBATCH -c 4

module add python/3.10
source $HOME/env_arbutus/bin/activate

cd /home/mila/c/charlotte.lange/scratch/neurips23/causalpaca/emulator/internal/
python upload_trained_models.py