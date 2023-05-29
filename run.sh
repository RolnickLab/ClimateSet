#!/bin/bash
#SBATCH --account=rrg-bengioy-ad         # Yoshua pays for your job
#SBATCH --cpus-per-task=2                # Ask for 6 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=20G                        # Ask for 32 GB of RAM
#SBATCH --time=72:00:00                   # The job will run for 3 hours
#SBATCH -o /scratch/venka97/nohup.out  # Write the log in $SCRATCH

module load python/3.10.2
source /home/venka97/scratch/dl/bin/activate

python -m data_building.builders.downloader --cfg /home/venka97/projects/def-drolnick/venka97/code/causalpaca/config.yaml
rsync -r --ignore-existing $SLURM_TMPDIR/causalpaca /home/venka97/scratch
