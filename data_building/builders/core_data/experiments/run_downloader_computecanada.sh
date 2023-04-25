#!/bin/bash

# TODO choose CPUs and make specs for compute canada
# TODO debug downloader after that
#SBATCH --cpus-per-task=1     # specify cpu
#SBATCH --gres=gpu:rtx8000:1  # specify gpu
#SBATCH --mem=150G            # specify memory
#SBATCH --time=50:00:00       # set runtime
#SBATCH --partition=long
#SBATCH -o /home/mila/c/charlotte.lange/scratch/causalpaca/slurms/slurm_downloader-%j.out        # set log dir to home

# Note running: sbatch data_building/builders/core_data/experiments/run_downloader_computecanada.sh


# 1. Run setup.sh (on cluster with venv)
source setup.sh -c -v
source env_data/bin/activate/

# 2. Copy code from scratch to $SLURM_TMPDIR/
cp -r /network/scratch/c/charlotte.lange/causalpaca/ $SLURM_TMPDIR/

# 3. Set Flags
export GPU=0
export CUDA_VISIBLE_DEVICES=0

# 4. Change working directory to $SLURM_TMPDIR
cd $SLURM_TMPDIR/causalpaca/

# 5. Run Python
echo "Running downloader.py ..."
python3.7 -m data_building.builders.downloader

# 6. Copy output to personal scratch (Mila cluster example)
#cp -r $SLURM_TMPDIR/causalpaca/tmp/* /network/scratch/c/charlotte.lange/causalpaca/tmp/

# try and copy to julia's scratch
cp -r $SLURM_TMPDIR/causalpaca/tmp/* /home/jkalt/projects/def-drolnick/jkalt/data/raw/


# 7. Experiment is finished
echo "Done."
