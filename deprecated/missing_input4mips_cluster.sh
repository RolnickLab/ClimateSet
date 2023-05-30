#!/bin/bash

#SBATCH --cpus-per-task=1                               # specify cpu

#SBATCH --gres=gpu:rtx8000:1                            # specify gpu

#SBATCH --mem=32G                                        # specify memory

#SBATCH --time=00:40:00                                  # set runtime

#SBATCH -o /home/mila/c/charlotte.lange/causalpaca/slurm/esgf_bash_-%j.out        # set log dir to home

# Note running: sbatch --partition=unkillable data/mother_data/missing_input4mips_cluster.sh

 5. Copy data and code from scratch to $SLURM_TMPDIR/
echo "pwd before copying data"
pwd
cp -r /network/scratch/c/charlotte.lange/causalpaca/  $SLURM_TMPDIR/
echo "pwd after copying data"
pwd
#rm -r $SLURM_TMPDIR/caiclone/results/
#cp -r /network/scratch/j/julia.kaltenborn/data/ $SLURM_TMPDIR/

# 6. Set Flags

#export GPU=0
#export CUDA_VISIBLE_DEVICES=0

# 7. Change working directory to $SLURM_TMPDIR

cd $SLURM_TMPDIR/causalpaca/data/data
echo "pwd after changing to TMP/causalpaca/data/data"
pwd
bash $SLURM_TMPDIR/causalpaca/data/mother_data/missing_input4mips.sh -H -o causalpaca < $SLURM_TMPDIR/causalpaca/data/mother_data/credentials.txt

# 8. Run Python
#echo "Running test file with easy verison"
#python3.9 testing/test_downloader.py

#echo "Running mother_data/downloader.py ..."
#python3.9 mother_data/downloader.py
#echo "pwd after running downloader"
#pwd



# 9. Copy output to scratch
cp -r $SLURM_TMPDIR/causalpaca/data/data/* /network/scratch/c/charlotte.lange/causalpaca/data/mother_data/data/

echo "done"
