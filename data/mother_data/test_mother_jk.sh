#!/bin/bash

#SBATCH --cpus-per-task=1                               # specify cpu

#SBATCH --gres=gpu:rtx8000:1                             # specify gpu

#SBATCH --mem=16G                                        # specify memory

#SBATCH --time=00:40:00                                  # set runtime

#SBATCH -o /home/mila/j/julia.kaltenborn/slurm-causalpaca/slurm-%j.out        # set log dir to home

# Note running: sbatch --partition=unkillable data/mother_data/test_mother_jk.sh

# 1. Load Python

module load python/3.9


# 3. Create or Set Up Environment

if [ -a env39/bin/activate ]; then
    echo "pwd before activating venv"
    pwd
    source env39/bin/activate

else
    echo "env39 does not exist"
    python3.9 -m venv env39
    source env39/bin/activate
    python3.9 -m pip install --upgrade pip
    pip3.9 install wheel setuptools

    # 4. Install requirements.txt if it exists

    if [ -a requirements2.txt ]; then
        pip3.9 install -r requirements2.txt
    fi

fi

# 5. Copy data and code from scratch to $SLURM_TMPDIR/
echo "pwd before copying data"
pwd
cp -r /network/scratch/j/julia.kaltenborn/causalpaca/  $SLURM_TMPDIR/
echo "pwd after copying data"
pwd
#rm -r $SLURM_TMPDIR/caiclone/results/
#cp -r /network/scratch/j/julia.kaltenborn/data/ $SLURM_TMPDIR/

# 6. Set Flags

export GPU=0
export CUDA_VISIBLE_DEVICES=0

# 7. Change working directory to $SLURM_TMPDIR

cd $SLURM_TMPDIR/causalpaca/data/
echo "pwd after changing to TMP/causalpaca/data/"
pwd

# 8. Run Python
#echo "Running test file with easy verison"
#python3.9 testing/test_downloader.py

echo "Running mother_data/downloader.py ..."
python3.9 mother_data/downloader.py
echo "pwd after running downloader"
pwd

# 9. Copy output to scratch
#cp -r $SLURM_TMPDIR/causalpaca/data/mother_data/data/* /network/scratch/c/charlotte.lange/causalpaca/data/data/

# try and copy to julia's scratch
cp -r $SLURM_TMPDIR/causalpaca/data/data/* /network/scratch/j/julia.kaltenborn/data/raw/


# 10. Experiment is finished

echo "Done."
