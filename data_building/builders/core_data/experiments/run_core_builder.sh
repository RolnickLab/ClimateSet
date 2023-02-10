#!/bin/bash

#SBATCH --cpus-per-task=1                               # specify cpu

#SBATCH --mem=16G                                        # specify memory

#SBATCH --time=04:00:00                                  # set runtime

#SBATCH -o /home/mila/c/charlotte.lange/slurm-%j.out        # set log dir to home


# 1. Load Python

module load python/3.7


# 3. Create or Set Up Environment

if [ -a env4/bin/activate ]; then

    source env4/bin/activate

else

    python3.7 -m venv env4
    source env4/bin/activate
    python3.7 -m pip install --upgrade pip
    pip3.7 install wheel setuptools
fi


# 4. Install requirements.txt if it exists

if [ -a requirements.txt ]; then

    pip3.7 install -r requirements2.txt

fi
#source /home/mila/c/charlotte.lange/causalpaca/bin/activate

# 5. Copy data and code from scratch to $SLURM_TMPDIR/

cp -r /network/scratch/c/charlotte.lange/causalpaca/  $SLURM_TMPDIR/
#rm -r $SLURM_TMPDIR/caiclone/results/
#cp -r /network/scratch/j/julia.kaltenborn/data/ $SLURM_TMPDIR/

# 6. Set Flags

export GPU=0
export CUDA_VISIBLE_DEVICES=0

# 7. Change working directory to $SLURM_TMPDIR

cd $SLURM_TMPDIR/causalpaca/data_building/

# 8. Run Python

echo "Running builders/core_data/build_core_dataset.py ..."
python3.7 builders/core_data/build_core_dataset.py


# 9. Copy output to scratch
cp -r $SLURM_TMPDIR/causalpaca/data/data/* /network/scratch/c/charlotte.lange/causalpaca/data/data/

# try and copy to julia's scratch
cp -r $SLURM_TMPDIR/causalpaca/data/data/* /network/scratch/j/julia.kaltenborn/data/raw/


# 10. Experiment is finished

echo "Done."
