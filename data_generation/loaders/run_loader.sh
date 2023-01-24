#!/bin/bash

#SBATCH --cpus-per-task=1                               # specify cpu

#SBATCH --gres=gpu:rtx8000:1                            # specify gpu

#SBATCH --mem=64G                                        # specify memory

#SBATCH --time=01:00:00                                  # set runtime

#SBATCH -o /home/mila/c/charlotte.lange/scratch/causalpaca/slurms/slurm_loader-%j.out        # set log dir to home

# Note running: sbatch --partition=unkillable data/run_loader.sh


module load python/3.7


# 3. Create or Set Up Environment

if [ -a env37/bin/activate ]; then
    source env37/bin/activate

else
    echo "env37 does not exist"
    python3.7 -m venv env37
    source env37/bin/activate
    python3.7 -m pip install --upgrade pip
    pip3.7 install wheel setuptools

    # 4. Install requirements.txt if it exists

    if [ -a requirements37.txt ]; then
        pip3.7 install -r requirements37.txt
    fi

fi


# 5. Copy data and code from scratch to $SLURM_TMPDIR/
echo "pwd before copying data"
pwd
cp -r /network/scratch/c/charlotte.lange/causalpaca/data/  $SLURM_TMPDIR/
echo "pwd after copying data"
pwd
ls
#rm -r $SLURM_TMPDIR/caiclone/results/
#cp -r /network/scratch/j/julia.kaltenborn/data/ $SLURM_TMPDIR/

# 6. Set Flags

export GPU=0
export CUDA_VISIBLE_DEVICES=0

# 7. Change working directory to $SLURM_TMPDIR

cd $SLURM_TMPDIR/
echo "pwd after changing to TMP/causalpaca/data/"
ls
pwd



echo "Running data_generation/loaders/datasets.py ..."
python3.7 data_generation/loaders/datasets.py

# 10. Experiment is finished

echo "Done."
