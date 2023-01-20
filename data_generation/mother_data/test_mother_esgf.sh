#!/bin/bash

#SBATCH --cpus-per-task=1                               # specify cpu

#SBATCH --gres=gpu:rtx8000:1                            # specify gpu

#SBATCH --mem=150G                                        # specify memory

#SBATCH --time=50:00:00                                  # set runtime

#SBATCH -o /home/mila/c/charlotte.lange/scratch/causalpaca/slurms/slurm_downloader-%j.out        # set log dir to home

# Note running: sbatch --partition=unkillable data/mother_data/test_mother_esgf.sh

# script to stard downloader

module load python/3.7


# 3. Create or Set Up Environment

if [ -a env37/bin/activate ]; then
    echo "activating env37"
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

cp -r /network/scratch/c/charlotte.lange/causalpaca/data/  $SLURM_TMPDIR/

# 6. Set Flags

export GPU=0
export CUDA_VISIBLE_DEVICES=0

# 7. Change working directory to $SLURM_TMPDIR

cd $SLURM_TMPDIR/causalpaca/data/

# 8. Run Python
echo "Running data/mother_data/downloader.py ..."
python3.7 data/mother_data/downloader.py


# 9. Copy output to scratch
cp -r $SLURM_TMPDIR/causalpaca/data/* /network/scratch/c/charlotte.lange/causalpaca/data/mother_data/data/

# try and copy to julia's scratch
cp -r $SLURM_TMPDIR/causalpaca/data/* /network/scratch/j/julia.kaltenborn/data/raw/


# 10. Experiment is finished

echo "Done."
