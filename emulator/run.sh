#!/bin/bash
#SBATCH --account=def-drolnick         # Yoshua pays for your job
#SBATCH --cpus-per-task=2                # Ask for CPUs
#SBATCH --gres=gpu:1                     # Ask for GPUs
#SBATCH --mem=5G                        # Ask for GB of RAM
#SBATCH --time=01:00:00                   # The job will run for x hours
#SBATCH -o /projects/def-drolnick/pandora/causalpaca/emulator/run.out  # Write the log in 

module load python/3.10.2


# condidional: activate environment or create and install from requirements + running setup.py install

#  Create or Set Up Environment

if [ -a /home/pandora/projects/def-drolnick/pandora/causalpaca/emulator/emulator_venv/bin/activate ]; then

    source /home/pandora/projects/def-drolnick/pandora/causalpaca/emulator/emulator_venv/bin/activate

else

    python -m venv /home/pandora/projects/def-drolnick/pandora/causalpaca/emulator/emulator_venv/
    source /home/pandora/projects/def-drolnick/pandora/causalpaca/emulator/emulator_venv/bin/activate
    pip install -U pip wheel setuptools

fi

# Install requirements.txt if it exists

if [ -a requirements_emulator.txt ]; then

    pip install -r /home/pandora/projects/def-drolnick/pandora/causalpaca/emulator/requirements_emulator.txt
    # setup emulator package
    cd /home/pandora/projects/def-drolnick/pandora/causalpaca
    pip install -e .

fi

cd /home/pandora/projects/def-drolnick/pandora/causalpaca/emulator
python run.py