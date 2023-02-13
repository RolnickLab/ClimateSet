#!/bin/bash

# TODO if else environment
# if module load python/3.9 is possible -> you are on a cluster
# if not -> presumably you are not on cluster


# 1. Load modules if available
module load python/3.9
# module load python/3.7
#module load cdo # TODO if else

# 2. Install CDO if not available as module
# TODO how to do this on the cluster? without sudo?
sudo apt-get install libnetcdf-dev libhdf5-dev
sudo apt-get install cdo
# TODO check if this already exists before installing it!!

# 3. Create venv
if [ -a env_data/bin/activate ]; then
    source env_data/bin/activate

else
    echo "env_data is created now"
    python3.9 -m venv env_data
    source env_data/bin/activate
    python3.9 -m pip install --upgrade pip
    pip3.9 install wheel setuptools

    # 4. Install requirements.txt if it exists

    if [ -a requirements_data.txt ]; then
        pip3.9 install -r requirements_data.txt
    fi

fi
