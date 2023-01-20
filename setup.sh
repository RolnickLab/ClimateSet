#!/bin/bash

# 1. Load modules if available
module load python/3.7
#module load cdo # TODO if else

# 2. Install CDO if not available as module
# TODO how to do this on the cluster? without sudo?
sudo apt-get install libnetcdf-dev libhdf5-dev
sudo apt-get install cdo

# 3. Create venv
if [ -a env37/bin/activate ]; then
    source env37/bin/activate

else
    echo "env37 is created now"
    python3.7 -m venv env37
    source env37/bin/activate
    python3.7 -m pip install --upgrade pip
    pip3.7 install wheel setuptools

    # 4. Install requirements.txt if it exists

    if [ -a requirements2.txt ]; then
        pip3.7 install -r requirements2.txt
    fi

fi
