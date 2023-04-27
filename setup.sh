#!/bin/bash

# TODO conda environment case

venv='false'
cluster='false'

while getopts vc flag; do
    case "${flag}" in
      v) venv='true' ;;
      c) cluster='true' ;;
    esac
done

if $cluster; then
  module load python/3.9 || echo "Python module cannot be loaded."
  if $venv; then
    module load cdo/2.0.5 || echo "CDO module cannot be loaded. Consider uding a conda environment instead"
  fi
else
  # install cdo in case we are local and use venv
  if $venv; then
    sudo apt-get install libnetcdf-dev libhdf5-dev
    sudo apt-get install cdo
  fi
fi

# Install and activate environments

# VENV case
if $venv; then
  if [ -a env_data/bin/activate ]; then
      source env_data/bin/activate

  else
      echo "env_data is created now"
      python -m venv env_data
      source env_data/bin/activate
      python -m pip install --upgrade pip
      pip install wheel setuptools

      # Install requirements.txt if it exists
      if [ -a requirements_data.txt ]; then
          pip install -r requirements_data.txt
      else
          echo "Requirements file missing"
      fi

  fi

# CONDA case
else
    # if conda environment does not exists already
    echo "TODO this is broken - you have to install conda yourself right now"
    exit 1
    if conda info --envs | grep -q conda_env_data; then
        source activate conda_env_data
    else
        conda create -n conda_env_data python=3.9
        source activate conda_env_data
        if [ -a requirements_data.txt ]; then
            pip install -r requirements_data.txt
            conda install -c conda-forge cdo
        else
            echo "Requirements file missing"
        fi
    fi
fi
