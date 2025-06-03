#!/bin/bash
# Copy: 
# source env_emulator/bin/activate 
# module load python/3.10


# Default values for flags
install=false
get_data=false
get_models=false
get_checkpoints=false
on_cluster=false
on_windows=false

# Parse command line options
while getopts dmcrwh opt; do
  case ${opt} in
    d )
      get_data=true
      ;;
    m )
      get_models=true
      ;;
    c )
      get_checkpoints=true
      ;;
    r ) 
      on_cluster=true
      ;;
    w )
      on_windows=true
      ;;
    h )
      echo "Usage: $0 [-d] [-m] [-c] [-r] [-w]"
      echo "Options:"
      echo "  -d: Data - Download ClimateSet data,"
      echo "  -m: Models - Download pretrained ,odels."
      echo "  -c: Checkpoints - Download ClimaX checkpoints"
      echo "  -r: Remote - Code run on the Mila cluster (and other clusters)."
      echo "  -w: Windows - Code run on a Windows System."

      exit 0
      ;;
    \? )
      echo "Invalid option: $OPTARG. Use -h for help." 1>&2
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))

# Set HYDRA_FULL_ERROR environment variable
export HYDRA_FULL_ERROR=1

# prepare dependencies
if [ "$on_cluster" = true ]; then
  # Load Python 3.10 module
  module load python/3.10 || { echo "Python module cannot be loaded."; exit 1; }
  module load libffi
  # Set PYTHONPATH to current directory
  # export PYTHONPATH=$(pwd)
  # on the cluster we already have poetry
else
  pip install poetry
fi

# Download data if -d flag is set
if [ "$get_data" = true ]; then
  python scripts/download_climateset_huggingface.py || { echo "Failed to download climateset data"; exit 1; }
fi

# Download models if -m flag is set
if [ "$get_models" = true ]; then
  bash scripts/download_pretrained_models_huggingface.py || { echo "Failed to download pretrained models"; exit 1; }
fi

# Run additional Bash script if -c flag is set
if [ "$get_checkpoints" = true ]; then
  bash scripts/download_climax_checkpoints.sh || { echo "Failed to download ClimaX checkpoints"; exit 1; }
fi

# Check if env_emulator folder exists
if [ ! -d "env_emulator" ]; then
    # Create a virtual environment
    python -m venv env_emulator || { echo "Failed to create virtual environment."; exit 1; }
    # Activate the virtual environment
    if [ "$on_windows" = false ]; then
        source env_emulator/bin/activate || { echo "Failed to activate virtual environment."; exit 1; }
    else 
        env_emulator/Scripts/activate || { echo "Failed to activate virtual environment."; exit 1; }
    fi 
    # Install the emulator package in editable mode 
    poetry install --all-groups || { echo "Failed to install emulator package."; exit 1; }
fi