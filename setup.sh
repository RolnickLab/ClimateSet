#!/bin/bash
# Copy: 
# source env_emulator/bin/activate 
# module load python/3.10


# Default values for flags
run_python=false
run_bash=false
run_checkpoints=false
no_cluster=false
on_windows=false

# Parse command line options
while getopts pbcnh opt; do
  case ${opt} in
    p )
      run_python=true
      ;;
    b )
      run_bash=true
      ;;
    c )
      run_checkpoints=true
      ;;
    n ) 
      no_cluster=true
      ;;
    w )
      on_windows=true
      ;;
    h )
      echo "Usage: $0 [-p] [-b] [-c] [-n] [-w]"
      echo "Options:"
      echo "  -p: Run python download_climateset.py"
      echo "  -b: Run bash download_climateset.sh if you are within Canada instead of -p"
      echo "  -c: Run bash download_climax_checkpoints.sh"
      echo "  -n: no access to the mila cluster"
      echo "  -w: On a Windows System"

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

if [ "$no_cluster" = false ]; then
  # Load Python 3.10 module
  module load python/3.10 || { echo "Python module cannot be loaded."; exit 1; }
  module load libffi
  # Set PYTHONPATH to current directory
  # export PYTHONPATH=$(pwd)
fi

if [ "$no_cluster" = true ]; then 
  pip install poetry
fi

# Run Python script if -p flag is set
if [ "$run_python" = true ]; then
  python download_climateset.py || { echo "Failed to run download_climateset.py"; exit 1; }
fi

# Run Bash script if -b flag is set
if [ "$run_bash" = true ]; then
  bash download_climateset.sh || { echo "Failed to run download_climateset.sh"; exit 1; }
fi

# Run additional Bash script if -c flag is set
if [ "$run_checkpoints" = true ]; then
  bash download_climax_checkpoints.sh || { echo "Failed to run download_climax_checkpoints.sh"; exit 1; }
fi

# Check if env_emulator folder exists
if [ ! -d "env_emulator" ]; then
    # Create a virtual environment
    python -m venv env_emulator || { echo "Failed to create virtual environment."; exit 1; }
fi

# Activate the virtual environment
if [ "$on_windows" = false ]; then
    source env_emulator/bin/activate || { echo "Failed to activate virtual environment."; exit 1; }
else 
    env_emulator/Scripts/activate || { echo "Failed to activate virtual environment."; exit 1; }
fi 

# Install the emulator package in editable mode 
poetry install --all-groups || { echo "Failed to install emulator package."; exit 1; }
