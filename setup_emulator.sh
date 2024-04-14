#!/bin/bash
# Copy: 
# source env_emulator/bin/activate 
# module load python/3.10
# export PYTHONPATH=$(pwd)


# Default values for flags
run_python=false
run_bash=false
run_checkpoints=false
no_cluster = false

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
      no_cluster = true
      ;;
    h )
      echo "Usage: $0 [-p] [-b] [-c] [-n]"
      echo "Options:"
      echo "  -p: Run python download_climateset.py"
      echo "  -b: Run bash download_climateset.sh if you are within Canada instead of -p"
      echo "  -c: Run bash download_climax_checkpoints.sh"
      echo "  -n: no access to the mila cluster"

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
  export PYTHONPATH=$(pwd)
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

echo "nothing"

# Check if env_emulator folder exists
if [ ! -d "env_emulator" ]; then
    # Create a virtual environment
    python -m venv env_emulator || { echo "Failed to create virtual environment."; exit 1; }
fi

# Activate the virtual environment
source env_emulator/bin/activate || { echo "Failed to activate virtual environment."; exit 1; }

# Install requirements
pip install -r requirements.txt || { echo "Failed to install requirements."; exit 1; }
pip install -r requirements_climax.txt || { echo "Failed to install requirements."; exit 1; }


# Change directory to emulator folder
cd emulator || { echo "Failed to change directory to emulator."; exit 1; }

# Install the emulator package in editable mode
pip install -e . || { echo "Failed to install emulator package."; exit 1; }
