#!/bin/bash
#SBATCH --job-name=upload_models
#SBATCH --output=upload_models.out
#SBATCH --error=error.out
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --mem=30Gb
#SBATCH --partition=long
#SBATCH -c 4

module add python/3.10
source $HOME/env_arbutus/bin/activate

# Uncomment unwanted experiments / models
# Be aware that the single emulator experiment folder containing ClimaX checkpoints is quite large (13Gb) and may take some time
declare -a experiments=("single_emulator" "finetuning_emulator" "super_emulator" )
declare -a models=("ClimaX_frozen" "ClimaX"  "CNN_LSTM_ClimateBench" "Unet")

echo "Downloading pretrained models' checkpoints..."

for exp in "${experiments[@]}"; do
    for m in "${models[@]}"; do
        path="pretrained_models/"$exp"/"$m""
        echo "Downloading "$exp" "$m""
        file=""$path".zip"
        link="https://object-arbutus.cloud.computecanada.ca/causalpaca/"$file"" 
        curl $link --output $file  --create-dirs # bild correct string
        # unpack
        unzip $file -d $path
        # remove zip
        rm -f $file
        echo "Finished "$exp" "$m""

    done
done

echo "Done. Finished downloading pretrained models. :)"

