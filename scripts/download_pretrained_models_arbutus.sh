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


