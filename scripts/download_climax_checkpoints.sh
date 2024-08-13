# Change the directory where the data will be downloaded below
mkdir -p "emulator/src/core/models/climax/pretrained_checkpoints/"
cd "emulator/src/core/models/climax/pretrained_checkpoints/"
echo "Downloading pretrained ClimaX weights from the original release..."
curl -L "https://huggingface.co/tungnd/climax/resolve/main/5.625deg.ckpt" --output ClimaX-5.625deg.ckpt
curl -L "https://huggingface.co/tungnd/climax/resolve/main/1.40625deg.ckpt" --output ClimaX-1.40625deg.ckpt
echo "Done."