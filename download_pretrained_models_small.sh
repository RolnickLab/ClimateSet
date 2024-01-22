
echo "Downloading pretrained models' checkpoints..."
curl https://object-arbutus.cloud.computecanada.ca/causalpaca/pretrained_models_small.tar.gz --output pretrained_models_small.tar.gz

echo "Done. Finished downloading the compressed files, now extracting!"

tar -xzf "pretrained_models_small.gz"; rm -f "pretrained_models_small.gz";


echo "Done. Finished downloading pretrained models. :)"
