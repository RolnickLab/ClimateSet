pwd

while true; do
    read -p "Do you wish to remove old files? " yn
    case $yn in
        [Yy]* ) rm -f ../Slurm/{*,.*}; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done


sbatch Slurm_A100/BatchSize/unet_4
sbatch Slurm_A100/BatchSize/unet_8
sbatch Slurm_A100/BatchSize/unet_16
sbatch Slurm_A100/BatchSize/unet_32
sbatch Slurm_A100/BatchSize/unet_64
sbatch Slurm_A100/BatchSize/unet_128
sbatch Slurm_A100/BatchSize/unet_256
sbatch Slurm_A100/BatchSize/unet_512

echo "Success - Go sleep and see your results in the morning!"