pwd

while true; do
    read -p "Do you wish to remove old files? " yn
    case $yn in
        [Yy]* ) rm -f ../Slurm/{*,.*}; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done


sbatch Slurm/A100/Baseline/climax
# sbatch Slurm/A100/Baseline/climax_36
sbatch Slurm/A100/Baseline/climax_frozen
# sbatch Slurm/A100/Baseline/climax_frozen_36
sbatch Slurm/A100/Baseline/convlstm
# sbatch Slurm/A100/Baseline/convlstm_36
sbatch Slurm/A100/Baseline/unet
# sbatch Slurm/A100/Baseline/unet_36

echo "Success - Go sleep and see your results in the morning!"