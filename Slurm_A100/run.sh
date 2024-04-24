pwd

while true; do
    read -p "Do you wish to remove old files? " yn
    case $yn in
        [Yy]* ) rm -f ../Slurm/{*,.*}; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done


# sbatch Slurm_A100/climax
# sbatch Slurm_A100/climax_36
# sbatch Slurm_A100/climax_frozen
# sbatch Slurm_A100/climax_frozen_36
sbatch Slurm_A100/convlstm
# sbatch Slurm_A100/convlstm_36
# sbatch Slurm_A100/unet
# sbatch Slurm_A100/unet_36

echo "Success - Go sleep and see your results in the morning!"