pwd

while true; do
    read -p "Do you wish to remove old files? " yn
    case $yn in
        [Yy]* ) rm -f ../Slurm/{*,.*}; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done


sbatch Slurm/climax
# sbatch Slurm/climax_36
# sbatch Slurm/climax_frozen
# sbatch Slurm/climax_frozen_36
#sbatch Slurm/convlstm
# sbatch Slurm/convlstm_36
# sbatch Slurm/unet
# sbatch Slurm/unet_36

echo "Success - Go sleep and see your results in the morning!"