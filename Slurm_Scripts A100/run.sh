pwd

while true; do
    read -p "Do you wish to remove old files? " yn
    case $yn in
        [Yy]* ) rm -f ../Slurm/{*,.*}; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done


sbatch Slurm_Scripts/climax
sbatch Slurm_Scripts/climax_36
sbatch Slurm_Scripts/climax_frozen
sbatch Slurm_Scripts/climax_frozen_36
sbatch Slurm_Scripts/convlstm
sbatch Slurm_Scripts/convlstm_36
sbatch Slurm_Scripts/unet
sbatch Slurm_Scripts/unet_36

echo "Success - Go sleep and see your results in the morning!"