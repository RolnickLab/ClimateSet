pwd

while true; do
    read -p "Do you wish to remove old files? " yn
    case $yn in
        [Yy]* ) rm -f ../Slurm/{*,.*}; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done

sbatch Slurm/MPT/climax
sbatch Slurm/MPT/climax_frozen
sbatch Slurm/MPT/convlstm
sbatch Slurm/MPT/unet



echo "Success - Go do something else now :)!"