pwd

while true; do
    read -p "Do you wish to remove old files? " yn
    case $yn in
        [Yy]* ) rm -f ../Slurm/{*,.*}; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done

sbatch Slurm/Pin_Memory/climax
sbatch Slurm/Pin_Memory/climax_frozen
sbatch Slurm/Pin_Memory/convlstm
sbatch Slurm/Pin_Memory/unet



echo "Success - Go do something else now :)!"