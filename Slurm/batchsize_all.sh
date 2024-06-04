pwd

while true; do
    read -p "Do you wish to remove old files? " yn
    case $yn in
        [Yy]* ) rm -f ../Slurm/{*,.*}; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done


# sbatch Slurm/BatchSize_convlstm/4
# sbatch Slurm/BatchSize_convlstm/8
# sbatch Slurm/BatchSize_convlstm/16
# sbatch Slurm/BatchSize_convlstm/32

# sbatch Slurm/BatchSize_unet/4
# sbatch Slurm/BatchSize_unet/8
# sbatch Slurm/BatchSize_unet/16
# sbatch Slurm/BatchSize_unet/32

sbatch Slurm/BatchSize_climax_frozen/4
sbatch Slurm/BatchSize_climax_frozen/8
sbatch Slurm/BatchSize_climax_frozen/16
sbatch Slurm/BatchSize_climax_frozen/32

sbatch Slurm/BatchSize_climax/4
sbatch Slurm/BatchSize_climax/8
sbatch Slurm/BatchSize_climax/16
sbatch Slurm/BatchSize_climax/32


echo "Success - Go do something else now :)!"