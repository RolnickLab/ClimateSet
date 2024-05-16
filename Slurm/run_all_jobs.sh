pwd

while true; do
    read -p "Do you wish to remove old files? " yn
    case $yn in
        [Yy]* ) rm -f ../Slurm/{*,.*}; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done


##  RUN 1
# sbatch Slurm/climax
# sbatch Slurm/climax_frozen
# sbatch Slurm/convlstm
# sbatch Slurm/unet
# sbatch Slurm/climax_36
sbatch Slurm/climax_frozen_36
sbatch Slurm/convlstm_36
# sbatch Slurm/unet_36

# sbatch Slurm/Pin_Memory/climax
# sbatch Slurm/Pin_Memory/climax_frozen
# sbatch Slurm/Pin_Memory/convlstm
# sbatch Slurm/Pin_Memory/unet

# # # RUN 2
# sbatch Slurm/BatchSize_climax/4
# sbatch Slurm/BatchSize_climax/8
# sbatch Slurm/BatchSize_climax/16
# sbatch Slurm/BatchSize_climax/32

sbatch Slurm/BatchSize_climax_frozen/4
sbatch Slurm/BatchSize_climax_frozen/8
sbatch Slurm/BatchSize_climax_frozen/16
sbatch Slurm/BatchSize_climax_frozen/32

# # # RUN 3
sbatch Slurm/BatchSize_convlstm/4
sbatch Slurm/BatchSize_convlstm/8
sbatch Slurm/BatchSize_convlstm/16
sbatch Slurm/BatchSize_convlstm/32

sbatch Slurm/BatchSize_unet/4
sbatch Slurm/BatchSize_unet/8
sbatch Slurm/BatchSize_unet/16
sbatch Slurm/BatchSize_unet/32

# RUN 4 -- CHANGE THE GROUP NAME TO A100!!!
# sbatch Slurm/A100/climax
# sbatch Slurm/A100/climax_frozen
# sbatch Slurm/A100/convlstm
# sbatch Slurm/A100/unet






# sbatch Slurm/MPT/climax
# sbatch Slurm/MPT/climax_frozen
# sbatch Slurm/MPT/convlstm
# sbatch Slurm/MPT/unet




echo "Success - Go sleep and see your results in the morning!"
