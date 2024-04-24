pwd

while true; do
    read -p "Do you wish to remove old files? " yn
    case $yn in
        [Yy]* ) rm -f ../Slurm/{*,.*}; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done


# sbatch Slurm_A100/BatchSize/convlstm_4
sbatch Slurm_A100/BatchSize/convlstm_8
# sbatch Slurm_A100/BatchSize/convlstm_16
# sbatch Slurm_A100/BatchSize/convlstm_32


# sbatch Slurm_A100/BatchSize/convlstm_64
# sbatch Slurm_A100/BatchSize/convlstm_128
# sbatch Slurm_A100/BatchSize/convlstm_256
# sbatch Slurm_A100/BatchSize/convlstm_512

echo "Success - Go sleep and see your results in the morning!"