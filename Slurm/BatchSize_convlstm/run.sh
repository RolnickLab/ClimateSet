pwd

while true; do
    read -p "Do you wish to remove old files? " yn
    case $yn in
        [Yy]* ) rm -f ../Slurm/{*,.*}; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done


sbatch Slurm/BatchSize_convlstm/4
sbatch Slurm/BatchSize_convlstm/8
sbatch Slurm/BatchSize_convlstm/16
sbatch Slurm/BatchSize_convlstm/32



echo "Success - Go sleep and see your results in the morning!"