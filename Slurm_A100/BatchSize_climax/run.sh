pwd

while true; do
    read -p "Do you wish to remove old files? " yn
    case $yn in
        [Yy]* ) rm -f ../Slurm/{*,.*}; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done


sbatch Slurm_A100/BatchSize_climax/4
sbatch Slurm_A100/BatchSize_climax/8
sbatch Slurm_A100/BatchSize_climax/16
sbatch Slurm_A100/BatchSize_climax/32
sbatch Slurm_A100/BatchSize_climax/64
sbatch Slurm_A100/BatchSize_climax/128
sbatch Slurm_A100/BatchSize_climax/256
sbatch Slurm_A100/BatchSize_climax/512

echo "Success - Go sleep and see your results in the morning!"