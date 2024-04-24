pwd

while true; do
    read -p "Do you wish to remove old files? " yn
    case $yn in
        [Yy]* ) rm -f ../Slurm/{*,.*}; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done


sbatch Slurm_A100/BatchSize_climax_frozen/4
sbatch Slurm_A100/BatchSize_climax_frozen/8
sbatch Slurm_A100/BatchSize_climax_frozen/16
sbatch Slurm_A100/BatchSize_climax_frozen/32
sbatch Slurm_A100/BatchSize_climax_frozen/64
sbatch Slurm_A100/BatchSize_climax_frozen/128
sbatch Slurm_A100/BatchSize_climax_frozen/256
sbatch Slurm_A100/BatchSize_climax_frozen/512

echo "Success - Go sleep and see your results in the morning!"