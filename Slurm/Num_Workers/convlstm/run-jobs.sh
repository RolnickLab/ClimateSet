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
sbatch Slurm/Num_Workers/convlstm/0
sbatch Slurm/Num_Workers/convlstm/2
sbatch Slurm/Num_Workers/convlstm/4
sbatch Slurm/Num_Workers/convlstm/6
sbatch Slurm/Num_Workers/convlstm/8


