

module load python/3.10
export PYTHONPATH=$(pwd)

# 2. Load DL Framework

module load cuda/10.0/cudnn/7.6
module load python/3.7/cuda/11.1/cudnn/8.0/pytorch/1.8.1
    

# 3. Create or Set Up Environment
deactivate

if [ -a env_new_emulator/bin/activate ]; then

source env_new_emulator/bin/activate
echo "activated"

echo $PYTHONPATH
dir
cd $(pwd)


export NCCL_BLOCKING_WAIT=1 #Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.


# 8. Run Python
export HYDRA_FULL_ERROR=1
echo "Running python test.py ..."
python emulator/run.py experiment=Batchsize_A100_Experiments/superemulator_convlstm_BS4.yaml seed=3423


# 9. Copy output to scratch
#cp /home/mila/f/felix-andreas.nahrstedt/Slurm/SC6-A100-basetest-convlstm_out.out 

# 10. Experiment is finished

echo "Experiment $EXP_NAME is concluded."