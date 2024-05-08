
## SLURM FILE

#!/bin/bash -l
#
# Allocate 2 nodes with 2 GPUs each for 24 hours
#SBATCH -p v100 --gres=gpu:v100:2 --nodes=1 --ntasks-per-node=2 --cpus-per-task=2 --time=24:00:00
#
# Job name
#SBATCH -J tf_job

# Activate conda environment
source /home/hpc/iwnt/iwnt106h/miniconda3/bin/activate /home/hpc/iwnt/iwnt106h/miniconda3/envs/lite_mono_new/

# Load required modules
module load cuda

# Set the number of GPUs and nodes
NUM_GPUS_PER_NODE=2
NUM_NODES=2

# SLURM environment variables
export CUDA_VISIBLE_DEVICES=0,1
export SLURM_PROCID=$SLURM_PROCID
export SLURM_NTASKS=$SLURM_NTASKS
export SLURM_NODELIST=$SLURM_NODELIST

# Run the Python script with distributed setup
cd /home/hpc/iwnt/iwnt106h/Pre_train_Swiftformer_Imagenet_multi_gpu/Lite-Mono/lite-mono-pretrain-code/

# Assuming your script is named main.py and contains the init_distributed_mode function

python -m torch.distributed.launch --nproc_per_node=2 main.py --data_path /home/woody/iwnt/iwnt106h/Imagenet/ --epochs 150 --auto_resume True

# Note: Adjust the command line arguments based on your actual script and requirements
