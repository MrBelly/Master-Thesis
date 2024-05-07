#!/bin/bash -l
#
# allocate 1 node for 24 hours
#SBATCH -p v100 --gres=gpu:v100:1 --time=24:00:00
#
#job name
#PBS -N tf_job
# activate conda env
source /home/hpc/iwnt/iwnt106h/miniconda3/bin/activate /home/hpc/iwnt/iwnt106h/miniconda3/envs/light_mono/
#load required modules
module load cuda
#run python script
cd /home/hpc/iwnt/iwnt106h/Lite-Mono/
python test_simple.py --load_weights_folder weights --image_path 000000.png


