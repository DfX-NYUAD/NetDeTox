#!/bin/bash

#Define the resource requirements here using #SBATCH
#SBATCH --job-name=b22_original
#SBATCH -p nvidia
#SBATCH --gres=gpu:1 
#SBATCH --mem=10G
#SBATCH -t 24:00:00
#SBATCH --output=ct_example_%j.out 


set -euo pipefail
module purge
module load gcc/9.2.0   
source ~/.bashrc
# conda activate circuit-tf
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

nvidia-smi
python -u ct_example.py