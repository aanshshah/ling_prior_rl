#!/bin/bash

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu --gres=gpu:2

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -t 24:00:00

# Specify a job name:
#SBATCH -J graph_dqn_warehouse_train

# Specify an output file
#SBATCH -o warehouse.out
#SBATCH -e warehouse.err


# Set up the environment by loading modules
module load cuda/9.2.148 cudnn/7.6.5

# source venv 
source ~/.bashrc
conda activate graph-rl

# Run a script
python scripts/warehouse_v1/train_dqn.py with env_configs/warehouse_v1/one_one.json model_configs/reduce.json model_configs/qhead.json hyper_configs/slow.json agent.opt.kwargs.lr=1e-4 agent.min_train_episodes=40000
conda deactivate

