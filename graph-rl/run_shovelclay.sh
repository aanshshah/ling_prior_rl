#!/bin/bash

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu --gres=gpu:2

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=16G
#SBATCH -c 6
#SBATCH -t 24:00:00

# Specify a job name:
#SBATCH -J graph_dqn_malmo_train

# Specify an output file
#SBATCH -o malmo_shovel_clay.out 
#SBATCH -e malmo_shovel_clay.err


# Set up the environment by loading modules
module load cuda/9.2.148 cudnn/7.6.5

# source venv 
source ../../reai_venv/bin/activate

# Run a script
unbuffer python scripts/malmo_v0/train_dqn.py with pickaxe_skyline.json model_configs/reduce.json model_configs/qhead.json hyper_configs/slow.json agent.opt.kwargs.lr=1e-4 agent.min_train_episodes=40000 addr=172.25.201.5 port=9001 agent.experiement_name='shovel_clay'
deactivate
