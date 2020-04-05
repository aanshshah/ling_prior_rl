#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=20G
#SBATCH -J THOR-KG
#SBATCH -o THOR_KG.out
#SBATCH -e THOR_KG.out

module load anaconda/3-5.2.0
cd ..
python create_graph.py --aug --syn all_relations all_thor all_thor