#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH -J THOR-KG
#SBATCH -o THOR_KG.out
#SBATCH -e THOR_KG.out

module load anaconda/3-5.2.0
cd ..
python --aug --syn create_graph.py all_thor all_relations all_thor