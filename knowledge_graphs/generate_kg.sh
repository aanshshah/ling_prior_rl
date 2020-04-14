#!/bin/bash
# Example usage: ./generate_kg.sh all_relations
cd graphs/
rm -r *
cd ..
python create_graph.py $1 all_minecraft_filtered minecraft_with_wordnet --aug --syn --depth 3 --wordnet
python create_graph.py $1 thor_savn_filtered thor_with_wordnet --aug --syn --depth 3 --wordnet
python create_graph.py $1 all_minecraft_filtered minecraft_without_wordnet --aug --syn --depth 3
python create_graph.py $1 thor_savn_filtered thor_without_wordnet --aug --syn --depth 3
python analyze_kg.py