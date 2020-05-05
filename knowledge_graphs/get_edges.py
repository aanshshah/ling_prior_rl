import numpy as np 
import json

def get_map(path):
    with open(path, 'r') as f:
        mapping = json.load(f)
    return mapping

word_to_idx = get_map("data/mappings/minecraft_v0_wordmap.json")
idx_to_word = {v : k for k, v in word_to_idx.items()}
relation_to_idx = get_map("data/mappings/relation_mapping_conceptnet.json")
idx_to_relation = {v : k for k, v in relation_to_idx.items()}

graph = np.load("graphs/minecraft_v0_conceptnet.npy")
edges = []
nodes = set()
for word_one_idx, entity_row in enumerate(graph):
    word_one = idx_to_word[word_one_idx]
    nodes.add(word_one)
    for word_two_idx, relations in enumerate(entity_row):
        for relation_idx, value in enumerate(relations):
            if relation_idx > 0: continue
            if value:
                word_two = idx_to_word[word_two_idx]
                relation_name = idx_to_relation[relation_idx]
                edges.append((word_one, word_two))

node_filename = "relatedTo_augmented_edges.txt"
edge_filenmae = "all_nodes.txt"

with open(node_filename, 'w') as f:
    f.write(str(nodes))
with open(edge_filenmae, 'w') as f:
    f.write(str(edges))
