import numpy as np
import json

def read_graph(graph_name):
    return np.load("../graphs/{0}".format(graph_name))

def create_mapping(graph, relation_to_idx, word_to_idx):
    relations = list(relation_to_idx.keys())
    words = list(word_to_idx.keys())
    idx_to_relation = {v : k for k, v in relation_to_idx.items()}
    idx_to_word = {v : k for k, v in word_to_idx.items()}
    all_relationships = {}
    for word_one_idx, entity_row in enumerate(graph):
        word_one = idx_to_word[word_one_idx]
        relation_map = all_relationships.get(word_one, {})
        for word_two_idx, relations in enumerate(entity_row):
            for relation_idx, value in enumerate(relations):
                if value:
                    word_two = idx_to_word[word_two_idx]
                    relation_name = idx_to_relation[relation_idx]
                    curr_relations = relation_map.get(relation_name, [])
                    curr_relations.append(word_two)
                    relation_map[relation_name] = curr_relations
        if relation_map:
            all_relationships[word_one] = relation_map
            
    return all_relationships

def get_map(path):
    with open(path, 'r') as f:
        mapping = json.load(f)
    return mapping

# def save_mapping(path, mapping):
#     with open(path, 'w') as f:
#         json.dump(dic)

def main():
    graph_name = "minecraft_v0_conceptnet.npy"
    word_to_idx = get_map("../data/mappings/minecraft_v0_wordmap.json")
    relation_to_idx = get_map("../data/mappings/relation_mapping_conceptnet.json")
    graph = read_graph(graph_name)
    all_relationships = create_mapping(graph, relation_to_idx, word_to_idx)
    print(all_relationships)

if __name__ == '__main__':
    main()