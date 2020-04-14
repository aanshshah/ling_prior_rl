import os
import numpy as np
import json
import pandas as pd

BASE = 'graphs/'
RELATION_MAP = 'data/mappings/relation_mapping_conceptnet.json'
MAP_BASE = 'data/mappings/'

def get_graph():
	files = os.listdir(BASE)
	name_to_graph = {}
	for file in files:
		if file != '.DS_Store' and 'reshaped' not in file:
			graph = np.load(os.path.join(BASE, file))
			name_to_graph[file] = graph
	return name_to_graph

def get_word_mapping(graph_name):
	path = os.path.join(MAP_BASE, graph_name[:-4]+'_wordmap.json')
	with open(path, 'r') as f:
		words_to_indx = json.load(f)
	idx_to_words = {}
	for word, idx in words_to_indx.items():
		idx_to_words[idx] = word
	all_words = list(words_to_indx.keys())
	return idx_to_words, all_words

def get_relation_mapping():
	with open(RELATION_MAP, 'r') as f:
		relations_to_index = json.load(f)
	idx_to_relations = {}
	for relation, idx in relations_to_index.items():
		idx_to_relations[idx] = relation
	all_relations = list(relations_to_index.keys())
	return idx_to_relations, all_relations

def calculate_connections(graph_mapping, relation_map, all_relations):
	connections = {}
	frequencies = {}
	for graph_name, graph in graph_mapping.items():
		idx_to_word, all_words = get_word_mapping(graph_name[:-11])
		connections[graph_name] = {r: 0 for r in all_relations}
		frequency = {w: {r: 0 for r in all_relations} for w in all_words}
		word_freq = {}
		for w in all_words:
			word_freq[w] = 0
		# word_freq = {w: for w in all_words}
		total_relations = 0
		for w1_i, row in enumerate(graph):
			word_one = idx_to_word[w1_i]
			for w2_i, relation in enumerate(row):
				word_two = idx_to_word[w2_i]
				for i, value in enumerate(relation):
					relation_name = relation_map[i]
					if value > 0:
						if word_one != word_two:
							word_freq[word_one] += 1
							word_freq[word_two] += 1
						total_relations += 1
						connections[graph_name][relation_name] += 1
						frequency[word_one][relation_name] += 1
						frequency[word_two][relation_name] += 1
		for word, freq in word_freq.items():
			frequency[word]['total'] = freq
		frequencies[graph_name] = frequency
		connections[graph_name]['total_relations'] = total_relations
		connections[graph_name]['total_entities'] = graph.shape[0]
	return connections, frequencies

def main():
	graph_mapping = get_graph()
	relation_map, all_relations = get_relation_mapping()
	connections, frequencies = calculate_connections(graph_mapping, relation_map, all_relations)
	df_connect = pd.DataFrame.from_dict(connections, orient='index')
	# df_freq = pd.DataFrame.from_dict({(i,j): frequency[i][j] for i in frequency.keys() for j in frequency[i].keys()}, orient='index')
	print(df_connect)
	for name, frequency in frequencies.items():
		print(name)
		df = pd.DataFrame.from_dict(frequency, orient='index')
		print(df.nlargest(n=20, columns='total'))

if __name__ == '__main__':
	main()