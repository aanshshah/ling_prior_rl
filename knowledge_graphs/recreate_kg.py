import argparse
import numpy as np
import pydot
import os
import json

BASE = 'graphs/'
RELATION_MAP = 'data/mappings/relation_mapping_conceptnet.json'
MAP_BASE = 'data/mappings/'
VIZ_BASE = 'visualizations/subsets/'
COLORS = ['black', 'blue', 'brown', 'green', 'orange', 'purple', 'red', 'yellow', 'indigo', 'turqoise',
          'pink', 'olive', 'crimson', 'steelblue', 'lime']


def get_graph():
	files = os.listdir(BASE)
	name_to_graph = {}
	for file in files:
		if file != '.DS_Store' and 'reshaped' not in file:
			graph = np.load(os.path.join(BASE, file))
			name_to_graph[file] = graph
	return name_to_graph

def get_relation_mapping():
	with open(RELATION_MAP, 'r') as f:
		relations_to_index = json.load(f)
	idx_to_relations = {}
	for relation, idx in relations_to_index.items():
		idx_to_relations[idx] = relation
	all_relations = list(relations_to_index.keys())
	return idx_to_relations, all_relations

def get_word_mapping(graph_name):
	path = os.path.join(MAP_BASE, graph_name[:-4]+'_wordmap.json')
	with open(path, 'r') as f:
		words_to_indx = json.load(f)
	idx_to_words = {}
	for word, idx in words_to_indx.items():
		idx_to_words[idx] = word
	all_words = list(words_to_indx.keys())
	return idx_to_words, all_words, words_to_indx

def get_mapping(thing_map):
	with open(thing_map, 'r') as f:
		thing_to_index = json.load(f)
	idx_to_thing = {}
	for thing, idx in thing_to_index.items():
		idx_to_thing[idx] = thing
	all_things = list(thing_to_index.keys())
	return idx_to_thing, all_things

def create_graphs(subset_relations, graphs, idx_to_relation, relations):
	for graph_name, graph in graphs.items():
		if 'without' in graph_name: continue
		pygraph = pydot.Dot(graph_type='graph')
		idx_to_word, words, word_to_idx = get_word_mapping(graph_name[:-11])
		pygraph = create_graph(pygraph, subset_relations, graph, idx_to_relation, idx_to_word, relations, words)
		write_graph(subset_relations, graph_name)

def write_graph(pygraph, subset_relations, graph_name):
	viz_name = os.path.join(VIZ_BASE, 'subset_{0}_{1}.png'.format("_".join(subset_relations), graph_name))
	pygraph.write_png(viz_name)
	print("Finished {0}".format(viz_name))

def create_graph_based_on_words(subset_relations, name_to_graph, idx_to_relation, relations, subset_words):
	def should_skip(name):
		if 'without' in name: return True
		if 'thor' in name: return True
		return False
	for graph_name, graph in name_to_graph.items():
		if should_skip(graph_name): continue
		pygraph = pydot.Dot(graph_type='graph')
		idx_to_word, words, word_to_idx = get_word_mapping(graph_name[:-11])
		word_to_node = {}
		for word in words:
			node = pydot.Node(word)
			word_to_node[word] = node
		word_idx_connections = []
		max_depth = 3
		curr_depth = 0
		for s_word in subset_words:
			word_idx = word_to_idx[s_word]
			word_idx_connections.append((word_idx, curr_depth))
		added = set()
		while word_idx_connections:
			word_idx, word_depth = word_idx_connections.pop()
			curr_depth = word_depth
			if word_depth > max_depth:
				break
			word_one = idx_to_word[word_idx]
			node_one = word_to_node[word_one]
			features = graph[word_idx]
			for connected_word_idx, relation_list in enumerate(features):
				word_two = idx_to_word[connected_word_idx]
				node_two = word_to_node[word_two]
				for relation_idx, value in enumerate(relation_list):
					relation_name = idx_to_relation[relation_idx]
					if value > 0 and relation_name in subset_relations:
						to_be_added = (word_one, word_two, relation_name)
						other_possible = (word_two, word_one, relation_name)
						if not (to_be_added in added or other_possible in added):
							pygraph.add_node(node_one)
							pygraph.add_node(node_two)
							edge = pydot.Edge(node_one, node_two, label=relation_name, color=COLORS[relation_idx])
							pygraph.add_edge(edge)
							word_idx_connections.append((connected_word_idx, curr_depth+1))
							added.add(to_be_added)
							added.add(other_possible)
		write_graph(pygraph, subset_relations, graph_name+'words')

def create_graph(pygraph, subset_relations, graph, idx_to_relation, idx_to_word, relations, words):
	word_to_node = {}
	for word in words:
		node = pydot.Node(word)
		word_to_node[word] = node
	for w1_i, row in enumerate(graph):
		word_one = idx_to_word[w1_i]
		node_one = word_to_node[word_one]
		for w2_i, relation in enumerate(row):
			word_two = idx_to_word[w2_i]
			node_two = word_to_node[word_two]
			for r_i, value in enumerate(relation):
				relation_name = idx_to_relation[r_i]
				if value > 0 and relation_name in subset_relations:
					pygraph.add_node(node_one)
					pygraph.add_node(node_two)
					edge = pydot.Edge(node_one, node_two, label=relation_name, color=COLORS[r_i])
					pygraph.add_edge(edge)
	return pygraph

def main(args):
	subset_relations = set(args.relations)
	subset_words = set(args.words)
	idx_to_relation, relations = get_relation_mapping()
	name_to_graph = get_graph()
	if args.relations and not args.words:
		create_graphs(subset_relations, name_to_graph, idx_to_relation, relations)
	if args.words:
		create_graph_based_on_words(subset_relations, name_to_graph, idx_to_relation, relations, subset_words)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--relations', nargs='+', help='relations to visualize separated by spaces')
	parser.add_argument('--words', nargs='+', help="Words to visualize all the words")
	args = parser.parse_args()
	main(args)