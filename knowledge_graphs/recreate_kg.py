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
	return idx_to_words, all_words

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
		idx_to_word, words = get_word_mapping(graph_name[:-11])
		pygraph = create_graph(pygraph, subset_relations, graph, idx_to_relation, idx_to_word, relations, words)
		viz_name = os.path.join(VIZ_BASE, 'subset_{0}_{1}.png'.format("_".join(subset_relations), graph_name))
		pygraph.write_png(viz_name)
		print("Finished {0}".format(viz_name))

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
	idx_to_relation, relations = get_relation_mapping()
	name_to_graph = get_graph()
	create_graphs(subset_relations, name_to_graph, idx_to_relation, relations)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('relations', nargs='+', help='relations to visualize separated by spaces')
	parser.add_argument('--all', action="store_false", help="Specify if you want all to visualize all the words", default=True)
	args = parser.parse_args()
	main(args)