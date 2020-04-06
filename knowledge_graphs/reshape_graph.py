import argparse
import numpy as np 
import os

BASE_PATH = 'graphs/'

def reshape_graph(graph, num_relations, num_entities):
	n = num_entities*num_relations
	new_graph = np.zeros(shape=(n,n))
	for i in range(num_entities):
		entity_relations = []
		for j in range(num_entities):
			relations = graph[i, j, :]
			entity_relations.extend(relations)
		new_graph[i] = entity_relations
	assert np.sum(new_graph) == np.sum(graph)
	return new_graph

def save_graph(graph, new_name):
	np.save(os.path.join(BASE_PATH, new_name), graph)

def get_num_entities(graph):
	assert graph.shape[0] == graph.shape[1]
	return graph.shape[0]

def get_num_relations(graph):
	return graph.shape[2]

def get_graph(path):
	return np.load(path)

def main(args):
	graph = get_graph(os.path.join(BASE_PATH, args.graph_name))
	num_entities = get_num_entities(graph)
	num_relations = get_num_relations(graph)
	reshaped_graph = reshape_graph(graph, num_relations, num_entities)
	save_graph(reshaped_graph, args.new_name)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('graph_name', help="Name of the numpy graph file")
	parser.add_argument('new_name', help="Name of the new numpy graph file")
	args = parser.parse_args()
	main(args)