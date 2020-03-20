import random
random.seed(0)
import requests
import numpy as np
import argparse
import pydot

BASE_URL = "http://api.conceptnet.io"
COLORS = ['black', 'blue', 'brown', 'green', 'orange', 'purple', 'red','yellow']

def format_request(relation, word_one, word_two=None):
	query = "/a/"
	if word_one and word_two:
		query += "[/r/{0}/,/c/en/{1}/,/c/en/{2}/]".format(relation, word_one, word_two)
	if not word_two:
		query += "[/r/{0}/,/c/en/{1}/]".format(relation, word_one)
	url = BASE_URL + query
	return requests.get(url).json()

def determine_relationship(relation, word_one, word_two=None):
	response = format_request(relation, word_one, word_two)
	return response['weight'] if response.get('weight') else 0

def create_graph(entities, relations):
	num_entities = len(entities)
	num_relations = len(relations)
	graph = np.zeros(shape=(num_entities, num_entities, num_relations))
	word_to_idx = generate_word_mappings(entities)
	relation_to_idx = generate_word_mappings(relations)
	for relation in relations:
		relation_idx = relation_to_idx[relation]
		for word_one in entities:
			word_one_idx = word_to_idx[word_one]
			for word_two in entities:
				graph[word_one_idx, word_to_idx[word_two], relation_idx] = \
				determine_relationship(relation, word_one, word_two)
	np.save('graphs/'+args.graph_name, graph)
	return graph 

def generate_word_mappings(entities):
	return {entities[i] : i for i in range(len(entities))}

def generate_index_mappings(entities):
	return {i: entities[i] for i in range(len(entities))}	

def parse(filename):
	with open(filename, 'r') as f:
		content = f.readlines()
		content = [x.strip() for x in content]
	return content

def visualize_graph(graph_matrix, entities, relations, graph_name):
	def generate_color_scheme():
		pass
	entity_map = generate_index_mappings(entities)
	relation_map = generate_index_mappings(relations)
	graph = pydot.Dot(graph_type='graph')
	word_to_node = {}
	for word in entity_map.values():
		node = pydot.Node(word)
		word_to_node[word] = node
		graph.add_node(node)

	for relation_i in range(graph_matrix.shape[2]):
		relation = relation_map[relation_i]
		for entity_i in range(graph_matrix.shape[0]):
			word_one = entity_map[entity_i]
			node_one = word_to_node[word_one]
			for entity_j in range(graph_matrix.shape[1]):
				word_two = entity_map[entity_j]
				node_two = word_to_node[word_two]
				if graph_matrix[entity_i, entity_j, relation_i] != 0:
					edge = pydot.Edge(node_one, node_two, label=relation, color=COLORS[relation_i])
					graph.add_edge(edge)
	graph_name = graph_name.split('/')[-1][:-4]
	graph.write_png('visualizations/{0}.png'.format(graph_name))

def main(args):
	relations = parse(args.relations_filename)
	entities = parse(args.entities_filename)
	if args.viz:
		graph_matrix = np.load(args.graph_name)
		visualize_graph(graph_matrix, entities, relations, args.graph_name)
	else:
		create_graph(entities, relations)
		

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('relations_filename', help="Name of the relations text file")
	parser.add_argument('entities_filename', help="Name of the entities text file")
	parser.add_argument('graph_name', help="Name of knowledge graph")
	parser.add_argument('--viz', action="store_true", help="Visualize an existing graph", default=False)
	args = parser.parse_args()
	main(args)