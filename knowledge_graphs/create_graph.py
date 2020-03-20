import requests
import numpy as np
import argparse

BASE_URL = "http://api.conceptnet.io"

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
	word_to_idx = {entities[i] : i for i in range(num_entities)}
	relation_to_idx = {relations[i] : i for i in range(num_relations)}
	for relation in relations:
		relation_idx = relation_to_idx[relation]
		for word_one in entities:
			word_one_idx = word_to_idx[word_one]
			for word_two in entities:
				graph[word_one_idx, word_to_idx[word_two], relation_idx] = \
				determine_relationship(relation, word_one, word_two)
	return graph 

def parse(filename):
	with open(filename, 'r') as f:
		content = f.readlines()
		content = [x.strip() for x in content]
	return content



def main(args):
	entities = parse(args.relations_filename) 
	relations = parse(args.entities_filename)
	graph = create_graph(entities, relations)
	np.save(args.graph_name, graph)
	return graph

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('relations_filename', help="Name of the relations text file")
	parser.add_argument('entities_filename', help="Name of the entities text file")
	parser.add_argument('graph_name', help="Full path/filename of where the knowledge graph should be saved")
	args = parser.parse_args()
	main(args)