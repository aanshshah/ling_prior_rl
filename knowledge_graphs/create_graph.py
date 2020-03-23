import random
random.seed(0)
import requests
import numpy as np
import argparse
import pydot
from nltk.corpus import wordnet as wn
import datetime
from time import sleep

BASE_URL = "http://api.conceptnet.io"
COLORS = ['black', 'blue', 'brown', 'green', 'orange', 'purple', 'red','yellow']
LOG_FILE = 'logs/log.txt'

def format_request(relation, word_one, word_two=None):
	query = "/a/"
	if word_one and word_two:
		query += "[/r/{0}/,/c/en/{1}/,/c/en/{2}/]".format(relation, word_one, word_two)
	if not word_two:
		query += "[/r/{0}/,/c/en/{1}/]".format(relation, word_one)
	url = BASE_URL + query
	i = 0
	response = {}
	with open(LOG_FILE, 'a') as f:
		while i < 100000:
			if i>0 and i % 1000 == 0:
				curr_time = str(datetime.datetime.now())
				log_line = "{4}: Try #{0} for relation: {1}, word 1: {2}, word 2: {3}".format(i, relation, word_one, word_two, curr_time)
				f.write(log_line + '\n')
			try:
				response = requests.get(url).json()
				break
			except:
				i += 1
		if i == 100000:
			f.write("{3}: words {0} and {1} failed with relation {2}".format(word_one, word_two, relation, curr_time) + '\n')
			response = {}
	return response

def determine_relationship(relation, word_one, word_two=None):
	try:
		response = format_request(relation, word_one, word_two)
	except:
		raise Exception("There was an error with relation: {0} and words {1} {2}".format(relation, word_one, word_two))
		exit()
	return response['weight'] if response.get('weight') else 0

def create_graph(graph_name, entities, relations):
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
	np.save(graph_name, graph)
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

def create_more_entities(entities):
	def generate_synonyms(new_word):
		return new_word.lemma_names()
	new_entities = set()
	for word in entities:
		for new_word in wn.synsets(word, pos=wn.NOUN):
			new_entities.add(new_word.name().split('.')[0])
			for hypernym in new_word.hypernyms():
				print(new_word.name(), hypernym.name())
				new_entities.add(hypernym.name().split('.')[0])
		new_entities.add(word)
	return list(new_entities)

def add_hypernyms(entities, filename, syn=False):
	def get_LCA_by_distance(w_1, w_2):
		ss_w1 = wn.synsets(w_1, pos=wn.NOUN)
		ss_w2 = wn.synsets(w_2, pos=wn.NOUN)
		midDist = float("inf")
		synObj1, synObj2 = None, None
		lca = None
		for word_one in ss_w1:
			for word_two in ss_w2:
				curDist = word_one.shortest_path_distance(word_two)
				if curDist != None and curDist < midDist:
					midDist = curDist
					lca = word_one.lowest_common_hypernyms(word_two)
					synObj1, synObj2 = word_one, word_two
		return (lca, midDist, synObj1, synObj2)
	def clean_LCA(lca):
		cleaned_LCA = []
		for word in lca:
			cleaned_LCA.append(word.name().split('.')[0])
		return cleaned_LCA

	def write(entities, filename):
		filename = filename.split('.')[0] + '_lca.txt'
		with open(filename, 'w') as f:
			for word in entities:
				f.write(word +'\n')
	def clean_word(word):
		return word.name().split('.')[0]
	new_entities = []
	for word_one in entities:
		for word_two in entities:
			if word_one == word_two: continue
			lca, _, s1, s2 = get_LCA_by_distance(word_one, word_two)
			s1 = clean_word(s1)
			s2 = clean_word(s2)
			lca = clean_LCA(lca)
			new_entities += lca
			if syn:
				new_entities += s1
				new_entities += s2
			
	all_entities = list(set(new_entities + entities))
	write(all_entities, filename)
	return all_entities

def visualize_graph(graph_matrix, entities, relations, graph_name):
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
	if args.aug:
		entities = add_hypernyms(entities, args.entities_filename, syn=args.syn)
	if args.viz:
		graph_matrix = np.load(args.graph_name)
		visualize_graph(graph_matrix, entities, relations, args.graph_name)
	else:
		create_graph(args.graph_name, entities, relations)
		

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('relations_filename', help="Name of the relations text file")
	parser.add_argument('entities_filename', help="Name of the entities text file")
	parser.add_argument('graph_name', help="Name of knowledge graph")
	parser.add_argument('--viz', action="store_true", help="Visualize an existing graph", default=False)
	parser.add_argument('--aug', action="store_true", help="Augment existing entities with lowest common hypernyms")
	parser.add_argument('--syn', action="store_true", help="Augment existing entities with lowest common hypernyms and synonyms")
	args = parser.parse_args()
	main(args)