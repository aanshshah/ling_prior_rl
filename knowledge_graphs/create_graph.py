import random
import os
random.seed(0)
import requests
import numpy as np
import argparse
from nltk.corpus import wordnet as wn
import time
import json
from heapq import nlargest

ENTITIES_BASE = 'data/entities/'
GRAPH_BASE = 'graphs/'
RELATIONS_BASE = 'data/relations'
VIZ_BASE = 'visualizations/'
MAPPING_BASE = 'data/mappings/'
CONCEPTNET_PATH = os.path.join(MAPPING_BASE, 'conceptnet_map.txt')

BASE_URL = "http://api.conceptnet.io"
COLORS = ['black', 'blue', 'brown', 'green', 'orange', 'purple', 'red','yellow', 'indigo', 'turqoise', 
			'pink', 'olive', 'crimson', 'steelblue', 'lime']
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
	while i < 1000000: #try request one million times 
		try:
			response = requests.get(url).json()
			break
		except:
			i += 1
			time.sleep(15)
	if i == 1000000: #this should never happen
		response = {} 
	return response

def determine_relationship(relation, word_one, word_two=None, conceptnet_map=None):
	if conceptnet_map:
		try: return conceptnet_map[word_one][relation][word_two]
		except: return 0
	else:
		try:response = format_request(relation, word_one, word_two)
		except: raise Exception("There was an error with relation: {0} and words {1} {2}".format(relation, word_one, word_two))
		return response.get('weight', 0)

def get_path(base, name, syn, conceptnet, ext):
	if conceptnet: embedding = 'conceptnet'
	else: embedding = 'wordnet'
	if syn: graph_path = os.path.join(base, name + '_{0}_with_syn.{1}'.format(embedding, ext))
	else: graph_path = os.path.join(base, name + '_{0}.{1}'.format(embedding, ext))
	return graph_path

def open_dict(filename):
	with open(filename, 'r') as f:
		data = json.load(f)
	return data

def save_dict(data, filename):
	with open(filename, 'w') as f:
		json.dump(data, f)

def reshape_graph(graph):
	def reshape(graph, num_relations, num_entities):
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

	def get_num_entities(graph):
		assert graph.shape[0] == graph.shape[1]
		return graph.shape[0]

	def get_num_relations(graph):
		return graph.shape[2]

	num_relations = get_num_relations(graph)
	num_entities = get_num_entities(graph)
	return reshape(graph, num_relations, num_entities)

def create_graph(graph_name, entities, relations, conceptnet_map, syn):
	num_entities = len(entities)
	num_relations = len(relations)
	graph = np.zeros(shape=(num_entities, num_entities, num_relations))
	word_to_idx = generate_word_mappings(entities, relation=False)
	relation_to_idx = generate_word_mappings(relations, relation=True)
	for relation in relations:
		relation_idx = relation_to_idx[relation]
		for word_one in entities:
			word_one = word_one.lower()
			word_one_idx = word_to_idx[word_one]
			for word_two in entities:
				word_two = word_two.lower()
				word_two_idx = word_to_idx[word_two]
				graph[word_one_idx, word_two_idx, relation_idx] = determine_relationship(relation, word_one, word_two, conceptnet_map)
	graph_path = get_path(GRAPH_BASE, graph_name, syn, conceptnet_map, ext='npy')
	word_map_name = 'word_mapping'
	word_map_path = get_path(MAPPING_BASE, word_map_name, syn, conceptnet_map, ext='json')
	save_dict(word_to_idx, word_map_path)
	relation_map_name = 'relation_mapping'
	relation_map_path = get_path(MAPPING_BASE, relation_map_name, syn, conceptnet_map, ext='json')
	save_dict(relation_to_idx, relation_map_path)
	reshaped_graph = reshape(graph)
	np.save(graph_path, reshaped_graph)
	return graph 

def generate_word_mappings(entities, relation=False):
	if relation:
		return {entities[i] : i for i in range(len(entities))}
	else:
		return {entities[i].lower() : i for i in range(len(entities))}

def generate_index_mappings(entities, relation=False):
	if relation:
		return {i: entities[i] for i in range(len(entities))}	
	else:
		return {i: entities[i].lower() for i in range(len(entities))}	

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
				new_entities.add(hypernym.name().split('.')[0])
		new_entities.add(word)
	return list(new_entities)

def write(entities, filename, conceptnet, syn=False):
	filename = get_path(ENTITIES_BASE, filename.split('.')[0], conceptnet, syn, ext='txt')
	with open(filename, 'w') as f:
		for word in entities:
			if word:
				f.write(word +'\n')

def sanitize_entities(entities):
	sanitized_entities = []
	for entity in entities:
		if entity:
			sanitized_entities.append(entity.lower())
	return sanitized_entities

def get_LCA_by_distance(w_1, w_2):
	ss_w1 = wn.synsets(w_1, pos=wn.NOUN)
	ss_w2 = wn.synsets(w_2, pos=wn.NOUN)
	midDist = float("inf")
	synObj1, synObj2 = None, None
	lca = None
	for word_one in ss_w1:
		for word_two in ss_w2:
			if word_one == word_two: continue
			curDist = word_one.shortest_path_distance(word_two)
			if curDist != None and curDist < midDist:
				midDist = curDist
				lca = word_one.lowest_common_hypernyms(word_two)
				synObj1, synObj2 = word_one, word_two
	return (lca, midDist, synObj1, synObj2)
def clean_word(word):
	return word.name().split('.')[0]

def get_hypernym_with_wordnet(w_1, w_2, syn):
	new_entities = []
	word_one = word_one.lower()
	word_two = word_two.lower()
	if word_one == word_two: return
	lca, _, s1, s2 = get_LCA_by_distance(word_one, word_two)
	if not (s1 and s2): return None
	s1 = clean_word(s1)
	s2 = clean_word(s2)
	if lca[0]: 
		lca = clean_word(lca[0])
		new_entities.append(lca)
	else: return None
	if syn:
		return lca, s1, s2
	else:
		return lca
		
def add_hypernyms_with_wordnet(entities, filename, syn=False):
	new_entities = []
	for word_one in entities:
		word_one = word_one.lower()
		for word_two in entities:
			word_two = word_two.lower()
			if word_one == word_two: continue
			lca, _, s1, s2 = get_LCA_by_distance(word_one, word_two)
			if not (s1 and s2): 
				print("Words do not exist in word net: {0} {1}".format(word_one, word_two))
				continue
			s1 = clean_word(s1)
			s2 = clean_word(s2)
			if lca[0]: 
				lca = clean_word(lca[0])
				new_entities.append(lca)
			else:
				print("Least common hypernym doesn't exist for words {0} {1}".format(word_one, word_two))
			if syn:
				new_entities.append(s1) 
				new_entities.append(s2)
	all_entities = sanitize_entities(list(set(new_entities + entities)))
	write(all_entities, filename, conceptnet=False, syn=syn)
	return all_entities

def add_hypernyms_with_conceptnet(entities, filename, conceptnet_map, depth, syn=0):
	def get_all_hypernyms_for_one_word(word, mapping, depth):
		all_hypernyms = set()
		relation = 'isA'
		relation_map = mapping.get(word, {})
		hypernyms = relation_map.get(relation, {})
		frontier = hypernyms.keys()
		curr_depth = 0
		while frontier:
			curr_word = frontier.pop()
			all_hypernyms.add(curr_word)
			if curr_depth == depth:
				for word in frontier:
					all_hypernyms.add(word)
					break
				curr_depth += 1
			neighbor_relations = mapping.get(curr_word, {})
			neighbor_words = neighbor_relations.get(relation, {})
			for word in neighbor_words.keys():
				frontier.append(word)
		return all_hypernyms
	def get_shared_hypernyms(word_one, word_two, mapping, depth):
		word_one_hypernyms = get_all_hypernyms_for_one_word(word_one, mapping, depth)
		word_two_hypernyms = get_all_hypernyms_for_one_word(word_two, mapping, depth)
		shared_hypernyms = word_one_hypernyms.union(word_two_hypernyms)
		return list(shared_hypernyms)
	def get_synonyms(word, mapping, n=5):
		relation = 'RelatedTo'
		relation_map = mapping.get(word, {})
		synonyms = relation_map.get(relation, {})
		synonyms = nlargest(n, synonyms, key = synonyms.get)
		return synonyms
	new_entities = []
	for word_one in entities:
		word_one = word_one.lower()
		for word_two in entities:
			word_two = word_two.lower()
			if word_one == word_two: continue
			shared_hypernyms = get_shared_hypernyms(word_one, word_two, conceptnet_map, depth)
			if shared_hypernyms: new_entities.extend(shared_hypernyms)
			if syn:
				word_one_synonyms = get_synonyms(word_one, conceptnet_map, n=syn)
				word_two_synonyms = get_synonyms(word_two, conceptnet_map, n=syn)
				if word_one_synonyms: new_entities.extend(word_one_synonyms)
				if word_two_synonyms: new_entities.extend(word_two_synonyms)
	all_entities = sanitize_entities(list(set(new_entities + entities)))
	write(all_entities, filename, conceptnet_map, syn)
	return all_entities

def visualize_graph(entities, relations, graph_name, syn, conceptnet, svg):
	def change_word_edge(entities):
		for i in range(len(entities)):
			entity = entities[i]
			if entity == 'edge':
				entities[i] = 'edgee'
		return entities
	def debug_graph_vizualization(graph, graph_name):
		debug_filepath = os.path.join(VIZ_BASE, 'debug/{0}.txt'.format(graph_name))
		with open(debug_filepath, 'w') as f:
			f.write(graph.to_string())
	import pydot
	entities = change_word_edge(entities)
	graph_path = get_path(GRAPH_BASE, graph_name, syn, conceptnet=conceptnet, ext='npy')
	graph_matrix = np.load(graph_path)
	entity_map = generate_index_mappings(entities, relation=False)
	relation_map = generate_index_mappings(relations, relation=True)
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
	debug_graph_vizualization(graph, graph_name)
	if svg:
		viz_path = get_path(VIZ_BASE, graph_name, syn, conceptnet, ext='svg')
		graph.write_svg(viz_path)
	else:
		viz_path = get_path(VIZ_BASE, graph_name, syn, conceptnet, ext='png')
		graph.write_png(viz_path)

def main(args):
	relations = parse(os.path.join(RELATIONS_BASE, args.relations_filename+'.txt'))
	entities = parse(os.path.join(ENTITIES_BASE, args.entities_filename+'.txt'))
	if not args.wordnet:
		conceptnet_map = open_dict(CONCEPTNET_PATH)
	if args.aug:
		start_aug = time.time()
		num_start_entities = len(entities)
		if args.wordnet:
			entities = add_hypernyms_with_wordnet(entities, args.entities_filename, syn=args.syn)
		else:
			entities = add_hypernyms_with_conceptnet(entities, args.entities_filename, conceptnet_map, depth=args.depth, syn=args.syn)
		num_entities = len(entities)
		num_entities_added = num_entities - num_start_entities
		end_aug = time.time()
		elapsed_aug = end_aug-start_aug
		print("Time required to add {1} hypernyms from {2} entities: {0} seconds".format(elapsed_aug, num_entities_added, num_start_entities)) 
	if args.graph_name:
		start_graph = time.time()
		if args.wordnet:
			conceptnet_map = None
		create_graph(args.graph_name, entities, relations, conceptnet_map, args.syn)
		end_graph = time.time()
		elapsed_graph = end_graph - start_graph
		print("Time required to create the graph: {0} seconds".format(elapsed_graph))
	if args.viz:
		start_viz = time.time()
		if args.wordnet:
			conceptnet_map = None
		visualize_graph(entities, relations, args.graph_name, args.syn, conceptnet_map, args.svg)
		end_viz = time.time()
		elapsed_viz = end_viz - start_viz
		print("Time required to visualize the graph: {0} seconds".format(elapsed_viz))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('relations_filename', help="Name of the relations text file")
	parser.add_argument('entities_filename', help="Name of the entities text file")
	parser.add_argument('graph_name', nargs='?', help="Optional; Provide the name of knowledge graph to create the graph (ndarray)", default=False)
	parser.add_argument('--depth', nargs='?', help="Depth to use for hypernyms in Conceptnet", default=2)
	parser.add_argument('--wordnet', action="store_true", help="Use wordnet to augment instead of conceptnet", default=False)
	parser.add_argument('--viz', action="store_true", help="Visualize an existing graph", default=False)
	parser.add_argument('--svg', action="store_true", help="Visualize an existing graph", default=False)
	parser.add_argument('--aug', action="store_true", help="Augment existing entities with lowest common hypernyms", default=False)
	parser.add_argument('--syn', nargs='?', help="Augment existing entities with lowest common hypernyms and synonyms", default=0)
	args = parser.parse_args()
	main(args)