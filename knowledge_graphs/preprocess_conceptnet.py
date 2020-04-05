import argparse
import json

def get_relations(relationpath):
	relations_set = set()

	with open(relationpath) as r:
		relations = r.readlines()
	for i in range(len(relations)):
		relation = relations[i].strip()
		relations[i] = relation
		relations_set.add('/r/{0}'.format(relation))
	return relations_set, relations

def prune_conceptnet(relations_set, filepath):
	new_entries = []
	old_total_lines = 0
	with open(filepath) as cn:
		for line in cn:
			old_total_lines += 1
			query, relation, word_one, word_two, data = line.split("\t")
			if relation not in relations_set: 
				continue
			elif word_one[:6] != '/c/en/' or word_two[:6] != '/c/en/':
				continue
			else:
				new_entries.append([query, relation, word_one, word_two, data])
	return new_entries, old_total_lines

def save_pruned_conceptnet(new_entries, filename):
	line_count = len(new_entries)
	with open(filename, 'w') as cn:
		for line in new_entries:
			cn.write("\t".join(line))
	return line_count

def parse(conceptnet_path):
	"""
	mapping = {"apple": 
				{
					"RelatedTo": {"fruit": 1.3, "orange": 1},
					"IsA": {"fruit": 3, "object": 3}
				} 
			  }
	"""
	def get_word_from_url(url):
		return url.split('/')[-1]
	def preprocess_line(line):
		query, relation, word_one, word_two, data = line.split("\t")
		relation = get_word_from_url(relation)
		word_one = get_word_from_url(word_one)
		word_two = get_word_from_url(word_two)
		data = json.loads(data)
		weight = data["weight"]
		return relation, word_one, word_two, weight
	mapping = {}
	with open(conceptnet_path, 'r') as cn:
		for line in cn:
			if not line: continue
			relation, word_one, word_two, weight = preprocess_line(line)
			relations_map = mapping.get(word_one, {})
			weight_map = relations_map.get(relation, {})
			weight_map[word_two] = weight
			relations_map[relation] = weight_map
			mapping[word_one] = relations_map
	return mapping

def save_dict(data, filename):
	with open(filename, 'w') as f:
		json.dump(data, f)

def main(args):
	filepath = 'data/conceptnet/conceptnet.csv'
	relationpath = 'data/relations/all_relations.txt'
	new_conceptnet_path = 'data/conceptnet/english_conceptnet.csv'
	if args.prune:
		relations_set, relations = get_relations(relationpath)
		pruned_net, old_total_lines = prune_conceptnet(relations_set, filepath)
		new_total_lines = save_pruned_conceptnet(pruned_net, new_conceptnet_path)
	if args.map:
		mapping = parse(new_conceptnet_path)
		conceptnet_mapping_path = "data/mappings/conceptnet_map.txt"
		save_dict(mapping, conceptnet_mapping_path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--prune', action="store_true", help="Prune conceptnet to only english words and relations", default=False)
	parser.add_argument('--map', action="store_true", help="Create conceptnet mapping from word to relation to word to weight", default=False)
	args = parser.parse_args()
	main(args)