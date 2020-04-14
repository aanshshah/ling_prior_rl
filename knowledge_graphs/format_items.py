import argparse
import json
import os

def write(new_items, entities):
	with open(new_items, 'w') as f:
		for entity in entities:
			f.write(entity + '\n')

def parse_lines(lines, env_type):
	entities = []
	for line in lines:
		formatted_line = line.split()
		if env_type == "minecraft":
			entity = formatted_line[0] 
		else:
			entity = formatted_line[1] 
		entities.append(entity)
	return entities

def create_format(original_items, fformat, ftype):
	with open(original_items) as f:
		lines = f.readlines()
	entities = parse_lines(lines, ftype)
	if fformat:
		write(original_items, entities)
	return entities

def create_mapping(new_items_path, original_items, mapping_name):
	def parse_new_items(path):
		entities = []
		with open(path, 'r') as f:
			lines = f.readlines()
		for line in lines:
			item = line.strip()
			entities.append(item.lower())
		return entities
	def save_dict(data, filename):
		with open(filename, 'w') as f:
			json.dump(data, f)
	new_items = parse_new_items(new_items_path)
	assert len(new_items) == len(original_items)
	num_items = len(new_items)
	original_to_new_items_map = {}
	for i in range(num_items):
		original_to_new_items_map[original_items[i]] = new_items[i]
	save_dict(original_to_new_items_map, mapping_name)

def main(args):
	BASE_PATH = 'data/entities'
	BASE_MAP = 'data/mappings'
	original_items_path = os.path.join(BASE_PATH, args.original)
	new_items_path = os.path.join(BASE_PATH, args.new)
	mapping_name = os.path.join(BASE_MAP, args.map_name)
	original_items = create_format(original_items_path, args.format, args.type)	
	if args.map:
		create_mapping(new_items_path, original_items, mapping_name)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--format', action="store_true", help="Format the original items", default=False)
	parser.add_argument('--original', help="Original items path name", default='all_minecraft_raw.txt')
	parser.add_argument('--new', help="New items path name", default='all_minecraft_filtered.txt')
	parser.add_argument('--map_name', help="Name of the mapping file to map original item to new item", default='minecraft_items_to_conceptnet_map.txt')
	parser.add_argument('--map', action="store_true", help="Map the original thor items/entities to similar ones understandable by conceptnet", default=False)
	parser.add_argument("--type", type=str, choices=["minecraft", "thor"], help="Choose type to format (minecraft or thor)")
	args = parser.parse_args()
	main(args)