

def write(entities):
	filename = 'data/entities/all_thor.txt'
	with open(filename, 'w') as f:
		for entity in entities:
			f.write(entity + '\n')

def parse_lines(lines):
	entities = []
	for line in lines:
		formatted_line = line.split()
		entity = formatted_line[1]
		entities.append(entity)
	return entities

def main():
	filename = '../wordnet/thor_items'
	with open(filename) as f:
		lines = f.readlines()
	entities = parse_lines(lines)
	write(entities)

if __name__ == '__main__':
	main()