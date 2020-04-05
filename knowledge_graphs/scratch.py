import json

def open_dict(filename):
	with open(filename, 'r') as f:
		data = json.load(f)
	return data

def main():
	cn_map = open_dict('data/mappings/conceptnet_map.txt')
	print(cn_map['pillow'])

if __name__ == '__main__':
	main()