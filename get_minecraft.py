

items = []

with open("mine_craft_items.txt","r") as open_file:
	for x in open_file.readlines():
		if ':' in x: continue

		split_terms = x.split()
		if split_terms[-1] == "Block":
			items.append(split_terms[-2].lower())
		else: 	items.append(split_terms[-1].lower())


