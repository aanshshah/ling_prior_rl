The create_graph script can create two different types of knowledge graphs: (a) one with the entities augmented the lowest common hypernym shared between each pair of words and (b) the former in addition to all the synonyms for each pair of words. This can be achieved with the --aug and --syn flags, respectively. The generated knowledge graphs are generated as stacked adjacency matrices and are saved as a numpy array. You can also save a visualization of the graph with the --viz flag. 

The script assumes that all the entities are stored in a text file in data/entities and are separated by a newline (see basic_entities.txt for an example). The same is true for relations (data/relations). When the --aug flag is used, the new entities file is stored with "\_lca" appended to it and '\_lca_with_syn' when the --syn flag is used. The same is true for the saved knowledge graphs. 

Usage: python create_graph.py --aug --syn --viz <relation_name> <entity_name> <graph_name> 
Example Usage:
python create_graph.py --aug --syn --viz basic_relations basic_entities basic_graph