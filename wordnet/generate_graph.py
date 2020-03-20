#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: Jianmei Ye
@file: WordTreeByBaseHTTP.py
@time: 4/21/17 4:31 PM
"""

import codecs
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
import sklearn
from optparse import OptionParser
import urlparse
import os
import pydot
import copy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import numpy as np
from tqdm import tqdm

os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\graphviz\\bin'
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\graphviz'
os.environ["PATH"] += os.pathsep + "C:\\Users\\zach_surf\\.conda\\envs\\wnet\\Lib\\site-packages"


print(os.environ["PATH"])
# exit()

from NLTKWordNet import NLTKWordNet


import io

def load_vectors(fname,use_vocab):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        if tokens[0] in use_vocab: data[tokens[0]] = map(float, tokens[1:])
    return data


class WordTree(BaseHTTPRequestHandler):
    def do_GET(self):
        test = NLTKWordNet()
        query = {}
        path = self.path
        fun = None
        if 'hyper?' in path:
            _, tmp = path.split('hyper?', 1)
            query = urlparse.parse_qs(tmp)
            fun = 'hyper'
        elif 'hypon?' in path:
            _, tmp = path.split('hypon?', 1)
            query = urlparse.parse_qs(tmp)
            fun = 'hypon'

        word1 = query.get("word1")
        word2 = query.get("word2")
        f = None
        try:
            if word1 and word2:
                lca, midDist, synObj1, synObj2 = test.getLCAByShortestDistance(word1[0], word2[0])
                out_graph = test.two_node_graph(synObj1, synObj2, fun)
                fileName = "output/"+str(synObj1.name()) + '_vs_' + str(synObj2.name()) + '_'+fun+'_.png'
                f = open(fileName, 'rb')
                self.send_response(200)
                self.send_header('Content-type', 'image/png')
        except KeyboardInterrupt:
            fileName = "templates/404.html"
            f = codecs.open(fileName, 'r', 'utf-8')
            print(f.read())
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
        self.end_headers()
        output = f.read()
        self.wfile.write(output) 
        f.close()
        return



class EntGraph():
    def __init__(self,word_list):
        self.test = NLTKWordNet()
        self.word_list = word_list
        self.pairs = []
        self.graphs = []
        self.syn_obs = []
        
        def clean_name(name):
            return name.split('.')[0].replace('"','')


        for w in word_list:
            for w_other in word_list:
                if w == w_other:
                    continue
                if (w_other,w) in self.pairs:
                    continue
                if (w,w_other) in self.pairs:
                    continue

                self.pairs.append((w,w_other))

        for word1,word2 in self.pairs:
            if word1 and word2:
                try:
                    lca, midDist, synObj1, synObj2 = self.test.getLCAByShortestDistance(word1, word2)
                    self.syn_obs.extend([clean_name(synObj1.name()),clean_name(synObj2.name())])
                    out_graph = self.test.two_node_graph(synObj1, synObj2, 'hyper')
                    self.graphs.append(out_graph)
                except IndexError:
                    print(word1,word2,"FAILED")
              
           
        
        all_nodes = set()
        all_edges = set()

        for g in self.graphs:
            for x in g.get_nodes():
               all_nodes.add(clean_name(x.get_name()))
            for edge in g.get_edges():
               all_edges.add((clean_name(edge.get_source()),clean_name(edge.get_destination())))

        general_graph = pydot.Dot(graph_type='digraph')
        for v in all_nodes:
            general_graph.add_node(pydot.Node(v))
        for src,dest in all_edges:
            general_graph.add_edge(pydot.Edge(src,dest))

        self.main_graph = general_graph
        self.syn_obs = set(self.syn_obs)


# def computer_number_of_clusters(object_to_vector,kmax=4):
#     sil = []
#     keys = sorted(object_to_vector.keys())
#     embs = [np.reshape(object_to_vector[k],(-1)) for k in keys]
#     # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
#     for k in range(2, kmax+1):
#       kmeans = KMeans(n_clusters = k).fit(embs)
#       labels = kmeans.labels_
#       sil.append((k,silhouette_score(embs, labels, metric = 'euclidean')))

#     return max(sil,key=lambda x: x[1])

# def make_clusters(object_to_vector):

#     num_clusters,value = computer_number_of_clusters(object_to_vector)
#     keys = sorted(object_to_vector.keys())
#     embs = [np.reshape(object_to_vector[k],(-1)) for k in keys]
#     kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embs)
#     word_2_cluster = {}
#     cluster_2_words = {}
#     for k,l in zip(keys,kmeans.labels_):
#         word_2_cluster[k] = l
#         if l not in cluster_2_words: cluster_2_words[l] = []
#         cluster_2_words[l].append(k)
#     return word_2_cluster,cluster_2_words


def add_most_sim_edges(graph,vocab,num,vertices,use_clusters=True):
    all_nodes = vertices
    metric = sklearn.metrics.pairwise.cosine_similarity

    object_to_vector = {}

    for k in all_nodes:
        if k in vocab or (k.split('_')[0] in vocab and k.split('_')[1] in vocab): 
            object_to_vector[k] = np.array(vocab[k])if "_" not in k else np.array(vocab[k.split('_')[0]])+np.array(vocab[k.split('_')[1]])
        else:
            object_to_vector[k] = np.zeros((1,300))

        object_to_vector[k] = object_to_vector[k].reshape(1,-1)

    # word_2_cluster,cluster_to_words = make_clusters(object_to_vector)

    # if use_clusters:
    #     clusters = sorted(cluster_to_words.keys())
    #     for c in clusters:
    #        graph.add_node(pydot.Node("CLUSTER_"+str(c)))
       

    for n in all_nodes:
        other_nodes = copy.copy(all_nodes)
        other_nodes.remove(n)
        
        nearest_neighbors = sorted(other_nodes,key = lambda x: metric(object_to_vector[n],object_to_vector[x]),reverse=True)[:num]

        # for neighbor in nearest_neighbors:
        #     graph.add_edge(pydot.Edge(n,neighbor,color='blue'))

        # if use_clusters:
        #     graph.add_edge(pydot.Edge(n,"CLUSTER_"+str(word_2_cluster[n]),color='green'))



# def get_minecraft_items():

#     items = []

#     with open("mine_craft_items.txt","r") as open_file:
#         for x in open_file.readlines():
#             if ':' in x: continue

#             split_terms = x.split()
#             if split_terms[-1] == "Block":
#                 items.append(split_terms[-2].lower())
#             else:   items.append(split_terms[-1].lower())

#     return list(set(items))


def remove_unnecc_edges(graph,keep_vertices):
    new_graph = None
    for n in tqdm(copy.deepcopy(graph).get_nodes()):
        new_graph =  pydot.Dot(graph_type='digraph', rankdir='BT')
        remove_edges = []
        add_edges = []
        # valid_vertices = set()
        # print(keep_vertices)
        # if n.get_name() in keep_vertices:
        #     new_graph.add_node(n)
        #     valid_vertices.add(n.get_name())
        # else:
        edges_from = [(x.get_source(),x.get_destination()) for x in graph.get_edges() if x.get_source() == n.get_name() ]
        edges_to = [(x.get_source(),x.get_destination()) for x in graph.get_edges() if x.get_destination() == n.get_name() ]
            # print(edges_to)
        if len(edges_from) == 1 and  len(edges_to) == 1 and n.get_name() not in keep_vertices:
            remove_edges.extend(edges_from)
            remove_edges.extend(edges_to)

            for x1,y1 in edges_to:
                for x2,y2 in edges_from:
                    add_edges.append((x1,y2))
      

        # for n in graph.get_nodes():
            #Delete is not relevenat or no edges


        for edge in graph.get_edges():
            if (edge.get_source(),edge.get_destination()) in remove_edges:
                continue
            new_graph.add_edge(edge)

        for source,dest in add_edges:
            new_graph.add_edge(pydot.Edge(source,dest))

        # for n in graph.get_nodes()

        graph = copy.deepcopy(new_graph)

    return new_graph









def Main():
    # server = HTTPServer(('127.0.0.1', 8080), WordTree)
    # print('Started http server')
    # server.serve_forever()

    # USE_MINECRAFT = False #Can set to true

    words = ["plank", "wood", "toolshed", "stick", "workbench", "cloth", "grass", "factory", "rope", "bridge", "iron", "bed", "axe", "shears", "gold", "gem", "worker"] #get_minecraft_items() if USE_MINECRAFT else ["box","gem","key","lock","player"] #["plank", "wood", "toolshed", "stick", "workbench", "cloth", "grass", "factory", "rope", "bridge", "iron", "bed", "axe", "shears", "gold", "gem", "worker"]
    entGraph = EntGraph(word_list = words) #["chicken","bowl","party","hat","shoe"]
    relevant_nodes = entGraph.syn_obs

    split_terms = []
    for n in relevant_nodes:
        if '_' in n:
            split_terms.extend(n.split('_'))

    # w2v = load_vectors("wiki-news-300d-1M.vec",list(relevant_nodes)+split_terms)
    # add_most_sim_edges(entGraph.main_graph,w2v,1,relevant_nodes)

    print(entGraph.main_graph)
    simple_graph = remove_unnecc_edges(entGraph.main_graph,relevant_nodes)
    entGraph.main_graph.write_svg("./output/boxworld.svg") #ex_all.svg")
    simple_graph.write_svg("./output/boxworld_simple.svg") #ex_simple.svg")



    # print(entGraph.main_graph)
   

    


if __name__ == "__main__":
    parser = OptionParser()
    Main()