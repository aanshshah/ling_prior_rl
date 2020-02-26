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

import numpy as np

os.environ["PATH"] += os.pathsep + 'C:/Program Files/graphviz/bin'
os.environ["PATH"] += os.pathsep + 'C:/Program Files/graphviz'

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

        general_graph = pydot.Dot(graph_type='graph')
        for v in all_nodes:
            general_graph.add_node(pydot.Node(v))
        for src,dest in all_edges:
            general_graph.add_edge(pydot.Edge(src,dest))

        self.main_graph = general_graph
        self.syn_obs = set(self.syn_obs)



# def make_clusters(object_to_vector):
#    keys = sorted(object_to_vector.keys())
#    embs = [np.reshape(object_to_vector[k],(-1)) for k in keys]
#    kmeans = KMeans(n_clusters=5, random_state=0).fit(embs)
#    for k,l in zip(keys,kmeans.labels_):
#         print(k,l)

def add_most_sim_edges(graph,vocab,num,vertices):
    all_nodes = vertices
    metric = sklearn.metrics.pairwise.cosine_similarity

    object_to_vector = {}

    for k in all_nodes:
        if k in vocab or (k.split('_')[0] in vocab and k.split('_')[1] in vocab): 
            object_to_vector[k] = np.array(vocab[k])if "_" not in k else np.array(vocab[k.split('_')[0]])+np.array(vocab[k.split('_')[1]])
        else:
            object_to_vector[k] = np.zeros((1,300))

        object_to_vector[k] = object_to_vector[k].reshape(1,-1)


    # make_clusters(object_to_vector)

    for n in all_nodes:
        other_nodes = copy.copy(all_nodes)
        other_nodes.remove(n)
        nearest_neighbors = sorted(other_nodes,key = lambda x: metric(object_to_vector[n],object_to_vector[x]),reverse=True)[:num]
        for neighbor in nearest_neighbors:
            graph.add_edge(pydot.Edge(n,neighbor,color='blue'))




def get_minecraft_items():

    items = []

    with open("mine_craft_items.txt","r") as open_file:
        for x in open_file.readlines():
            if ':' in x: continue

            split_terms = x.split()
            if split_terms[-1] == "Block":
                items.append(split_terms[-2].lower())
            else:   items.append(split_terms[-1].lower())

    return list(set(items))




def Main():
    # server = HTTPServer(('127.0.0.1', 8080), WordTree)
    # print('Started http server')
    # server.serve_forever()

    USE_MINECRAFT = False #Can set to true

    words = get_minecraft_items()[:20] if USE_MINECRAFT else ["plank", "wood", "toolshed", "stick", "workbench", "cloth", "grass", "factory", "rope", "bridge", "iron", "bed", "axe", "shears", "gold", "gem", "worker"]
    entGraph = EntGraph(word_list = words) #["chicken","bowl","party","hat","shoe"]

    relevant_nodes = entGraph.syn_obs

    split_terms = []
    for n in relevant_nodes:
        if '_' in n:
            split_terms.extend(n.split('_'))

    w2v = load_vectors("wiki-news-300d-1M.vec",list(relevant_nodes)+split_terms)
    add_most_sim_edges(entGraph.main_graph,w2v,3,relevant_nodes)


    print(entGraph.main_graph)

    entGraph.main_graph.write_png("./output/result3.png")


    # print(entGraph.main_graph)
   

    


if __name__ == "__main__":
    parser = OptionParser()
    Main()