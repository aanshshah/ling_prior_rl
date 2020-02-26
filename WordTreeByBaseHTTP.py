#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: Jianmei Ye
@file: WordTreeByBaseHTTP.py
@time: 4/21/17 4:31 PM
"""

import codecs
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
from optparse import OptionParser
import urlparse
import os
import pydot

os.environ["PATH"] += os.pathsep + 'C:\Program Files\graphviz\bin'
os.environ["PATH"] += os.pathsep + 'C:\Program Files\graphviz'

from NLTKWordNet import NLTKWordNet


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
                lca, midDist, synObj1, synObj2 = self.test.getLCAByShortestDistance(word1, word2)
                out_graph = self.test.two_node_graph(synObj1, synObj2, 'hyper')
                self.graphs.append(out_graph)
              
           
        
        all_nodes = set()
        all_edges = set()

        for g in self.graphs:
            for x in g.get_nodes():
               all_nodes.add(x.get_name())
            for edge in g.get_edges():
               all_edges.add((edge.get_source(),edge.get_destination()))

        general_graph = pydot.Dot(graph_type='graph')
        for v in all_nodes:
            general_graph.add_node(pydot.Node(v))
        for src,dest in all_edges:
            general_graph.add_edge(pydot.Edge(src,dest))

        self.main_graph = general_graph


def Main():
    # server = HTTPServer(('127.0.0.1', 8080), WordTree)
    # print('Started http server')
    # server.serve_forever()

    entGraph = EntGraph(word_list = ["chicken","bowl","party","hat","shoe"])
    print(entGraph.main_graph)
   

    


if __name__ == "__main__":
    parser = OptionParser()
    Main()