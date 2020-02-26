#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: Jianmei Ye
@file: NLTKWordNet.py
@time: 4/21/17 4:31 PM
"""

from cStringIO import StringIO

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import heapq

from nltk.corpus import wordnet as wn
import pydot



class NLTKWordNet(object):
    def getSynsetList(self, word):
        return wn.synsets(word)

    def getSysetListWithPos(self, word, pos):
        return wn.synsets(word, pos=pos)

    def getDefinition(self, synObj):
        return wn.synset(synObj).definition()

    def getRootHypernyms(self, synObj):
        return synObj.root_hypernyms()

    def getLCA(self, synObj1, synObj2):
        return synObj1.lowest_common_hypernyms(synObj2)

    def getPathSimilarity(self, synObj1, synObj2):
        return synObj1.path_similarity(synObj2)

    def getLCHPathSimilarity(self, synObj1, synObj2):
        return synObj1.lch_similarity(synObj2)

    def getWUPPathSimilarity(self, synObj1, synObj2):
        return synObj1.wup_similarity(synObj2)

    def getShortestPathDistanceBetweenTwoWord(self, synObj1, synObj2):
        return synObj1.shortest_path_distance(synObj2)

    def getLCAByShortestDistance(self, word1, word2):
        ss_w1 = [list(self.getSysetListWithPos(word1, pos=wn.NOUN))[0]]
        ss_w2 = [list(self.getSysetListWithPos(word2, pos=wn.NOUN))[0]]
        lca = None
        midDist = float('inf')
        synObj1, synObj2 = None, None
        for s1 in ss_w1:
            for s2 in ss_w2:
                curDist = self.getShortestPathDistanceBetweenTwoWord(s1, s2)
                if curDist != None and curDist < midDist:
                    midDist = curDist
                    lca = self.getLCA(s1, s2)
                    synObj1, synObj2 = s1, s2
        return (lca, midDist, synObj1, synObj2)

    def getLCAByPathSimilarity(self, word1, word2):
        ss_w1 = self.getSynsetList(word1)
        ss_w2 = self.getSynsetList(word2)
        lca = None
        maxSim = float('-inf')
        for s1 in ss_w1:
            for s2 in ss_w2:
                curSim = self.getPathSimilarity(s1, s2)
                if curSim > maxSim:
                    maxSim = curSim
                    lca = self.getLCA(s1, s2)
        return lca, maxSim

    def getTopKNearestLCA(self, word1, word2, k=3):
        topk = set()
        ss_w1 = self.getSynsetList(word1)
        ss_w2 = self.getSynsetList(word2)
        lcaTuple = []
        for s1 in ss_w1:
            for s2 in ss_w2:
                curDist = self.getShortestPathDistanceBetweenTwoWord(s1, s2)
                if curDist:
                    midDist = curDist
                    lca = self.getLCA(s1, s2)
                    heapq.heappush(lcaTuple, (midDist, lca))
        while len(topk) < k:
            tmp = heapq.heappop(lcaTuple)
            topk.add(tmp[1][0])
        return list(topk)

    # def closure_graph_hypernyms(self, synObj):
        seen = set()
        graph = pydot.Dot(graph_type='digraph', rankdir='BT')

        def recurse(s):
            if not s in seen:
                seen.add(s)
                graph.add_node(pydot.Node(str(s.name())))
                for s1 in s.hypernyms():
                    graph.add_node(pydot.Node(str(s1.name())))
                    graph.add_edge(pydot.Edge(str(s.name()), str(s1.name())))
                    recurse(s1)

        recurse(synObj)
        pngNam = str(synObj.name()) + '_hyper_.png'
        graph.write_png("output/"+pngNam)
        # return graph

    def closure_graph_hyponyms(self, synObj):
        seen = set()
        graph = pydot.Dot(graph_type='digraph', rankdir='LR')

        def recurse(s):
            if not s in seen:
                seen.add(s)
                graph.add_node(pydot.Node(str(s.name())))
                for s1 in s.hyponyms():
                    graph.add_node(pydot.Node(str(s1.name())))
                    graph.add_edge(pydot.Edge(str(s.name()), str(s1.name())))
                    recurse(s1)

        recurse(synObj)
        pngNam = str(synObj.name()) + '_hypon_.png'
        graph.write_png(pngNam)

    def two_node_graph(self, synObj1, synObj2, fun):
        lca = self.getLCA(synObj1, synObj2)[0]
        if fun == 'hyper':
            method = lambda f: f.hypernyms()
            graph = pydot.Dot(graph_type='digraph', rankdir='BT',
                              label=str(synObj1.name()).split('.')[0] + '  VS   ' + str(synObj2.name()).split('.')[0],
                              fontsize="24.0")
        elif fun == 'hypon':
            method = lambda f: f.hyponyms()
            graph = pydot.Dot(graph_type='digraph', rankdir='LR',
                              label=str(synObj1.name()).split('.')[0] + '  VS   ' + str(synObj2.name()).split('.')[0],
                              fontsize="24.0")

        seen = set()

        # print str(synObj1.name()) + ' vs ' + str(synObj2.name())
        def recurse(s, color):
            if not s in seen:
                seen.add(s)
                if color:
                    if s == lca:
                        graph.add_node(pydot.Node(str(s.name()), color='red'))
                else:
                    graph.add_node(pydot.Node(str(s.name())))
                for s1 in method(s):
                    if color:
                        if s1 == lca:
                            graph.add_node(pydot.Node(str(s1.name()), color='red'))
                    else:
                        graph.add_node(pydot.Node(str(s1.name())))
                    graph.add_edge(pydot.Edge(str(s.name()), str(s1.name())))
                    recurse(s1, color)

        recurse(synObj1, True)
        recurse(synObj2, False)
        pngNam = str(synObj1.name()) + '_vs_' + str(synObj2.name()) + '_'+fun+'_.png'
        #graph.write_png("./output/"+pngNam)
        # png_str = graph.create_png()
        # sio = StringIO()
        # sio.write(png_str)
        # sio.seek(0)
        # img = mpimg.imread(sio)

        # plot the image
        # imgplot = plt.imshow(img, aspect='equal')
        # plt.show(block=False)
        print ("Image generated!")
        return graph




