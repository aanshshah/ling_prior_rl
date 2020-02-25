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
                test.two_node_graph(synObj1, synObj2, fun)
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


def Main():
    server = HTTPServer(('127.0.0.1', 8080), WordTree)
    print('Started http server')
    server.serve_forever()



if __name__ == "__main__":
    parser = OptionParser()
    Main()