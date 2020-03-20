#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: Jianmei Ye
@file: WordTreeByFlask.py
@time: 4/17/17 2:50 PM
"""
from flask import Flask, send_file, request, render_template, abort

from NLTKWordNet import NLTKWordNet

app = Flask(__name__,static_path='')

@app.route("/hyper", methods = ['GET'])
def buildHyperTree():
    test = NLTKWordNet()
    word1,word2 = None, None
    for arg in request.args:
        if arg.lower() == 'word1':
            word1 = request.args.get(arg)
        elif arg.lower() == 'word2':
            word2 = request.args.get(arg)
    fun = 'hyper'
    try:
        if word1 and word2:
            lca, midDist, synObj1, synObj2 = test.getLCAByShortestDistance(word1, word2)
            test.two_node_graph(synObj1, synObj2, fun)
            fileName = str(synObj1.name()) + '_vs_' + str(synObj2.name()) + '_hyper_.png'
            return send_file("output/"+fileName, mimetype='image/png')
    except:
        abort(404)


@app.route("/hypon", methods=['GET'])
def buildHyponTree():
    test = NLTKWordNet()
    word1, word2 = None, None
    for arg in request.args:
        if arg.lower() == 'word1':
            word1 = request.args.get(arg)
        elif arg.lower() == 'word2':
            word2 = request.args.get(arg)
    fun = 'hypon'
    try:
        if word1 and word2:
            lca, midDist, synObj1, synObj2 = test.getLCAByShortestDistance(word1, word2)
            test.two_node_graph(synObj1, synObj2, fun)
            fileName = str(synObj1.name()) + '_vs_' + str(synObj2.name()) + '_hypon_.png'
            return send_file("output/"+fileName, mimetype='image/png')
    except:
        abort(404)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

def Main():
    from werkzeug.serving import make_server
    httpd = make_server('127.0.0.1', 8080, app, threaded=True)
    print ("Serving http on port 8080...")
    httpd.serve_forever()

if __name__ == "__main__":
    Main()
