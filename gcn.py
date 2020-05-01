""" Some code borrowed from https://github.com/tkipf/pygcn."""

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.models as models
# from utils.net_util import norm_col_init, weights_init
import scipy.sparse as sp
import numpy as np

# from datasets.glove import Glove

# from .model_io import ModelOutput


# def normalize_adj(adj):
#     adj = sp.coo_matrix(adj)
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

class GCN(torch.nn.Module):
    def __init__(self,args=None):
        super().__init__() 
        n = 5
        self.n = n
        self.objects = torch.arange(0,self.n)
        self.node_to_game_char = {i:i+1 for i in self.objects.tolist()}

        # get and normalize adjacency matrix.
        A_raw = torch.eye(self.n) #torch.load("") #./data/gcn/adjmat.dat")
        A = A_raw #normalize_adj(A_raw).tocsr().toarray()
        self.A = torch.nn.Parameter(torch.Tensor(A))

        # self.get_word_embed = nn.Linear(300, 512)
        # self.get_class_embed = nn.Linear(1000, 512)

        self.W0 = nn.Linear(16, 32, bias=False)
        self.W1 = nn.Linear(32, 32, bias=False)
        self.W2 = nn.Linear(32, 32, bias=False)

        self.get_obj_emb = nn.Embedding(self.n, 16)
        self.final_mapping = nn.Linear(32, 16)

    def gcn_embed(self):
        # x = self.resnet18[0](state)
        # x = x.view(x.size(0), -1)
        # x = torch.sigmoid(self.resnet18[1](x))
        # class_embed = self.get_class_embed(x)
        # word_embed = self.get_word_embed(self.all_glove.detach())
        # x = torch.cat((class_embed.repeat(self.n, 1), word_embed), dim=1)
        
        nodes = self.objects #.view(1,self.n)
        node_embeddings = self.get_obj_emb(nodes)

        x = torch.mm(self.A, node_embeddings)
        x = F.relu(self.W0(x))
        x = torch.mm(self.A, x)
        x = F.relu(self.W1(x))
        x = torch.mm(self.A, x)
        x = F.relu(self.W2(x))
        # x = x.view(1, self.n)
        x = self.final_mapping(x)

        return x

    def add_state_info(self,game_state):
        #game_state = (1,10,10)
        game_state_embed = self.get_obj_emb(game_state.view(-1,game_state.shape[-2]*game_state.shape[-1]))
        game_state_embed = game_state_embed.view(game_state.shape[0],game_state.shape[1],game_state.shape[2],-1)
        # print(game_state_embed.shape)
        node_embeddings = self.gcn_embed()
        for n,embedding in zip(self.objects.tolist(),node_embeddings):
            indx = (game_state == self.node_to_game_char[n]).nonzero()
            game_state_embed[indx[:, 0], indx[:, 1], indx[:, 2]] = embedding
        return game_state_embed

test = GCN()
new_state = test.add_state_info(torch.ones((1,10,10)).long())
print(new_state)