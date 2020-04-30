# ******************************************************************************
# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing

from graphrl.modules.graph.wnet import WeightNet
from graphrl.modules.nn import get_activation_gain


class DGConv(MessagePassing):
    def __init__(self,
                 num_in_feats,
                 num_out_feats,
                 num_edge_in_feats,
                 wnet_hidden_units,
                 bias=True,
                 use_orth_init=False):
        super(DGConv, self).__init__('add')
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.num_edge_in_feats = num_edge_in_feats
        self.use_orth_init = use_orth_init

        self.wnet = WeightNet(wnet_hidden_units, (self.num_edge_in_feats, ),
                              (self.num_in_feats, self.num_out_feats),
                              use_orth_init=use_orth_init)
        self.skip = nn.Linear(num_in_feats, num_out_feats, bias=False)
        if self.use_orth_init:
            nn.init.orthogonal_(self.skip.weight.data,
                                get_activation_gain('relu'))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_bias()

    def reset_bias(self):
        if self.bias is not None:
            if self.use_orth_init:
                nn.init.constant_(self.bias.data, 0)
            else:
                fan_in = self.num_in_feats
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, edge_index, edge_feats, edge_norm=None):
        return self.propagate(edge_index,
                              x=x,
                              edge_feats=edge_feats,
                              edge_norm=edge_norm,
                              size=list((x.size(0), x.size(0))))

    def message(self, x_j, edge_feats, edge_norm):
        edge_weights = self.wnet(edge_feats)
        out = torch.bmm(x_j[:, None, :], edge_weights)[:, 0, :]

        if edge_norm is not None:
            out = out * edge_norm[:, None]
        if self.bias is not None:
            out = out + self.bias[None, :]

        return out

    def update(self, aggr_out, x):
        skip_x = self.skip(x)
        aggr_out = aggr_out + skip_x
        return aggr_out

    def __repr__(self):
        return 'DGCONV'
