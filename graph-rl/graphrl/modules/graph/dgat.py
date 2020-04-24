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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.utils import softmax, scatter_

from graphrl.modules.graph.wnet import WeightNet


class DGAT(nn.Module):
    def __init__(self, num_in_feats, num_out_feats, num_edge_in_feats, wnet_hidden_units, heads=1, concat=True, negative_slope=0.2, bias=True, flow='source_to_target', aggr='add'):
        super(DGAT, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.num_edge_in_feats = num_edge_in_feats
        self.flow = flow
        self.aggr = aggr

        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope

        self.src_wnet = WeightNet(wnet_hidden_units, (self.num_edge_in_feats,), (self.num_in_feats, heads * self.num_out_feats))
        self.dst_wnet = WeightNet(wnet_hidden_units, (self.num_edge_in_feats,), (self.num_in_feats, heads * self.num_out_feats))

        self.self_weights = nn.Parameter(torch.Tensor(self.num_in_feats, heads * self.num_out_feats))

        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * num_out_feats))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(self.heads * self.num_out_feats))
        elif bias and not self.concat:
            self.bias = nn.Parameter(torch.Tensor(self.num_out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_bias()

    def reset_bias(self):
        zeros(self.bias)
        glorot(self.self_weights)

    def forward(self, x, edge_index, edge_feats):
        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)

        x_i = torch.index_select(x, 0, edge_index[i])
        x_j = torch.index_select(x, 0, edge_index[i])

        src_weights = self.src_wnet(edge_feats)
        dst_weights = self.dst_wnet(edge_feats)
        x_i = torch.bmm(x_i[:, None, :], src_weights)[:, 0, :].view(-1, self.heads, self.num_out_feats)
        x_j = torch.bmm(x_j[:, None, :], dst_weights)[:, 0, :].view(-1, self.heads, self.num_out_feats)

        x_self = torch.matmul(x, self.self_weights).view(-1, self.heads, self.num_out_feats)

        self_idxs = torch.arange(end=x.size(0), dtype=edge_index.dtype, device=edge_index.device)
        alpha_idxs = torch.cat([edge_index[0], self_idxs], dim=0)

        raw_alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        raw_alpha_self = (torch.cat([x_self, x_self], dim=-1) * self.att).sum(dim=-1)
        raw_alpha = torch.cat([raw_alpha, raw_alpha_self], dim=0)
        raw_alpha = F.leaky_relu(raw_alpha, self.negative_slope)
        alpha = softmax(raw_alpha, alpha_idxs, x.size(0))

        messages = torch.cat([x_j, x_self], dim=0)
        messages = messages * alpha[:, :, None]
        aggr_out = scatter_(self.aggr, messages, alpha_idxs, dim_size=x.size(0))

        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.num_out_feats)
        else:
            aggr_out = aggr_out.mean(dim=1)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return 'DGAT'
