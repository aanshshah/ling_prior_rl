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

import collections
import math
import numpy as np
import torch
import torch.nn as nn


class WeightNet(nn.Sequential):
    """ Weight generating network. """

    def __init__(self, hidden_units, in_shape, out_shape, name='wnet'):
        self.name = name

        # needed to cat output to correct shape
        self.out_shape = out_shape

        layers = []
        in_feats = int(np.prod(in_shape))
        for ix, out_feats in enumerate(hidden_units):
            layers.append((name + '_lin%s' % ix, nn.Linear(in_feats, out_feats)))
            layers.append((name + '_relu%s' % ix, nn.ReLU(inplace=True)))
            in_feats = out_feats

        # final layer doesn't have ReLu (weights are positive and negative)
        weight_size = int(np.prod(out_shape))
        layers.append((name + '_weightlin', nn.Linear(in_feats, weight_size)))

        super().__init__(collections.OrderedDict(layers))

    def forward(self, input):
        bsize = input.shape[0]
        out = super().forward(input.view(bsize, -1))
        return out.view(-1, *self.out_shape)


class DynamicGraphConvMessages(nn.Module):
    def __init__(self, num_in_feats, num_out_feats, num_edge_in_feats, wnet_hidden_units, dynamic_bias=False):
        super(DynamicGraphConvMessages, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.num_edge_in_feats = num_edge_in_feats

        if dynamic_bias:
            self.bias_wnet = WeightNet(wnet_hidden_units, (self.num_edge_in_feats,), (self.num_out_feats,))
        else:
            self.bias_wnet = None

        self.wnet = WeightNet(wnet_hidden_units, (self.num_edge_in_feats,), (self.num_in_feats, self.num_out_feats))

    def forward(self, input_features, adj, rel):
        bz, num = input_features.size(0), input_features.size(1)
        input_features = input_features[:, :, None, None, :].repeat(1, 1, num, 1, 1)

        input_features = input_features.view(-1, 1, self.num_in_feats)
        weights = self.wnet(rel.view(-1, self.num_edge_in_feats)).view(bz, num, num, self.num_in_feats, self.num_out_feats)
        weights = weights.view(-1, self.num_in_feats, self.num_out_feats)

        message_features = torch.bmm(input_features, weights).view(bz, num, num, self.num_out_feats)
        message_features = message_features * adj[:, :, :, None]

        if self.bias_wnet is not None:
            dynamic_bias = self.bias_wnet(rel.view(-1, self.num_edge_in_feats)).view(bz, num, num, self.num_out_feats)
            message_features = message_features + dynamic_bias
        return message_features


class DynamicGraphConv(nn.Module):
    def __init__(self, num_in_feats, num_out_feats, num_edge_in_feats, wnet_hidden_units, bias=False, dynamic_bias=False, use_reverse_edges=False,
                 reduction=torch.mean, use_skip=False):
        super(DynamicGraphConv, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.num_edge_in_feats = num_edge_in_feats
        self.reduction = reduction
        self.use_skip = use_skip
        self.use_reverse_edges = use_reverse_edges

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.num_out_feats,))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(torch.Tensor(self.num_out_feats, self.num_in_feats))
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None
        self.forward_messages = DynamicGraphConvMessages(num_in_feats, num_out_feats, num_edge_in_feats, wnet_hidden_units, dynamic_bias=dynamic_bias)
        if self.use_reverse_edges:
            self.reverse_messages = DynamicGraphConvMessages(num_in_feats, num_out_feats, num_edge_in_feats, wnet_hidden_units, dynamic_bias=dynamic_bias)
        self.skip_fc = nn.Linear(num_in_feats, num_out_feats)

    def forward(self, input_features, adj, rel):
        if self.use_skip:
            skip_features = self.skip_fc(input_features)

        if self.use_reverse_edges:
            forward_adj_sum = adj.sum(1)
            backward_adj_sum = adj.sum(2)
            norm = torch.max(forward_adj_sum + backward_adj_sum, torch.ones_like(forward_adj_sum))
        else:
            forward_adj_sum = adj.sum(1)
            norm = torch.max(forward_adj_sum, torch.ones_like(forward_adj_sum))

        message_features = self.forward_messages(input_features, adj, rel)
        if self.use_reverse_edges:
            reverse_message_features = self.reverse_messages(input_features, adj.permute(0, 2, 1).contiguous(), rel.permute(0, 2, 1, 3).contiguous())
            message_features = message_features + reverse_message_features
        output_features = torch.sum(message_features, 1)

        if self.use_skip:
            output_features = output_features + skip_features
            norm = norm + 1

        if self.bias is not None:
            output_features = output_features + self.bias[None, None, :]
            norm = norm + 1

        # output_features = output_features / norm[:, :, None]

        return output_features
