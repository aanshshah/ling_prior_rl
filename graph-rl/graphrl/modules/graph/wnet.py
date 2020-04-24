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
import collections
import numpy as np
from graphrl.modules.nn import get_activation_gain


class WeightNet(nn.Sequential):
    """ Weight generating network. """

    def __init__(self, hidden_units, in_shape, out_shape, name='wnet', use_orth_init=False):
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
        if use_orth_init:
            layer = nn.Linear(in_feats, weight_size)
            bias_data = torch.zeros(out_shape, dtype=torch.float32)
            nn.init.orthogonal_(bias_data, get_activation_gain('relu'))
            layer.bias.data = bias_data.view(-1)
            nn.init.orthogonal_(layer.weight.data, 0.01)
            layers.append((name + '_weightlin', layer))
        else:
            layers.append((name + '_weightlin', nn.Linear(in_feats, weight_size)))

        super().__init__(collections.OrderedDict(layers))

    def forward(self, input):
        bsize = input.shape[0]
        out = super().forward(input.view(bsize, -1))
        return out.view(-1, *self.out_shape)
