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

from torch_scatter import scatter_mean

from graphrl.modules.nn import get_activation_gain, init_module


def kg_into_sg_pool_inner(kg_node_feats, obs):
    bz, height, width = obs.shape
    channels = kg_node_feats.size(2)

    kg_node_feats = kg_node_feats.permute(0, 2, 1)

    sg_node_feats = torch.gather(kg_node_feats, 2, obs.view(bz, 1, height * width).repeat(1, channels, 1))
    sg_node_feats = sg_node_feats.view(bz, channels, height, width)
    return sg_node_feats


def sg_into_kg_pool_inner(sg_node_feats, obs, num_kg_nodes):
    bz, channels, height, width = sg_node_feats.shape
    sg_node_feats = sg_node_feats.view(bz, channels, height * width)
    obs = obs.view(bz, -1)[:, None, :].repeat(1, channels, 1)
    output_feats = scatter_mean(sg_node_feats, obs, 2, dim_size=num_kg_nodes)

    output_feats = output_feats.permute(0, 2, 1)
    return output_feats


class KGIntoSGPool(nn.Module):
    def forward(self, batch):
        obs = batch['obs']
        kg_node_feats = batch['kg_node_feats']
        sg_node_feats = kg_into_sg_pool_inner(kg_node_feats, obs)

        new_batch = dict(batch)
        new_batch['sg_node_feats'] = sg_node_feats
        return new_batch


class KGSGWrapperOp(nn.Module):
    def __init__(self, inner, get_keys, replace_keys, single_result):
        super(KGSGWrapperOp, self).__init__()
        self.inner = inner
        self.get_keys = get_keys
        self.replace_keys = replace_keys
        self.single_result = single_result

    def forward(self, batch):
        arguments = [batch[key] for key in self.get_keys]
        results = self.inner(*arguments)
        if self.single_result:
            results = [results]

        new_batch = dict(batch)
        for key, val in zip(self.replace_keys, results):
            new_batch[key] = val

        return new_batch


class KGFlattenUnflatten(nn.Module):
    def __init__(self, inner):
        super(KGFlattenUnflatten, self).__init__()
        self.inner = inner

    def forward(self, batch):
        kg_node_feats = batch['kg_node_feats']
        bz, num_nodes = kg_node_feats.size(0), kg_node_feats.size(1)
        kg_node_feats = kg_node_feats.contiguous().view(bz * num_nodes, -1)

        new_batch_1 = dict(batch)
        new_batch_1['kg_node_feats'] = kg_node_feats

        new_batch_2 = self.inner(new_batch_1)
        kg_node_feats = new_batch_2['kg_node_feats']
        kg_node_feats = kg_node_feats.view(bz, num_nodes, -1)

        new_batch_3 = dict(new_batch_2)
        new_batch_3['kg_node_feats'] = kg_node_feats

        return new_batch_3


class KGSGIntoSGConv(nn.Module):
    def __init__(self, sg_in_feats, sg_out_feats, kg_in_feats, use_orth_init=False):
        super(KGSGIntoSGConv, self).__init__()
        self.sg_in_feats = sg_in_feats
        self.sg_out_feats = sg_out_feats
        self.kg_in_feats = kg_in_feats

        self.kg_fc = nn.Linear(self.kg_in_feats, self.sg_out_feats)
        self.sg_conv = nn.Conv2d(sg_in_feats, sg_out_feats, 3, padding=1)
        if use_orth_init:
            init_module(self.kg_fc, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), get_activation_gain('relu'))
            init_module(self.sg_conv, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), get_activation_gain('relu'))

    def forward(self, batch):
        obs = batch['obs']
        kg_node_feats = batch['kg_node_feats']
        sg_node_feats = batch['sg_node_feats']

        sg_node_feats = self.sg_conv(sg_node_feats)
        kg_node_feats = self.kg_fc(kg_node_feats)
        kg_into_sg_feats = kg_into_sg_pool_inner(kg_node_feats, obs)
        sg_node_feats = sg_node_feats + kg_into_sg_feats

        new_batch = dict(batch)
        new_batch['sg_node_feats'] = sg_node_feats
        return new_batch


class KGSGIntoKGConv(nn.Module):
    def __init__(self, sg_in_feats, kg_in_feats, kg_out_feats, use_orth_init=False):
        super(KGSGIntoKGConv, self).__init__()
        self.sg_in_feats = sg_in_feats
        self.kg_in_feats = kg_in_feats
        self.kg_out_feats = kg_out_feats

        self.kg_fc = nn.Linear(kg_in_feats, kg_out_feats)
        self.sg_conv = nn.Conv2d(sg_in_feats, kg_out_feats, 1)
        if use_orth_init:
            init_module(self.kg_fc, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), get_activation_gain('relu'))
            init_module(self.sg_conv, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), get_activation_gain('relu'))

    def forward(self, batch):
        obs = batch['obs']
        kg_node_feats = batch['kg_node_feats']
        sg_node_feats = batch['sg_node_feats']

        num_kg_nodes = kg_node_feats.size(1)

        sg_node_feats = self.sg_conv(sg_node_feats)
        sg_into_kg_feats = sg_into_kg_pool_inner(sg_node_feats, obs, num_kg_nodes)

        kg_node_feats = self.kg_fc(kg_node_feats)
        kg_node_feats = sg_into_kg_feats + kg_node_feats

        new_batch = dict(batch)
        new_batch['kg_node_feats'] = kg_node_feats
        return new_batch
