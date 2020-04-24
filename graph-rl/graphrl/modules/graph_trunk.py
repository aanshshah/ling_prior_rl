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

import torch.nn as nn
from torch_geometric.nn import RGCNConv, GCNConv

from graphrl.modules.nn import SpatialReduce, MLP, get_activation_cls
from graphrl.modules.graph.sgkg import KGIntoSGPool, KGSGIntoSGConv, KGSGWrapperOp, KGFlattenUnflatten, KGSGIntoKGConv
from graphrl.modules.graph.dgconv import DGConv
from graphrl.modules.graph.dgat import DGAT
from graphrl.modules.nn import init_module, get_activation_gain


def make_activation_layer(activation_name, activation_key):
    activation_obj = get_activation_cls(activation_name)()

    def inner(x):
        return [activation_obj(x)]

    return KGSGWrapperOp(inner, [activation_key], [activation_key], single_result=False)


class GraphTrunk(nn.Module):
    def __init__(self, graph_layer_params, named_graph_params, final_graph_hidden_size, fc_hidden_sizes, use_orth_init):
        super(GraphTrunk, self).__init__()
        self.graph_layer_params = graph_layer_params
        self.fc_hidden_sizes = fc_hidden_sizes
        self.use_orth_init = use_orth_init

        seq = []

        for layer_params in graph_layer_params:
            layer_type = layer_params['type']
            layer_args = list(layer_params.get('args', ()))

            new_layer_args = []
            for arg in layer_args:
                if isinstance(arg, str) and arg.startswith('named_'):
                    arg = arg.split('named_')[1]
                    arg = named_graph_params[arg]
                new_layer_args.append(arg)
            layer_args = new_layer_args

            layer_kwargs = layer_params.get('kwargs', {})
            layer = self.make_layer(layer_type, layer_args, layer_kwargs)
            seq.append(layer)
        self.graph_stack = nn.Sequential(*seq)

        self.spatial_reduce = SpatialReduce(reduction='mean')
        self.mlp = MLP(final_graph_hidden_size, fc_hidden_sizes, 'relu')

    def make_layer(self, layer_type, layer_args, layer_kwargs):
        if layer_type == 'kg_sg_pool':
            return KGIntoSGPool()
        elif layer_type == 'kg_rgcnconv':
            rgconv_layer = RGCNConv(*layer_args, **layer_kwargs)
            return KGFlattenUnflatten(KGSGWrapperOp(rgconv_layer, ['kg_node_feats', 'kg_edges', 'kg_rels'], ['kg_node_feats'], single_result=True))
        elif layer_type == 'kg_dgconv':
            dgconv_layer = DGConv(*layer_args, **layer_kwargs, use_orth_init=self.use_orth_init)
            return KGFlattenUnflatten(KGSGWrapperOp(dgconv_layer, ['kg_node_feats', 'kg_edges', 'kg_rels'], ['kg_node_feats'], single_result=True))
        elif layer_type == 'kg_dgat':
            dgconv_layer = DGAT(*layer_args, **layer_kwargs)
            return KGFlattenUnflatten(KGSGWrapperOp(dgconv_layer, ['kg_node_feats', 'kg_edges', 'kg_rels'], ['kg_node_feats'], single_result=True))
        elif layer_type == 'kg_gcn':
            gcn_layer = GCNConv(*layer_args, **layer_kwargs)
            return KGFlattenUnflatten(KGSGWrapperOp(gcn_layer, ['kg_node_feats', 'kg_edges'], ['kg_node_feats'], single_result=True))
        elif layer_type == 'activation':
            return make_activation_layer(*layer_args, **layer_kwargs)
        elif layer_type == 'kgsg_sg_conv':
            return KGSGIntoSGConv(*layer_args, **layer_kwargs, use_orth_init=self.use_orth_init)
        elif layer_type == 'kgsg_kg_conv':
            return KGSGIntoKGConv(*layer_args, **layer_kwargs, use_orth_init=self.use_orth_init)
        elif layer_type == 'sg_conv':
            conv_layer = nn.Conv2d(*layer_args, **layer_kwargs)
            if self.use_orth_init:
                init_module(conv_layer, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), get_activation_gain('relu'))
            return KGSGWrapperOp(conv_layer, ['sg_node_feats'], ['sg_node_feats'], single_result=True)
        else:
            raise ValueError('Unknown layer type {}.'.format(layer_type))

    def forward(self, batch):
        batch = self.graph_stack(batch)

        sg_node_feats = batch['sg_node_feats']
        x = self.spatial_reduce(sg_node_feats)
        x = self.mlp(x)
        return x
