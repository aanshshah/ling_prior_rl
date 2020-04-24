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

from graphrl.modules.nn import Lambda, BatchFlatten, MLP, ConvStack, SpatialReduce, get_activation_cls, compute_conv_output_shape, init_module, get_activation_gain, MiniResidualStack


class LayerTrunk(nn.Module):
    def __init__(self, input_shape, all_layer_params):
        super(LayerTrunk, self).__init__()

        seq = []

        for layer_params in all_layer_params:
            layer_type = layer_params['type']
            layer_args = list(layer_params.get('args', ()))
            layer_kwargs = dict(layer_params.get('kwargs', {}))
            extras = layer_params.get('extras', {})
            input_shape, layer = self.make_layer(input_shape, layer_type, layer_args, layer_kwargs, extras)
            seq.append(layer)
        self.seq = nn.Sequential(*seq)
        self.final_shape = input_shape

    def get_output_shape(self):
        return self.final_shape

    def make_layer(self, input_shape, layer_type, layer_args, layer_kwargs, extras):
        if layer_type == 'activation':
            activation_obj = get_activation_cls(layer_args[0])()
            return input_shape, activation_obj
        elif layer_type == 'conv':
            layer_args = [input_shape[0]] + layer_args
            conv_layer = nn.Conv2d(*layer_args, **layer_kwargs)
            init_module(conv_layer, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), get_activation_gain('relu'))
            input_shape = compute_conv_output_shape(input_shape, *layer_args[1:], **layer_kwargs)
            return input_shape, conv_layer
        elif layer_type == 'spatial_reduce':
            if len(input_shape) != 3:
                raise ValueError
            input_shape = input_shape[:1]
            return input_shape, SpatialReduce(layer_args[0])
        elif layer_type == 'flatten':
            input_shape = [np.prod(input_shape)]
            return input_shape, BatchFlatten()
        elif layer_type == 'linear':
            layer_args = [input_shape[0]] + layer_args
            linear_layer = nn.Linear(*layer_args, **layer_kwargs)
            init_module(linear_layer, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), get_activation_gain('relu'))
            return [layer_args[1]], linear_layer
        else:
            raise ValueError('Unknown layer type {}'.format(layer_type))

    def forward(self, x):
        return self.seq(x)


class MLPTrunk(nn.Module):
    def __init__(self, input_shape, hidden_sizes, activation):
        super(MLPTrunk, self).__init__()
        layers = []
        layers.append(Lambda(lambda x: x.float()))
        layers.append(BatchFlatten())
        layers.append(MLP(int(np.prod(input_shape)), hidden_sizes, activation))
        self.seq = nn.Sequential(*layers)
        self.hidden_sizes = list(hidden_sizes)

    def get_output_shape(self):
        return [self.hidden_sizes[-1]]

    def forward(self, x):
        return self.seq(x)


class SiameseMLPTrunk(nn.Module):
    def __init__(self, input_shape, siamese_hidden_sizes, joint_hidden_sizes, activation, ordered_keys, use_orth_init=True):
        super(SiameseMLPTrunk, self).__init__()
        self.idv_mlp = MLP(int(np.prod(input_shape)), siamese_hidden_sizes, activation, use_orth_init=use_orth_init)
        self.joint_mlp = MLP(siamese_hidden_sizes[-1] * len(ordered_keys), joint_hidden_sizes, activation, use_orth_init=use_orth_init)
        self.ordered_keys = ordered_keys

    def forward(self, x):
        vals = [x[key] for key in self.ordered_keys]
        bz = vals[0].size(0)
        vals = [val.view(bz, -1) for val in vals]
        stacked = torch.stack(vals, 1)
        hidden_stacked = self.idv_mlp(stacked)
        hidden = hidden_stacked.view(bz, -1)
        return self.joint_mlp(hidden)


class ConvMLPTrunk(nn.Module):
    def __init__(self,
                 input_shape,
                 conv_out_cs, conv_filter_sizes, conv_paddings, conv_strides,
                 fc_hidden_sizes,
                 activation):
        super(ConvMLPTrunk, self).__init__()
        self.conv_stack = ConvStack(input_shape[0], conv_out_cs, conv_filter_sizes, conv_paddings, conv_strides, activation)
        conv_output_shape = self.conv_stack.compute_output_shape(input_shape)
        self.mlp_trunk = MLPTrunk(conv_output_shape, fc_hidden_sizes, activation)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.mlp_trunk(x)
        return x

    def get_output_shape(self):
        return self.mlp_trunk.get_output_shape()


class MiniResidualTrunk(nn.Module):
    def __init__(self,
                 input_shape,
                 conv_out_cs, conv_filter_sizes, conv_paddings, conv_strides,
                 residual_cs, num_residual_blocks,
                 fc_hidden_sizes,
                 activation,
                 reduction):
        super(MiniResidualTrunk, self).__init__()
        seq = []
        seq.append(ConvStack(input_shape[0], conv_out_cs, conv_filter_sizes, conv_paddings, conv_strides, activation))
        seq.append(nn.Conv2d(conv_out_cs[-1], residual_cs, 3, stride=1, padding=1))
        seq.append(MiniResidualStack(residual_cs, num_residual_blocks, activation))
        seq.append(SpatialReduce(reduction))
        self.mlp_trunk = MLPTrunk(residual_cs, fc_hidden_sizes, activation)
        seq.append(self.mlp_trunk)

        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)

    def get_output_shape(self):
        return self.mlp_trunk.get_output_shape()


NATURE_FC_HIDDEN_SIZE = 512


class NatureTrunk(nn.Module):
    def __init__(self, input_shape, activation):
        super(NatureTrunk, self).__init__()
        permuted_shape = (input_shape[2], input_shape[0], input_shape[1])
        self.conv_mlp_trunk = ConvMLPTrunk(input_shape=permuted_shape,
                                           conv_out_cs=[32, 64, 64], conv_filter_sizes=[8, 4, 3], conv_paddings=[0, 0, 0], conv_strides=[4, 2, 1],
                                           fc_hidden_sizes=[NATURE_FC_HIDDEN_SIZE],
                                           activation=activation)

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 3, 1, 2)
        x = x / 255.
        return self.conv_mlp_trunk(x)

    def get_output_shape(self):
        return self.conv_mlp_trunk.get_output_shape()


class ConvReduceMLPTrunk(nn.Module):
    def __init__(self,
                 input_shape,
                 conv_out_cs, conv_filter_sizes, conv_paddings, conv_strides,
                 fc_hidden_sizes,
                 activation,
                 reduction):
        super(ConvReduceMLPTrunk, self).__init__()
        seq = []

        conv_stack = ConvStack(input_shape[0], conv_out_cs, conv_filter_sizes, conv_paddings, conv_strides, activation)
        seq.append(conv_stack)

        seq.append(SpatialReduce(reduction))

        self.mlp_trunk = MLPTrunk(conv_out_cs[-1], fc_hidden_sizes, activation)
        seq.append(self.mlp_trunk)

        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)

    def get_output_shape(self):
        return self.mlp_trunk.get_output_shape()


ALL_TRUNK_CONFIG = {
    'trunk_type': '',
    'hidden_sizes': [],
    'conv_out_cs': [], 'conv_filter_sizes': [], 'conv_paddings': [], 'conv_strides': [],
    'fc_hidden_sizes': [],
    'reduction': '',
    'activation': 'relu',
    'all_layer_params': []
}
