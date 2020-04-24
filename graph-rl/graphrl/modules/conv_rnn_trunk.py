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

from graphrl.modules.nn import MLP, ConvStack, SpatialReduce


class ConvRNNTrunk(nn.Module):
    def __init__(self,
                 input_shape,
                 conv_out_cs, conv_filter_sizes, conv_paddings, conv_strides,
                 rnn_type, rnn_num_layers, rnn_hidden_size,
                 fc_hidden_sizes,
                 reduction,
                 activation):
        super(ConvRNNTrunk, self).__init__()
        self.conv_stack = ConvStack(input_shape[1], conv_out_cs, conv_filter_sizes, conv_paddings, conv_strides, activation)
        self.spatial_reduce = SpatialReduce(reduction)
        self.rnn_type = rnn_type

        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(conv_out_cs[-1], rnn_hidden_size, rnn_num_layers)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(conv_out_cs[-1], rnn_hidden_size, rnn_num_layers)
        else:
            raise ValueError('Unknown rnn type: {}'.format(self.rnn_type))

        self.mlp = MLP(rnn_hidden_size * rnn_num_layers, fc_hidden_sizes, activation)

    def forward(self, x):
        t, bz, c, h, w = x.shape

        x = x.view(t * bz, c, h, w)
        x = self.conv_stack(x)
        x = self.spatial_reduce(x)

        c = x.size(1)

        x = x.view(t, bz, c)

        _, h = self.rnn(x)

        if self.rnn_type == 'lstm':
            h = h[0]

        h = h.permute(1, 0, 2)
        h = h.view(bz, -1)

        h = self.mlp(h)

        return h
