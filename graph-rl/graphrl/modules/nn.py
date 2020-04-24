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


def init_module(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


def get_activation_cls(activation):
    if activation == 'relu':
        return nn.ReLU
    else:
        raise ValueError('Unknown activation.')


def get_activation_gain(activation):
    if activation == 'relu':
        return nn.init.calculate_gain('relu')
    else:
        raise ValueError('Unknown activation.')


class BatchFlatten(nn.Module):
    def forward(self, x):
        bz = x.size(0)
        return x.view(bz, -1)


class Fork(nn.Module):
    def __init__(self, *args):
        super(Fork, self).__init__()
        self.layers = nn.ModuleList(list(args))

    def forward(self, x):
        results = []
        for i, module in enumerate(self.layers):
            res = module(x)
            results.append(res)
        return tuple(results)


class Lambda(nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation, use_orth_init=True):
        super(MLP, self).__init__()
        seq = []

        for hidden_size in hidden_sizes:
            linear = nn.Linear(input_size, hidden_size)
            if use_orth_init:
                init_module(linear, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), get_activation_gain(activation))
            seq.append(linear)
            seq.append(get_activation_cls(activation)())
            input_size = hidden_size
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class SpatialReduce(nn.Module):
    def __init__(self, reduction):
        super(SpatialReduce, self).__init__()
        self.reduction = reduction

    def forward(self, x):
        bz, c = x.size(0), x.size(1)
        x = x.view(bz, c, -1)

        if self.reduction == 'max':
            x = torch.max(x, 2)[0]
        elif self.reduction == 'mean':
            x = torch.mean(x, 2)
        else:
            raise ValueError('Unknown reduction {}.'.format(self.reduction))
        return x


def compute_conv_output_shape(input_shape, out_channels, kernel_size, stride, padding):
    _, h, w = input_shape

    h = (h - kernel_size + 2 * padding) // stride + 1
    w = (w - kernel_size + 2 * padding) // stride + 1

    return out_channels, h, w


class MiniResidualBlock(nn.Module):
    def __init__(self, channels, activation):
        super(MiniResidualBlock, self).__init__()
        self.channels = channels
        self.activation = activation

        self.conv1 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)

    def forward(self, x):
        activation = getattr(F, self.activation)
        h = self.conv1(activation(x))
        h = self.conv2(activation(h))
        x = x + h
        return x


class MiniResidualStack(nn.Module):
    def __init__(self, channels, num_blocks, activation):
        super(MiniResidualStack, self).__init__()
        seq = []
        for i in range(num_blocks):
            seq.append(MiniResidualBlock(channels, activation))
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class ConvStack(nn.Module):
    def __init__(self, in_c, out_cs, filter_sizes, paddings, strides, activation):
        super(ConvStack, self).__init__()

        self.out_cs = out_cs
        self.filter_sizes = filter_sizes
        self.paddings = paddings
        self.strides = strides

        seq = []
        for out_c, filter_size, padding, stride in zip(out_cs, filter_sizes, paddings, strides):
            conv = nn.Conv2d(in_c, out_c, filter_size, stride=stride, padding=padding)
            init_module(conv, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), get_activation_gain(activation))
            seq.append(conv)
            seq.append(get_activation_cls(activation)())
            in_c = out_c
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)

    def compute_output_shape(self, input_shape):
        for out_c, filter_size, padding, stride in zip(self.out_cs, self.filter_sizes, self.paddings, self.strides):
            input_shape = compute_conv_output_shape(input_shape, out_c, filter_size, stride, padding)
        return input_shape
