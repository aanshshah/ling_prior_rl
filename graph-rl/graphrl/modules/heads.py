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
import torch.distributions as dist

from graphrl.modules.nn import Fork, init_module


class CategoricalHead(nn.Module):
    def __init__(self, input_size, num_actions, use_orth_init=True):
        super(CategoricalHead, self).__init__()

        self.fc = nn.Linear(input_size, num_actions)
        if use_orth_init:
            init_module(self.fc, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)

    def forward(self, x):
        logits = self.fc(x)
        return dist.Categorical(logits=logits)


class ValueHead(nn.Module):
    def __init__(self, input_size, use_orth_init=True):
        super(ValueHead, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        if use_orth_init:
            init_module(self.fc, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

    def forward(self, x):
        return self.fc(x)[:, 0]


class QHead(nn.Module):
    def __init__(self, input_size, num_actions):
        super(QHead, self).__init__()
        self.fc = nn.Linear(input_size, num_actions)
        init_module(self.fc, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

    def forward(self, x):
        return self.fc(x)


class CategoricalValueHead(nn.Module):
    def __init__(self, input_size, num_actions, use_orth_init=True):
        super(CategoricalValueHead, self).__init__()
        categorical_head = CategoricalHead(input_size, num_actions, use_orth_init=use_orth_init)
        value_head = ValueHead(input_size, use_orth_init=use_orth_init)

        self.fork = Fork(categorical_head, value_head)

    def forward(self, x):
        return self.fork(x)
