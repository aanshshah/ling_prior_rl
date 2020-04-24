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
import torch.nn as nn

from graphrl.models.utils import get_cls


class TrunkAndHead(nn.Module):
    def __init__(self, input_shape, num_actions, trunk, head):
        super(TrunkAndHead, self).__init__()

        self.trunk = get_cls(trunk['path'], trunk['name'])(input_shape, **trunk['kwargs'])
        trunk_output_size = int(np.prod(self.trunk.get_output_shape()))
        self.head = get_cls(head['path'], head['name'])(trunk_output_size, num_actions, **head['kwargs'])

    def forward(self, x):
        return self.head(self.trunk(x))
