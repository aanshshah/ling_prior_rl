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
from torch.utils.data.dataloader import default_collate


class RewardFilter(object):
    def __call__(self, value):
        return value

    def __str__(self):
        return 'RewardFilter()'


class RewardRescaleFilter(RewardFilter):
    def __init__(self, scale):
        super(RewardRescaleFilter, self).__init__()
        self.scale = scale

    def __call__(self, value):
        return value / self.scale

    def __str__(self):
        return 'RewardRescaleFilter({})'.format(self.scale)


class ObservationFilter(object):
    def __call__(self, values):
        return default_collate(values)

    def output_space(self, input_space):
        return input_space

    def __str__(self):
        return 'ObservationFilter()'


class AtariObservationFilter(ObservationFilter):
    def __call__(self, values):
        values = [np.array(value) for value in values]
        values = default_collate(values)
        return values

    def __str__(self):
        return 'AtariObservationFilter()'.format()


class RNNObservationFilter(ObservationFilter):
    def __init__(self, batch_first):
        super(RNNObservationFilter, self).__init__()
        self.batch_first = batch_first

    def __call__(self, values):
        values = [np.array(value) for value in values]

        if self.batch_first:
            values = np.stack(values, 0)
        else:
            values = np.stack(values, 1)

        values = default_collate(values)
        return values

    def __str__(self):
        return 'RNNObservationFilter()'.format()
