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


class Schedule(object):
    def value(self, step):
        raise NotADirectoryError


class ConstantSchedule(Schedule):
    def __init__(self, constant):
        super(ConstantSchedule, self).__init__()
        self.constant = constant

    def value(self, step):
        return self.constant


class LinearSchedule(Schedule):
    def __init__(self, initial_value, final_value, decay_steps):
        super(LinearSchedule, self).__init__()
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_steps = decay_steps

    def value(self, step):
        delta = (self.final_value - self.initial_value) / float(self.decay_steps)
        current_value = self.initial_value + delta * step
        if self.initial_value < self.final_value:
            current_value = np.clip(current_value, self.initial_value, self.final_value)
        else:
            current_value = np.clip(current_value, self.final_value, self.initial_value)
        return current_value

    def __str__(self):
        return 'LinearSchedule{{initial:{}, final:{}, steps:{}}}'.format(
            self.initial_value,
            self.final_value,
            self.decay_steps
        )
