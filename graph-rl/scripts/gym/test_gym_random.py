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

import gym
import torch
import sacred

from graphrl.sacred.custom_file_observer import CustomFileStorageOption
from graphrl.agents.random_agent import RandomAgentParams
from graphrl.environments.atari_wrappers import make_atari_env


ex = sacred.Experiment('test_gym_random')


@ex.config
def config():
    env = {
        'name': 'CartPole-v0',
        'atari': False
    }


agent_params = RandomAgentParams(ex)
agent_params.set_sacred_defaults()


@ex.automain
def main(_config, _seed):
    torch.manual_seed(_seed)

    env_name = _config['env']['name']
    use_atari = _config['env']['atari']

    def env_func():
        if use_atari:
            env = make_atari_env(env_name, use_lazy_frames=False)
        else:
            env = gym.make(env_name)
        return env

    agent_params.load_sacred_config()
    agent_params.env_func = env_func
    agent_params.mode = 'test'

    agent = agent_params.make_agent()
    agent.run()
