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
import sacred
import torch

from graphrl.sacred.custom_file_observer import CustomFileStorageOption
from graphrl.agents.deep_q_agent import DeepQAgentParams
from graphrl.environments.atari_wrappers import make_atari_env
from graphrl.agents.filters import AtariObservationFilter
from graphrl.models.utils import get_cls


ex = sacred.Experiment('train_gym_dqn')


@ex.config
def config():
    env = {
        'name': 'CartPole-v0',
        'atari': False
    }


agent_params = DeepQAgentParams(ex)
agent_params.set_sacred_defaults()


@ex.capture(prefix='arch')
def build_net(input_shape, num_actions, path, name, kwargs):
    model_cls = get_cls(path, name)
    return model_cls(input_shape=input_shape, num_actions=num_actions, **kwargs)


@ex.automain
def main(_seed, _run, _config):
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

    input_space = env_func().observation_space
    if use_atari:
        agent_params.obs_filter = AtariObservationFilter()
        input_space = agent_params.obs_filter.output_space(input_space)

    input_shape = input_space.shape
    num_actions = env_func().action_space.n

    online_q_net = build_net(input_shape=input_shape, num_actions=num_actions)
    target_q_net = build_net(input_shape=input_shape, num_actions=num_actions)
    agent_params.online_q_net = online_q_net
    agent_params.target_q_net = target_q_net

    agent = agent_params.make_agent()
    agent.run()
