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
import torch.nn as nn

from graphrl.agents.filters import RewardRescaleFilter
from graphrl.agents.ppo_agent import PPOAgentParams
from graphrl.modules.heads import CategoricalValueHead
from graphrl.modules.trunks import build_trunk, ALL_TRUNK_CONFIG
from graphrl.environments.atari_wrappers import make_atari_env


ex = sacred.Experiment('train_gym_ppo')


@ex.config
def config():
    env = {
        'name': 'CartPole-v0',
        'atari': False
    }

    arch = {
        'trunk': ALL_TRUNK_CONFIG
    }

    reward_scale = 200.


agent_params = PPOAgentParams(ex)
agent_params.set_sacred_defaults()


@ex.capture(prefix='arch')
def build_net(input_shape, num_actions, trunk):
    trunk, trunk_output_size = build_trunk(input_shape=input_shape, **trunk)
    head = CategoricalValueHead(trunk_output_size, num_actions)
    return nn.Sequential(trunk, head)


@ex.automain
def main(_seed, _run, _config, reward_scale):
    torch.manual_seed(_seed)

    env_name = _config['env']['name']
    use_atari = _config['env']['atari']

    def env_func():
        if use_atari:
            env = make_atari_env(env_name, use_lazy_frames=False)
        else:
            env = gym.make(env_name)
        return env

    input_shape = env_func().observation_space.shape
    num_actions = env_func().action_space.n

    agent_params.load_sacred_config()
    agent_params.env_func = env_func
    agent_params.reward_filter = RewardRescaleFilter(reward_scale)

    policy_value_net = build_net(input_shape=input_shape, num_actions=num_actions)
    agent_params.policy_value_net = policy_value_net

    agent = agent_params.make_agent()
    agent.run()
