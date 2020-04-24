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

import sacred
import torch
import torch.nn as nn

from graphrl.agents.actor_critic_agent import ActorCriticAgentParams
from graphrl.modules.heads import CategoricalValueHead
from graphrl.modules.trunks import build_trunk, ALL_TRUNK_CONFIG
from graphrl.environments.coins_maze.coins_maze_v0 import make_coins_maze_env
from graphrl.environments.wrappers import RenderEnv


ex = sacred.Experiment('train_coins_maze_ac')


@ex.config
def config():
    env = {
        'artfile': 'assets/coins_maze_art/6x7/art_6x7_1.txt',
        'render': False
    }

    arch = {
        'trunk': ALL_TRUNK_CONFIG
    }


agent_params = ActorCriticAgentParams(ex)
agent_params.set_sacred_defaults()


@ex.capture(prefix='arch')
def build_net(input_shape, num_actions, trunk):
    trunk, trunk_output_size = build_trunk(input_shape=input_shape, **trunk)
    head = CategoricalValueHead(trunk_output_size, num_actions)
    return nn.Sequential(trunk, head)


@ex.capture(prefix='env')
def build_env(artfile, render):
    env = make_coins_maze_env(artfile, encode_onehot=True)
    if render:
        env = RenderEnv(env)
    return env


@ex.automain
def main(_seed, _run):
    torch.manual_seed(_seed)

    input_shape = build_env().observation_space.shape
    num_actions = build_env().action_space.n

    agent_params.load_sacred_config()
    agent_params.env_func = build_env

    policy_value_net = build_net(input_shape=input_shape, num_actions=num_actions)
    agent_params.policy_value_net = policy_value_net

    agent = agent_params.make_agent()
    agent.run()
