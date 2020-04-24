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
import numpy as np

from graphrl.sacred.custom_file_observer import CustomFileStorageOption
from graphrl.sacred.config import maybe_add_slack
from graphrl.agents.actor_critic_agent import ActorCriticAgentParams
from graphrl.modules.heads import CategoricalValueHead
from graphrl.modules.trunks import build_trunk, ALL_TRUNK_CONFIG
from graphrl.environments.ai2thor_visnav.ai2thor_visnav_v0 import AI2ThorVisnav


ex = sacred.Experiment('train_warehouse_v1_ac')
maybe_add_slack(ex)


@ex.config
def arch_config():
    arch = {
        'trunk': ALL_TRUNK_CONFIG,
        'head': {
            'use_orth_init': False
        }
    }


@ex.config
def config():
    env = {
        'train': {
            'h5_dir': '/localdisk/ai2thor_visnav_data',
            'scene_name': 'bedroom_04'
        },
        'test': {
            'h5_dir': '/localdisk/ai2thor_visnav_data',
            'scene_name': 'bedroom_04'
        }
    }

    arch = {
        'trunk': {
            'trunk_type': 'mlp',
            'hidden_sizes': [256, 512]
        }
    }


agent_params = ActorCriticAgentParams(ex)
agent_params.set_sacred_defaults()


@ex.capture(prefix='arch')
def build_net(input_shape, num_actions, trunk, head):
    trunk, trunk_output_size = build_trunk(input_shape=input_shape, **trunk)
    head = CategoricalValueHead(trunk_output_size, num_actions, use_orth_init=head['use_orth_init'])
    return nn.Sequential(trunk, head)


@ex.automain
def main(_seed, env, _config):
    torch.manual_seed(_seed)

    def train_env_func():
        train_env = AI2ThorVisnav(**env['train'])
        return train_env

    test_envs = [AI2ThorVisnav(**env['test'])]

    input_shape = train_env_func().observation_space.shape
    num_actions = train_env_func().action_space.n

    agent_params.load_sacred_config()
    agent_params.train_env_func = train_env_func
    agent_params.test_envs = test_envs

    policy_value_net = build_net(input_shape=input_shape, num_actions=num_actions)
    agent_params.policy_value_net = policy_value_net
    agent_params.reward_filter = lambda reward: np.clip(reward, -1, 1)

    agent = agent_params.make_agent()
    agent.run()
