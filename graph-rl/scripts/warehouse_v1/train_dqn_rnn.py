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
import glob
import os
import json

from graphrl.sacred.custom_file_observer import CustomFileStorageOption
from graphrl.sacred.config import maybe_add_slack
from graphrl.agents.deep_q_agent import DeepQAgentParams
from graphrl.modules.heads import QHead
from graphrl.modules.conv_rnn_trunk import ConvRNNTrunk
from graphrl.environments.warehouse.warehouse_v1 import make_warehouse_env
from graphrl.environments.wrappers import RenderEnv, SampleEnv, CustomFrameStack
from graphrl.agents.stopping import DoesntTrainStoppingCondition
from graphrl.environments.pycolab_wrappers import OneHotEnv
from graphrl.agents.filters import RNNObservationFilter


ex = sacred.Experiment('train_warehouse_v1_dqn')
maybe_add_slack(ex)


@ex.config
def config():
    rootdir = './'

    env = {
        'train': {
            'artfile_folder': 'environments/simple/train',
            'render': False,
            'boxes': ['b'],
            'buckets': ['B'],
            'bucket_to_boxes': [('B', ['b'])],
            'character_map': [],
            'history_num_steps': 4,

            'kg': {
                'file': './',
                'entities': ['A', 'x', 'B', 'b', ' ', 'X'],
                'entities_from_file': False
            }
        },
        'test': {
            'artfile_folder': 'environments/simple/train',
            'render': False,
            'boxes': ['b'],
            'buckets': ['B'],
            'bucket_to_boxes': [('B', ['b'])],
            'character_map': [],
            'history_num_steps': 4,

            'kg': {
                'file': './',
                'entities': ['A', 'x', 'B', 'b', ' ', 'X'],
                'entities_from_file': False
            }
        }
    }

    arch = {
        'trunk': {
            'conv_out_cs': [],
            'conv_filter_sizes': [],
            'conv_paddings': [],
            'conv_strides': [],
            'fc_hidden_sizes': [],
            'rnn_type': 'rnn',
            'rnn_num_layers': 0,
            'rnn_hidden_size': 0
        }
    }

    stop = {
        'min_episodes': 5000,
        'doesnt_train_episodes': 200,
        'bad_reward': 0
    }


agent_params = DeepQAgentParams(ex)
agent_params.set_sacred_defaults()


@ex.capture(prefix='arch')
def build_net(input_shape, num_actions, trunk):
    trunk_module = ConvRNNTrunk(input_shape=input_shape, **trunk)
    trunk_output_size = trunk['fc_hidden_sizes'][-1]
    head = QHead(trunk_output_size, num_actions)
    return nn.Sequential(trunk_module, head)


def load_entities(file, entities, entities_from_file):
    if entities_from_file:
        with open(file, 'r') as f:
            kg_dict = json.load(f)
        entities = kg_dict['entities']
    return entities


def build_envs(artfile_folder, boxes, buckets, bucket_to_boxes, character_map, render, kg, should_print, phase, history_num_steps, **kwargs):
    artfiles = glob.glob(os.path.join(artfile_folder, '*.txt'))
    bucket_to_boxes = dict(bucket_to_boxes)
    character_map = dict(character_map)

    entities = load_entities(**kg)

    if should_print:
        print('Phase: {}'.format(phase))
        print('Reading artfile folder: {}'.format(artfile_folder))
        for i, artfile in enumerate(artfiles):
            print('{}) {}'.format(i + 1, artfile))
        print('Entities: {}'.format(entities))

    def env_func(artfile):
        env = make_warehouse_env(artfile,
                                 boxes=boxes,
                                 buckets=buckets,
                                 bucket_to_boxes=bucket_to_boxes,
                                 character_map=character_map)
        env = OneHotEnv(env, [ord(c) for c in entities])
        env = CustomFrameStack(env, history_num_steps, 0)
        if render:
            env = RenderEnv(env)
        return env

    envs = [env_func(artfile) for artfile in artfiles]
    return envs


@ex.capture(prefix='stop')
def add_stopping_params(params, min_episodes, doesnt_train_episodes, bad_reward):
    stopping_cond = DoesntTrainStoppingCondition(min_episodes, doesnt_train_episodes, bad_reward)
    params.custom_stopping_cond = stopping_cond


@ex.automain
def main(_seed, env, rootdir, _config):
    torch.manual_seed(_seed)

    os.chdir(rootdir)

    def train_env_func(should_print=False):
        train_envs = build_envs(**env['train'], should_print=should_print, phase='train')
        train_env = SampleEnv(train_envs)
        return train_env

    train_env_func(should_print=True)
    test_envs = build_envs(**env['test'], should_print=True, phase='test')

    input_space = train_env_func().observation_space
    agent_params.obs_filter = RNNObservationFilter(batch_first=False)
    input_space = agent_params.obs_filter.output_space(input_space)

    input_shape = input_space.shape
    num_actions = train_env_func().action_space.n

    agent_params.load_sacred_config()
    agent_params.train_env_func = train_env_func
    agent_params.test_envs = test_envs
    add_stopping_params(params=agent_params)

    online_q_net = build_net(input_shape=input_shape, num_actions=num_actions)
    target_q_net = build_net(input_shape=input_shape, num_actions=num_actions)
    agent_params.online_q_net = online_q_net
    agent_params.target_q_net = target_q_net

    agent = agent_params.make_agent()
    agent.run()
