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
from graphrl.agents.ppo_agent import PPOAgentParams
from graphrl.modules.heads import CategoricalValueHead
from graphrl.modules.trunks import build_trunk, ALL_TRUNK_CONFIG
from graphrl.environments.warehouse.warehouse_v1 import make_warehouse_env
from graphrl.environments.wrappers import RenderEnv, SampleEnv
from graphrl.agents.stopping import DoesntTrainStoppingCondition
from graphrl.environments.pycolab_wrappers import OneHotEnv


ex = sacred.Experiment('train_warehouse_v1_ppo')
maybe_add_slack(ex)


@ex.config
def arch_config():
    arch = {
        'trunk': ALL_TRUNK_CONFIG
    }


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
            'use_movable': False,

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
            'use_movable': False,

            'kg': {
                'file': './',
                'entities': ['A', 'x', 'B', 'b', ' ', 'X'],
                'entities_from_file': False
            }
        }
    }

    arch = {
        'trunk': {
            'trunk_type': 'mlp',
            'hidden_sizes': [256, 512]
        }
    }

    stop = {
        'min_episodes': 5000,
        'doesnt_train_episodes': 200,
        'bad_reward': 0
    }


agent_params = PPOAgentParams(ex)
agent_params.set_sacred_defaults()


@ex.capture(prefix='arch')
def build_net(input_shape, num_actions, trunk):
    trunk, trunk_output_size = build_trunk(input_shape=input_shape, **trunk)
    head = CategoricalValueHead(trunk_output_size, num_actions)
    return nn.Sequential(trunk, head)


def build_envs(artfile_folder, boxes, buckets, bucket_to_boxes, character_map, render, kg, should_print, phase, use_movable, **kwargs):
    artfiles = glob.glob(os.path.join(artfile_folder, '*.txt'))
    num_buckets_file = os.path.join(artfile_folder, 'num_buckets.listfile')
    bucket_to_boxes = dict(bucket_to_boxes)
    character_map = dict(character_map)

    with open(kg['file']) as f:
        kg_dict = json.load(f)

    with open(num_buckets_file) as f:
        num_buckets_list = list(f.read().split())
        num_buckets_list = [int(n) for n in num_buckets_list]

    if should_print:
        print('Phase: {}'.format(phase))

        print('Reading artfile folder: {}'.format(artfile_folder))
        for i, (artfile, num_buckets) in enumerate(zip(artfiles, num_buckets_list)):
            print('{}) artfile: {}, num buckets: {}'.format(i + 1, artfile, num_buckets))

        print('KG entities: {}'.format(kg_dict['entities']))
        print('KG node num feats: {}'.format(kg_dict['num_node_feats']))
        print('KG edge num feats: {}'.format(kg_dict['num_edge_feats']))
        print('KG')
        print('KG nodes')
        for node_dict in kg_dict['nodes']:
            if node_dict['feature_type'] == 'one_hot':
                print('Node: {}, idx: {}, node feature length: {}, node feature argmax: {}'.format(node_dict['node'], kg_dict['entities'].index(node_dict['node']), node_dict['feature_len'], node_dict['feature_idx']))
            elif node_dict['feature_type'] == 'list':
                print('Node: {}, features: {}'.format(node_dict['node'], node_dict['feats']))
            else:
                raise ValueError('Unknown node feature type {}.'.format(node_dict['feature_type']))

    def env_func(artfile, num_buckets):
        env = make_warehouse_env(artfile,
                                 num_buckets=num_buckets,
                                 boxes=boxes,
                                 buckets=buckets,
                                 bucket_to_boxes=bucket_to_boxes,
                                 character_map=character_map,
                                 use_movable=use_movable)
        env = IndexToFeatsEnv(env, kg_dict)
        if render:
            env = RenderEnv(env)
        return env

    envs = [env_func(artfile, num_buckets) for artfile, num_buckets in zip(artfiles, num_buckets_list)]
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

    input_shape = train_env_func().observation_space.shape
    num_actions = train_env_func().action_space.n

    agent_params.load_sacred_config()
    agent_params.train_env_func = train_env_func
    agent_params.test_envs = test_envs
    add_stopping_params(params=agent_params)

    policy_value_net = build_net(input_shape=input_shape, num_actions=num_actions)
    agent_params.policy_value_net = policy_value_net
    agent_params.add_episode_metric('episode_success', sum)

    agent = agent_params.make_agent()
    agent.run()
