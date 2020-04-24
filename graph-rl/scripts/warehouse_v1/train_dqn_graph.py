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
import re

from graphrl.sacred.config import maybe_add_slack
from graphrl.sacred.custom_file_observer import CustomFileStorageOption
from graphrl.agents.deep_q_agent import DeepQAgentParams
from graphrl.modules.heads import QHead
from graphrl.environments.warehouse.warehouse_v1 import make_warehouse_env
from graphrl.environments.wrappers import RenderEnv, SampleEnv
from graphrl.environments.graph_env import GraphEnv
from graphrl.agents.stopping import DoesntTrainStoppingCondition
from graphrl.modules.graph_trunk import GraphTrunk


ex = sacred.Experiment('train_warehouse_v1_dqn_graph')
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
            'dont_crop_adj': False,
            'one_hot_edges': True,

            'kg': {
                'file': 'knowledge_graphs/simple.json'
            }
        },
        'test': {
            'artfile_folder': 'environments/simple/train',
            'render': False,
            'boxes': ['b'],
            'buckets': ['B'],
            'bucket_to_boxes': [('B', ['b'])],
            'character_map': [],
            'dont_crop_adj': False,
            'one_hot_edges': True,

            'kg': {
                'file': 'knowledge_graphs/simple.json'
            }
        }
    }

    arch = {
        'trunk': {
            'fc_hidden_sizes': [],
            'use_orth_init': False,
            'final_graph_hidden_size': 0,
            'graph_layer_params': []
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
def build_net(num_actions, num_node_feats, num_edge_feats, trunk, _config):
    named_graph_params = {'num_node_feats': num_node_feats, 'num_edge_feats': num_edge_feats}
    trunk_module = GraphTrunk(**trunk, named_graph_params=named_graph_params)
    trunk_output_size = trunk['fc_hidden_sizes'][-1]
    head = QHead(trunk_output_size, num_actions)
    return nn.Sequential(trunk_module, head)


def get_artfiles(artfile_folder):
    artfiles = glob.glob(os.path.join(artfile_folder, '*.txt'))
    final_filenames = [os.path.split(x)[-1] for x in artfiles]
    final_filenames = [x.split('.')[0] for x in final_filenames]
    final_integers = [int(next(re.finditer(r'\d+$', x)).group(0)) for x in final_filenames]
    idxs = list(sorted(range(len(artfiles)), key=lambda i: final_integers[i]))
    artfiles = [artfiles[idx] for idx in idxs]
    return artfiles


def build_envs(artfile_folder, boxes, buckets, bucket_to_boxes, character_map, render, dont_crop_adj, one_hot_edges, kg, phase, should_print):
    artfiles = get_artfiles(artfile_folder)

    bucket_to_boxes = dict(bucket_to_boxes)
    character_map = dict(character_map)

    with open(kg['file']) as f:
        kg_dict = json.load(f)

    if should_print:
        print('Phase: {}'.format(phase))

        print('Reading artfile folder: {}'.format(artfile_folder))
        for i, artfile in enumerate(artfiles):
            print('{}) {}'.format(i + 1, artfile))

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

        print('KG edges')
        for edge_dict in kg_dict['edges']:
            print('Source: {}, dst: {}, edge feature length: {}, edge feature {}'.format(edge_dict['src'], edge_dict['dst'], edge_dict['feature_len'], edge_dict['feature_idx']))

    def env_func(artfile):
        env = make_warehouse_env(artfile,
                                 boxes=boxes,
                                 buckets=buckets,
                                 bucket_to_boxes=bucket_to_boxes,
                                 character_map=character_map)
        if render:
            env = RenderEnv(env)
        env = GraphEnv(env=env, kg_dict=kg_dict, dont_crop_adj=dont_crop_adj, one_hot_edges=one_hot_edges)
        return env
    envs = [env_func(artfile) for artfile in artfiles]
    return envs


@ex.capture(prefix='stop')
def add_stopping_params(params, min_episodes, doesnt_train_episodes, bad_reward):
    stopping_cond = DoesntTrainStoppingCondition(min_episodes, doesnt_train_episodes, bad_reward)
    params.custom_stopping_cond = stopping_cond


@ex.automain
def main(_seed, _run, env, rootdir):
    torch.manual_seed(_seed)

    os.chdir(rootdir)

    with open(env['train']['kg']['file']) as f:
        train_kg_dict = json.load(f)
    num_node_feats = train_kg_dict['num_node_feats']
    num_edge_feats = train_kg_dict['num_edge_feats']

    def train_env_func(should_print=False):
        train_envs = build_envs(**env['train'], should_print=should_print, phase='train')
        train_env = SampleEnv(train_envs)
        return train_env

    train_env_func(should_print=True)
    test_envs = build_envs(**env['test'], should_print=True, phase='test')

    num_actions = train_env_func().action_space.n

    agent_params.load_sacred_config()
    agent_params.train_env_func = train_env_func
    agent_params.test_envs = test_envs
    add_stopping_params(params=agent_params)

    agent_params.obs_filter = GraphEnv.batch_observations

    online_q_net = build_net(num_actions=num_actions, num_node_feats=num_node_feats, num_edge_feats=num_edge_feats)
    target_q_net = build_net(num_actions=num_actions, num_node_feats=num_node_feats, num_edge_feats=num_edge_feats)
    agent_params.online_q_net = online_q_net
    agent_params.target_q_net = target_q_net

    agent_params.add_episode_metric('episode_success', sum)

    agent = agent_params.make_agent()
    agent.run()
