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
import glob
import os
import json
import re

from graphrl.sacred.custom_file_observer import CustomFileStorageOption
from graphrl.sacred.config import maybe_add_slack
from graphrl.agents.actor_critic_agent import ActorCriticAgentParams
from graphrl.models.utils import get_cls
from graphrl.environments.warehouse.warehouse_v1 import make_warehouse_env
from graphrl.environments.wrappers import RenderEnv, SampleEnv
from graphrl.environments.pycolab_wrappers import IndexToFeatsEnv


ex = sacred.Experiment('train_warehouse_v1_ac')
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


agent_params = ActorCriticAgentParams(ex)
agent_params.set_sacred_defaults()


@ex.capture(prefix='arch')
def build_net(input_shape, num_actions, path, name, kwargs):
    model_cls = get_cls(path, name)
    return model_cls(input_shape=input_shape, num_actions=num_actions, **kwargs)


def load_entities(file, entities, entities_from_file):
    if entities_from_file:
        with open(file, 'r') as f:
            kg_dict = json.load(f)
        entities = kg_dict['entities']
    return entities


def get_artfiles(artfile_folder):
    artfiles = glob.glob(os.path.join(artfile_folder, '*.txt'))
    final_filenames = [os.path.split(x)[-1] for x in artfiles]
    final_filenames = [x.split('.')[0] for x in final_filenames]
    final_integers = [int(next(re.finditer(r'\d+$', x)).group(0)) for x in final_filenames]
    idxs = list(sorted(range(len(artfiles)), key=lambda i: final_integers[i]))
    artfiles = [artfiles[idx] for idx in idxs]
    return artfiles


def build_envs(_run, artfile_folder, boxes, buckets, bucket_to_boxes, character_map, render, kg, should_print, should_save, phase, use_movable, **kwargs):
    artfiles = get_artfiles(artfile_folder)
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

            if should_save:
                save_filename = os.path.join(*os.path.split(artfile)[-2:])
                _run.add_artifact(artfile, save_filename)

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


@ex.automain
def main(_run, _seed, env, rootdir, _config):
    torch.manual_seed(_seed)

    os.chdir(rootdir)

    def train_env_func(should_print=False, should_save=False):
        train_envs = build_envs(_run, **env['train'], should_print=should_print, should_save=should_save, phase='train')
        train_env = SampleEnv(train_envs)
        return train_env

    train_env_func(should_print=True, should_save=True)
    test_envs = build_envs(_run, **env['test'], should_print=True, should_save=True, phase='test')

    input_shape = train_env_func().observation_space.shape
    num_actions = train_env_func().action_space.n

    agent_params.load_sacred_config()
    agent_params.train_env_func = train_env_func
    agent_params.test_envs = test_envs

    policy_value_net = build_net(input_shape=input_shape, num_actions=num_actions)
    agent_params.policy_value_net = policy_value_net
    agent_params.add_episode_metric('episode_success', sum)

    agent = agent_params.make_agent()
    agent.run()
