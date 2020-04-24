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

from graphrl.environments.warehouse.warehouse_v1 import make_warehouse_env
from graphrl.environments.wrappers import RenderEnv
from graphrl.environments.pycolab_wrappers import IndexToFeatsEnv


ex = sacred.Experiment('train_warehouse_v1_run_moves')


@ex.config
def config():
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


def build_envs(_run, artfile_folder, boxes, buckets, bucket_to_boxes, character_map, render, kg, should_print, should_save, phase, use_movable, **kwargs):
    artfiles = glob.glob(os.path.join(artfile_folder, '*.txt'))
    num_buckets_file = os.path.join(artfile_folder, 'num_buckets.listfile')
    moves_file = os.path.join(artfile_folder, 'moves.listfile')
    bucket_to_boxes = dict(bucket_to_boxes)
    character_map = dict(character_map)

    with open(kg['file']) as f:
        kg_dict = json.load(f)

    with open(num_buckets_file) as f:
        num_buckets_list = list(f.read().split())
        num_buckets_list = [int(n) for n in num_buckets_list]

    with open(moves_file) as f:
        env_moves_list = list(f.read().split())
        env_moves_list = [[int(x) for x in line.split(',')] for line in env_moves_list]

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

    envs = [(artfile, env_moves, num_buckets, env_func(artfile, num_buckets)) for artfile, env_moves, num_buckets in zip(artfiles, env_moves_list, num_buckets_list)]
    return envs


@ex.automain
def main(_run, _seed, env, _config):
    torch.manual_seed(_seed)

    train_envs = build_envs(_run, **env['train'], should_print=True, should_save=True, phase='train')
    test_envs = build_envs(_run, **env['test'], should_print=True, should_save=True, phase='test')

    all_envs = train_envs + test_envs

    for artfile, env_moves, num_buckets, env in all_envs:
        print('Validating {} with {} buckets using moves: {}'.format(artfile, num_buckets, env_moves))

        env.reset()

        buckets_found = 0
        total_success = 0
        done = False

        for move in env_moves:
            _, reward, done, info = env.step(move)
            total_success += info['episode_success']
            if reward > 2.95:
                buckets_found += 1
            if done:
                break

        if buckets_found != num_buckets:
            raise ValueError('Failed.')
        if total_success != 1:
            raise ValueError('Falied success={}'.format(total_success))
