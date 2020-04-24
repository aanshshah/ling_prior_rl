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

from graphrl.sacred.config import maybe_add_slack
from graphrl.sacred.custom_file_observer import CustomFileStorageOption
from graphrl.agents.deep_q_agent import DeepQAgentParams
from graphrl.models.utils import get_cls
from graphrl.environments.wrappers import RenderEnv, SampleEnv
from graphrl.environments.pacman.pacman_gym_v1 import PacmanEnv, EnlargeEnv
from graphrl.environments.pycolab_wrappers import OneHotEnv


ex = sacred.Experiment('train_pacman_v1_dqn')
maybe_add_slack(ex)


@ex.config
def config():
    rootdir = './'

    env = {
        'train': {
            'layout_folder': 'assets/pacman/smallGrid',
            'ghost_type': 'random',
            'render': False,
            'enlarge': False,

            'kg': {
                'file': './',
                'entities': ['%', ' ', '.', 'G', 'H', 'o', 'P'],
                'entities_from_file': False
            }
        },
        'test': {
            'layout_folder': 'assets/pacman/smallGrid',
            'ghost_type': 'random',
            'render': False,
            'enlarge': False,

            'kg': {
                'file': './',
                'entities': ['%', ' ', '.', 'G', 'H', 'o', 'P'],
                'entities_from_file': False
            }
        }
    }

    eps = {
        'eps_type': 'linear',
        'constant_value': 0.1,
        'initial_value': 1.,
        'final_value': 0.01,
        'decay_steps': 10000
    }


agent_params = DeepQAgentParams(ex)
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


def build_envs(layout_folder, ghost_type, render, kg, should_print, phase, enlarge, **kwargs):
    layout_files = glob.glob(os.path.join(layout_folder, '*.lay'))
    entities = load_entities(**kg)

    if should_print:
        print('Phase: {}'.format(phase))
        print('Reading layout folder: {}'.format(layout_folder))
        for i, layout_file in enumerate(layout_files):
            print('{}) {}'.format(i + 1, layout_file))
        print('Entities: {}'.format(entities))

    def env_func(layout_file):
        env = PacmanEnv(layout_file, ghost_type)
        if enlarge:
            env = EnlargeEnv(env)
        if render:
            env = RenderEnv(env)
        env = OneHotEnv(env, [ord(c) for c in entities])
        return env

    envs = [env_func(layout_file) for layout_file in layout_files]
    return envs


@ex.automain
def main(_seed, _run, env, rootdir):
    torch.manual_seed(_seed)

    os.chdir(rootdir)

    def train_env_func(should_print=False):
        train_envs = build_envs(**env['train'], should_print=should_print, phase='train')
        train_env = SampleEnv(train_envs)
        return train_env

    train_env_func(should_print=True)
    test_envs = build_envs(**env['test'], should_print=True, phase='train')

    input_shape = train_env_func().observation_space.shape
    num_actions = train_env_func().action_space.n

    agent_params.load_sacred_config()
    agent_params.train_env_func = train_env_func
    agent_params.test_envs = test_envs

    online_q_net = build_net(input_shape=input_shape, num_actions=num_actions)
    target_q_net = build_net(input_shape=input_shape, num_actions=num_actions)
    agent_params.online_q_net = online_q_net
    agent_params.target_q_net = target_q_net

    agent_params.add_episode_metric('episode_success', sum)

    agent = agent_params.make_agent()
    agent.run()
