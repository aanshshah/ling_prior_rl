import sacred
import glob
import os
import json
import torch
import torch.nn as nn

from graphrl.sacred.config import maybe_add_slack
from graphrl.sacred.custom_file_observer import CustomFileStorageOption
from graphrl.agents.deep_q_agent import DeepQAgentParams
from graphrl.modules.heads import QHead
from graphrl.environments.graph_env import GraphEnv
from graphrl.modules.graph_trunk import GraphTrunk
from graphrl.environments.pacman.pacman_gym_v1 import PacmanEnv, EnlargeEnv
from graphrl.environments.wrappers import RenderEnv, SampleEnv


ex = sacred.Experiment('train_pacman_v1_dqn_graph')
maybe_add_slack(ex)


@ex.config
def config():
    rootdir = './'

    env = {
        'train': {
            'layout_folder': 'assets/pacman/smallGrid',
            'ghost_type': 'random',
            'render': False,
            'dont_crop_adj': False,
            'one_hot_edges': True,
            'enlarge': False,

            'kg': {
                'file': './'
            }
        },
        'test': {
            'layout_folder': 'assets/pacman/smallGrid',
            'ghost_type': 'random',
            'render': False,
            'dont_crop_adj': False,
            'one_hot_edges': True,
            'enlarge': False,

            'kg': {
                'file': './'
            }
        }
    }

    arch = {
        'trunk': {
            'fc_hidden_sizes': [],
            'final_graph_hidden_size': 0,
            'graph_layer_params': []
        }
    }


agent_params = DeepQAgentParams(ex)
agent_params.set_sacred_defaults()


@ex.capture(prefix='arch')
def build_net(num_actions, num_node_feats, num_edge_feats, trunk):
    named_graph_params = {'num_node_feats': num_node_feats, 'num_edge_feats': num_edge_feats}
    trunk_module = GraphTrunk(**trunk, named_graph_params=named_graph_params)
    trunk_output_size = trunk['fc_hidden_sizes'][-1]
    head = QHead(trunk_output_size, num_actions)
    return nn.Sequential(trunk_module, head)


def build_envs(layout_folder, ghost_type, render, dont_crop_adj, kg, one_hot_edges, phase, enlarge, should_print):
    layout_files = glob.glob(os.path.join(layout_folder, '*.lay'))

    with open(kg['file']) as f:
        kg_dict = json.load(f)

    if should_print:
        print('Reading layout folder: {}'.format(layout_folder))
        for i, layout_file in enumerate(layout_files):
            print('{}) {}'.format(i + 1, layout_file))

        print('KG entities: {}'.format(kg_dict['entities']))
        print('KG node num feats: {}'.format(kg_dict['num_node_feats']))
        print('KG edge num feats: {}'.format(kg_dict['num_edge_feats']))
        print('KG')
        print('KG nodes')
        for node_dict in kg_dict['nodes']:
            print('Node: {}, idx: {}, node feature length: {}, node feature argmax: {}'.format(node_dict['node'], kg_dict['entities'].index(node_dict['node']), node_dict['feature_len'], node_dict['feature_idx']))

        print('KG edges')
        for edge_dict in kg_dict['edges']:
            print('Source: {}, dst: {}, edge feature length: {}, edge feature {}'.format(edge_dict['src'], edge_dict['dst'], edge_dict['feature_len'], edge_dict['feature_idx']))

    def env_func(layout_file):
        env = PacmanEnv(layout_file, ghost_type)
        if enlarge:
            env = EnlargeEnv(env)
        if render:
            env = RenderEnv(env)
        env = GraphEnv(env=env, kg_dict=kg_dict, dont_crop_adj=dont_crop_adj, one_hot_edges=one_hot_edges)
        return env

    envs = [env_func(layout_file) for layout_file in layout_files]
    return envs


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

    test_envs = build_envs(**env['test'], should_print=True, phase='test')

    num_actions = train_env_func().action_space.n

    agent_params.load_sacred_config()
    agent_params.train_env_func = train_env_func
    agent_params.test_envs = test_envs

    agent_params.obs_filter = GraphEnv.batch_observations

    online_q_net = build_net(num_actions=num_actions, num_node_feats=num_node_feats, num_edge_feats=num_edge_feats)
    target_q_net = build_net(num_actions=num_actions, num_node_feats=num_node_feats, num_edge_feats=num_edge_feats)
    agent_params.online_q_net = online_q_net
    agent_params.target_q_net = target_q_net

    agent_params.add_episode_metric('episode_success', sum)

    agent = agent_params.make_agent()
    agent.run()
