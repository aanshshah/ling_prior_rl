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

import argparse
import json
import os
from utils import create_dir


parser = argparse.ArgumentParser()
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--no_edges', action='store_true', default=False)
parser.add_argument('--fully_connected', action='store_true', default=False)
parser.add_argument('--fully_connected_distinct', action='store_true', default=False)
parser.add_argument('--same_edge_feats', action='store_true', default=False)
args = parser.parse_args()

output_file = args.output_file
no_edges = args.no_edges
fully_connected = args.fully_connected
fully_connected_distinct = args.fully_connected_distinct
same_edge_feats = args.same_edge_feats


if 'ENV_ROOT_DIR' in os.environ:
    output_file = os.path.join(os.environ['ENV_ROOT_DIR'], output_file)

create_dir(output_file)

PLAYER_CHARACTER = 'P'
GHOST_CHARACTER = 'G'
SCARED_GHOST_CHARACTER = 'H'
CAPSULE_CHARACTER = 'o'
COIN_CHARACTER = '.'
BACKGROUND_CHARACTER = ' '
WALL_CHARACTER = '%'

kg_entities = [PLAYER_CHARACTER, GHOST_CHARACTER, SCARED_GHOST_CHARACTER, CAPSULE_CHARACTER, COIN_CHARACTER, BACKGROUND_CHARACTER, WALL_CHARACTER]

kg_dict = {}
kg_dict['entities'] = kg_entities
kg_dict['nodes'] = []
kg_dict['edges'] = []
kg_dict['num_node_feats'] = len(kg_entities)

for i, entity in enumerate(kg_entities):
    node_d = {}
    node_d['node'] = entity
    node_d['feature_type'] = 'one_hot'
    node_d['feature_idx'] = i
    node_d['feature_len'] = len(kg_entities)
    kg_dict['nodes'].append(node_d)


def add_edge(src, dst, feature_idx, feature_len):
    if same_edge_feats:
        feature_idx = 0
        feature_len = 1

    edge_d = {}
    edge_d['src'] = src
    edge_d['dst'] = dst
    edge_d['feature_idx'] = feature_idx
    edge_d['feature_len'] = feature_len
    kg_dict['edges'].append(edge_d)


if no_edges:
    kg_dict['num_edge_feats'] = 1
elif fully_connected:
    for i, n1 in enumerate(kg_entities):
        for n2 in kg_entities[i + 1:]:
            add_edge(n1, n2, 0, 1)
            add_edge(n2, n1, 0, 1)
    kg_dict['num_edge_feats'] = 1
elif fully_connected_distinct:
    feature_len = len(kg_entities) * (len(kg_entities) - 1) * 2
    cur_idx = 0

    for i, n1 in enumerate(kg_entities):
        for n2 in kg_entities[i + 1:]:
            add_edge(n1, n2, cur_idx, feature_len)
            cur_idx += 1
            add_edge(n2, n1, cur_idx, feature_len)
            cur_idx += 1
    kg_dict['num_edge_feats'] = feature_len
else:
    feature_types = {
        'impassable': 0,
        'eats_coin': 1,
        'eats_capsule': 2,
        'eats_player': 3,
        'eats_scared_ghost': 4,
        'becomes_scared': 5
    }

    for c in [PLAYER_CHARACTER, GHOST_CHARACTER, SCARED_GHOST_CHARACTER]:
        add_edge(c, WALL_CHARACTER, feature_types['impassable'], len(feature_types))
    add_edge(PLAYER_CHARACTER, COIN_CHARACTER, feature_types['eats_coin'], len(feature_types))
    add_edge(PLAYER_CHARACTER, CAPSULE_CHARACTER, feature_types['eats_capsule'], len(feature_types))
    add_edge(PLAYER_CHARACTER, SCARED_GHOST_CHARACTER, feature_types['eats_scared_ghost'], len(feature_types))
    add_edge(GHOST_CHARACTER, PLAYER_CHARACTER, feature_types['eats_player'], len(feature_types))
    add_edge(GHOST_CHARACTER, SCARED_GHOST_CHARACTER, feature_types['becomes_scared'], len(feature_types))

    kg_dict['num_edge_feats'] = len(feature_types)

with open(output_file, 'w') as f:
    json.dump(kg_dict, f, indent=4)
