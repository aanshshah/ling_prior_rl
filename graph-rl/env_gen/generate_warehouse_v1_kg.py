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


def parse_bucket_to_boxes(l, use_raw_ascii):
    bucket_to_boxes = {}
    cur_bucket = None
    cur_boxes = None
    should_add = False

    for i, c in enumerate(l):
        if use_raw_ascii:
            c = chr(int(c))

        if str.isupper(c):
            if should_add:
                bucket_to_boxes[cur_bucket] = cur_boxes
            cur_bucket = c
            cur_boxes = []
            should_add = True
        else:
            cur_boxes.append(c)

    if should_add:
        bucket_to_boxes[cur_bucket] = cur_boxes

    return bucket_to_boxes


parser = argparse.ArgumentParser()
parser.add_argument('--bucket_to_boxes', type=str, nargs='+', required=True)
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--use_colors', action='store_true', default=False)
parser.add_argument('--use_background_entity', action='store_true', default=False)
parser.add_argument('--use_negative_fills', action='store_true', default=False)
parser.add_argument('--use_agent_filled', action='store_true', default=False)
parser.add_argument('--no_edges', action='store_true', default=False)
parser.add_argument('--fully_connected', action='store_true', default=False)
parser.add_argument('--fully_connected_distinct', action='store_true', default=False)
parser.add_argument('--same_edge_feats', action='store_true', default=False)
parser.add_argument('--use_raw_ascii', action='store_true', default=False)
args = parser.parse_args()

bucket_to_boxes = parse_bucket_to_boxes(args.bucket_to_boxes, args.use_raw_ascii)
output_file = args.output_file
use_background_entity = args.use_background_entity
use_negative_fills = args.use_negative_fills
use_agent_filled = args.use_agent_filled
no_edges = args.no_edges
fully_connected = args.fully_connected
fully_connected_distinct = args.fully_connected_distinct
use_colors = args.use_colors
same_edge_feats = args.same_edge_feats

if 'ENV_ROOT_DIR' in os.environ:
    output_file = os.path.join(os.environ['ENV_ROOT_DIR'], output_file)

bucket_entities = []
box_entities = set()

for bucket_c, bucket_box_cs in bucket_to_boxes.items():
    bucket_entities.append(bucket_c)
    box_entities.update(bucket_box_cs)

box_entities = list(box_entities)

AGENT_CHARACTER = 'A'
WALL_CHARACTER = '+'
FILLED_CHARACTER = 'X'

kg_entities = [WALL_CHARACTER, AGENT_CHARACTER, FILLED_CHARACTER] + bucket_entities + box_entities

if use_background_entity:
    kg_entities.append(' ')

entity_to_type = {}

char_ls = [[WALL_CHARACTER], [AGENT_CHARACTER], [FILLED_CHARACTER], bucket_entities, box_entities]
if use_background_entity:
    char_ls.append([[' ']])
for i, char_l in enumerate(char_ls):
    for c in char_l:
        entity_to_type[c] = i
num_entity_types = len(char_ls)


entity_to_color = {}
for i, (bucket_c, bucket_box_cs) in enumerate(bucket_to_boxes.items()):
    entity_to_color[bucket_c] = i
    for box_c in bucket_box_cs:
        entity_to_color[box_c] = i
num_entity_colors = len(bucket_to_boxes)


kg_dict = {}
kg_dict['entities'] = kg_entities
kg_dict['nodes'] = []
kg_dict['edges'] = []

if use_colors:
    kg_dict['num_node_feats'] = num_entity_types + num_entity_colors

    for i, entity in enumerate(kg_entities):
        node_d = {}
        node_d['node'] = entity
        node_d['feature_len'] = num_entity_types + num_entity_colors

        type_feat = [0] * num_entity_types
        color_feat = [0] * num_entity_colors
        type_feat[entity_to_type[entity]] = 1
        if entity in entity_to_color:
            color_feat[entity_to_color[entity]] = 1
        node_d['feats'] = type_feat + color_feat
        node_d['feature_type'] = 'list'
        kg_dict['nodes'].append(node_d)
else:
    kg_dict['num_node_feats'] = len(kg_entities)

    for i, entity in enumerate(kg_entities):
        node_d = {}
        node_d['node'] = entity
        node_d['feature_idx'] = i
        node_d['feature_len'] = len(kg_entities)
        node_d['feature_type'] = 'one_hot'
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
        'fills': 0,
        'pushes': 1,
        'impassable': 2
    }

    if use_negative_fills:
        feature_types['negative_fills'] = len(feature_types)

    if use_agent_filled:
        feature_types['agent_filled'] = len(feature_types)

    for bucket, boxes in bucket_to_boxes.items():
        for box in boxes:
            add_edge(box, bucket, feature_types['fills'], len(feature_types))

    if use_negative_fills:
        for bucket in bucket_to_boxes:
            for box in box_entities:
                if box not in bucket_to_boxes[bucket]:
                    add_edge(box, bucket, feature_types['negative_fills'], len(feature_types))

    for box in box_entities:
        add_edge(AGENT_CHARACTER, box, feature_types['pushes'], len(feature_types))

    for bucket in bucket_to_boxes:
        add_edge(AGENT_CHARACTER, bucket, feature_types['impassable'], len(feature_types))
    add_edge(AGENT_CHARACTER, WALL_CHARACTER, feature_types['impassable'], len(feature_types))

    if use_agent_filled:
        add_edge(AGENT_CHARACTER, FILLED_CHARACTER, feature_types['agent_filled'], len(feature_types))

    kg_dict['num_edge_feats'] = len(feature_types)


with open(output_file, 'w') as f:
    json.dump(kg_dict, f, indent=4)
