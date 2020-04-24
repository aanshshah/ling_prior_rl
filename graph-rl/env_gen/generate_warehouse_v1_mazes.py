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

import numpy as np
import argparse
import os

from env_gen_utils import make_maps


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
parser.add_argument('--maze_size', type=int, default=10)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num_train', type=int, default=5)
parser.add_argument('--num_test', type=int, default=5)
parser.add_argument('--train_bucket_to_boxes', type=str, nargs='+', required=True)
parser.add_argument('--test_bucket_to_boxes', type=str, nargs='+', required=True)
parser.add_argument('--min_num_buckets', type=int, default=1)
parser.add_argument('--max_num_buckets', type=int, default=1)
parser.add_argument('--unique_buckets', action='store_true', default=False)
parser.add_argument('--min_num_pulls', type=int, default=4)
parser.add_argument('--max_num_pulls', type=int, default=7)
parser.add_argument('--min_random_steps', type=int, default=3)
parser.add_argument('--max_random_steps', type=int, default=6)
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--use_raw_ascii', action='store_true', default=False)
args = parser.parse_args()


np.random.seed(args.seed)  # set seed for repeatability

output_dir = args.dir

if 'ENV_ROOT_DIR' in os.environ:
    output_dir = os.path.join(os.environ['ENV_ROOT_DIR'], output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)


PLAYER_CHARACTER = 'A'
BORDER_CHARACTER = '+'

phase_to_buckets_to_boxes = {
    'train': parse_bucket_to_boxes(args.train_bucket_to_boxes, args.use_raw_ascii),
    'test': parse_bucket_to_boxes(args.test_bucket_to_boxes, args.use_raw_ascii)
}

prev_mazes = set()


for phase, nmazes in [('train', args.num_train), ('test', args.num_test)]:
    phase_dir = os.path.join(output_dir, phase)
    if not os.path.isdir(phase_dir):
        os.mkdir(phase_dir)

    num_buckets_list, mazes, map_moves_list = make_maps(nmazes, prev_mazes=prev_mazes, bucket_to_boxes=phase_to_buckets_to_boxes[phase],
                                                        map_height=args.maze_size, map_width=args.maze_size,
                                                        min_num_buckets=args.min_num_buckets, max_num_buckets=args.max_num_buckets,
                                                        unique_buckets=args.unique_buckets,
                                                        min_num_pulls=args.min_num_pulls, max_num_pulls=args.max_num_pulls,
                                                        min_random_steps=args.min_random_steps, max_random_steps=args.max_random_steps,
                                                        agent_char=PLAYER_CHARACTER, border_char=BORDER_CHARACTER)
    prev_mazes.update(mazes)
    for idx, maze in enumerate(mazes):
        with open(os.path.join(phase_dir, phase + str(idx) + '.txt'), 'w') as f:
            print(maze, end='', file=f)

    with open(os.path.join(phase_dir, 'num_buckets.listfile'), 'w') as f:
        for num_buckets in num_buckets_list:
            print(num_buckets, file=f)

    with open(os.path.join(phase_dir, 'moves.listfile'), 'w') as f:
        for moves in map_moves_list:
            moves = [str(m) for m in moves]
            print(','.join(moves), file=f)
