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
import sys


def try_reach_squares(map_height, map_width, barriers, dest_squares, start_square):
    q = [start_square]
    visited = set([start_square])
    backtrace = {start_square: (None, None)}

    reached_square = None

    while q:
        node_x, node_y = q.pop(0)

        if (node_x, node_y) in dest_squares:
            reached_square = (node_x, node_y)
            break

        for move_x, move_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_x, next_y = node_x + move_x, node_y + move_y

            if 0 <= next_x < map_height and 0 <= next_y < map_width and (next_x, next_y) not in barriers and (next_x, next_y) not in visited:
                q.append((next_x, next_y))
                visited.add((next_x, next_y))
                backtrace[(next_x, next_y)] = ((node_x, node_y), (move_x, move_y))

    if reached_square is None:
        return None, None, None, False

    path = [reached_square]
    moves = []

    cur_square = reached_square
    while True:
        cur_square, cur_move = backtrace[cur_square]
        if cur_square is None:
            break
        path.append(cur_square)
        moves.append(cur_move)

    path = list(reversed(path))
    moves = list(reversed(moves))

    return reached_square, path, moves, True


def try_pull_box(map_height, map_width, barriers, agent_square, box_square, num_pulls):
    def get_walks(map_height, map_width, barriers, agent_square, box_square):
        barriers = barriers + [box_square]
        box_x, box_y = box_square
        dest_squares = [(box_x - 1, box_y), (box_x + 1, box_y), (box_x, box_y - 1), (box_x, box_y + 1)]

        for dest_square in dest_squares:
            reached_square, _, moves, success = try_reach_squares(map_height, map_width, barriers, [dest_square], agent_square)
            if success:
                yield reached_square, moves

    def get_pull(map_height, map_width, barriers, agent_square, box_square):
        old_agent_square = agent_square
        agent_x, agent_y = agent_square
        box_x, box_y = box_square

        for move_x, move_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if box_x + move_x == agent_x and box_y + move_y == agent_y:
                next_x, next_y = agent_x + move_x, agent_y + move_y

                if 0 <= next_x < map_height and 0 <= next_y < map_width and (next_x, next_y) not in barriers:
                    return (next_x, next_y), old_agent_square, (move_x, move_y), True
        return None, None, None, False

    q = [(agent_square, box_square, 0, [])]
    visited = set([box_square])

    while q:
        cur_agent_square, cur_box_square, cur_num_pulls, cur_moves = q.pop(0)

        if cur_num_pulls == num_pulls:
            yield (cur_agent_square, cur_box_square, cur_moves)
        else:
            agent_walks = get_walks(map_height, map_width, barriers, cur_agent_square, cur_box_square)
            for next_agent_square, next_agent_moves in agent_walks:
                next_agent_square, next_box_square, pull_move, pull_success = get_pull(map_height, map_width, barriers, next_agent_square, cur_box_square)
                next_num_pulls = cur_num_pulls + 1
                if pull_success and next_box_square not in visited:
                    q.append((next_agent_square, next_box_square, next_num_pulls, cur_moves + next_agent_moves + [pull_move]))
                    visited.add(next_box_square)


def sample_reach_squares():

    cases = [
        {
            'map_height': 10,
            'map_width': 10,
            'barriers': [],
            'dest_squares': [(3, 3), (4, 4)],
            'start_square': (4, 4)
        },
        {
            'map_height': 10,
            'map_width': 10,
            'barriers': [],
            'dest_squares': [(4, 4), (5, 5)],
            'start_square': (3, 3)
        },
        {
            'map_height': 8,
            'map_width': 4,
            'barriers': [(5, 0), (5, 1), (5, 2)],
            'dest_squares': [(0, 0), (0, 1), (0, 2)],
            'start_square': (7, 0)
        }
    ]

    for case in cases:
        map_height, map_width, barriers, dest_squares, start_square = case['map_height'], case['map_width'], case['barriers'], case['dest_squares'], case['start_square']

        print('Map height: ', map_height, 'Map width: ', map_width, 'Barriers: ', barriers, 'Dest squares: ', dest_squares, 'Start square: ', start_square)

        reached_square, path, moves, success = try_reach_squares(map_height, map_width, barriers, dest_squares, start_square)

        if success:
            print('Reached: ', reached_square, 'Path: ', path, 'Moves: ', moves)
        else:
            print('Failed')

        print()


def sample_pull_box():
    cases = [
        {
            'map_height': 5,
            'map_width': 5,
            'barriers': [],
            'agent_square': (2, 1),
            'box_square': (1, 1),
            'num_pulls': 2
        },
        {
            'map_height': 2,
            'map_width': 2,
            'barriers': [],
            'agent_square': (0, 0),
            'box_square': (1, 1),
            'num_pulls': 3
        }
    ]

    for case in cases:
        map_height, map_width, barriers, agent_square, box_square, num_pulls = case['map_height'], case['map_width'], case['barriers'], case['agent_square'], case['box_square'], case['num_pulls']

        print('Map height: ', map_height, 'Map width: ', map_width, 'Barriers: ', barriers, 'Agent square: ', agent_square, 'Box square: ', box_square, 'Num pulls: ', num_pulls)

        pulls = list(try_pull_box(map_height, map_width, barriers, agent_square, box_square, num_pulls))
        print('Pulls: ', pulls)

    print()


def initial_map(map_height, map_width, bucket_to_boxes, min_num_buckets, max_num_buckets, unique_buckets):
    num_buckets = np.random.randint(min_num_buckets, max_num_buckets + 1)

    if unique_buckets:
        bucket_choices = list(bucket_to_boxes.keys())
    else:
        bucket_choices = []
        for bucket, boxes in bucket_to_boxes.items():
            for _ in range(len(boxes)):
                bucket_choices.append(bucket)

    np.random.shuffle(bucket_choices)
    buckets = bucket_choices[:num_buckets]

    boxes = []
    for bucket in buckets:
        box_choices = bucket_to_boxes[bucket]
        box_choices = [box for box in box_choices if box not in boxes]
        box = box_choices[np.random.randint(len(box_choices))]
        boxes.append(box)

    square_choices = [(i, j) for i in range(map_height) for j in range(map_width)]
    np.random.shuffle(square_choices)
    squares = square_choices[:num_buckets + 1]
    agent_square = squares[0]
    squares = squares[1:]

    return num_buckets, buckets, boxes, squares, agent_square


def sample_initial_map():
    cases = [
        {
            'map_height': 10,
            'map_width': 10,
            'bucket_to_boxes': {'B': ['b'], 'C': ['c']},
            'min_num_buckets': 2,
            'max_num_buckets': 2,
            'unique_buckets': True
        },
        {
            'map_height': 10,
            'map_width': 10,
            'bucket_to_boxes': {'B': ['b', 'c'], 'D': ['d', 'e']},
            'min_num_buckets': 2,
            'max_num_buckets': 2,
            'unique_buckets': False
        },
        {
            'map_height': 1,
            'map_width': 3,
            'bucket_to_boxes': {'B': ['b', 'c']},
            'min_num_buckets': 2,
            'max_num_buckets': 2,
            'unique_buckets': False
        }
    ]

    for case in cases:
        map_height, map_width, bucket_to_boxes, min_num_buckets, max_num_buckets, unique_buckets = case['map_height'], case['map_width'], case['bucket_to_boxes'], case['min_num_buckets'], case['max_num_buckets'], case['unique_buckets']
        print('Map height', map_height, 'Map width', map_width, 'Bucket to boxes', bucket_to_boxes, 'Min num buckets', min_num_buckets, 'Max num buckets', max_num_buckets, 'Unique buckets', unique_buckets)

        buckets, boxes, squares, agent_square = initial_map(map_height, map_width, bucket_to_boxes, min_num_buckets, max_num_buckets, unique_buckets)
        print('Buckets', buckets, 'boxes', boxes, 'Squares', squares, 'Agent square', agent_square)


def make_map_inner_no_border(map_height, map_width, bucket_to_boxes, min_num_buckets, max_num_buckets, unique_buckets, min_num_pulls, max_num_pulls, min_random_steps, max_random_steps, agent_char):
    num_buckets, buckets, boxes, bucket_squares, agent_square = initial_map(map_height, map_width, bucket_to_boxes, min_num_buckets, max_num_buckets, unique_buckets)
    box_squares = []
    all_moves = []

    for box, bucket_square in zip(boxes, bucket_squares):
        barriers = box_squares + bucket_squares
        num_pulls = np.random.randint(min_num_pulls, max_num_pulls + 1)
        pulls = list(try_pull_box(map_height, map_width, barriers, agent_square, bucket_square, num_pulls))
        if len(pulls) == 0:
            return 0, None, None, False
        agent_square, box_square, pull_moves = pulls[np.random.randint(len(pulls))]
        box_squares.append(box_square)
        all_moves.extend(pull_moves)

    agent_x, agent_y = agent_square
    random_steps = np.random.randint(min_random_steps, max_random_steps + 1)
    barriers = box_squares + bucket_squares
    for i in range(random_steps):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        moves = [(move_x, move_y) for (move_x, move_y) in moves if 0 <= agent_x + move_x < map_height and 0 <= agent_y + move_y < map_width and
                 (agent_x + move_x, agent_y + move_y) not in barriers]
        if len(moves) == 0:
            break
        move_x, move_y = moves[np.random.randint(len(moves))]
        agent_x, agent_y = agent_x + move_x, agent_y + move_y
        barriers.append((agent_x, agent_y))
        all_moves.append((move_x, move_y))
    agent_square = (agent_x, agent_y)

    map_chars = [[' ' for _ in range(map_width)] for _ in range(map_height)]
    map_chars[agent_square[0]][agent_square[1]] = agent_char

    for box_square, box in zip(box_squares, boxes):
        map_chars[box_square[0]][box_square[1]] = box
    for bucket_square, bucket in zip(bucket_squares, buckets):
        map_chars[bucket_square[0]][bucket_square[1]] = bucket

    new_all_moves = list(reversed(all_moves))
    move_converter = {
        (1, 0): 0,
        (-1, 0): 1,
        (0, 1): 2,
        (0, -1): 3
    }
    new_all_moves = [move_converter[move] for move in new_all_moves]

    return num_buckets, map_chars, new_all_moves, True


def make_map_inner(map_height, map_width, bucket_to_boxes, min_num_buckets, max_num_buckets, unique_buckets, min_num_pulls, max_num_pulls, min_random_steps, max_random_steps, agent_char, border_char):
    num_buckets, map_chars, map_moves, map_success = make_map_inner_no_border(map_height - 2, map_width - 2, bucket_to_boxes, min_num_buckets, max_num_buckets, unique_buckets, min_num_pulls, max_num_pulls, min_random_steps, max_random_steps, agent_char)

    if not map_success:
        return num_buckets, None, None, False

    map_chars = [[border_char] + line + [border_char] for line in map_chars]
    map_chars = [[border_char for _ in range(map_width)]] + map_chars + [[border_char for _ in range(map_width)]]

    return num_buckets, map_chars, map_moves, True


def make_maps(num_maps, prev_mazes=None, **kwargs):
    all_num_buckets = []
    map_strings = []
    all_map_moves = []

    for _ in range(num_maps):
        map_string = None
        map_success = False
        while not map_success:
            num_buckets, map_chars, map_moves, map_success = make_map_inner(**kwargs)
            if map_success:
                map_chars = [''.join(line) for line in map_chars]
                map_string = '\n'.join(map_chars).strip()
                if map_string in map_strings:
                    map_success = False
                elif prev_mazes is not None and map_string in prev_mazes:
                    map_success = False
                else:
                    all_num_buckets.append(num_buckets)
                    map_strings.append(map_string)
                    all_map_moves.append(map_moves)

    return all_num_buckets, map_strings, all_map_moves


def sample_make_maps():
    cases = [
        {
            'num_maps': 8,
            'map_height': 10,
            'map_width': 10,
            'bucket_to_boxes': {'B': ['b']},
            'min_num_buckets': 1,
            'max_num_buckets': 1,
            'unique_buckets': True,
            'min_num_pulls': 4,
            'max_num_pulls': 7,
            'min_random_steps': 0,
            'max_random_steps': 12,
            'agent_char': 'A',
            'border_char': '+'
        },
        {
            'num_maps': 8,
            'map_height': 10,
            'map_width': 10,
            'bucket_to_boxes': {'B': ['b'], 'C': ['c']},
            'min_num_buckets': 1,
            'max_num_buckets': 2,
            'unique_buckets': True,
            'min_num_pulls': 4,
            'max_num_pulls': 7,
            'min_random_steps': 0,
            'max_random_steps': 12,
            'agent_char': 'A',
            'border_char': '+'
        }
    ]

    for i, case in enumerate(cases):
        print('Test case', i + 1)
        maps = make_maps(**case)

        for j, map_string in enumerate(maps):
            print('Map', j + 1)
            print(map_string)


if __name__ == '__main__':
    seed = int(sys.argv[1])
    np.random.seed(seed)
    # sample_reach_squares()
    # sample_pull_box()
    # sample_initial_map()
    sample_make_maps()
