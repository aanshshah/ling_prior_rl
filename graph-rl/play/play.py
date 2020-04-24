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

from graphrl.environments.warehouse.warehouse_v1 import make_warehouse_env


def main():
    artfile = 'environments/multiple_five_five_100_20/test/test2.txt'
    boxes = ["b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
    buckets = ["B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
    bucket_to_boxes = dict([["B", ["b"]], ["C", ["c"]], ["D", ["d"]], ["E", ["e"]], ["F", ["f"]], ["G", ["g"]], ["H", ["h"]], ["I", ["i"]], ["J", ["j"]], ["K", ["k"]]])
    character_map = {}

    env = make_warehouse_env(artfile, boxes, buckets, bucket_to_boxes, character_map=character_map)
    env.reset()
    env.render()
    done = False

    while not done:
        action = int(input())
        _, _, done, _ = env.step(action)
        env.render()


if __name__ == '__main__':
    main()
