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

import os
import subprocess
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

if not os.path.exists('environments'):
    os.mkdir('environments')

dir_path = os.path.dirname(os.path.realpath(__file__))
scripts = list(Path(dir_path).glob(os.path.join('**', '*.sh')))
processes = []

print('Generating files with seed {}'.format(args.seed))

with open(os.devnull, 'w') as FNULL:
    for s in scripts:
        print('Starting {}'.format(s))
        p = subprocess.Popen(['sh', str(s), str(args.seed)])
        processes.append(p)

    status = [False] * len(scripts)
    completed = 0

    while completed < len(scripts):
        for i in range(len(scripts)):
            retcode = processes[i].poll()
            if status[i] or retcode is None:
                continue
            if retcode == 0:
                print('SUCCESS {} finished'.format(scripts[i]))
            else:
                print(retcode)
                print('{} FAILED'.format(scripts[i]))
            status[i] = True
            completed += 1
