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
import argparse
import glob
import collections
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    args = parser.parse_args()

    indir = os.path.expanduser(args.indir)
    outdir = os.path.expanduser(args.outdir)
    mode = args.mode

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    def split_name_into_name_and_time(name):
        new_name = name.split('_')[:-1]
        time = name.split('_')[-1].split('.pth')[0]
        new_name = '_'.join(new_name)
        new_name = os.path.split(new_name)[-1]
        return new_name, time

    def sort_time_list(time_list):
        time_list = list(time_list)
        if 'best' in time_list:
            has_best = True
            time_list.remove('best')
        else:
            has_best = False
        time_list = [int(x) for x in time_list]
        time_list = list(sorted(time_list))
        time_list = [str(x) for x in time_list]
        if has_best:
            time_list.append('best')
        return time_list

    def get_nets_map():
        nets_map = collections.defaultdict(list)
        net_names = glob.glob(os.path.join(indir, '*.pth'))
        for name in net_names:
            name, time = split_name_into_name_and_time(name)
            nets_map[name].append(time)
        return {k: sort_time_list(v) for k, v in nets_map.items()}

    if mode == 'list':
        nets_map = get_nets_map()
        for name, time_list in nets_map.items():
            print('Name: {}, times: {}'.format(name, time_list))
    else:
        nets_map = get_nets_map()
        for name in nets_map:
            best_file = os.path.join(indir, '{}_{}.pth'.format(name, mode))
            out_file = os.path.join(outdir, '{}.pth'.format(name))
            shutil.copyfile(best_file, out_file)


if __name__ == '__main__':
    main()
