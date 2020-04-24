#!/usr/bin/env python

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

# coding=utf-8
from __future__ import division, print_function, unicode_literals
import json
import os
import os.path
import tempfile

from shutil import copyfile

from sacred.commandline_options import CommandLineOption
from sacred.dependencies import get_digest
from sacred.observers.base import RunObserver
from sacred.utils import FileNotFoundError
from sacred import optional as opt
from sacred.serializer import flatten


DEFAULT_FILE_STORAGE_PRIORITY = 20


class CustomFileStorageObserver(RunObserver):
    VERSION = 'CustomFileStorageObserver-0.7.0'

    @classmethod
    def create(cls, basedir, resource_dir=None, source_dir=None,
               template=None, priority=DEFAULT_FILE_STORAGE_PRIORITY):
        resource_dir = resource_dir or os.path.join(basedir, '_resources')
        source_dir = source_dir or os.path.join(basedir, '_sources')
        if template is not None:
            if not os.path.exists(template):
                raise FileNotFoundError("Couldn't find template file '{}'"
                                        .format(template))
        else:
            template = os.path.join(basedir, 'template.html')
            if not os.path.exists(template):
                template = None
        return cls(basedir, resource_dir, source_dir, template, priority)

    def __init__(self, basedir, resource_dir, source_dir, template,
                 priority=DEFAULT_FILE_STORAGE_PRIORITY):
        self.basedir = basedir
        self.resource_dir = resource_dir
        self.source_dir = source_dir
        self.template = template
        self.priority = priority
        self.dir = None
        self.run_entry = None
        self.config = None
        self.info = None
        self.cout = ""
        self.cout_write_cursor = 0

    def queued_event(self, ex_info, command, host_info, queue_time, config,
                     meta_info, _id):
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)
        if _id is None:
            self.dir = tempfile.mkdtemp(prefix='run_', dir=self.basedir)
        else:
            self.dir = os.path.join(self.basedir, str(_id))
            os.mkdir(self.dir)

        self.run_entry = {
            'experiment': dict(ex_info),
            'command': command,
            'host': dict(host_info),
            'meta': meta_info,
            'status': 'QUEUED',
        }
        self.config = config
        self.info = {}

        self.save_json(self.run_entry, 'run.json')
        self.save_json(self.config, 'config.json')

        for s, m in ex_info['sources']:
            self.save_file(s)

        return os.path.relpath(self.dir, self.basedir) if _id is None else _id

    def save_sources(self, ex_info):
        base_dir = ex_info['base_dir']
        source_info = []
        for s, m in ex_info['sources']:
            abspath = os.path.join(base_dir, s)
            store_path, md5sum = self.find_or_save(abspath, self.source_dir)
            # assert m == md5sum
            source_info.append([s, os.path.relpath(store_path, self.basedir)])
        return source_info

    def started_event(self, ex_info, command, host_info, start_time, config,
                      meta_info, _id):
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)
        if _id is None:
            for i in range(200):
                dir_nrs = [int(d) for d in os.listdir(self.basedir)
                           if os.path.isdir(os.path.join(self.basedir, d)) and
                           d.isdigit()]
                _id = max(dir_nrs + [0]) + 1
                self.dir = os.path.join(self.basedir, str(_id))
                try:
                    os.mkdir(self.dir)
                    break
                except Exception:  # Catch race conditions
                    if i > 100:
                        # After some tries,
                        # expect that something other went wrong
                        raise
        else:
            self.dir = os.path.join(self.basedir, str(_id))
            os.mkdir(self.dir)

        ex_info['sources'] = self.save_sources(ex_info)

        self.run_entry = {
            'experiment': dict(ex_info),
            'command': command,
            'host': dict(host_info),
            'start_time': start_time.isoformat(),
            'meta': meta_info,
            'status': 'RUNNING',
            'resources': [],
            'artifacts': [],
            'heartbeat': None
        }
        self.config = config
        self.info = {}
        self.cout = ""
        self.cout_write_cursor = 0

        self.save_json(self.run_entry, 'run.json')
        self.save_json(self.config, 'config.json')
        self.save_cout()

        return os.path.relpath(self.dir, self.basedir) if _id is None else _id

    def find_or_save(self, filename, store_dir):
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
        source_name, ext = os.path.splitext(os.path.basename(filename))
        md5sum = get_digest(filename)
        store_name = source_name + '_' + md5sum + ext
        store_path = os.path.join(store_dir, store_name)
        if not os.path.exists(store_path):
            copyfile(filename, store_path)
        return store_path, md5sum

    def save_json(self, obj, filename):
        with open(os.path.join(self.dir, filename), 'w') as f:
            json.dump(flatten(obj), f, sort_keys=True, indent=2)

    def save_file(self, filename, target_name=None):
        target_name = target_name or os.path.basename(filename)
        target_folder = os.path.join(self.dir, os.path.join(*os.path.split(target_name)[:-1]))
        os.makedirs(target_folder, exist_ok=True)
        copyfile(filename, os.path.join(self.dir, target_name))

    def save_cout(self):
        with open(os.path.join(self.dir, 'cout.txt'), 'ab') as f:
            f.write(self.cout[self.cout_write_cursor:].encode("utf-8"))
            self.cout_write_cursor = len(self.cout)

    def render_template(self):
        if opt.has_mako and self.template:
            from mako.template import Template
            template = Template(filename=self.template)
            report = template.render(run=self.run_entry,
                                     config=self.config,
                                     info=self.info,
                                     cout=self.cout,
                                     savedir=self.dir)
            _, ext = os.path.splitext(self.template)
            with open(os.path.join(self.dir, 'report' + ext), 'w') as f:
                f.write(report)

    def heartbeat_event(self, info, captured_out, beat_time, result):
        self.info = info
        self.run_entry['heartbeat'] = beat_time.isoformat()
        self.run_entry['result'] = result
        self.cout = captured_out
        self.save_cout()
        self.save_json(self.run_entry, 'run.json')
        if self.info:
            self.save_json(self.info, 'info.json')

    def completed_event(self, stop_time, result):
        self.run_entry['stop_time'] = stop_time.isoformat()
        self.run_entry['result'] = result
        self.run_entry['status'] = 'COMPLETED'

        self.save_json(self.run_entry, 'run.json')
        self.render_template()

    def interrupted_event(self, interrupt_time, status):
        self.run_entry['stop_time'] = interrupt_time.isoformat()
        self.run_entry['status'] = status
        self.save_json(self.run_entry, 'run.json')
        self.render_template()

    def failed_event(self, fail_time, fail_trace):
        self.run_entry['stop_time'] = fail_time.isoformat()
        self.run_entry['status'] = 'FAILED'
        self.run_entry['fail_trace'] = fail_trace
        self.save_json(self.run_entry, 'run.json')
        self.render_template()

    def resource_event(self, filename):
        store_path, md5sum = self.find_or_save(filename, self.resource_dir)
        self.run_entry['resources'].append([filename, store_path])
        self.save_json(self.run_entry, 'run.json')

    def artifact_event(self, name, filename, metadata=None, content_type=None):
        self.save_file(filename, name)
        self.run_entry['artifacts'].append(name)
        self.save_json(self.run_entry, 'run.json')

    def log_metrics(self, metrics_by_name, info):
        """Store new measurements into metrics.json.
        """
        metrics_dir = os.path.join(self.dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)

        for metric_name, metric_ptr in metrics_by_name.items():
            safe_metric_name = metric_name.replace('.', '-')
            metric_file = os.path.join(metrics_dir, '{}.txt'.format(safe_metric_name))

            with open(metric_file, 'a') as f:
                for value, step, timestamp in zip(metric_ptr["values"], metric_ptr["steps"], metric_ptr["timestamps"]):
                    timestamp = str(timestamp.isoformat())
                    result = '{},{},{}'.format(step, value, timestamp)
                    print(result, file=f)

    def __eq__(self, other):
        if isinstance(other, CustomFileStorageObserver):
            return self.basedir == other.basedir
        return False

    def __ne__(self, other):
        return not self.__eq__(other)


class CustomFileStorageOption(CommandLineOption):
    """Add a file-storage observer to the experiment."""

    short_flag = 'K'
    arg = 'BASEDIR'
    arg_description = "Base-directory to write the runs to"

    @classmethod
    def apply(cls, args, run):
        run.observers.append(CustomFileStorageObserver.create(args))
