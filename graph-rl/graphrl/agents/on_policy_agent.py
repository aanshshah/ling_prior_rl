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

from graphrl.agents.agent import Agent, AgentParams


ON_POLICY_AGENT_DEFAULT_PARAMS = {
    'truncate_path_steps': 0,
    'update_freq_steps': 1000
}


class OnPolicyAgentParams(AgentParams):
    def __init__(self, *args):
        super(OnPolicyAgentParams, self).__init__(*args)
        self.agent_class = OnPolicyAgent

        for k, v in ON_POLICY_AGENT_DEFAULT_PARAMS.items():
            setattr(self, k, v)

    def make_default_config(self):
        config = super(OnPolicyAgentParams, self).make_default_config()
        config.update(ON_POLICY_AGENT_DEFAULT_PARAMS)
        return config

    def load_config(self, config):
        super(OnPolicyAgentParams, self).load_config(config)

        for k in ON_POLICY_AGENT_DEFAULT_PARAMS:
            if k in config:
                setattr(self, k, config[k])


class OnPolicyAgent(Agent):
    def __init__(self, params):
        super(OnPolicyAgent, self).__init__(params=params)

        # Collect paths here as training progresses
        self.paths = []
        self.observations = []
        self.actions = []
        self.rewards = []
        self.steps_collected = 0

        self.last_update_step = 0

    def train_on_env_reset(self, obs):
        self.observations = [[obs[i]] for i in range(len(obs))]
        self.actions = [[] for _ in range(len(obs))]
        self.rewards = [[] for _ in range(len(obs))]

    def update_model(self):
        self.last_update_step = self.CTR_TRAIN_STEPS

        paths = self.paths
        self.paths = []
        self.steps_collected = 0

        self.train_on_paths(paths)

    def train_on_env_step(self, obs, action, reward, done, aborted, info):
        for i in range(len(obs)):
            self.observations[i].append(obs[i])
            self.actions[i].append(action[i])
            self.rewards[i].append(reward[i])

        for i in range(len(obs)):
            if done[i]:
                path = {
                    'observation': self.observations[i][:-1],
                    'next_observation': self.observations[i][1:],
                    'action': self.actions[i],
                    'reward': self.rewards[i],
                    'done': [False for _ in range(len(self.actions[i]) - 1)] + [True]
                }
                self.paths.append(path)
                self.steps_collected += len(self.actions[i])

                self.observations[i] = [self.observations[i][-1]]
                self.actions[i] = []
                self.rewards[i] = []

            elif aborted[i]:
                if len(self.observations[i]) > 2:
                    path = {
                        'observation': self.observations[i][:-2],
                        'next_observation': self.observations[i][1:-1],
                        'action': self.actions[i][:-1],
                        'reward': self.rewards[i][:-1],
                        'done': [False for _ in range(len(self.actions[i]) - 1)]
                    }
                    self.paths.append(path)
                    self.steps_collected += len(self.actions[i]) - 1

                self.observations[i] = [self.observations[i][-1]]
                self.actions[i] = []
                self.rewards[i] = []

            elif len(self.actions[i]) >= self.params.truncate_path_steps:
                path = {
                    'observation': self.observations[i][:-1],
                    'next_observation': self.observations[i][1:],
                    'action': self.actions[i],
                    'reward': self.rewards[i],
                    'done': [False for _ in range(len(self.actions[i]))]
                }
                self.paths.append(path)
                self.steps_collected += len(self.actions[i])

                self.observations[i] = [self.observations[i][-1]]
                self.actions[i] = []
                self.rewards[i] = []

        if self.steps_collected >= self.params.update_freq_steps:
            self.update_model()
