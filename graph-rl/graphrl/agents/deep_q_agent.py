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
import torch
import torch.nn.functional as F
from graphrl.agents.replay_buffer import PrioritizedReplayBuffer

from graphrl.agents.agent import Agent, AgentParams
from graphrl.agents.optimizer import OptimizerParams
from graphrl.agents.schedule import LinearSchedule, ConstantSchedule
from graphrl.agents.utils import move_to_device


DEEPQ_AGENT_DEFAULT_PARAMS = {
    'batch_size': 32,
    'gamma': 0.99,
    'double_dqn': False,
    'use_huber_loss': False,
    'heatup_steps': 0,
    'update_freq_steps': 1000,
    'update_target_weights_freq_steps': 10000,
    'train_epsilon': {
        'type': 'linear',
        'constant_value': 0.1,
        'initial_value': 1,
        'final_value': 0.1,
        'decay_steps': 1000000
    },
    'test_epsilon': 0.05,

    'replay_buffer': {
        'size': 1000000,
        'per_alpha': 0,
        'per_beta': {
            'type': 'constant',
            'constant_value': 0.4,
            'initial_value': 1,
            'final_value': 0.1,
            'decay_steps': 1000000
        }
    }
}


class DeepQAgentParams(AgentParams):
    def __init__(self, *args):
        super(DeepQAgentParams, self).__init__(*args)
        self.agent_class = DeepQNetwork

        for k, v in DEEPQ_AGENT_DEFAULT_PARAMS.items():
            setattr(self, k, v)

        self.train_epsilon_schedule = None
        self.optimizer_params = OptimizerParams()
        self.online_q_net = None
        self.target_q_net = None

    def make_default_config(self):
        config = super(DeepQAgentParams, self).make_default_config()
        config.update(DEEPQ_AGENT_DEFAULT_PARAMS)
        config['opt'] = self.optimizer_params.make_default_config()

        return config

    def load_config(self, config):
        super(DeepQAgentParams, self).load_config(config)

        for k in DEEPQ_AGENT_DEFAULT_PARAMS:
            if k in config:
                setattr(self, k, config[k])

        train_epsilon_config = config['train_epsilon']

        if train_epsilon_config['type'] == 'linear':
            self.train_epsilon_schedule = LinearSchedule(train_epsilon_config['initial_value'], train_epsilon_config['final_value'], train_epsilon_config['decay_steps'])
        elif train_epsilon_config['type'] == 'constant':
            self.train_epsilon_schedule = ConstantSchedule(train_epsilon_config['constant_value'])
        else:
            raise ValueError('Unknown schedule: {}.'.format(train_epsilon_config['type']))

        per_beta_config = self.replay_buffer['per_beta']
        if per_beta_config['type'] == 'linear':
            self.per_beta_schedule = LinearSchedule(per_beta_config['initial_value'], per_beta_config['final_value'], per_beta_config['decay_steps'])
        elif per_beta_config['type'] == 'constant':
            self.per_beta_schedule = ConstantSchedule(per_beta_config['constant_value'])
        else:
            raise ValueError('Unknown schedule: {}'.format(per_beta_config['type']))

        self.optimizer_params.load_config(config['opt'])


class DeepQNetwork(Agent):
    def __init__(self, params):

        online_q_net = params.online_q_net.to(params.device)
        target_q_net = params.target_q_net.to(params.device)

        nets = {
            'online_q_net': online_q_net,
            'target_q_net': target_q_net
        }

        super(DeepQNetwork, self).__init__(params=params, nets=nets)

        self.online_q_net = online_q_net
        self.target_q_net = target_q_net
        self.optimizer = self.params.optimizer_params.make_optimizer(self.online_q_net)

        self.replay_buffer = PrioritizedReplayBuffer(self.params.replay_buffer['size'], self.params.replay_buffer['per_alpha'])

        # Store an obs until we see the effect
        self.obs = None

        self.last_update_step = 0
        self.last_update_target_weights_step = 0

    def train_on_env_reset(self, obs):
        self.obs = obs

    def train_on_env_step(self, obs, action, reward, done, aborted, info):

        for i in range(len(self.obs)):
            if not aborted[i]:
                self.replay_buffer.add(self.obs[i], action[i], reward[i], obs[i], done[i])
        self.obs = obs

        self.maybe_update()

    def act(self, obs, training):
        num_envs = len(obs)
        random_actions = [self.action_space.sample() for _ in range(num_envs)]
        if self.mode == 'train' and self.CTR_TRAIN_STEPS < self.params.heatup_steps:
            return random_actions

        self.online_q_net.eval()

        obs = self.filter_obs(obs)
        obs = move_to_device(obs, self.params.device)
        with torch.no_grad():
            qs = self.online_q_net(obs)
            best_actions = qs.argmax(1).cpu().numpy()

        if training:
            epsilon = self.params.train_epsilon_schedule.value(self.CTR_TRAIN_STEPS - self.params.heatup_steps)
            self.log_scalar_debug('train.epsilon.bystep', epsilon, self.CTR_TRAIN_STEPS)
        else:
            epsilon = self.params.test_epsilon

        use_optimal = np.random.rand(num_envs) > epsilon
        actions = np.where(use_optimal, best_actions, random_actions)

        return actions

    def maybe_update(self):
        if self.CTR_TRAIN_STEPS < self.params.heatup_steps:
            return

        if self.CTR_TRAIN_STEPS - self.last_update_target_weights_step >= self.params.update_target_weights_freq_steps:
            self.target_q_net.load_state_dict(self.online_q_net.state_dict())
            self.last_update_target_weights_step = self.CTR_TRAIN_STEPS

        if self.CTR_TRAIN_STEPS - self.last_update_step >= self.params.update_freq_steps:
            self.update()
            self.last_update_step = self.CTR_TRAIN_STEPS

    def update(self):
        beta = self.params.per_beta_schedule.value(self.CTR_TRAIN_STEPS - self.params.heatup_steps)
        obs, actions, rewards, next_obs, dones, weights, idxs = self.replay_buffer.sample(self.params.batch_size, beta)

        obs = self.filter_obs(obs)
        next_obs = self.filter_obs(next_obs)

        obs = move_to_device(obs, self.params.device)
        next_obs = move_to_device(next_obs, self.params.device)

        rewards, dones, weights = [np.array(x, dtype=np.float32) for x in [rewards, dones, weights]]
        actions = np.array(actions)
        actions, rewards, dones, weights = [torch.from_numpy(x).to(self.params.device) for x in [actions, rewards, dones, weights]]

        self.online_q_net.train()
        self.target_q_net.eval()

        self.optimizer.zero_grad()

        qs_from_online = self.online_q_net(obs)
        value_preds = qs_from_online.gather(1, actions[:, None])[:, 0]

        next_qs_from_target = self.target_q_net(next_obs).detach()

        if self.params.double_dqn:
            next_qs_from_online = self.online_q_net(next_obs).detach()
            next_actions = next_qs_from_online.argmax(1)
        else:
            next_actions = next_qs_from_target.argmax(1)

        next_values = next_qs_from_target.gather(1, next_actions[:, None])[:, 0]
        value_targets = rewards + (1 - dones) * self.params.gamma * next_values

        if self.params.use_huber_loss:
            losses = F.smooth_l1_loss(value_preds, value_targets, reduction='none')
            abs_td_errors = torch.abs(value_preds - value_targets)
        else:
            losses = F.mse_loss(value_preds, value_targets, reduction='none')
            abs_td_errors = torch.abs(value_preds - value_targets)

        loss = torch.mean(losses * weights)

        loss.backward()
        self.optimizer.step()

        loss = float(loss)
        self.log_scalar_debug('train.loss.bystep', loss, self.CTR_TRAIN_STEPS)

        mean_abs_td_error = float(torch.mean(abs_td_errors))
        self.log_scalar_debug('train.abs_td_error.bystep', mean_abs_td_error, self.CTR_TRAIN_STEPS)
        abs_td_errors = abs_td_errors.cpu().detach().numpy()
        self.replay_buffer.update_priorities(idxs, abs_td_errors)
