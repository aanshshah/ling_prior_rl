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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SequentialSampler

from graphrl.agents.agent import Agent, AgentParams
from graphrl.agents.optimizer import OptimizerParams
from graphrl.agents.utils import move_to_device


ACTOR_CRITIC_AGENT_DEFAULT_PARAMS = {
    'batch_size': 32,
    'gamma': 0.99,
    'gae_lambda': 0.96,
    'policy_loss_weight': 1.0,
    'value_loss_weight': 0.5,
    'policy_entropy_weight': 0.0,
    'value_loss_type': 'mse',
    'clip_grads_by_norm': False,
    'max_grad_norm': 0.5,
    'loss_clip_param': 0.0,
    'update_freq_steps_per_env': 5,
    'stochastic_test_policy': False,
    'normalize_advantages': False,
    'use_gae': False
}


class ActorCriticAgentParams(AgentParams):
    def __init__(self, *args):
        super(ActorCriticAgentParams, self).__init__(*args)
        self.optimizer_params = OptimizerParams()
        self.policy_value_net = None
        self.agent_class = ActorCriticAgent

    def make_default_config(self):
        config = super(ActorCriticAgentParams, self).make_default_config()
        config.update(ACTOR_CRITIC_AGENT_DEFAULT_PARAMS)
        config['opt'] = self.optimizer_params.make_default_config()
        return config

    def load_config(self, config):
        super(ActorCriticAgentParams, self).load_config(config)

        for k in ACTOR_CRITIC_AGENT_DEFAULT_PARAMS:
            if k in config:
                setattr(self, k, config[k])
        self.optimizer_params.load_config(config['opt'])


class ActorCriticAgent(Agent):
    def __init__(self, params):

        policy_value_net = params.policy_value_net.to(params.device)

        nets = {
            'policy_value_net': policy_value_net
        }

        super(ActorCriticAgent, self).__init__(params=params, nets=nets)
        '''
        policy_value_net.forward shoulf return a pair (policy, value)
        '''

        self.policy_value_net = policy_value_net
        self.optimizer = self.params.optimizer_params.make_optimizer(self.policy_value_net)

        self.observations = [[] for _ in range(self.params.num_train_envs)]
        self.rewards = torch.zeros(self.params.update_freq_steps_per_env, self.params.num_train_envs).to(self.device)
        self.actions = torch.zeros(self.params.update_freq_steps_per_env, self.params.num_train_envs).to(self.device)
        self.values = torch.zeros(self.params.update_freq_steps_per_env + 1, self.params.num_train_envs).to(self.device)
        self.action_log_probs = torch.zeros(self.params.update_freq_steps_per_env, self.params.num_train_envs).to(self.device)
        self.returns = torch.zeros(self.params.update_freq_steps_per_env + 1, self.params.num_train_envs).to(self.device)
        self.advantages = torch.zeros(self.params.update_freq_steps_per_env + 1, self.params.num_train_envs).to(self.device)
        self.masks = torch.ones(self.params.update_freq_steps_per_env + 1, self.params.num_train_envs).to(self.device)
        self.bad_masks = torch.ones(self.params.update_freq_steps_per_env + 1, self.params.num_train_envs).to(self.device)
        self.valid_indices = []
        self.steps_collected = 0

    def add_obs_to_buffer(self, obs):
        for i in range(self.params.num_train_envs):
            self.observations[i].append(obs[i])

    def act(self, obs, training):
        self.policy_value_net.eval()

        obs = self.filter_obs(obs)
        obs = move_to_device(obs, self.params.device)

        with torch.no_grad():
            action_dist, values = self.policy_value_net(obs)

        if training or self.params.stochastic_test_policy:
            action = action_dist.sample()
        else:
            action = action_dist.logits.argmax(1)

        if training:
            self.values[self.steps_collected] = values.detach()
            self.action_log_probs[self.steps_collected] = action_dist.log_prob(action).detach()

        return action.cpu().numpy()

    def train_on_env_reset(self, obs):
        self.add_obs_to_buffer(obs)

    def add_final_values(self):
        self.policy_value_net.eval()

        obs = [self.observations[i][-1] for i in range(self.params.num_train_envs)]

        obs = self.filter_obs(obs)
        obs = move_to_device(obs, self.params.device)

        with torch.no_grad():
            _, values = self.policy_value_net(obs)
            self.values[-1] = values.detach()

    def add_returns(self):
        if self.params.use_gae:
            gae = 0
            for step in reversed(range(self.params.update_freq_steps_per_env)):
                td_residual = (self.rewards[step] + self.masks[step + 1] * self.params.gamma * self.values[step + 1]) - self.values[step]
                gae = td_residual + self.params.gamma * self.params.gae_lambda * self.masks[step + 1] * gae
                gae = gae * self.bad_masks[step + 1]
                self.advantages[step] = gae
            self.returns = self.advantages + self.values
        else:
            self.returns[-1] = self.values[-1]
            for step in reversed(range(self.params.update_freq_steps_per_env)):
                self.returns[step] = (self.returns[step + 1] * self.params.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                    + self.values[step] * (1 - self.bad_masks[step + 1])
            self.advantages = self.returns - self.values

        if self.params.normalize_advantages:
            time_idxs, env_idxs = zip(*self.valid_indices)
            advantages = self.advantages[time_idxs, env_idxs]
            advantage_mean, advantage_std = torch.mean(advantages), torch.std(advantages)
            self.advantages = (self.advantages - advantage_mean) / (advantage_std + 1e-5)

    def update_model(self):
        self.policy_value_net.train()
        sampler = BatchSampler(SequentialSampler(range(len(self.valid_indices))), self.params.batch_size, drop_last=False)

        self.optimizer.zero_grad()

        LOSS_NAMES = ['loss', 'policy_loss', 'policy_entropy', 'value_loss', 'explained_variance']

        loss_sums = {k: torch.tensor(0, device=self.device, dtype=torch.float32)
                     for k in LOSS_NAMES}

        for sample_idxs_batch in sampler:
            idxs = [self.valid_indices[idx] for idx in sample_idxs_batch]
            time_idxs, env_idxs = zip(*idxs)

            obs = [self.observations[env_idx][time_idx] for time_idx, env_idx in zip(time_idxs, env_idxs)]
            obs = self.filter_obs(obs)
            obs = move_to_device(obs, self.params.device)

            returns = self.returns[time_idxs, env_idxs]
            advantages = self.advantages[time_idxs, env_idxs]
            actions = self.actions[time_idxs, env_idxs]
            action_dist, value_preds = self.policy_value_net(obs)

            policy_loss = -torch.mean(action_dist.log_prob(actions) * advantages.float())
            policy_entropy = torch.mean(action_dist.entropy())

            if self.params.value_loss_type == 'huber':
                value_loss = F.smooth_l1_loss(value_preds, returns)
            elif self.params.value_loss_type == 'mse':
                value_loss = F.mse_loss(value_preds, returns)
            else:
                raise ValueError('Unknown value loss type')

            explained_variance = 1 - torch.var(returns - value_preds) / torch.var(returns)

            loss = self.params.policy_loss_weight * policy_loss
            loss = loss - self.params.policy_entropy_weight * policy_entropy
            loss = loss + self.params.value_loss_weight * value_loss

            loss_scale = float(len(sample_idxs_batch)) / len(self.valid_indices)
            (loss * loss_scale).backward()

            for k, kloss in zip(LOSS_NAMES,
                                [loss, policy_loss, policy_entropy, value_loss, explained_variance]):
                loss_sums[k] += kloss.detach() * len(sample_idxs_batch)

        if self.params.clip_grads_by_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.policy_value_net.parameters(), self.params.max_grad_norm)

        self.optimizer.step()

        loss_avgs = {k: float(v / len(self.valid_indices)) for k, v in loss_sums.items()}

        for k, v in loss_avgs.items():
            self.log_train_to_sacred_debug(k, v)

        if self.params.clip_grads_by_norm:
            self.log_train_to_sacred_debug('grad_norm', float(grad_norm))

    def reset_buffers(self):
        self.observations = [[self.observations[i][-1]] for i in range(self.params.num_train_envs)]
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.steps_collected = 0
        self.valid_indices = []

    def train_on_env_step(self, obs, action, reward, done, aborted, info):
        self.add_obs_to_buffer(obs)

        self.rewards[self.steps_collected] = torch.from_numpy(reward).to(self.params.device)
        self.actions[self.steps_collected] = torch.from_numpy(action).to(self.params.device)
        self.masks[self.steps_collected + 1] = 1 - torch.from_numpy(done.astype(np.float32)).to(self.params.device)
        self.bad_masks[self.steps_collected + 1] = 1 - torch.from_numpy(aborted.astype(np.float32)).to(self.params.device)

        for i in range(self.params.num_train_envs):
            if not aborted[i]:
                self.valid_indices.append((self.steps_collected, i))

        self.steps_collected += 1

        if self.steps_collected == self.params.update_freq_steps_per_env:
            self.add_final_values()
            self.add_returns()
            self.update_model()
            self.reset_buffers()
