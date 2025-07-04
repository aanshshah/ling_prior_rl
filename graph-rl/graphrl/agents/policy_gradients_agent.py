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

import torch
import torch.utils.data as data
import numpy as np
import collections

from graphrl.agents.on_policy_agent import OnPolicyAgent, OnPolicyAgentParams
from graphrl.agents.optimizer import OptimizerParams
from graphrl.agents.utils import PathsDataset, move_to_device, compute_returns, make_collate_fn


POLICY_GRADIENT_AGENT_DEFAULT_PARAMS = {
    'gamma': 0.99,
    'batch_size': 32,
    'use_future_return': True,
    'returns_normalizer': 'none',
    'policy_loss_weight': 1.,
    'policy_entropy_weight': 0.
}


class PolicyGradientsAgentParams(OnPolicyAgentParams):
    def __init__(self, *args):
        super(PolicyGradientsAgentParams, self).__init__(*args)
        self.optimizer_params = OptimizerParams()
        self.agent_class = PolicyGradientsAgent

    def make_default_config(self):
        config = super(PolicyGradientsAgentParams, self).make_default_config()
        config.update(POLICY_GRADIENT_AGENT_DEFAULT_PARAMS)
        config['opt'] = self.optimizer_params.make_default_config()
        return config

    def load_config(self, config):
        super(PolicyGradientsAgentParams, self).load_config(config)

        for k in POLICY_GRADIENT_AGENT_DEFAULT_PARAMS:
            if k in config:
                setattr(self, k, config[k])
        self.optimizer_params.load_config(config['opt'])


class PolicyGradientsAgent(OnPolicyAgent):
    def __init__(self, params):
        super(PolicyGradientsAgent, self).__init__(params=params)

        self.policy_net = self.params.policy_net.to(self.params.device)
        self.optimizer = self.params.optimizer_params.make_optimizer(self.policy_net)

        self.episodes_with_timestep = np.zeros((50,), dtype=np.int32)
        self.mean_timestep_returns = np.zeros((50,), dtype=np.float32)

    def add_returns_to_paths(self, paths):
        rewards = [path['reward'] for path in paths]
        returns = [compute_returns(rewards=paths[i]['reward'],
                                   gamma=self.params.gamma,
                                   use_future_return=self.params.use_future_return)
                   for i in range(len(rewards))]

        if self.params.returns_normalizer == 'none':
            pass
        elif self.params.returns_normalizer == 'episode':
            for return_ in returns:
                return_mean = np.mean(return_)
                return_std = np.std(return_)

                if return_std == 0:
                    return_.fill(0)
                else:
                    return_ -= return_mean
                    return_ /= return_std
        elif self.params.returns_normalizer == 'timestep':
            for return_ in returns:
                if len(return_) > len(self.episodes_with_timestep):
                    length_to_add = len(return_) - len(self.episodes_with_timestep)
                    self.episodes_with_timestep = np.concatenate([self.episodes_with_timestep, [0] * length_to_add], 0)
                    self.mean_timestep_returns = np.concatenate([self.mean_timestep_returns, [0.] * length_to_add], 0)

                for i in range(len(return_)):
                    self.episodes_with_timestep[i] += 1
                    self.mean_timestep_returns[i] -= self.mean_timestep_returns[i] / self.episodes_with_timestep[i]
                    self.mean_timestep_returns[i] += return_[i] / self.episodes_with_timestep[i]

            for return_ in returns:
                return_ -= self.mean_timestep_returns[:len(return_)]
        else:
            raise ValueError('Unknown returns normalizer: {}'.format(self.returns_normalizer))

        for path, return_ in zip(paths, returns):
            path['return'] = return_

    def act(self, obs, training):
        self.policy_net.eval()

        obs = self.filter_obs(obs)
        obs = move_to_device(obs, self.params.device)
        with torch.no_grad():
            action_dist = self.policy_net(obs)
        if training:
            action = action_dist.sample()
        else:
            action = action_dist.logits.argmax(1)
        return action.cpu().numpy()

    def backward_on_batch(self, batch, loss_scale):
        batch = move_to_device(batch, self.params.device)
        obss, actions, returns = batch['observation'], batch['action'], batch['return']
        action_dist = self.policy_net(obss)
        policy_loss = -torch.mean(action_dist.log_prob(actions) * returns.float())
        policy_entropy = torch.mean(action_dist.entropy())

        loss = self.params.policy_loss_weight * policy_loss
        loss -= self.params.policy_entropy_weight * policy_entropy
        (loss * loss_scale).backward()

        loss, policy_loss, policy_entropy = float(loss), float(policy_loss), float(policy_entropy)
        bz = len(obss)

        losses = {'loss': loss, 'policy_loss': policy_loss, 'policy_entropy': policy_entropy}

        return bz, losses

    def train_on_paths(self, paths):
        self.add_returns_to_paths(paths)

        keys = ['observation', 'action', 'return']
        filters = {'observation': self.filter_obs}
        train_dataset = PathsDataset(paths, keys=keys, filters=filters)
        train_dataloader = data.DataLoader(train_dataset,
                                           collate_fn=make_collate_fn({'observation': self.filter_obs}),
                                           batch_size=self.params.batch_size,
                                           pin_memory=self.params.device.type == 'cuda',
                                           shuffle=True)

        losses_sums = collections.defaultdict(float)
        bz_sum = 0

        self.policy_net.train()
        self.optimizer.zero_grad()
        for i, batch in enumerate(train_dataloader):
            bz, losses = self.backward_on_batch(batch, float(len(batch)) / len(train_dataset))

            bz_sum += bz
            for k, v in losses.items():
                losses_sums[k] += v * bz

        self.optimizer.step()

        for k, v in losses_sums.items():
            v = v / bz_sum
            self.log_train_to_sacred(k, v / bz_sum)
