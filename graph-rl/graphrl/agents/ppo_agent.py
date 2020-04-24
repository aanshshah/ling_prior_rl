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
import torch.nn.functional as F
import torch.nn as nn

from graphrl.agents.actor_critic_agent import ActorCriticAgent, ActorCriticAgentParams
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from graphrl.agents.utils import move_to_device


PPO_AGENT_DEFAULT_PARAMS = {
    'num_ppo_epochs': 4,
    'loss_clip_param': 0.0
}


class PPOAgentParams(ActorCriticAgentParams):
    def __init__(self, *args):
        super(PPOAgentParams, self).__init__(*args)
        self.policy_value_net = None
        self.agent_class = PPOAgent

    def make_default_config(self):
        config = super(PPOAgentParams, self).make_default_config()
        config.update(PPO_AGENT_DEFAULT_PARAMS)
        return config

    def load_config(self, config):
        super(PPOAgentParams, self).load_config(config)

        for k in PPO_AGENT_DEFAULT_PARAMS:
            if k in config:
                setattr(self, k, config[k])


class PPOAgent(ActorCriticAgent):
    def __init__(self, params):
        super(PPOAgent, self).__init__(params=params)
        '''
        policy_value_net.forward should return a pair (policy, value)
        '''
        self.policy_value_net = self.params.policy_value_net.to(self.params.device)

    def update_model(self):
        self.policy_value_net.train()
        sampler = BatchSampler(SequentialSampler(range(len(self.valid_indices))), self.params.batch_size, drop_last=False)

        loss_sums = {k: torch.tensor(0, device=self.device, dtype=torch.float32) for k in ['loss', 'policy_loss', 'policy_entropy', 'value_loss']}

        for epoch in range(self.params.num_ppo_epochs):
            for sample_idxs_batch in sampler:
                self.optimizer.zero_grad()

                idxs = [self.valid_indices[idx] for idx in sample_idxs_batch]
                time_idxs, env_idxs = zip(*idxs)

                obs = [self.observations[env_idx][time_idx] for time_idx, env_idx in zip(time_idxs, env_idxs)]
                obs = self.filter_obs(obs)
                obs = move_to_device(obs, self.params.device)

                returns = self.returns[time_idxs, env_idxs]
                advantages = self.advantages[time_idxs, env_idxs]
                actions = self.actions[time_idxs, env_idxs]
                old_log_probs = self.action_log_probs[time_idxs, env_idxs]
                old_values = self.values[time_idxs, env_idxs]

                action_dist, value_preds = self.policy_value_net(obs)
                new_log_probs = action_dist.log_prob(actions)
                log_prob_ratio = torch.exp(new_log_probs - old_log_probs)
                clipped_ratio = torch.clamp(log_prob_ratio, 1.0 - self.params.loss_clip_param, 1.0 + self.params.loss_clip_param)
                policy_loss = -torch.mean(torch.min(log_prob_ratio * advantages, clipped_ratio * advantages))
                policy_entropy = torch.mean(action_dist.entropy())

                if self.params.value_loss_type == 'huber':
                    value_loss = F.smooth_l1_loss(value_preds, returns)
                elif self.params.value_loss_type == 'mse':
                    value_loss = F.mse_loss(value_preds, returns)
                elif self.params.value_loss_type == 'clipped_value':
                    clipped_values = old_values + (value_preds - old_values).clamp(-self.params.loss_clip_param, self.params.loss_clip_param)
                    value_loss = torch.max((value_preds - returns).pow(2), (clipped_values - returns).pow(2)).mean()
                else:
                    raise ValueError('Unknown value loss type')

                loss = self.params.policy_loss_weight * policy_loss
                loss = loss - self.params.policy_entropy_weight * policy_entropy
                loss = loss + self.params.value_loss_weight * value_loss

                loss_scale = float(len(sample_idxs_batch)) / self.params.batch_size
                if self.params.clip_grads_by_norm:
                    nn.utils.clip_grad_norm_(self.policy_value_net.parameters(), self.params.max_grad_norm)
                (loss * loss_scale).backward()
                self.optimizer.step()

                for k, kloss in zip(['loss', 'policy_loss', 'policy_entropy', 'value_loss'], [loss, policy_loss, policy_entropy, value_loss]):
                    loss_sums[k] += kloss.detach() * len(sample_idxs_batch)

        loss_avgs = {k: float(v / (len(self.valid_indices) * self.params.num_ppo_epochs)) for k, v in loss_sums.items()}

        for k, v in loss_avgs.items():
            self.log_train_to_sacred(k, v)
