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


class StoppingCondition(object):
    def report_train_episode(self, episode_reward):
        pass

    def report_eval_episode(self, env_idx, episode_reward):
        pass

    def should_stop(self, agent):
        return False


class DoesntTrainStoppingCondition(StoppingCondition):
    def __init__(self, min_episodes, doesnt_train_episodes, bad_reward):
        super(DoesntTrainStoppingCondition, self).__init__()
        self.min_episodes = min_episodes
        self.doesnt_train_episodes = doesnt_train_episodes
        self.bad_reward = bad_reward

        self.train_rewards = []

    def report_train_episode(self, episode_reward):
        self.train_rewards.append(episode_reward)

    def should_stop(self, agent):
        if agent.CTR_TRAIN_EPISODES < self.min_episodes:
            return False
        recent_rewards = self.train_rewards[-self.doesnt_train_episodes:]
        if len(recent_rewards) < self.doesnt_train_episodes:
            return False
        should_stop = True
        for reward in recent_rewards:
            if reward > self.bad_reward:
                should_stop = False
        return should_stop
