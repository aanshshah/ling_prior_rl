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
import os
import time
import numpy as np
import tempfile
import collections
from graphrl.environments.continuous_env import ContinuousEnv
from graphrl.environments.vec_env.dummy_vec_env import DummyVecEnv
from graphrl.environments.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder

from graphrl.agents.filters import RewardFilter, ObservationFilter
from graphrl.agents.stopping import StoppingCondition


AGENT_DEFAULT_PARAMS = {
    'mode': 'train',
    'min_train_steps': 10000000000,
    'use_min_train_steps': False,
    'min_train_episodes': 100000,
    'use_min_train_episodes': True,
    'no_progress_steps': 50000,
    'use_no_progress_steps': False,
    'max_steps_per_train_episode': None,
    'max_steps_per_test_episode': None,
    'use_eval_freq_steps': False,
    'eval_freq_steps': 5000,
    'use_eval_freq_episodes': True,
    'eval_freq_episodes': 50,
    'test_episodes': 10,
    'device_name': 'cuda',
    'print_actions': False,
    'num_train_envs': 1,
    'should_load_nets': False,
    'load_nets_folder': './model',
    'save_freq_steps': 500000,
    'save_dir': './saved_models',
    'experiment_name': 'pickaxe_stone',
    'tensorboard_logdir': './log',
    'use_tensorboard': False,
    'record_video': False,
    'video_interval': 20,
    'video_length': 500,
    'video_path': '/localdisk/graphrl_videos',
    'log_debug': False
}


class AgentParams(object):
    def __init__(self, ex=None):
        super(AgentParams, self).__init__()

        for k, v in AGENT_DEFAULT_PARAMS.items():
            setattr(self, k, v)

        # Filters
        # NOTE: the Agent class takes care of applying the reward filter but not the observation filter.
        # This is to allow memory-expensive processing such as stacking frames to be done as late as possible.
        # Subclasses should call self.filter_obs to use the observation filter.
        self.reward_filter = RewardFilter()
        self.obs_filter = ObservationFilter()
        self.agent_class = Agent
        self.sacred_run = None
        self.train_env_func = None
        self.test_envs = []
        self.custom_stopping_cond = StoppingCondition()
        self.device = None
        self.episode_metrics = {}

        self.ex = ex
        if self.ex is not None:
            @self.ex.capture
            def get_run_func(_run):
                return _run

            @self.ex.capture
            def get_config_func(_config):
                return _config

            self.get_run_func = get_run_func
            self.get_config_func = get_config_func

    def add_episode_metric(self, name, agg_func):
        self.episode_metrics[name] = agg_func

    def set_sacred_defaults(self):
        if self.ex is None:
            raise ValueError('No sacred experiment.')
        self.ex.add_config(agent=self.make_default_config())

    def load_sacred_config(self):
        if self.ex is None:
            raise ValueError('No sacred experiment.')
        self.load_config(self.get_config_func()['agent'])
        self.sacred_run = self.get_run_func()

    def make_default_config(self):
        return dict(AGENT_DEFAULT_PARAMS)

    def load_config(self, config):
        for k in AGENT_DEFAULT_PARAMS:
            if k in config:
                setattr(self, k, config[k])

    def make_agent(self):
        self.device = torch.device(self.device_name)
        return self.agent_class(params=self)

    @property
    def env(self):
        return self.train_env_func()

    @env.setter
    def env(self, value):
        self.train_env_func = lambda: value
        self.test_envs = [value]

    @property
    def env_func(self):
        return self.train_env_func

    @env_func.setter
    def env_func(self, value):
        self.train_env_func = value
        self.test_envs = [value()]


class Agent(object):
    def __init__(self, params, nets=None):
        super(Agent, self).__init__()
        self.params = params
        self.mode = self.params.mode

        def train_env_func():
            env = self.params.train_env_func()
            env = ContinuousEnv(env, max_steps=self.params.max_steps_per_train_episode)
            return env
        self.action_space = train_env_func().action_space

        if self.params.num_train_envs > 1:
            train_vec_env = SubprocVecEnv([train_env_func for _ in range(self.params.num_train_envs)])
        else:
            train_vec_env = DummyVecEnv([train_env_func])

        if self.params.record_video:
            train_vec_env = VecVideoRecorder(train_vec_env, self.params.video_path, record_video_trigger=lambda x: x % self.params.video_interval == 0, video_length=self.params.video_length)

        self.train_vec_env = train_vec_env
        self.test_envs = self.params.test_envs
        self.sacred_run = self.params.sacred_run
        self.device = self.params.device

        self.custom_stopping_cond = self.params.custom_stopping_cond

        # this counter is incremented when an action is taken.
        self.CTR_TRAIN_STEPS = 0
        # this counter is incremented when an episode ends or is aborted
        self.CTR_TRAIN_EPISODES = 0

        self.best_eval_results = None
        self.best_eval_steps = 0

        self.last_eval_episode = 0
        self.last_eval_step = 0
        self.last_save_step = 0

        self.episode_metrics = self.params.episode_metrics

        self.training_observations = None
        self.training_episode_metrics = [[] for _ in range(self.params.num_train_envs)]
        self.training_rewards = [0 for _ in range(self.params.num_train_envs)]
        self.training_steps = [0 for _ in range(self.params.num_train_envs)]

        if nets is None:
            nets = {}
        self.nets = nets

        print('Nets')
        for name, net in self.nets.items():
            print('Net {} has {} parameters.'.format(name, sum(p.numel() for p in net.parameters())))
            print(net)

        self.load_nets()

        self.open_files = {}

        if self.params.use_tensorboard:
            import tensorboardX
            self.summary_writer = tensorboardX.SummaryWriter(log_dir=self.params.tensorboard_logdir)

    # Subclasses should implement these methods

    def act(self, obs, training):
        raise NotImplementedError

    def train_on_env_step(self, obs, action, reward, done, aborted, info):
        pass

    def train_on_env_reset(self, obs):
        pass

    # Methods below this are implemented here

    def filter_obs(self, obs_list):
        return self.params.obs_filter(obs_list)

    def info_to_metrics(self, info):
        metrics = {}
        for k in self.params.episode_metrics:
            if k in info:
                metrics[k] = info[k]

        if 'episode' in info:
            for k, v in info['episode'].items():
                metrics['episode_{}'.format(k)] = v
        return metrics

    def reduce_metrics(self, metrics_list):
        grouped_metrics = collections.defaultdict(list)
        for m in metrics_list:
            for k, v in m.items():
                grouped_metrics[k].append(v)

        reduced_metrics = {}
        for k, vals in grouped_metrics.items():
            if k in self.params.episode_metrics:
                reduced_metrics[k] = self.params.episode_metrics[k](vals)
            elif 'episode' in k:
                if len(vals) > 1:
                    raise ValueError('Invalid episode metric {}'.format(k))
                reduced_metrics[k] = vals[0]
            else:
                raise ValueError('Unknown metric {}.'.format(k))
        return reduced_metrics

    def run_train_steps(self):
        finished_rewards = []
        finished_steps = []
        finished_completeds = []
        finished_episode_metrics = []

        start_time = time.time()

        actions = self.act(self.training_observations, training=True)

        observations, rewards, dones, infos = self.train_vec_env.step(actions)
        filtered_rewards = self.params.reward_filter(rewards)

        aborteds = [info['abort'] for info in infos]
        aborteds = np.array(aborteds)

        self.train_on_env_step(observations, actions, filtered_rewards, dones, aborteds, infos)

        self.training_observations = observations

        for i in range(self.params.num_train_envs):
            if not aborteds[i]:
                self.training_rewards[i] += rewards[i]
                self.CTR_TRAIN_STEPS += 1
                self.training_steps[i] += 1

                self.training_episode_metrics[i].append(self.info_to_metrics(infos[i]))

            if dones[i] or aborteds[i]:
                finished_rewards.append(self.training_rewards[i])
                finished_steps.append(self.training_steps[i])
                finished_completeds.append(dones[i])

                finished_episode_metrics.append(self.reduce_metrics(self.training_episode_metrics[i]))

                self.training_rewards[i] = 0
                self.training_episode_metrics[i] = []
                self.training_steps[i] = 0

                self.CTR_TRAIN_EPISODES += 1

        end_time = time.time()
        episode_speed = self.params.num_train_envs / float(end_time - start_time)

        return finished_rewards, finished_episode_metrics, finished_steps, finished_completeds, episode_speed

    def run_test_episode(self, env):
        rewards = []
        episode_metrics_list = []
        episode_steps = 0
        done = False

        start_time = time.time()

        obs = env.reset()

        while True:
            action = int(self.act([obs], training=False)[0])
            if self.params.print_actions:
                print('Action: {}'.format(action))
            obs, reward, done, info = env.step(action)

            episode_metrics_list.append(self.info_to_metrics(info))

            rewards.append(reward)
            episode_steps += 1

            if done:
                break

            if self.params.max_steps_per_test_episode is not None and episode_steps >= self.params.max_steps_per_test_episode:
                break

        end_time = time.time()
        episode_speed = episode_steps / float(end_time - start_time)

        episode_metrics = self.reduce_metrics(episode_metrics_list)

        episode_reward = sum(rewards)
        episode_completed = int(done)

        self.custom_stopping_cond.report_train_episode(episode_reward)

        return episode_reward, episode_metrics, episode_steps, episode_completed, episode_speed

    def log_scalar(self, name, value, step):
        self.sacred_run.log_scalar(name, value, step)
        if self.params.use_tensorboard:
            tensorboard_name = name.replace('.', '/')
            self.summary_writer.add_scalar(tensorboard_name, value, step)

    def log_scalar_debug(self, name, value, step):
        if self.params.log_debug:
            self.log_scalar(name, value, step)

    def log_episode_to_sacred(self, episode_reward, episode_metrics, episode_steps, episode_completed, episode_speed, phase, suffix, index):
        self.log_scalar('{}.episode_reward.{}'.format(phase, suffix), episode_reward, index)
        self.log_scalar('{}.episode_steps.{}'.format(phase, suffix), episode_steps, index)
        self.log_scalar('{}.episode_completed.{}'.format(phase, suffix), episode_completed, index)
        self.log_scalar_debug('{}.episode_speed.{}'.format(phase, suffix), episode_speed, index)

        for k, v in episode_metrics.items():
            self.log_scalar('{}.{}.{}'.format(phase, k, suffix), v, index)

    def log_train_to_sacred(self, name, value):
        self.log_scalar('train.{}.bystep'.format(name), value, self.CTR_TRAIN_STEPS)
        self.log_scalar('train.{}.byepisode'.format(name), value, self.CTR_TRAIN_EPISODES)

    def log_train_to_sacred_debug(self, name, value):
        if self.params.log_debug:
            self.log_train_to_sacred(name, value)

    def evaluate(self):
        results = []
        for i, env in enumerate(self.test_envs):
            episode_reward, episode_metrics, episode_steps, episode_completed, episode_speed = self.run_test_episode(env)
            name = 'eval_test_{}'.format(i + 1)
            self.log_episode_to_sacred(episode_reward, episode_metrics, episode_steps, episode_completed, episode_speed, name, 'bystep', self.CTR_TRAIN_STEPS)
            self.log_episode_to_sacred(episode_reward, episode_metrics, episode_steps, episode_completed, episode_speed, name, 'byepisode', self.CTR_TRAIN_EPISODES)
            self.custom_stopping_cond.report_eval_episode(i, episode_reward)
            results.append(episode_reward)
            log_str = 'Step {}. Evaluation episode no on {}. Reward: {:.2f}, steps: {}, completed: {}, speed: {:.2f} f/s'.format(self.CTR_TRAIN_STEPS, name, episode_reward, episode_steps, episode_completed, episode_speed)
            for k, v in sorted(episode_metrics.items()):
                log_str = log_str + ', {}: {:.2f}'.format(k.replace('_', ' '), v)
            print(log_str)

        if self.best_eval_results is None or all(cur_result > prev_result for cur_result, prev_result in zip(results, self.best_eval_results)):
            self.best_eval_results = results
            self.best_eval_steps = self.CTR_TRAIN_STEPS
            self.save_nets(is_best=True)
        self.last_eval_episode = self.CTR_TRAIN_EPISODES
        self.last_eval_step = self.CTR_TRAIN_STEPS

    def test(self):
        all_rewards = []
        for i, env in enumerate(self.test_envs):
            test_rewards = []
            # print('Testing on test env {}'.format(i))
            for j in range(self.params.test_episodes):
                episode_reward, episode_metrics, episode_steps, episode_completed, episode_speed = self.run_test_episode(env)
                all_rewards.append(episode_reward)
                self.log_episode_to_sacred(episode_reward, episode_metrics, episode_steps, episode_completed, episode_speed, 'test_{}'.format(i), 'byepisode', j)
                test_rewards.append(episode_reward)

                print('Test env {}, episode no {}. Reward: {:.2f}, steps: {}, completed: {}, speed: {:.2f} f/s'
                      .format(i, j, episode_reward, episode_steps, episode_completed, episode_speed))

            print('Test env {}, Mean reward: {}, std reward: {}'.format(i, np.mean(all_rewards), np.std(all_rewards)))

    def load_nets(self):
        if not self.params.should_load_nets:
            return
        for name, net in self.nets.items():
            net.load_state_dict(torch.load(os.path.join(self.params.load_nets_folder, '{}.pth'.format(name)), map_location=self.device_name))

    def save_nets(self, is_best=False):
        if not os.path.isdir(AGENT_DEFAULT_PARAMS['save_dir']):
            os.makedirs(AGENT_DEFAULT_PARAMS['save_dir'] + '/' + AGENT_DEFAULT_PARAMS['experiment_name'])
        for name, net in self.nets.items():
            torch.save(net.state_dict(), AGENT_DEFAULT_PARAMS['save_dir'] + '/' + AGENT_DEFAULT_PARAMS['experiment_name'] + '/{}_{}.pth'.format(name, self.CTR_TRAIN_STEPS))
            new_file, filename = tempfile.mkstemp()
            torch.save(net.state_dict(), filename) 
            self.sacred_run.add_artifact(filename, name='{}_{}.pth'.format(name, self.CTR_TRAIN_STEPS))

            if is_best:
                print('Saving best model at step {}'.format(self.CTR_TRAIN_STEPS))
                self.sacred_run.add_artifact(filename, name='{}_best.pth'.format(name))

        print('Saving at step {}'.format(self.CTR_TRAIN_STEPS))
        self.last_save_step = self.CTR_TRAIN_STEPS

    def check_min_train_steps(self):
        if self.params.use_min_train_steps:
            return self.CTR_TRAIN_STEPS > self.params.min_train_steps
        return True

    def check_min_train_episodes(self):
        if self.params.use_min_train_episodes:
            return self.CTR_TRAIN_EPISODES > self.params.min_train_episodes
        return True

    def check_no_progress_steps(self):
        if self.params.use_no_progress_steps:
            return self.CTR_TRAIN_STEPS - self.best_eval_steps > self.params.no_progress_steps
        return True

    def train(self):
        self.training_observations = self.train_vec_env.reset()
        self.train_on_env_reset(self.training_observations)

        speeds = []

        while True:

            # Check stopping conditions
            if self.check_min_train_steps() and self.check_min_train_episodes() and self.check_no_progress_steps():
                break

            if self.custom_stopping_cond.should_stop(self):
                break

            episode_rewards, episode_metrics_list, episode_steps, episode_completeds, speed = self.run_train_steps()

            speeds.append(speed)
            speeds = speeds[-100:]

            for episode_reward, episode_metrics, episode_steps, episode_completed in zip(episode_rewards, episode_metrics_list, episode_steps, episode_completeds):
                mean_speed = np.mean(speeds)

                self.log_episode_to_sacred(episode_reward, episode_metrics, episode_steps, episode_completed, mean_speed, 'train', 'bystep', self.CTR_TRAIN_STEPS)
                self.log_episode_to_sacred(episode_reward, episode_metrics, episode_steps, episode_completed, mean_speed, 'train', 'byepisode', self.CTR_TRAIN_EPISODES)
                log_str = 'Step {}. Train episode no {}. Reward: {:.2f}, steps: {}, completed: {}, speed: {:.2f} f/s'.format(self.CTR_TRAIN_STEPS, self.CTR_TRAIN_EPISODES, episode_reward, episode_steps, episode_completed, mean_speed)
                for k, v in sorted(episode_metrics.items()):
                    log_str = log_str + ', {}: {:.2f}'.format(k.replace('_', ' '), v)
                print(log_str)

            if (self.params.use_eval_freq_episodes and self.CTR_TRAIN_EPISODES - self.last_eval_episode >= self.params.eval_freq_episodes)\
                    or (self.params.use_eval_freq_steps and self.CTR_TRAIN_STEPS - self.last_eval_step >= self.params.eval_freq_steps):
                self.evaluate()

            if self.CTR_TRAIN_STEPS - self.last_save_step >= self.params.save_freq_steps:
                self.save_nets()

        self.evaluate()
        self.save_nets()

    def run(self):
        if self.mode == 'train':
            self.train()
        elif self.mode == 'test':
            self.test()
        else:
            raise ValueError('Agent mode must be train or test.')
