# -*- coding: utf-8 -*-
import gym
import gym.spaces

import os
import h5py
import numpy as np
import random
from graphrl.environments.ai2thor_visnav.constants import ACTION_SIZE
from graphrl.environments.ai2thor_visnav.constants import SCREEN_WIDTH
from graphrl.environments.ai2thor_visnav.constants import SCREEN_HEIGHT
from graphrl.environments.ai2thor_visnav.constants import HISTORY_LENGTH


class AI2ThorVisnav(gym.Env):
    def __init__(self, **config):
        super(AI2ThorVisnav, self).__init__()

        # configurations
        self.scene_name = config.get('scene_name', 'bedroom_04')
        self.random_start = config.get('random_start', True)
        self.n_feat_per_locaiton = config.get('n_feat_per_locaiton', 1)  # 1 for no sampling
        self.terminal_state_id = config.get('terminal_state_id', 0)

        self.h5_dir = config.get('h5_dir', 'data')
        self.h5_file_path = os.path.join(self.h5_dir, '{}.h5'.format(self.scene_name))
        self.h5_file = h5py.File(self.h5_file_path, 'r')

        self.locations = self.h5_file['location'][()]
        self.rotations = self.h5_file['rotation'][()]
        self.n_locations = self.locations.shape[0]

        self.terminals = np.zeros(self.n_locations)
        self.terminals[self.terminal_state_id] = 1
        self.terminal_states, = np.where(self.terminals)

        self.transition_graph = self.h5_file['graph'][()]
        self.shortest_path_distances = self.h5_file['shortest_path_distance'][()]

        self.history_length = HISTORY_LENGTH
        self.screen_height = SCREEN_HEIGHT
        self.screen_width = SCREEN_WIDTH

        # we use pre-computed fc7 features from ResNet-50
        # self.s_t = np.zeros([self.screen_height, self.screen_width, self.history_length])
        self.s_t = np.zeros([2048, self.history_length])
        self.s_target = self._tiled_state(self.terminal_state_id)

        self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.float32, shape=(2048, self.history_length))
        self.action_space = gym.spaces.Discrete(ACTION_SIZE)

        self.reset()

    def _tiled_state(self, state_id):
        k = random.randrange(self.n_feat_per_locaiton)
        f = self.h5_file['resnet_feature'][state_id][k][:, np.newaxis]
        return np.tile(f, (1, self.history_length))

    def reset(self):
        # randomize initial state
        while True:
            k = random.randrange(self.n_locations)
            min_d = np.inf

            # check if target is reachable
            for t_state in self.terminal_states:
                dist = self.shortest_path_distances[k][t_state]
                min_d = min(min_d, dist)
            # min_d = 0  if k is a terminal state
            # min_d = -1 if no terminal state is reachable from k
            if min_d > 0:
                break

        # reset parameters
        self.current_state_id = k
        self.s_t = self._tiled_state(self.current_state_id)

        self.reward = 0
        self.collided = False
        self.terminal = False

        return self._make_obs()

    def _make_obs(self):
        return {'obs': self.s_t, 'target': self.s_target}

    def _reward(self, terminal, collided):
        # positive reward upon task completion
        if terminal:
            return 10.0
        return -0.01

    @property
    def state(self):
        # read from hdf5 cache
        k = random.randrange(self.n_feat_per_locaiton)
        return self.h5_file['resnet_feature'][self.current_state_id][k][:, np.newaxis]

    def step(self, action):
        assert not self.terminal, 'step() called in terminal state'
        k = self.current_state_id
        if self.transition_graph[k][action] != -1:
            self.current_state_id = self.transition_graph[k][action]
            if self.terminals[self.current_state_id]:
                self.terminal = True
                self.collided = False
            else:
                self.terminal = False
                self.collided = False
        else:
            self.terminal = False
            self.collided = True

        self.reward = self._reward(self.terminal, self.collided)
        self.s_t = np.append(self.s_t[:, 1:], self.state, axis=1)

        return self._make_obs(), self.reward, self.terminal, {}
