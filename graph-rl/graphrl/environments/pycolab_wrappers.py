import numpy as np
import gym
import gym.spaces
import time


def load_art(artfile):
    '''
    Loads ascii art from a file.
    Returns:
        list(str): art that can be read by pycolab
        list(str): the characters used in the art
        (int, int): the shape of the art
    '''
    with open(artfile, 'r') as f:
        lines = f.read().split('\n')

    characters = set()
    for line in lines:
        characters.update(set(line))
    characters = list(sorted(list(characters)))

    rows = len(lines)
    cols = 0 if len(lines) == 0 else len(lines[0])

    return lines, characters, (rows, cols)


class PycolabMazeEnv(gym.Env):
    def __init__(self, make_game_function, num_actions, height, width, character_map={}):
        super(PycolabMazeEnv, self).__init__()
        self.make_game_function = make_game_function
        self.reward_range = (-float('inf'), float('inf'))
        self.action_space = gym.spaces.Discrete(num_actions)
        self.observation_space = gym.spaces.Box(low=0, high=255, dtype=np.uint8, shape=(height, width))
        self.character_map = character_map
        self.val_map = {ord(old): ord(new) for old, new in self.character_map.items()}

    def convert_board(self, board):
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                val = board[i, j]
                if val in self.val_map:
                    board[i, j] = self.val_map[val]
        return board

    def reset(self):
        self.game = self.make_game_function()
        self.ui = None
        obs, _, _ = self.game.its_showtime()
        self.total_reward = 0
        return self.convert_board(obs.board.copy())

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('Invalid action: {}'.format(action))
        obs, reward, _ = self.game.play(action)
        if reward is None:
            reward = 0
        self.total_reward += reward
        return self.convert_board(obs.board.copy()), reward, self.game.game_over, {}

    def render(self, mode='human'):
        if mode != 'human':
            raise ValueError('Invalid mode: {}'.format(mode))

        board_lines = []
        for line in self.game._board.board:
            line = line.tostring().decode('ascii')
            line = ''.join([self.character_map.get(c, c) for c in line])
            board_lines.append(line)
        board_str = '\n'.join(board_lines)
        print(board_str)
        print('Total reward: {}'.format(self.total_reward))
        # if self.game.game_over:
        #     print('Game over')
        time.sleep(0.3)


class OneHotEnv(gym.ObservationWrapper):
    def __init__(self, env, values):
        super(OneHotEnv, self).__init__(env)
        height, width = env.observation_space.shape
        self.values = values
        self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.float32, shape=(len(values), height, width))

    def observation(self, obs):
        layers = [obs == value for value in self.values]
        onehot = np.stack(layers, 0).astype(np.float32)
        return onehot


class IndexToFeatsEnv(gym.ObservationWrapper):
    def __init__(self, env, kg_dict):
        super(IndexToFeatsEnv, self).__init__(env)
        self.kg_dict = kg_dict
        self.entities = kg_dict['entities']
        self.num_entities = len(kg_dict['entities'])
        self.ord_to_idx = {ord(c): i + 1 for i, c in enumerate(self.entities)}
        self.reindex_func = np.vectorize((lambda x: self.ord_to_idx.get(x, 0)))
        self.num_node_feats = self.kg_dict['num_node_feats']

        height, width = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.float32, shape=(self.num_node_feats, height, width))

        kg_node_feats = np.zeros((self.num_entities + 1, self.num_node_feats), dtype=np.float32)
        for node_dict in kg_dict['nodes']:
            node_feats_type = node_dict['feature_type']
            node_idx = self.entities.index(node_dict['node']) + 1
            if node_feats_type == 'one_hot':
                feature_idx = node_dict['feature_idx']
                kg_node_feats[node_idx, feature_idx] = 1
            elif node_feats_type == 'list':
                kg_node_feats[node_idx, :] = node_dict['feats']
            else:
                raise ValueError('Unknown node feature type {}.'.format(node_feats_type))
        self.kg_node_feats = kg_node_feats

    def observation(self, obs):
        obs = self.reindex_func(obs)

        obs_shape = obs.shape
        obs = obs.flatten()
        feats = self.kg_node_feats[obs].T
        new_obs_shape = [-1] + list(obs_shape)
        feats = feats.reshape(*new_obs_shape)
        return feats


class OneHotWithEqualFeatsEnv(gym.ObservationWrapper):
    def __init__(self, env, values, equal_values):
        super(OneHotWithEqualFeatsEnv, self).__init__(env)
        height, width = env.observation_space.shape
        self.values = values
        self.equal_values = equal_values
        self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.float32, shape=(len(values) + len(equal_values), height, width))

    def observation(self, obs):
        layers = [obs == value for value in self.values]

        for eqvals in self.equal_values:
            layer = np.max(np.stack([obs == value for value in eqvals], 0), 0)
            layers.append(layer)

        onehot = np.stack(layers, 0).astype(np.float32)
        return onehot
