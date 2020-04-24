import gym
from gym import spaces
import random
import numpy as np
from collections import deque


class RenderEnv(gym.ObservationWrapper):
    def observation(self, observation):
        self.render()
        return observation


class PermuteEnv(gym.ObservationWrapper):
    def __init__(self, env, axes):
        super(PermuteEnv, self).__init__(env)

        self.axes = axes

        new_shape = []
        for axis in axes:
            new_shape.append(env.observation_space.shape[axis])
        new_shape = tuple(new_shape)
        self.observation_space = spaces.Box(low=np.min(env.observation_space.low), high=np.max(env.observation_space.high), dtype=env.observation_space.dtype, shape=new_shape)

    def observation(self, observation):
        return np.transpose(observation, self.axes)


class MapEnv(gym.ObservationWrapper):
    def __init__(self, env, m):
        super(MapEnv, self).__init__(env)
        self.m = m

        def func(x):
            return self.m[x]
        self.vec_func = np.vectorize(func)

    def observation(self, observation):
        return self.vec_func(observation)


class SampleEnv(gym.Env):
    def __init__(self, envs):
        super(SampleEnv, self).__init__()
        self.envs = envs
        self._cur_env = None
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.reward_range = self.envs[0].reward_range

    def reset(self):
        self._cur_env = random.choice(self.envs)
        return self._cur_env.reset()

    def step(self, action):
        return self._cur_env.step(action)

    def render(self, *args):
        self._cur_env.render(*args)


class CustomLazyFrames(object):
    def __init__(self, frames, axis):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._axis = axis

    def __array__(self, dtype=None):
        out = np.stack(self._frames, axis=self._axis)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def copy(self):
        return CustomLazyFrames(list(self._frames), self._axis)


class CustomFrameStack(gym.Wrapper):
    def __init__(self, env, k, axis):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.axis = axis
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = list(env.observation_space.shape)
        shp.insert(self.axis, self.k)
        self.observation_space = spaces.Box(low=0, high=255, shape=tuple(shp), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return CustomLazyFrames(list(self.frames), axis=self.axis)


def play(env):
    env = RenderEnv(env)
    env.reset()
    done = False
    while not done:
        action = int(input())
        env.step(action)
