# Taken from berkeleydeeprlcourse/hw3

import numpy as np
from collections import deque
import gym
from gym import spaces
from baselines.common.atari_wrappers import make_atari, EpisodicLifeEnv, FireResetEnv, ClipRewardEnv, WarpFrame
from baselines.bench.monitor import Monitor


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=2)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def copy(self):
        return LazyFrames(self._frames)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, use_lazy_frames):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.use_lazy_frames = use_lazy_frames
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

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
        ob = LazyFrames(list(self.frames))
        if not self.use_lazy_frames:
            ob = np.array(ob)
        return ob


def wrap_deepmind(env, skip=4, clip_rewards=True, frames_stack=4, use_lazy_frames=True):
    env = Monitor(env, None)
    env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    env = FrameStack(env, frames_stack, use_lazy_frames=use_lazy_frames)
    return env


def make_atari_env(name, use_lazy_frames=True):
    env = make_atari(name)
    env = wrap_deepmind(env, use_lazy_frames=use_lazy_frames)
    return env
