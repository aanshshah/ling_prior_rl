import gym


class ContinuousEnv(gym.Wrapper):
    def __init__(self, env, max_steps=None):
        super(ContinuousEnv, self).__init__(env)
        self.cur_step = 0
        self.max_steps = max_steps

    def reset(self):
        self.cur_step = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['abort'] = False

        if done:
            obs = self.reset()
        elif self.max_steps is not None and self.cur_step >= self.max_steps:
            obs = self.reset()
            reward = 0
            done = False
            info['abort'] = True
        else:
            self.cur_step += 1

        return obs, reward, done, info
