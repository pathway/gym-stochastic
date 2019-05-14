import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.toy_text import discrete


class WetChicken1d(discrete.DiscreteEnv):
    debug_out = False
    metadata = {
        'river_velocity': 1.0,
        'river_waterfall_x': 3.0,
        'river_start_x': 0.0,
        'river_turb': 2.5,

        'deltax_paddleback': -2.0,
        'deltax_hold': -1.0,
    }

    def __init__(self):
        self.seed()
        self.observation_space = spaces.Box(0, 1.0, (1,), dtype=np.float16)
        self.action_space = spaces.Discrete(3)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        return [ self.x ]

    def reset(self):
        self.x = 0
        self.step_n = 0
        start_reward=0
        start_done=False
        return self._get_obs()

    def step(self, action):
        done = False

        self.step_n += 1

        obs = np.zeros( (1), dtype=np.float16 )

        v = self.metadata['river_velocity']

        if action==0:       # drift
            deltax = 0
        elif action==1:     # hold
            deltax = self.metadata['deltax_hold']
        elif action == 2:   # paddleback
            deltax = self.metadata['deltax_paddleback']

        newx = self.x + v + deltax

        s = self.metadata['river_turb']
        turbx = np.random.uniform(-1*s,s)

        finalx = newx + turbx

        # fell off?
        if finalx > self.metadata['river_waterfall_x']:
            if self.debug_out: print('FELL')
            finalx=0.0
            done=True

        reward=finalx
        self.last_obs = obs

        obs[0]=finalx

        if self.debug_out:
            print('self.last_obs, deltax, turbx, obs, reward, done', self.last_obs, deltax, turbx, obs, reward, done)
        return obs, reward, done, {}


    def render(self, mode="human"):
        return

from gym.envs.registration import registry, register, make, spec


register(
    id='WetChicken1d-v0',
    entry_point='wetchicken1d:WetChicken1d',
    max_episode_steps=200,
    reward_threshold=20,
)

