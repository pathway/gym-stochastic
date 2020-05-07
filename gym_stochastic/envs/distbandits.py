import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.toy_text import discrete
import toolz
from gym_stochastic.envs.dist_utils import *



class DistributionContextualBanditEnv(gym.Env):
    """
    Bandit environment base to allow agents to interact with the class n-armed contextual bandit

    I started from the excellent https://github.com/JKCooper2/gym-bandits

    Modified:
    - allow arbitrary distributions for payout prob and reward values
    - add observation: random vector of values in [0.0,1.0]
    - allow for dependence of payout prob and reward values, on observation as well as arm

    p_dist_fn:  payout_prob: probability of delivering a reward for arm k after obs o
    can be defines as:
        fn(o,k)     Function returning
        [p1,p2..]   List of probs per-arm
        p1          Scalar prob for all arms

    r_dist_fn:  Function returning potential reward value for arm k (conditional on payout p_dist_fn) after obs o
    can be defines as:
        fn(o,k)     Function returning reward value
        [p1,p2..]   List of fixed rewards per-arm
        p1          Scalar reward for all arms

    obs_dim:    Size of observation space

    """
    def __init__(self, arms, p_dist_fn, r_dist_fn, obs_dim=2, seed=None):
        self.p_dist_fn = p_dist_fn
        self.r_dist_fn = r_dist_fn
        self.obs_dim = obs_dim

        self.n_bandits = arms
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.Box(-1.0,1.0,(obs_dim,) )

        self._seed(seed)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        self.obs = unit_vector(self.obs_dim)
        return self.obs

    def step(self, action):
        assert self.action_space.contains(action)

        self.obs = self._get_obs()
        reward = 0
        done = True

        # any reward?
        payout_prob = evaluate_sampler( self.p_dist_fn, self.obs, action  )
        if np.random.uniform() < payout_prob:
            reward = evaluate_sampler(self.r_dist_fn, self.obs, action)

        return self.obs, reward, done, {}

    def reset(self):
        return self._get_obs()

    def render(self, mode='human', close=False):
        pass



# all actions give full reward each time
env_happyland = DistributionContextualBanditEnv( arms=4, p_dist_fn=1.0, r_dist_fn=1.0,  )

# always reward, usually +1, sometimes -10, regardless of arm or obs
env_russian_roulette = DistributionContextualBanditEnv( arms=6, p_dist_fn=1.0, r_dist_fn=[1,1,1,-10,1,1],  )


def _test_bandit(e):
    print("---")
    for i in range(0,100):
        action = e.action_space.sample()
        obs, reward, done, info = e.step(action)
        print( "obs, action, reward, done, info", obs, action, reward, done, info )
        if done:
            e.reset()
