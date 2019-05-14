from gym.envs.registration import register

register(
    id='WetChicken1d-v0',
    entry_point='gym_stochastic.envs:WetChicken1dEnv',
)
