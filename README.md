# gym-stochastic

Reinforcement learning gyms for experimenting with stochasticity.

#### DistributionContextualBanditEnv-v0

- based on the wonderful  https://github.com/JKCooper2/gym-bandits (under [MIT license](https://github.com/JKCooper2/gym-bandits/commit/1aba0c6897346e31c2935c13249ae35ca4766121#diff-9879d6db96fd29134fc802214163b95a) )
- Generalized further
```
import gym_stochastic
from gym_stochastic.envs.dist_utils import *

arms_r_comp = get_sampler__composite_perarm( 
    sub_samplers=[
        get_sampler__composite_select([
            get_reward_sampler__fixed_norm_arm(5.0,1.0),
            get_reward_sampler__fixed_norm_arm(-20.0,5.0),]
        ),
        get_sampler__composite_select([
            get_reward_sampler__fixed_norm_arm(-10.0,1.0),
            get_reward_sampler__fixed_uniform_arm(5.0,25.0),],
            dist=[0.1, 0.9,] )] )

env=gym.make('DistributionContextualBanditEnv-v0',arms=2, p_dist_fn=1.0, r_dist_fn=arms_r_comp ), 
```
- Above config results in arm-reward histograms:
![Env1](https://github.com/pathway/gym-stochastic/blob/master/notebooks/env1.png)

#### WetChicken1d-v0

1 dimensional Wet Chicken as described in section 4.1 of:

Alexander Hans and Steffen Udluft. Efficient uncertainty propagation for reinforcement learning
with limited data. In ICANN, pp. 70–79. Springer, 2009.
https://www.tu-ilmenau.de/fileadmin/media/neurob/publications/conferences_int/2009/Hans-ICANN-2009.pdf

I was unable to locate a copy of the original reference:

V. Tresp. The wet game of chicken. Siemens AG, CT IC 4, Technical Report, 1994.
