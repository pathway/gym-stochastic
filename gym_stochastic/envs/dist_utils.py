import numpy as np
import toolz
import random

def unit_vector(dims):
    vec = [np.random.normal(0, 1) for i in range(dims)]
    mag = sum(x ** 2 for x in vec) ** .5
    return [x / mag for x in vec]


def evaluate_sampler( sampler_fn, obs, action ):
    """
    :param sampler_fn:
        fn(o,k)     Function returning sample value
        [p1,p2..]   List of values per-arm
        p1          Scalar value for all arms

    :param obs:
    :param action:
    :return:
    """
    if callable(sampler_fn):
        sample = sampler_fn(obs, action)
    elif type(sampler_fn) == list:
        sample = sampler_fn[action]
    else:
        sample = sampler_fn  # assume float or int
    return sample



def contextual__obs_dot(obs, arm, p_values_max=0.5, v_dot=None ):
    """
    :param arm:     arm count
    :param p_values:    either scalar (same for all arms) or vector (prob [0,1.0] to return reward
    :param dot:     scalar (implide vector), or vector to dot product with
    :return:        1 (yes reward) or 0 (no reward)

    """
    els = [ o_i *v_i for o_i,v_i in zip(obs,v_dot) ]
    dot_prod = sum( els )
    return dot_prod



def get_sampler__composite_mult(sub_samplers=[]):
    def sampler__composite_mult(obs,arm):
        rs = []
        r_mult = None
        for sub in sub_samplers:
            r = evaluate_sampler(sub, obs, arm)
            rs.append(r)
            if r_mult == None:
                r_mult = r
            else:
                r_mult *= r
        return r_mult
    return sampler__composite_mult




def get_sampler__composite_sum(sub_samplers=[]):
    def sampler__composite_sum(obs,arm):
        r_mult=0
        for sub in sub_samplers:
            r = evaluate_sampler(sub,obs,arm)
            r_mult+=r
        return r_mult
    return sampler__composite_sum



def get_sampler__composite_perarm(sub_samplers=[],):
    def sampler__composite_perarm(obs,arm,):
        choice = sub_samplers[ arm ]
        return evaluate_sampler( choice, obs,arm)
    return sampler__composite_perarm



def get_sampler__composite_select(sub_samplers=[],dist=1.0):
    def sampler__composite_uniform(obs,arm,dist=1.0):
        list_of_candidates = np.arange(0,len(sub_samplers))
        if type(dist)==list:
            dist_list = [ d/ sum(dist)  for d in dist ]
        else:
            dist_list = [ dist / len(sub_samplers), ] * len(sub_samplers)
        #print(probability_distribution_list)
        #probability_distribution = [0.2, 0.2, 0.6]
        draw = np.random.choice(list_of_candidates, 1,p=dist_list)[0]
        choice = sub_samplers[ draw ]
        return evaluate_sampler(choice,obs,arm)
    return toolz.partial(sampler__composite_uniform, dist=dist)



def get_reward_sampler__fixed_uniform_arm(min=0.0, max=1.0):
    def reward_sampler__fixed_uniform_arm(obs, arm):
        """
        rew_means and rew_vars can be scalar, or vector (value per arm)
        """
        r = np.random.uniform( min, max )
        return r
    return reward_sampler__fixed_uniform_arm


def get_reward_sampler__fixed_norm_arm(rew_means=0.0, rew_vars=1.0):
    def reward_sampler__fixed_norm_arm(obs, arm):
        """
        rew_means and rew_vars can be scalar, or vector (value per arm)
        """
        if type(rew_means)==list:
            p_mean = rew_means[arm]
            p_var = rew_vars[arm]
        else:
            p_mean = rew_means
            p_var = rew_vars

        r = np.random.normal( p_mean, p_var )
        return r
    return reward_sampler__fixed_norm_arm

contextual_payout_prob_sampler__obs_dot_v1 = toolz.partial( contextual__obs_dot, v_dot=[-1.0,0.0] )
contextual_payout_prob_sampler__obs_dot_v2 = toolz.partial( contextual__obs_dot, v_dot=[0.0,1.0] )



reward_sampler__fixed_simplegauss = get_reward_sampler__fixed_norm_arm()
reward_sampler__fixed_gauss_badgood = get_reward_sampler__fixed_norm_arm( [-0.5,1.0,],[2.0,1.0,], )


reward_sampler__comp1 =  get_sampler__composite_mult( sub_samplers=[reward_sampler__fixed_simplegauss, contextual_payout_prob_sampler__obs_dot_v1 ] )

payout_prob_sampler__comp1 =  get_sampler__composite_mult( sub_samplers=[contextual_payout_prob_sampler__obs_dot_v2, contextual_payout_prob_sampler__obs_dot_v1 ] )

