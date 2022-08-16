import numpy as np
import multiprocess
from scipy.stats import uniform, randint
from functools import partial

def sim_anneal(init_state,init_temp,
        num_iters,move_func,engine,
        cores=7,n_samples=150):
    s = init_state
    T = init_temp
    with multiprocess.Pool(cores) as pool:
        for k in range(num_iters):
            T = init_temp*(1-(k+1)/num_iters)
            s_next = move_func(s)
            #s_cost = cost_func(s)
            s_cost = np.mean(engine.query(seed_prime=s,pool=pool,n_sim=n_samples))
            s_next_cost = np.mean(engine.query(seed_prime=s_next,pool=pool,n_sim=n_samples)) 
            # compute acceptance probability
            if s_next_cost < s_cost:
                prob = 1
            else:
                prob = np.exp(-(s_next_cost-s_cost)/T)
            # make the jump
            if prob >= random.random():
                s = s_next
                print("cost",s_next_cost)
            else:
                print("cost",s_cost)
    return s

def move_seed(S,budget=None):
    if budget is None:
        budget = np.sum(S)
    nonzero_indices = np.nonzero(S)[0]
    nonzero_vals = S[nonzero_indices]
    rand_select = randint.rvs(0,budget)
    # what index does that correspond to?
    # first index where rolling sum > rand_select
    rolling_sum = np.cumsum(nonzero_vals)
    #print(np.argmax(rolling_sum>rand_select))
    #print(nonzero_indices)
    index_to_subtract = nonzero_indices[np.argmax(rolling_sum>rand_select)]
    index_to_add = randint.rvs(0,len(S)-1)
    to_return = S
    to_return[index_to_subtract] = to_return[index_to_subtract]-1
    to_return[index_to_add] = to_return[index_to_add]+1
    return to_return

#%%




