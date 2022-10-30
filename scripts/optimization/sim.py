import numpy as np
import pandas as pd
import multiprocess
from pathlib import Path
import sys
import random

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import scripts.optimization.vacc as vacc
import scripts.simple_test.test_case as test
import config.config as config

# CONFIG STUFF HERE
pop_vec, vacc_rate, seed, dist = test.generate(3)
c = 0.09 # a random feasible c

def sample_points():
    """
    sample_points helps finding a feasible vector under the two constraints
    """ 
    num_people = np.sum(pop_vec * c)
    vector = np.random.rand(1,3) * 0.1 # etting random initial value
    diff = np.sum(np.dot(vector,pop_vec)) - num_people 
    vector = vector.reshape(vector.shape[1])  
    flag = False
    while(flag == False):
        if (diff < 0):
            for i in range(vector.shape[0]):
                seed = random.randint(0,vector.shape[0])
                i = (i + seed) % vector.shape[0]
                max_num = vacc_rate[i] * pop_vec[i] - vector[i] * pop_vec[i]
                vector[i] += min((-diff)/pop_vec[i], max_num/pop_vec[i])
                diff += max_num
                if (diff >=0):
                    flag = True
                    break
        elif(diff > 0):
            for i in range(vector.shape[0]):
                seed = random.randint(0,vector.shape[0])
                i = (i + seed) % vector.shape[0]
                max_num = vector[i] * pop_vec[i]
                vector[i] -= min(max_num/pop_vec[i],diff/pop_vec[i])
                diff -= max_num
                if (diff >=0):
                    flag = True
                    break
        else:
            print("lucky")

    return vector


        

# optimizer oracle configuration
opt_config = {
    'obj':"attacksize",   # objective function
    'V_repr':"ratio",     # represent vacc rates as a ratio: [0,1]
    'constraint_bnd':0.09 # set c=0.05 (percentage can go down by 5%)
}
vacc_df = pd.DataFrame({'pop': pop_vec,
                    'vacc': vacc_rate}) # TODO the api should not be like this it does not make sense
# plug all arguments into oracle
oracle = vacc.VaccRateOptEngine(
        opt_config=opt_config,
        V_0=vacc_df['vacc'],            # vector, number of people vaccinated in each region
        seed=seed,                      # vector, number of starting infected in each region
        sim_config=config.TSIR_CONFIG,  # constant hyperparameters
        pop=vacc_df,                    # vector, populations at each region
        distances=dist)                 # distance matrix

if __name__ == '__main__':
    with multiprocess.Pool() as p:
        # query the vector where we uniformly distribute the vaccination decrease over all districts
        good_values = [0] #starting with 0
        num_iter = 10
        for i in range(num_iter): # 
            vector = sample_points()
            result, sim_pool = oracle.query(V_delta=vector,
                                            pool=p,
                                            n_sim=150,
                                            return_sim_pool=True)
            print(np.mean(result))
            if(np.mean(result) > good_values[-1]):
                good_values.append(np.mean(result))
        print(good_values)