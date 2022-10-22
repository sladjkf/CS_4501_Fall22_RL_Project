import numpy as np
import pandas as pd
import multiprocess
from pathlib import Path
import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import scripts.optimization.vacc as vacc
import scripts.simple_test.test_case as test
import config.config as config

if __name__ == '__main__':
    pop_vec, vacc_rate, seed, dist = test.generate(3)

    # optimizer oracle configuration
    opt_config = {
        'obj':"attacksize",   # objective function
        'V_repr':"ratio",     # represent vacc rates as a ratio: [0,1]
        'constraint_bnd':0.05 # set c=0.05 (percentage can go down by 5%)
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

    # setup for multithreading using 5 processes
    with multiprocess.Pool() as p:
        # query the vector where we uniformly distribute the vaccination decrease over all districts
        result, sim_pool = oracle.query(V_delta=0.05*np.ones_like(vacc_rate),
                                        pool=p,
                                        n_sim=150,
                                        return_sim_pool=True)
        print(np.mean(result))