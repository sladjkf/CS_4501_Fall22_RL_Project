import numpy as np
import pandas as pd
import multiprocess
from pathlib import Path
import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import scripts.optimization.vacc as vacc
import scripts.simple_test.test_case as test_case
import config

pop_vec, vacc_rate, seed, dist = test_case.generate(3)

# plug all arguments into oracle
oracle = vacc.VaccRateOptEngine(
        opt_config=config.OPT_CONFIG,
        V_0=vacc_rate,            # vector, number of people vaccinated in each region
        seed=seed,                      # vector, number of starting infected in each region
        sim_config=config.TSIR_CONFIG,  # constant hyperparameters
        pop=pop_vec,                    # vector, populations at each region HEH???????
        distances=dist)                 # distance matrix


if __name__ == '__main__':
    with multiprocess.Pool() as p:
        # query the vector where we uniformly distribute the vaccination decrease over all districts
        result = oracle.query(V_delta=0.05*np.ones_like(vacc_rate),
                                        pool=p,
                                        n_sim=150,
                                        return_sim_pool=False)
        print(np.mean(result))