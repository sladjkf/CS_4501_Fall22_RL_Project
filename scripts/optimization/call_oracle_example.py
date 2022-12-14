"""
Example file demonstrating how to setup and call the oracle
with a very simple example problem instance.
"""
import numpy as np
import pandas as pd
import multiprocess
from pathlib import Path
import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import scripts.optimization.vacc as vacc

if __name__ == '__main__':
        # NOTE: we assume the labeling is consistent
        # between vacc_df and dist_mat
        # set up population dataframe
        # with vaccination rates

        pop = np.array([1200,1200,1500,2000])
        vacc_rate = np.array([0.9,0.9,0.9,0.9])

        # set up distance matrix
        # all pairwise distances
        dist_mat = np.array([
                [0,1,5,5],
                [1,0,5,5],
                [5,5,0,10],
                [5,5,10,0],
        ])

        # setup the configuration
        # for the disease simulation
        tsir_config = {
            "iters":75,    # number of iterations to run sim for
            "tau1":0.7,    # gravity model parameters
            "tau2":1.2,
            "rho": 0.97,
            "theta": 0.05,
            "alpha": 0.97, # mixing rate
            "beta": 7      # disease infectiousness
        }
        # arguments for optimizer oracle
        sim_params = {
                'config':tsir_config,  # contains all disease parameters
                'pop':pop,
                'distances':dist_mat
        }
        # optimizer oracle configuration
        opt_config = {
            'obj':"attacksize",   # objective function
            'V_repr':"ratio",     # represent vacc rates as a ratio: [0,1]
            'constraint_bnd':0.05 # set c=0.05 (percentage can go down by 5%)
        }
        I = np.array([0,0,0,1])   # seeding: set outbreak to begin in district 4
        # plug all arguments into oracle
        engine = vacc.VaccRateOptEngine(
                opt_config=opt_config,
                V_0=vacc_rate, seed=I,
                sim_config=tsir_config,
                pop=pop,
                distances=np.array(dist_mat))
        # setup for multithreading using 5 threads
        with multiprocess.Pool(5) as p:
            # query the vector where we uniformly distribute the vaccination decrease over all districts
            result, sim_pool = engine.query(V_delta=0.05*np.ones(4),pool=p,n_sim=150, return_sim_pool=True)
            