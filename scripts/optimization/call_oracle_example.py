"""
Example file demonstrating how to setup and call the oracle
with a very simple example problem instance.
"""
import numpy as np
import pandas as pd
import scripts.optimization.vacc as vacc
import multiprocess
from pathlib import Path
import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import scripts.optimization.vacc as vacc

# setup for multithreading using 5 threads
with multiprocess.Pool(5) as p:
    # query the vector where we uniformly distribute the vaccination decrease over all districts
    result, sim_pool = engine.query(V_delta=0.05*np.ones(4),pool=p,n_sim=150, return_sim_pool=True)
print(result)
sim_pool.plot_paths()
#PYTHONPATH=$(pwd) python3 scripts/optimization/call_oracle_example.py
