#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 12:11:01 2022

@author: nicholasw
"""

#%%

# imports
import numpy as np
import sys

sys.path.append("/run/media/nicholasw/9D47-AA7C/Summer 2022 (C4GC with BII)/measles_metapop/")
from gravity import *
#%%
network_df = random_network(n=100)
network_df=grenfell_network()
config = {
    "iters":10,
    
    "tau1":1,
    "tau2":1,
    "rho":1,
    "theta":0.1/5000000,
    
    "alpha":0.97,
    "beta":22,
    
    "birth": 0.17/36
}

initial_state = np.zeros((len(network_df.index),2))
initial_state[0,0]=1

#%%

sim = spatial_tSIR(config,network_df,initial_state)
sim.run_simulation(verbose=True)

#%%
