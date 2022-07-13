#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
project_path = "/run/media/nick/9D47-AA7C/Summer 2022 (C4GC with BII)/measles_metapop/{}"

sys.path.append(project_path.format("scripts"))
#sys.path.append("/run/media/nicholasw/9D47-AA7C/Summer 2022 (C4GC with BII)/measles_metapop/")
from spatial_tsir import *
#%%
core_pop = 5000000
sat_pop= 100000
#network_df = random_network(n=100)
network_df=grenfell_network(core_pop=core_pop, sat_pop=sat_pop)
distances = get_distances(network_df)
#%%

infected = np.ones(len(network_df.index))
params = {
        "tau1":1,
        "tau2":1,
        "rho":1,
        "theta":1e-6
}

gravity(network_df, distances, infected, params, parallel=False)
