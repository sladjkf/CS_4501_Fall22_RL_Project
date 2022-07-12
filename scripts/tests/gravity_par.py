#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
core_pop = 5000000
sat_pop= 100000
#network_df = random_network(n=100)
network_df=grenfell_network(core_pop=core_pop, sat_pop=sat_pop)

distances = get_distances(network_df)
