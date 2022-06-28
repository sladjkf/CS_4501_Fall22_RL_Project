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
#from gravity import *
#%%

core_pop = 5000000
sat_pop= 100000
#network_df = random_network(n=100)
network_df=grenfell_network(core_pop=core_pop, sat_pop=sat_pop)
config = {
    "iters":36*10,
    
    "tau1":1,
    "tau2":1,
    "rho":1,
    "theta":0.01/core_pop,
    
    "alpha":0.90,
    "beta":24.6,
    
    "birth": 0.17/36
}

initial_state = np.zeros((len(network_df.index),2))
initial_state[10,0]=100

#%%

sim = spatial_tSIR(config,network_df,initial_state)
sim.run_simulation(verbose=False)

#%%

# plot correlation against distance?

import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
import plotnine
from functools import reduce

plt.scatter(get_distances(sim.patch_pop)[0,:], sim.correlation_matrix()[0,:])
plt.show()
#sns.regplot(get_distances(sim.patch_pop)[50,:], sim.correlation_matrix()[50,:],lowess=True)


series_num = [1] + list(range(10,250,10))

selected_dists = get_distances(sim.patch_pop)[series_num,:]
selected_cors = sim.correlation_matrix()[series_num,:]

series_data = reduce(
       lambda x,y: pd.merge(x,y,how='outer'),
       [pd.DataFrame({"dist":selected_dists[i,:], "cor":selected_cors[i,:], "id":series_num[i]}) 
        for i in range(selected_dists.shape[0])]
)


#series_data = pd.DataFrame({"dist":get_distances(sim.patch_pop)[series_num,:].reshape(,1),
#                            "cor":sim.correlation_matrix()[series_num,:]})
plotnine.options.figure_size=(11,11)
(
 ggplot(series_data,aes(x='dist',y='cor')) +
 geom_point(alpha=0.25)+
 geom_smooth(method='lowess',color='red')+
 facet_wrap('id')
)

#%%
ts_matrix = sim.get_ts_matrix()
cross_correlations = [np.correlate(
    ts_matrix.iloc[:,0],
    ts_matrix.iloc[:,k])[0] for k in range(1,250)]

print(len(cross_correlations))
print(len(get_distances(sim.patch_pop)[1:,:]))
plt.show()
plt.scatter(get_distances(sim.patch_pop)[1:,:], cross_correlations)

#%%

# sort network_df by distances to core?

network_df['distances'] = np.sqrt(network_df['x']**2 + network_df['y']**2)
network_df.sort_values(by='distances').head(25)

network_df[network_df['distances'] <= 51]