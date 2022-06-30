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
sat_pop= 150000
#network_df = random_network(n=100)
network_df=grenfell_network(core_pop=core_pop, sat_pop=sat_pop)


# setup time_varying beta
not_term = np.array([1,8,15,16,17,18,19,23,26])-1
term = np.array([x for x in np.arange(0,26) if x not in not_term])
beta_t = np.zeros(26)
beta_t[not_term] = 24.6
beta_t[term]= 33.3
print(beta_t)
config = {
    "iters":36*10,
    
    "tau1":1,
    "tau2":1,
    "rho":1,
    "theta":0.015/core_pop,
    
    "alpha":0.97,
    "beta":24.6,
    "beta_t":beta_t,
    
    "birth": 0.17/26
}

initial_state = np.zeros((len(network_df.index),2))
initial_state[0,0]=1

#%%
# plot grenfell network

plt.scatter(data=network_df,x='x',y='y')

#%%
# run simulation
sim = spatial_tSIR(config,network_df,initial_state)
sim.run_simulation(verbose=True)

#%%

# plot correlation of cities surrounding core against distance

import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
import plotnine
from functools import reduce

#plt.scatter(get_distances(sim.patch_pop)[0,:], sim.correlation_matrix()[0,:])

sns.regplot(get_distances(sim.patch_pop)[0,:], sim.correlation_matrix()[0,:],lowess=True)
plt.show()

series_num = [1] + list(range(10,250,10))

selected_dists = get_distances(sim.patch_pop)[series_num,:]
selected_cors = sim.correlation_matrix()[series_num,:]

series_data = reduce(
       lambda x,y: pd.merge(x,y,how='outer'),
       [pd.DataFrame({"dist":selected_dists[i,:], "cor":selected_cors[i,:], "id":series_num[i]}) 
        for i in range(selected_dists.shape[0])]
)

series_data['dist_sqroot'] = np.sqrt(series_data['dist'])

#series_data = pd.DataFrame({"dist":get_distances(sim.patch_pop)[series_num,:].reshape(,1),
#                            "cor":sim.correlation_matrix()[series_num,:]})
plotnine.options.figure_size=(11,11)
(
 ggplot(series_data,aes(x='dist_sqroot',y='cor')) +
 geom_point(alpha=0.25)+
 geom_smooth(method='lowess',color='red')+
 facet_wrap('id')
)
#%%

# find some way to compute inter-peak times?
# 1. get vector of peak timings (find local maxima of series)
# 2. take differencec of peak timings
# 3. take average to get average inter peak timings?


# nothing particularly conclusive, it seems...
# oh, no - it was the mixing exponent that was screwing stuff up...
# get a weakly increasing trend.
def get_avg_peak_diff(s1,s2,peaks=4):
    s1_peak_ind = s1.sort_values(ascending=False).head(peaks).index
    s2_peak_ind = s2.sort_values(ascending=False).head(peaks).index
    
    s1_peak_ind = np.array(s1_peak_ind)
    s2_peak_ind = np.array(s2_peak_ind)
    
    return np.mean(s2_peak_ind-s1_peak_ind)

ts_matrix = sim.get_ts_matrix()
get_avg_peak_diff(ts_matrix.iloc[:,0], ts_matrix.iloc[:,1])

peak_timings = np.array(
    [get_avg_peak_diff(ts_matrix.iloc[:,0], ts_matrix.iloc[:,k]) for k in range(0,ts_matrix.shape[1])]
)

sns.regplot(x=get_distances(sim.patch_pop)[0,:],y=peak_timings,x_jitter=1)

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