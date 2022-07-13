"""
Simulate the tSIR model using the VA ZCTA spatial resolution.
"""
#%% standard imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#project_path = "/d/Summer 2022 (C4GC with BII)/measles_metapop/"
project_path = "D:/Summer 2022 (C4GC with BII)/measles_metapop/{}"

import sys
sys.path.append(project_path.format("scripts/"))

from spatial_tsir import *

#%% load data

va_zcta_distances = pd.read_csv(project_path.format("data/VA_zipcodes_cleaned/VA_zipcodes_dist_haversine.csv"),index_col=0)
va_zcta_pop = pd.read_csv(project_path.format("data/VA_zipcodes_cleaned/VA_zcta_pop.csv")).dropna()

# cast columns
va_zcta_pop['pop'] = va_zcta_pop['pop'].astype('int')
va_zcta_pop['patch_id'] = va_zcta_pop['patch_id'].astype('int').astype('str')

va_zcta_distances.index = va_zcta_distances.index.astype('str')
va_zcta_distances.columns = va_zcta_distances.columns.astype('str')

#%% quick data validation

# check that ordering of the population vector and distance matrix match
assert all(va_zcta_pop['patch_id'].astype('int') == va_zcta_distances.index.astype('int'))
assert all(va_zcta_pop['patch_id'].astype('int') == va_zcta_distances.columns.astype('int'))

#%% data validation (zeros in distance matrix)
# find entries in the distance matrix with 0 distances

# these seem to arise from very close zip codes
# or zip codes where one is contained within another.
# just replace with half km?

# done for haversine distance matrix
more_than_1_zero = va_zcta_distances[np.sum(va_zcta_distances == 0,axis=1)>1].index
to_assign = 0.5
for code in more_than_1_zero:
    row = va_zcta_distances.loc[code,:]
    print(row[row == 0])
    link = row[row==0]
va_zcta_distances.loc['22211','22214'] = to_assign
va_zcta_distances.loc['22214','22211'] = to_assign
va_zcta_distances.loc['22801','22807'] = to_assign
va_zcta_distances.loc['22807','22801'] = to_assign


#%% setup simulation (config)

# reasonable to start with constant beta/contact rate

# retrieved from "A Gravity Model for Epidemic Metapopulations"
beta_t = [1.24, 1.14,1.16,1.31,1.24,1.12,1.06,1.02,0.94,0.98,1.06,1.08,
          0.96,0.92,0.92,0.86,0.76,0.63,0.62,0.83,1.13,1.20,1.11,1.02,1.04,1.08]
beta_t = np.array(beta_t)*30

# birth rate?
# https://www.marchofdimes.org/peristats/data?reg=99&top=2&stop=1&lev=1&slev=4&obj=1&sreg=51
birth_rate = 60/1000/28

config = {
    "iters":60,
    
    "tau1":1,
    "tau2":1.4,
    "rho":1,
    "theta":0.015/max(va_zcta_pop['pop']),
    "alpha":0.97,
    "beta":24.6
    #"birth":birth_rate,
    #"beta_t":beta_t
    #"birth_t":cases['birth_per_cap']
}

#%% setup simulation (initial state)
initial_state = np.zeros((len(va_zcta_pop.index),2))
# in lieu of more detailed local immunization data, initialize with fixed immunity rate?
immun_rate = 0.975

# initialize immune/recovered
R = np.int64(np.floor(va_zcta_pop['pop']*immun_rate))

# initialize infected? (let's say top 5)
top_5_pop = va_zcta_pop.sort_values(by='pop').tail()
I = np.array(va_zcta_pop['patch_id'].isin(top_5_pop['patch_id']).astype(int))

initial_state[:,0] = I
initial_state[:,1] = R

#%%
sim = spatial_tSIR(config,
        va_zcta_pop,
        initial_state,
        distances=np.array(va_zcta_distances))

sim.run_simulation(verbose=False)
#%%
sim.plot_epicurve()
sim.plot_epicurve(total=True)
