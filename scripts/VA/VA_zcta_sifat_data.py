# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 11:40:49 2022

@author: nrw5cq


"""
#%% standard imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# windows machine at the office
project_path = "D:/Summer 2022 (C4GC with BII)/measles_metapop/{}"
import sys
sys.path.append(project_path.format("scripts/"))

from spatial_tsir import *

#%% load data, filter and align indices

# vaccination data and population data from sifat
vacc_df = pd.read_csv(project_path.format("data/VA_zipcodes_cleaned/ZC_immunization_sifat.csv"))
# drop 0 population entries, they won't affect the simulation
vacc_df = vacc_df[vacc_df['population'] > 0].reset_index(drop=True)
vacc_df.rename({'population':'pop'},axis=1,inplace=True)

# load distance matrix computed from nominatim and geopy distance function
dist_df = pd.read_csv(project_path.format("data/VA_zipcodes_cleaned/ZC_distance_sifat_nom_geopy.csv"))
# need to replace 0's in distance matrix to avoid divide by zero in gravity formula
default_dist = 0.5
dist_df.loc[dist_df[np.isclose(dist_df['distKM'],0)].index,'distKM']=default_dist
# convert to matrix
dist_mat = dist_df.pivot(index='zipcode1',columns='zipcode2',values='distKM')
dist_mat = dist_mat.replace(np.nan,0)

# align matrix
dist_mat = dist_mat.loc[vacc_df['zipcode'],vacc_df['zipcode']]

#%% simulation configuration

# reasonable to start with constant beta/contact rate

# retrieved from "A Gravity Model for Epidemic Metapopulations"
beta_t = [1.24, 1.14,1.16,1.31,1.24,1.12,1.06,1.02,0.94,0.98,1.06,1.08,
          0.96,0.92,0.92,0.86,0.76,0.63,0.62,0.83,1.13,1.20,1.11,1.02,1.04,1.08]
beta_t = np.array(beta_t)*30

# birth rate?
# https://www.marchofdimes.org/peristats/data?reg=99&top=2&stop=1&lev=1&slev=4&obj=1&sreg=51
birth_rate = 60/1000/28

config = {
    "iters":360,
    
    "tau1":1,
    "tau2":1.4,
    "rho":1,
    "theta":0.015/max(vacc_df['pop']),
    "alpha":0.97,
    "beta":24.6
    #"birth":birth_rate
    #"beta_t":beta_t
    #"birth_t":cases['birth_per_cap']
}

#%% setup initial state

initial_state = np.zeros((len(vacc_df.index),2))

R = vacc_df['pop'] - vacc_df['nVaccCount']

# one way - again, try seeding top 5 counties
top_5 = vacc_df.sort_values(by='pop',ascending=False).head(5)
# another way - try seeding the top 5 unvaccinated counties (by raw numbers)
top_5 = vacc_df.sort_values(by='nVaccCount',ascending=False).head(5)
# another way - try seeding the top 5 unvaccinated counties (by ratio)
vacc_df['ratio'] = vacc_df['nVaccCount']/vacc_df['pop']
top_5 = vacc_df.sort_values(by='ratio',ascending=False).head(5)

I = np.zeros(len(vacc_df.index))
np.put(I,top_5.index,1)

initial_state[:,0] = I
initial_state[:,1] = R

#%% singlethread simulation
sim = spatial_tSIR(config,
        vacc_df,
        initial_state,
        distances=np.array(dist_mat))

sim.run_simulation()

#%% 
