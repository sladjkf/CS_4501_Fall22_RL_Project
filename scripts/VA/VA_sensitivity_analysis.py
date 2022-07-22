"""
Run a lot of simulations for a simulation experiment..
"""

####### imports and setup design parameters #####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys

# argparse - get seed strat,project path, # of cores to use
seeding_strat = ""
project_path = ""
cores = 0

sys.path.append(project_path.format("scripts/"))
from spatial_tsir import *

# design parameters - hardcoded in
betas = np.linspace(9,27,6)
inf_to_allocate = np.arange(5,10)
draws = 8000
time_range = 100


###### load and clean VA data ######

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

##### load and setup fixed parameters ####

params = np.load(project_path.format("outputs/log_calib_grav_params_jul22.npy"))

config = {
    "iters":time_range,
    "tau1":params[0],
    "tau2":params[1],
    "rho":params[2],
    "theta":params[3],
    "alpha":0.97,
}

##### run simulations ########

vacc_df['ratio'] = vacc_df['nVaccCount']/vacc_df['pop']

for i_to_all in inf_to_allocate:
    for beta in betas:
        initial_state = np.zeros((len(vacc_df.index),2))
        R = vacc_df['pop'] - vacc_df['nVaccCount']
        if seeding_strat == "pop":
            # one way - again, try seeding top 5 counties
            selected = vacc_df.sort_values(by='pop',ascending=False).head(i_to_all)
        if seeding_strat == "vax_ratio":
            selected = vacc_df.sort_values(by='ratio',ascending=False).head(i_to_all)
            # another way - try seeding the top 5 unvaccinated counties (by ratio)
            # another way - try seeding the top 5 unvaccinated counties (by raw numbers)
        if seeding_strat == "vax_raw":
            top_5 = vacc_df.sort_values(by='nVaccCount',ascending=False).head(i_to_all)
        if seeding_strat == "unif":
        if seeding_strat == "outflux":
        I = np.zeros(len(vacc_df.index))
        np.put(I,selected.index,1)
        initial_state[:,0] = I
        initial_state[:,1] = R

        save_name = "{}_".format(seeding_strat,beta,i_to_all)

        sim_pool = ...
        sim_pool.run_simulation()
        sim_pool.save()

