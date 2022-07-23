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
from copy import deepcopy

# argparse - get seed strat,project path, # of cores to use

parser = argparse.ArgumentParser(description="run a bunch of simulations")
parser.add_argument('--seeding_strat',type=str)
parser.add_argument('--project_path',type=str)
parser.add_argument('--cores',type=int)
args = parser.parse_args()

seeding_strat = args.seeding_strat
project_path = args.project_path + "{}" # so format works right
cores = args.cores

project_path.format("scripts/")
sys.path.append(project_path.format("scripts/"))

from spatial_tsir import *

# design parameters - hardcoded in
betas = np.linspace(8,27,6)

inf_to_allocate = np.arange(5,10)
DRAWS = 8000
TIME_RANGE = 100


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

outflow_rank = pd.read_csv(project_path.format("outputs/jul_22_grav_zip_outflow_ranking.csv"))

##### load and setup fixed parameters ####

params = np.load(project_path.format("outputs/log_calib_grav_params_jul22.npy"))

config = {
    "iters":TIME_RANGE,
    "tau1":params[0],
    "tau2":params[1],
    "rho":params[2],
    "theta":params[3],
    "alpha":0.97,
}

##### run simulations ########

vacc_df['ratio'] = vacc_df['nVaccCount']/vacc_df['pop']

for k in inf_to_allocate:
    # keep the randomly selected seed constant
    # across the simulation.
    # help isolate effect of changing beta
    if seeding_strat == "unif":
        selected = vacc_df.sample(k)
    for beta in betas:
        initial_state = np.zeros((len(vacc_df.index),2))
        R = vacc_df['pop'] - vacc_df['nVaccCount']
        if seeding_strat == "pop":
            # one way - again, try seeding top 5 counties
            selected = vacc_df.sort_values(by='pop',ascending=False).head(k)
        elif seeding_strat == "vax_ratio":
            selected = vacc_df.sort_values(by='ratio',ascending=False).head(k)
            # another way - try seeding the top 5 unvaccinated counties (by ratio)
            # another way - try seeding the top 5 unvaccinated counties (by raw numbers)
        elif seeding_strat == "vax_raw":
            selected = vacc_df.sort_values(by='nVaccCount',ascending=False).head(k)
        elif seeding_strat == "unif":
            print(selected)
        elif seeding_strat == "outflux":
            selected = vacc_df[vacc_df['zipcode'].isin(outflow_rank.head(k)['zip'])]
        else:
            print("invalid seeding strategy!")
            sys.exit(-1)

        I = np.zeros(len(vacc_df.index))
        np.put(I,selected.index,1)
        initial_state[:,0] = I
        initial_state[:,1] = R

        this_config= deepcopy(config)
        this_config['beta'] = beta

        save_name = "{}_{}_{}_VA_analysis.save".format(seeding_strat,int(beta),int(k))

        sim_pool = spatial_tSIR_pool(this_config,
                vacc_df,
                initial_state,
                n_sim=DRAWS,
                distances=np.array(dist_mat))
        sim_pool.run_simulation(threads=cores)
        sim_pool.save(project_path.format("outputs/va_sensitivity_analysis/"+save_name))

