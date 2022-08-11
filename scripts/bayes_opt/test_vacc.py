"""
test the vaccination objective function routine.
"""

# Standard imports
import sys
project_path = "/home/nick/Documents/4tb_sync/UVA GDrive/Summer 2022 (C4GC with BII)/measles_metapop/{}"
sys.path.append(project_path.format("scripts/bayes_opt/"))
import vacc
import numpy as np
import pandas as pd

#%%
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

#%%

#tSIR parameters
params = np.load(project_path.format("outputs/log_calib_grav_params_jul22.npy"))
tsir_config = {
    "iters":75,
    "tau1":params[0],
    "tau2":params[1],
    "rho":params[2],
    "theta":params[3],
    "alpha":0.97,
    "beta":3
}
sim_params = {
        'config':tsir_config,
        'pop':vacc_df,
        'distances':np.array(dist_mat)
}

#%%
top_5 = vacc_df.sort_values(by='pop',ascending=False).head(5)
I = np.zeros(len(vacc_df.index))
np.put(I,top_5.index,1)
# optimization parameters
opt_config = {
    'obj':"attacksize",
    'V_repr':"max_ratio",
    'constraint_bnd':0.05,
    "attacksize_cutoff":1000
}

#%%
import multiprocess

V_0 = (vacc_df['pop']-vacc_df['nVaccCount'])/(vacc_df['pop'])
V_0 = (vacc_df['pop']-vacc_df['nVaccCount'])/(max(vacc_df['pop']))
engine = vacc.VaccRateOptEngine(
        opt_config=opt_config,
        V_0=V_0, seed=I,
        sim_config=tsir_config,
        pop=vacc_df,
        distances=np.array(dist_mat))
V_prime = engine.V_0.copy()
V_prime[512] = V_prime[512]-0.8

with multiprocess.Pool(12) as p:
    engine.query(engine.V_0,pool=p,n_sim=50)
    result = engine.query(V_prime,pool=p,n_sim=50)
