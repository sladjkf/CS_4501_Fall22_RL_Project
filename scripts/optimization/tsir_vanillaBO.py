"""
Run a standard Bayesian optimization on the tSIR model
for the vaccination problem.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import multiprocess

from optimization import BayesianOptimization

import scripts.optimization.vacc as vacc

#def test(q):
#    return q@np.array([1,2,3,4,5])
def attacksize_mean(engine, pool, n_sim, q):
    """
    Small wrapper for optimization oracle
    needed for minimization routine
    Use functools partial to get a single function out
    """
    return np.mean(engine.query(V_prime=q, pool=pool, n_sim=n_sim))

# NOTE: BayesianOptimization library doesn't take arraylike
# We need to make a function with 705 arguments!

#### load data ###
with open("project_dir.txt") as f:
    project_path = f.read().strip() + "{}"
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

### tSIR parameters ###
params = np.load(project_path.format("outputs/log_calib_grav_params_jul22.npy"))
tsir_config = {
    "iters":100,
    "tau1":params[0],
    "tau2":params[1],
    "rho":params[2],
    "theta":params[3],
    "alpha":0.97,
    "beta":7
}
sim_params = {
        'config':tsir_config,
        'pop':vacc_df,
        'distances':np.array(dist_mat)
}

### initialize seed (population) ###
top_5 = vacc_df.sort_values(by='pop',ascending=False).head(5)
I = np.zeros(len(vacc_df.index))
np.put(I,top_5.index,1)
# optimization parameters
opt_config = {
    'obj':"attacksize",
    'V_repr':"ratio",
    'constraint_bnd':0.05,
    "attacksize_cutoff":1000
}
V_0 = (vacc_df['pop']-vacc_df['nVaccCount'])/(vacc_df['pop'])
engine = vacc.VaccRateOptEngine(
        opt_config=opt_config,
        V_0=V_0, seed=I,
        sim_config=tsir_config,
        pop=vacc_df,
        distances=np.array(dist_mat))

### setup bayesian optimization ###
def convert_ub_lb_vec_to_dict(lb,ub):
    """
    Convert a pair of upper-lower bound vectors into a dictionary
    Use the convention "x1,x2,x3,..."
    """
    pbounds_dict = {}
    for i,pair in enumerate(zip(lb,ub)):
        lb_i,ub_i = pair
        pbounds_dict["x{}".format(i)] = (lb_i,ub_i)
    return pbounds_dict

def turn_vector_fn_to_kw_fn(func,dim):
    """
    The BayesianOptimization library expects a function with multiple (keyword) arguments
    rather than a function with a single vector-valued argument.
    So, this is a helper function to transform a vector-valued function to a keyword function.
    """
    def fn(**kwargs):
        dim = len(kwargs)
        arg = np.zeros(dim)
        for i, pair in enumerate(kwargs.items()):
            key,value = pair
            arg[i] = value
        return func(arg)
    return fn

with multiprocess.Pool(12) as p:
    objective_vector = lambda x: attacksize_mean(q=x, engine=engine, pool=p, n_sim=50)
    objective = turn_vector_fn_to_kw_fn(objective_vector,len(V_0))
    ub = np.array(V_0)
    lb = np.array(V_0 - 0.08)
    pbounds = convert_ub_lb_vec_to_dict(lb,ub)
    optimizer = BayesianOptimization(
            f=objective,
            pbounds=pbounds,
            verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1)
    optimizer.maximize(
            init_points=15,
            n_iter=10,
            acq='ei'
        )
    print(optimizer.max)
    #except:
    #    print(engine.eval_history)
    #    print(np.mean(engine.eval_history['output'],axis=1))


