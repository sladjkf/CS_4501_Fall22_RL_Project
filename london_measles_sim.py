#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# load data
#project_path = "/run/media/nick/9D47-AA7C/Summer 2022 (C4GC with BII)/measles_metapop/{}"
project_path = "/run/media/nicholasw/9D47-AA7C/Summer 2022 (C4GC with BII)/measles_metapop/{}"

distances = pd.read_csv(project_path.format("EW_city_distances.csv"),index_col=0)

pop = pd.read_csv(project_path.format("EW_pop_cleaned.csv"),index_col='city')
pop = pop.reindex(distances.index)

cases = pd.read_csv(project_path.format("EW_cases_cleaned.csv"))
#%%
# import module
import sys
#sys.path.append("/run/media/nick/9D47-AA7C/Summer 2022 (C4GC with BII)/measles_metapop/")
sys.path.append("/run/media/nicholasw/9D47-AA7C/Summer 2022 (C4GC with BII)/measles_metapop/")
from spatial_tsir import *

#%%
# simulation configuration 
beta_t = [1.24, 1.14,1.16,1.31,1.24,1.12,1.06,1.02,0.94,0.98,1.06,1.08,
          0.96,0.92,0.92,0.86,0.76,0.63,0.62,0.83,1.13,1.20,1.11,1.02,1.04,1.08]
beta_t = np.array(beta_t)*30


#beta_t = np.array(
#    [1.11,1.11,1.10,1.09,1.06,1.03,1.01,.984,.963,.940,.910,.871,
#     .829,.789,.760,.749,.760,.793,.842,.895,.943,.980,1.01,1.03,1.05,1.08])*1e-5

config = {
    "iters":500,
    
    "tau1":1,
    "tau2":1.5,
    "rho":1,

    "theta":0.07/pop.loc['London','pop'],
    #"theta":1.56e-5,
    
    "alpha":0.97,
    "beta":24.6,
    "beta_t":beta_t,
   
    #"birth": 0.17/26,
    "birth_t":cases['birth_per_cap']
}

initial_state = np.zeros((len(pop.index),2))

# 1. seed the first city with 1 case
if False:
    initial_state[0,0]=1

# 2. try using some fixed immunity rate 
# to decrease the initial massive outbreak?
if False:
    immun_rate = .99
    initial_state[:,1]=np.array(pop['pop'])*immun_rate

    initial_state[:,0] = np.array(
            cases\
                    .iloc[0,:]\
                    .loc[~cases.columns.isin(['start','end','birth_per_cap','Unnamed: 0'])]\
                    .reindex(distances.index)
    )

# 3. everyone is immune at the start of the outbreak,
# except for the designated I cases

# initialize infections
initial_state[:,0] = np.array(
        cases\
                .iloc[0,:]\
                .loc[~cases.columns.isin(['start','end','birth_per_cap','Unnamed: 0'])]\
                .reindex(distances.index)
)

# initialize recovereds
initial_state[:,1]=pop['pop'] - np.int64(initial_state[:,0])
initial_state[:,1]=initial_state[:,1] - initial_state[:,1]*(1-.975)

#%%
sim = spatial_tSIR(config,
        pop,
        initial_state,
        distances=np.array(distances))

sim.run_simulation(verbose=False)
sim.plot_epicurve()
#%%
import copy
from multiprocess import Pool
import time
import random
# Calibration procedure?
# use the two somewhat ad-hoc loss functions
# and then some optimization routine to find the optimal ones?

# TODO: may need multithreading for calibration to actually be efficient
# or precompute?
def OLS_loss(actual,
        sim_params,
        tau1,tau2,rho,theta,immun_rate,
        runs_per_iter=20,cores=4):
    print(tau1,tau2,rho,theta,immun_rate,end=": ")
    """
    actual: dataframe of observed cases
    sim_param: simulation parameters. dictionary
        - config: config file
        - pop: dataframe of populations
        - initial_state: matrix denoting initial state of the simulation
        - distances: distance matrix
    tau1,tau2,rho: calibration parameters
    """
    cfg_copy = copy.deepcopy(sim_params['config'])

    cfg_copy['tau1'] = tau1
    cfg_copy['tau2'] = tau2
    cfg_copy['rho'] = rho
    cfg_copy['theta'] = theta


    pop, distances = sim_params['pop'], sim_params['distances']

    # setup initial state
    initial_state = np.zeros((len(pop.index),2))
    # initialize infections
    initial_state[:,0] = np.array(actual.iloc[0,:])
    # initialize recovereds
    initial_state[:,1]=pop['pop'] - np.int64(initial_state[:,0])
    initial_state[:,1]=initial_state[:,1] - np.int64(initial_state[:,1]*(1-immun_rate))

    # serial method
    #runs = []
    #for i in range(0,runs_per_iter):
    #    sim = spatial_tSIR(config,
    #            pop,
    #            initial_state,
    #            distances)
    #    sim.run_simulation(verbose=False)
    #    result = sim.get_ts_matrix()
    #    runs.append(result)

    # parallelize
    p = Pool(cores)
    def run_sim(x):
        np.random.seed(int(time.time()) + x*13)
        sim = spatial_tSIR(config,
                pop,
                initial_state,
                distances)
        sim.run_simulation(verbose=False)
        return sim.get_ts_matrix()
    runs = p.map_async(run_sim,range(0,runs_per_iter)).get()
    p.terminate()
    
    scores = []
    avg_score = 0
    for run in runs:
        # go column by column (over each patch)
        this_run_score = 0
        for k in range(0,actual.shape[1]):
            observed_y = actual.iloc[:,k]
            sim_y = run.iloc[:,k]
            # filter nonzero
            observed_y_nonzero = observed_y[observed_y > 0]
            sim_y_nonzero = sim_y.iloc[observed_y_nonzero.index]
            # how are they not the same length lmao
            #assert len(observed_y_nonzero) == len(sim_y_nonzero), "{},{}".format(len(observed_y_nonzero),len(sim_y_nonzero))
            this_length = len(observed_y_nonzero.index)
            this_pop = pop.iloc[k,0]
            this_run_score += sum((observed_y_nonzero - sim_y_nonzero)**2)/(this_length*this_pop)
            #print("iter {}".format(k),this_run_score)
            if np.isnan(this_run_score):
                print((np.isnan(observed_y_nonzero - sim_y_nonzero).any()))
                diff = observed_y_nonzero - sim_y_nonzero
                print(diff[np.isnan(diff)])
                nan_els = diff[np.isnan(diff)]
                fnanind = nan_els.index[0]
                print(observed_y_nonzero.iloc[fnanind],sim_y_nonzero.iloc[fnanind])
                print((observed_y_nonzero.iloc[fnanind]-sim_y_nonzero.iloc[fnanind])**2)

                print(observed_y_nonzero.iloc[nan_els.index])
                print(sim_y_nonzero.iloc[nan_els.index])

                print(observed_y_nonzero.iloc[nan_els.index] - sim_y_nonzero.iloc[nan_els.index])
                return
                #print("sim: {},obs: {}".format(
                #    np.isnan(sim_y_nonzero).any(), 
                #    np.isnan(observed_y_nonzero).any()
                #    ))

        #print(this_run_score)
        scores.append(this_run_score)
        avg_score += this_run_score
    avg_score = avg_score/len(runs)
    to_return = (avg_score,np.std(scores)/np.sqrt(runs_per_iter))
    print(to_return)
    return to_return

OLS_loss(actual,sim_params,1.5,1,1,.07/pop.loc['London','pop'],.6)

#%%

from scipy.optimize import minimize

def OLS_loss_partial(x):
    return OLS_loss(actual,sim_params,x[0],x[1],x[2],x[3],x[4])[0]

minimize(
        OLS_loss_partial,
        np.array([1,1.5,1,0.015/1000,.6]),
        bounds = [
            (0,None),
            (0,None),
            (0,None),
            (0,None),
            (0,1)
            ]
        )

#%% 
# testing the above function... first 400 timesteps?
# need to make sure columns and rows, etc. match
actual = cases\
        .loc[:,~cases.columns.isin(['start','end','birth_per_cap','Unnamed: 0'])]\
        .iloc[0:400,:]\
        .loc[:,pop.index]

sim_params = {}
sim_params['config'], sim_params['pop'], sim_params['initial_state'], sim_params['distances'] =\
        config, pop, initial_state, np.array(distances)

#%%
def corr_loss(actual,config,theta):
