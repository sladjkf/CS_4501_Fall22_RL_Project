#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# load data

# change this based on where you have stored the project
project_path = "/run/media/nick/9D47-AA7C/Summer 2022 (C4GC with BII)/measles_metapop/{}"

distances = pd.read_csv(project_path.format("data/EW_measles/EW_city_distances.csv"),index_col=0)

pop = pd.read_csv(project_path.format("data/EW_measles/EW_pop_cleaned.csv"),index_col='city')
pop = pop.reindex(distances.index)

cases = pd.read_csv(project_path.format("data/EW_measles/EW_cases_cleaned.csv"))

#%%
# import module
import sys
sys.path.append(project_path.format("scripts/EW_measles/"))
from spatial_tsir import *

#%%
# simulation configuration 

# retrieved from "A Gravity Model for Epidemic Metapopulations"
beta_t = [1.24, 1.14,1.16,1.31,1.24,1.12,1.06,1.02,0.94,0.98,1.06,1.08,
          0.96,0.92,0.92,0.86,0.76,0.63,0.62,0.83,1.13,1.20,1.11,1.02,1.04,1.08]
beta_t = np.array(beta_t)*30


#beta_t = np.array(
#    [1.11,1.11,1.10,1.09,1.06,1.03,1.01,.984,.963,.940,.910,.871,
#     .829,.789,.760,.749,.760,.793,.842,.895,.943,.980,1.01,1.03,1.05,1.08])*1e-5

config = {
    "iters":599,
    
    "tau1":1,
    "tau2":1.4,
    "rho":1,
    "theta":0.015/pop.loc['London','pop'],

    # parameters acquired through DE optimization routine
    #"tau1":0.48265995,
    #"tau2":0.78866368,
    #"rho":0.01060361,
    #"theta":0.06251831,

    "alpha":0.97,
    "beta":24.6,
    "beta_t":beta_t,
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

immun_rate=.9621218
# initialize infections
initial_state[:,0] = np.array(
        cases\
                .iloc[0,:]\
                .loc[~cases.columns.isin(['start','end','birth_per_cap','Unnamed: 0'])]\
                .reindex(distances.index)
)

# initialize recovereds
initial_state[:,1]=pop['pop'] - np.int64(initial_state[:,0])
initial_state[:,1]=initial_state[:,1] - initial_state[:,1]*(1-immun_rate)

#%%
sim = spatial_tSIR(config,
        pop,
        initial_state,
        distances=np.array(distances)*.3)

sim.run_simulation(verbose=False)
sim.plot_epicurve()

#%%
# testing simulations and loss functions
# let's say, first 400 time steps
# need to make sure columns and rows, etc. match
actual = cases\
        .loc[:,~cases.columns.isin(['start','end','birth_per_cap','Unnamed: 0'])]\
        .iloc[0:600,:]\
        .loc[:,pop.index]

sim_params = {}
sim_params['config'], sim_params['pop'], sim_params['initial_state'], sim_params['distances'] =\
        config, pop, initial_state, np.array(distances)

#%%

# plot each simulated time series against the actual
# do a bunch of stupid data manipulations

# first, full nonnormalized data
sim_to_merge = sim.get_ts_matrix()
sim_to_merge.columns = actual.columns

actual_to_merge = actual.copy(deep=True)

actual_to_merge['time'] = actual_to_merge.index
sim_to_merge['time'] = sim_to_merge.index

# melt the dataframes to get them into long format
sim_to_merge = pd.melt(sim_to_merge,id_vars='time',value_vars=list(distances.index))
actual_to_merge = pd.melt(actual_to_merge,id_vars='time',value_vars=list(distances.index))

# add "source"
sim_to_merge['source'] = "sim"
actual_to_merge['source'] = "obs"

# cast to int to suppress warnings
sim_to_merge['value'] = np.int64(sim_to_merge['value'])
actual_to_merge['value'] = np.int64(actual_to_merge['value'])

full = sim_to_merge.merge(actual_to_merge,how='outer')

# rename columns
full = full.rename(columns = {'variable':'city', 'value':'cases'})

#%%

# normalized data
sim_tm_norm= sim.get_ts_matrix()
sim_tm_norm.columns = actual.columns
sim_tm_norm = sim_tm_norm/sim_tm_norm.max()

actual_tm_norm = actual.copy(deep=True)
actual_tm_norm = actual_tm_norm/actual_tm_norm.max()

actual_tm_norm['time'] = actual_tm_norm.index
sim_tm_norm['time'] = sim_tm_norm.index

# same steps..
# melt the dataframes to get them into long format
sim_tm_norm = pd.melt(sim_tm_norm,id_vars='time',value_vars=list(distances.index))
actual_tm_norm = pd.melt(actual_tm_norm,id_vars='time',value_vars=list(distances.index))

# add "source"
sim_tm_norm['source'] = "sim"
actual_tm_norm['source'] = "obs"

full_norm = sim_tm_norm.merge(actual_tm_norm,how='outer')

#%%

# non normalized plot: how does the scale compare?
from plotnine import *
from plotnine.themes import element_text
(ggplot(full,aes(x='time',y='cases',color='source'))+
        geom_line()+
        facet_wrap('~city', scales='free_y')+
        ggtitle("E&W 7 cities: tau1={:.2f},tau2={:.2f},rho={:.2f},theta={:.2e},immun_rate={:.2f}".format(
            config['tau1'],config['tau2'], config['rho'], config['theta'], immun_rate
            )
        )+
        theme(
            text=element_text(size=13),
            subplots_adjust={'wspace':0.20}
        )
)

# normalized plot: how does the synchrony compare?

(ggplot(full_norm,aes(x='time',y='value',color='source'))+
        geom_line()+
        facet_wrap('~variable'))

#%%

# more diagnostic
# normalized data
sim_norm= sim.get_ts_matrix()
sim_norm.columns = actual.columns
sim_norm = sim_norm/sim_norm.max()

actual_norm = actual/actual.max()

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
        beta_scaling=1,
        distance_scaling=1,
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
    cfg_copy['beta_t'] = cfg_copy['beta_t']*beta_scaling


    pop, distances = sim_params['pop'], sim_params['distances']*distance_scaling

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
            # how are they not the same length lol
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
        np.array([1,1.5,1,0.015/1000,.975]),
        bounds = [
            (0,None),
            (0,None),
            (0,None),
            (0,None),
            (0,1)
            ]
        )


#%%
def intra_corr_loss(actual,
        sim_params,
        tau1,tau2,rho,theta,immun_rate,
        beta_scaling=1,
        runs_per_iter=20,cores=4):
    """
    describes correlations of "satellite cities" with "core city"
    """
    cfg_copy = copy.deepcopy(sim_params['config'])

    cfg_copy['tau1'] = tau1
    cfg_copy['tau2'] = tau2
    cfg_copy['rho'] = rho
    cfg_copy['theta'] = theta
    cfg_copy['beta_t'] = cfg_copy['beta_t']*beta_scaling

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

    # TODO: should I filter out zero observations?
    corr_observed = np.corrcoef(actual.T)

    scores = []
    for run in runs:
        # compute correlations to the city in column 0
        # which is assumed to be the core city?
        corr_matrix = np.corrcoef(run.T)

        # 0th column and 0th row are all the same...
        abs_diff_vec = np.abs(corr_observed[:,0] - corr_matrix[:,0])
        scores.append((1/len(abs_diff_vec))*sum(abs_diff_vec))

    to_return = (np.mean(scores),np.std(scores)/np.sqrt(runs_per_iter))
    print(to_return)
    return to_return

#%%
def corr_with_actual(actual,
        sim_params,
        tau1,tau2,rho,theta,immun_rate,
        beta_scaling=1,
        runs_per_iter=50,cores=5):
    """
    describe correlation of the simulation data with that of the observed data
    """
    print(tau1,tau2,rho,theta,immun_rate,end=": ")
    # setup parameters and run simulations
    cfg_copy = copy.deepcopy(sim_params['config'])

    cfg_copy['tau1'] = tau1
    cfg_copy['tau2'] = tau2
    cfg_copy['rho'] = rho
    cfg_copy['theta'] = theta
    cfg_copy['beta_t'] = cfg_copy['beta_t']*beta_scaling

    pop, distances = sim_params['pop'], sim_params['distances']

    # setup initial state
    initial_state = np.zeros((len(pop.index),2))
    # initialize infections
    initial_state[:,0] = np.array(actual.iloc[0,:])
    # initialize recovereds
    initial_state[:,1]=pop['pop'] - np.int64(initial_state[:,0])
    initial_state[:,1]=initial_state[:,1] - np.int64(initial_state[:,1]*(1-immun_rate))

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
    for run in runs:
        if len(run.index) != len(actual.index):
            print("dataframe lengths do not match, sim is {} and actual is {}".format(len(run.index),len(actual.index)))
            return
        corrs = [np.corrcoef(run.iloc[:,k], actual.iloc[:,k])[0,1] for k in range(0,len(run.columns))]
        scores.append(np.mean(corrs))
        
    to_return = (np.mean(scores),np.std(scores),np.std(scores)/np.sqrt(runs_per_iter))
    print(to_return)
    return to_return

#%%
tau1 = np.linspace(0,2,6)
tau2 = np.linspace(0,2,6)
rho = np.linspace(0,2,6)
theta = np.linspace(0,0.1,6)/pop.loc['London','pop']
#vax_rate = np.linspace(.7,.99,6)

grid = pd.DataFrame(
        [(x0,x1,x2,x3) for x0 in tau1 for x1 in tau2 for x2 in rho for x3 in theta]
        )
print(grid)
grid.apply(func=lambda row: corr_loss(actual,sim_params,row[0],row[1],row[2],row[3],.975), axis=1)

#%%

from scipy.optimize import minimize
from scipy.optimize import differential_evolution
def corr_loss_partial(x):
    return 1-corr_with_actual(actual,sim_params,x[0],x[1],x[2],x[3],x[4])[0]

#minimize(
#        lambda x: 1-corr_loss_partial(x),
#        np.array([1,1.5,1,0.015/1000,.975]),
#        bounds = [
#            (0,None),
#            (0,None),
#            (0,None),
#            (0,None),
#            (0,1)
#            ]
#        )

differential_evolution(
        func = corr_loss_partial,
        #np.array([1,1.5,1,0.015/1000,.975]),
        bounds = [(0,3),
            (0,3),
            (0,3),
            (0,1),
            (0,1)]
        )




#%%
#[corr_loss(actual,sim_params,1,1.5,1,x/pop.loc['London','pop'],.975) for x in np.linspace(0,0.06,10)]
#[corr_loss(actual,sim_params,1,1.5,1,0.01/pop.loc['London','pop'],.975,beta_scaling=x) for x in np.linspace(.5,1.5,10)]


[OLS_loss(actual,sim_params,1,1.5,1,0.01/pop.loc['London','pop'],.975,beta_scaling=x) for x in np.linspace(.5,1.5,10)]
