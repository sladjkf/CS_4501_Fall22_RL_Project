"""
load configuration file and initialize the function
"""
import configparser
import pandas as pd
import numpy as np
from scripts.optimization import vacc
#from scripts.optimization import vacc_bayes_opt
from ConstrainedLaMCTS.LAMCTS import bayes_opt
import multiprocess

config_dir = "config/5_by_5/{}"

sim_param_config = configparser.ConfigParser()
sim_param_config.optionxform = str  # retain case sensitivity

# load simulation files
with open(config_dir.format("params.cfg")) as config_file:
    sim_param_config.read_file(config_file)
    tsir_config_str = dict(sim_param_config['tsir_config'])
    tsir_config = {}
    for k, v in tsir_config_str.items():
        if k == 'iters':
            tsir_config['iters'] = int(tsir_config_str['iters'])
        else:
            tsir_config[k] = float(tsir_config_str[k])
    opt_config = dict(sim_param_config['opt_config'])
    opt_config['constraint_bnd'] = float(opt_config['constraint_bnd'])

# load vaccination/population data and distance matrix
vacc_df = pd.read_csv(config_dir.format("pop.csv"))
dist_list = pd.read_csv(config_dir.format("dist.csv"))
dist_mat = dist_list.pivot(index='0', columns='1', values='2')

seed = pd.read_csv(config_dir.format(sim_param_config['seed']['seed']),header=None)
seed = np.array(seed).flatten()

v = vacc.VaccProblemLAMCTSWrapper(
        opt_config = opt_config, 
        V_0= vacc_df['vacc'], 
        seed = seed,
        sim_config = tsir_config, 
        pop = vacc_df, 
        distances = np.array(dist_mat),
        negate=True, scale=True,
        cores=12, n_sim=100,
        output_dir = "/home/nick/Documents/4tb_sync/UVA GDrive/Semesters/Semester 7/CS 4501/project/CS_4501_Fall22_RL_Project/output/5by5/",
        name="5by5"
    )


from ConstrainedLaMCTS.LAMCTS.lamcts import MCTS

P = np.array(vacc_df['pop'])
c = opt_config['constraint_bnd']

# agent = MCTS(
#              lb = np.zeros(25),      # the lower bound of each problem dimensions
#              ub = np.ones(25),       # the upper bound of each problem dimensions
#              dims = 25,              # the problem dimensions
#              ninits = 150,           # the number of random samples used in initializations 
#              A_ineq = np.array([P]),
#              b_ineq = np.array([c*np.sum(P)]),
#              A_eq = None, b_eq = None,
#              func = v,               # function object to be optimized
#              Cp = 10,              # Cp for MCTS
#              leaf_size = 5, # tree leaf size
#              kernel_type = 'linear', #SVM configruation
#              gamma_type = "auto",    #SVM configruation
#              solver_type = 'turbo'
#              )

# agent.search(iterations = 200)

agent = MCTS(
             lb = np.zeros(25),      # the lower bound of each problem dimensions
             ub = np.ones(25),       # the upper bound of each problem dimensions
             dims = 25,              # the problem dimensions
             ninits = 150,           # the number of random samples used in initializations 
             A_ineq = np.array([P]),
             b_ineq = np.array([c*np.sum(P)]),
             A_eq = None, b_eq = None,
             func = v,               # function object to be optimized
             Cp = 10,              # Cp for MCTS
             leaf_size = np.inf, # tree leaf size
             kernel_type = 'linear', #SVM configruation
             gamma_type = "auto",    #SVM configruation
             solver_type = 'bo'
             )

agent.search(iterations = 1354)

# with multiprocess.Pool(15) as pool:
#     bayes_opt.vacc_bayes_opt_w_constr(
#         150,
#         v.engine, 
#         1354-150, pool,
#         noise_prior=1,
#         acq="UCB",
#         tolerance=1e-6, max_iters=1354)