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
import torch
import argparse
import os.path
parser = argparse.ArgumentParser(
        prog = "Run optimization"
    )

parser.add_argument('--cfg_dir', required=True)
parser.add_argument('--out_dir', required=True)
parser.add_argument('--name', required=True)
parser.add_argument('--dims',type=int,required=True)

parser.add_argument('--threads', type=int, default=12)
parser.add_argument('--sim_draws', type=int, default=100)
parser.add_argument('--n_init_pts',type=int,default=150)
parser.add_argument('--iters', type=int,default=250)
parser.add_argument('--method', default="lamcts")
parser.add_argument('--load', type=str, default="")

args = parser.parse_args()
print(args)

assert os.path.exists(args.cfg_dir)
assert os.path.exists(args.out_dir)

#config_dir = "config/5_by_5/{}"
config_dir = args.cfg_dir + "{}"

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
dist_mat = dist_list.pivot(index='zipcode1', columns='zipcode2', values='distKM')
dist_mat = dist_mat.fillna(0)

seed = pd.read_csv(config_dir.format(sim_param_config['seed']['seed']),header=None)
seed = np.array(seed).flatten()


torch.set_num_threads(args.threads)
v = vacc.VaccProblemLAMCTSWrapper(
        opt_config = opt_config, 
        V_0= vacc_df['vacc'], 
        seed = seed,
        sim_config = tsir_config, 
        pop = vacc_df, 
        distances = np.array(dist_mat),
        negate=True, scale=True,
        cores=args.threads, n_sim=args.sim_draws,
        output_dir = args.out_dir,
        name=args.name
    )


from ConstrainedLaMCTS.LAMCTS.lamcts import MCTS

P = np.array(vacc_df['pop'])
c = opt_config['constraint_bnd']

if args.method == "lamcts":
    if args.load != "":
        print("hello!")
        agent = MCTS(
                 lb = np.zeros(args.dims),      # the lower bound of each problem dimensions
                 ub = np.ones(args.dims),       # the upper bound of each problem dimensions
                 dims = args.dims,              # the problem dimensions
                 ninits = 0,           # the number of random samples used in initializations 
                 A_ineq = np.array([P]),
                 b_ineq = np.array([c*np.sum(P)]),
                 A_eq = None, b_eq = None,
                 func = v,               # function object to be optimized
                 Cp = 10,              # Cp for MCTS
                 leaf_size = 5, # tree leaf size
                 kernel_type = 'linear', #SVM configruation
                 gamma_type = "auto",    #SVM configruation
                 solver_type = 'turbo',
                 num_threads = args.threads
                 )
        agent.load_agent(load_path=args.load)
        agent.search(iterations=args.iters)
    else:
        print("else branch")
        agent = MCTS(
                 lb = np.zeros(args.dims),      # the lower bound of each problem dimensions
                 ub = np.ones(args.dims),       # the upper bound of each problem dimensions
                 dims = args.dims,              # the problem dimensions
                 ninits = args.n_init_pts,           # the number of random samples used in initializations 
                 A_ineq = np.array([P]),
                 b_ineq = np.array([c*np.sum(P)]),
                 A_eq = None, b_eq = None,
                 func = v,               # function object to be optimized
                 Cp = 10,              # Cp for MCTS
                 leaf_size = 5, # tree leaf size
                 kernel_type = 'linear', #SVM configruation
                 gamma_type = "auto",    #SVM configruation
                 solver_type = 'turbo',
                 num_threads = args.threads
                 )
        agent.search(iterations = args.iters)
    #agent.dump(name=args.name+"mcts_agent", out_dir=args.out_dir)
elif args.method == "bo":
    agent = MCTS(
                 lb = np.zeros(args.dims),      # the lower bound of each problem dimensions
                 ub = np.ones(args.dims),       # the upper bound of each problem dimensions
                 dims = args.dims,              # the problem dimensions
                 ninits = args.n_init_pts,           # the number of random samples used in initializations 
                 A_ineq = np.array([P]),
                 b_ineq = np.array([c*np.sum(P)]),
                 A_eq = None, b_eq = None,
                 func = v,               # function object to be optimized
                 Cp = 10,              # Cp for MCTS
                 leaf_size = np.inf, # tree leaf size
                 kernel_type = 'linear', #SVM configruation
                 gamma_type = "auto",    #SVM configruation
                 solver_type = 'bo',
                 num_threads = args.num_threads
                 )
    agent.search(iterations = args.iters)

# with multiprocess.Pool(15) as pool:
#     bayes_opt.vacc_bayes_opt_w_constr(
#         150,
#         v.engine, 
#         1354-150, pool,
#         noise_prior=1,
#         acq="UCB",
#         tolerance=1e-6, max_iters=1354)
