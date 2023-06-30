"""
This script is used for running the vaccination scenario optimization problems.
Optimization results are outputted in the specified directory (see below) and two files are created:
- '..best_trace.csv', which stores the best optima seen so far for each iteration of the optimization
- '..samples.csv', which stores all evaluations of the objective functions in the order they were computed

A description of the runtime arguments that may be used with this program are as follows:

REQUIRED settings:
--cfg_dir: 
	(Required, str) Give the path to the directory containing configuration info to set up the simulation.
	For this project, configuration directories can contain these files.
	- pop.csv: A csv file which lists zipcodes and their populations. Has the columns:
		- 'id': zipcode number
		- 'pop': population
		- 'vacc': vaccination rate
	- dist.csv: A csv file which lists all the pairwise distances between zipcodes. Has the columns:
		- 'zipcode1, zipcode2, distKM': For each row, 'distKM' is the distance in KM between zipcode1, zipcode2.
	- params.cfg: A ConfigParser compliant file which defines parameters for the tSIR simulation and optimization.
	  Please see the 'config' folder for example configurations.
	- A seeding file: This is a csv file which specifies where the initial infected case(s) is/are.
	  It is a vector with the same # of rows as pop.csv. Each row contains a number specifying the number of initial cases
	  the corresponding zipcode in pop.csv.
	All files should share a common indexing, e.g the first row of pop.csv should correspond to the first row of the seeding file,
	and the zipcodes in dist.csv should match those in pop.csv.

--out_dir: (Required, str) Give the path to the directory to output the optimization results in.

--name: (Required, str) Give an identifier string which will be included in the optimization result output filenames.

--dims: (Required, str) Specify the dimension of the problem. 
  This is typically the number of zipcodes, or if aggregating, the number of aggregation regions.

COMPUTATION settings:

--threads: How many total processes can be run in parallel using multiprocess.
  This is typically how many cores you have available for your use.

--sim_workers: How many total simulations are run in parallel at the same time.
  Set equal to 1 for serial evaluation. If using more than 1, you usually want to 
  set batch_size the same as sim_workers.

--threads_per_sim: How many processes to allocate to each simulation. Usually, this should
  be equal to (threads)/(sim_workers), e.g 20 threads and 4 sim_workers is 5 threads_per_sim.

--batch_size: How many data points batched TurBO is allowed to acquire at a time via Thompson sampling. 
  Usually, this should be equal to sim_workers.

--sim_draws: How many simulation replicates are run per objective function evaluation.
  Since the tSIR simulation is stochastic, the objective is usually a sample statistic of 
  simulation replicates (i.e mean). This argument specifies the number of samples to draw in one 
  objective function evaluation.

--n_init_pts: How many initial points to draw to initialize LAMCTS.

--iters: The total number of LAMCTS iterations allowed.

--samples: The total number of samples allowed.

--method: Use LAMCTS or vanilla BO (almost always use LAMCTS).

--load: Load a dumped LAMCTS_agent file. Specify the path to the file.
  This is a pickle file which the LAMCTS program saves upon successful completion.

--load_samples (str) : Load samples from a previous run. The format is
  "path/to/samples.csv,path/to/best_trace.csv."

AGGREGATE problem options:

These options should only be specified if the aggregate problem is to be run.
In the aggregate problem, the feasible set of solutions is constrained by grouping
sets of zipcodes together into larger spatial units (e.g counties). This enables the
simulation model to still run at a finer level of spatial aggregation, while reducing
the level of the larger optimization problem.

Each zipcode is assumed to belong to one county only.

--agg_size: Number of dimensions (larger spatial units) in the aggregated problem. 
  If aggregating, --dims=--agg_size.
--agg_mapping: Path to file specifying the aggregation mapping.
  This is a text file/csv where each row contains one number, with each row corresponding
  to rows of pop.csv. The number indicates which county the zipcode is mapped into.

SURVEILLANCE problem options:

--surv_mapping: Path to file specifying the surveillance mapping.
  Similar to agg_mapping.

--surv_constrs: Comma-delimited string specifying constraints for each surveillance region.
  e.g "0.05,0.05,0.05,0.05,0.05" if each surveillance region is to have a 5% constraint, and there
  are 5 regions. The ordering follows the labels in the surv_mapping file.

LAMCTS hyperparameters:

--cp: LAMCTS hyperparameter. Determines UCT bonus applied to nodes of the search tree.
--leaf_size: LAMCTS hyperparameter. Determines splitting thresholds for nodes of the search tree.
--hopsy_thin: hopsy hyperparameter. Determines thinning for MCMC sampler chains.
"""
import configparser
import pandas as pd
import numpy as np
from my_scripts.optimization import vacc
#from scripts.optimization import vacc_bayes_opt
#from ConstrainedLaMCTS.LAMCTS import bayes_opt
import multiprocess
import multiprocessing
import torch
import argparse
import os.path
import sys
from ConstrainedLaMCTS.LAMCTS.lamcts import MCTS
from scipy.optimize import linprog

# hopefully mitigate memory issues from copying a large amount of state
# from the parent process. Will be slower, but maybe that's ok?

if __name__ == "__main__":
    multiprocess.set_start_method('spawn')

    parser = argparse.ArgumentParser(
            prog = "Run optimization"
        )

    parser.add_argument('--cfg_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--dims',type=int,required=True)

    parser.add_argument('--threads', type=int, default=40)
    parser.add_argument('--threads_per_sim', type=int, default=20)
    parser.add_argument('--sim_workers', type=int, default=2)
    parser.add_argument('--batch_size',type=int, default=2)

    parser.add_argument('--sim_draws', type=int, default=100)
    parser.add_argument('--n_init_pts',type=int,default=150)
    parser.add_argument('--iters', type=int,default=250)
    parser.add_argument('--samples',type=int, default=100000)
    parser.add_argument('--method', default="lamcts")
    parser.add_argument('--load', type=str, default="")
    # provide a comma separated string: "pathtosamples.csv,path_to_best_trace.csv"
    parser.add_argument('--load_samples', type=str, default="")

    # county level optimization
    parser.add_argument('--agg_size', type=int, default=None)
    parser.add_argument('--agg_mapping', type=str, default=None)

    # surveillance regions
    parser.add_argument('--surv_mapping', type=str, default=None) # vector with length = # of regions?
    parser.add_argument('--surv_constrs', type=str, default=None) # comma separated list of constraints

    # optimizer hyperparameters
    parser.add_argument("--cp",type=float,default=0.1)
    parser.add_argument("--leaf_size",type=int,default=100)
    parser.add_argument("--hopsy_thin",type=int,default=150)

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
            elif k == 'grav_variant':
                tsir_config['grav_variant'] = tsir_config_str['grav_variant']
            else:
                tsir_config[k] = float(tsir_config_str[k])
        opt_config = dict(sim_param_config['opt_config'])
        opt_config['constraint_bnd'] = float(opt_config['constraint_bnd'])

    print(tsir_config)
    print(opt_config)

    # load vaccination/population data and distance matrix
    vacc_df = pd.read_csv(config_dir.format("pop.csv"))
    dist_list = pd.read_csv(config_dir.format("dist.csv"))
    dist_mat = dist_list.pivot(index='zipcode1', columns='zipcode2', values='distKM')
    dist_mat = dist_mat.fillna(0)

    seed = pd.read_csv(config_dir.format(sim_param_config['seed']['seed']),header=None)
    seed = np.array(seed).flatten()


    torch.set_num_threads(args.threads)

    # should use aggregate in opt_config

    do_aggregate = args.agg_size is not None and args.agg_mapping is not None

    if do_aggregate:
        agg_mapping = pd.read_csv(args.agg_mapping)
        agg_mapping = np.array(agg_mapping['mapping'])
        v = vacc.VaccProblemLAMCTSWrapper(
            opt_config = opt_config, 
            V_0= vacc_df['vacc'], 
            seed = seed,
            sim_config = tsir_config, 
            pop = vacc_df, 
            distances = np.array(dist_mat),
            negate=True, scale=True,
            cores=args.threads_per_sim, n_sim=args.sim_draws,
            output_dir = args.out_dir,
            name=args.name,
            agg_vector=agg_mapping,
            agg_size=args.agg_size,
            save_memory=True
        )
    else:
        v = vacc.VaccProblemLAMCTSWrapper(
            opt_config = opt_config, 
            V_0= vacc_df['vacc'], 
            seed = seed,
            sim_config = tsir_config, 
            pop = vacc_df, 
            distances = np.array(dist_mat),
            negate=True, scale=True,
            cores=args.threads_per_sim, n_sim=args.sim_draws,
            output_dir = args.out_dir,
            name=args.name,
            save_memory=True
        )

    if do_aggregate:
        P = np.zeros(args.agg_size)
        for zipcode_index, county_index in enumerate(agg_mapping):
            P[county_index] += vacc_df['pop'][zipcode_index]
        c = opt_config['constraint_bnd']
        # upper bound by the least vaccinated in each county
        ub = np.ones(args.agg_size)*np.inf
        for zipcode_index, county_index in enumerate(agg_mapping):
            this_zip_vacc = vacc_df.loc[zipcode_index,'vacc']
            if ub[county_index] > this_zip_vacc:
                ub[county_index] = this_zip_vacc
        v.ub = ub
        v.lb = np.zeros(args.dims)
        #for index, element in enumerate(ub):
            # if there are redundant cities (no zipcodes assigned)
            # just give it an upper bound of 1
            # essentially just a dummy var, doesn't affect opt
        #    if np.isinf(element):
        #        ub[index] = 1
    else:
        P = np.array(vacc_df['pop'])
        c = opt_config['constraint_bnd']
        ub = np.array(vacc_df['vacc'])

    A_ineq = [P]
    b_ineq = [c*np.sum(P)]
    #A_ineq = []
    #b_ineq = []

    # if running VHHA constrained version
    if args.surv_mapping is not None and args.surv_constrs is not None:
        surv_map_vec = np.array(pd.read_csv(args.surv_mapping)['surv_mapping'])
        surv_constrs = np.float64(args.surv_constrs.split(","))
        num_regions = np.max(surv_map_vec)+1
        assert len(surv_constrs) == num_regions, "number of constraint bounds provided did not match indices in mapping vector. maybe you forgot to 0-index?, \nsurv_constrs: {}\n num_regions: {}".format(surv_constrs,num_regions)
        for this_region_index, perc_constr in zip(range(num_regions),surv_constrs):
            if np.isinf(perc_constr):
                continue
            this_indicator_vec = np.int64(surv_map_vec == this_region_index)
            constr_LHS = P*this_indicator_vec # elementwise product
            constr_RHS = perc_constr * (P @ this_indicator_vec)
            A_ineq.append(constr_LHS)
            b_ineq.append(constr_RHS)

    A_ineq = np.array(A_ineq)
    b_ineq = np.array(b_ineq)

    print("population vector\n", P)
    print("rhs constraint bound\n",c)
    print("upper bound vector\n",ub)
    print("A_ineq\n",A_ineq)
    print("b_ineq\n",b_ineq)

    lp_bounds = [(0,up) for up in ub]
    test_feas = linprog(c=np.zeros(len(P)), A_ub=A_ineq, b_ub=b_ineq, bounds=lp_bounds)
    if test_feas.status==2:
        raise ValueError("Problem appears to be infeasible")
    elif test_feas.status==0:
        print("Problem appears to be feasible")

    if args.method == "lamcts":
        if args.load != "" and args.load_samples == "":
            print("hello!")
            agent = MCTS(
                     lb = np.zeros(args.dims),      # the lower bound of each problem dimensions
                     ub = ub,       # the upper bound of each problem dimensions
                     dims = args.dims,              # the problem dimensions
                     ninits = 0,           # the number of random samples used in initializations 
                     A_ineq = A_ineq,
                     b_ineq = b_ineq,
                     A_eq = None, b_eq = None,
                     func = v,               # function object to be optimized
                     Cp = args.cp,              # Cp for MCTS
                     leaf_size = args.leaf_size, # tree leaf size
                     kernel_type = 'linear', #SVM configruation
                     gamma_type = "auto",    #SVM configruation
                     solver_type = 'turbo',
                     num_threads = args.threads,
                     sim_workers = args.sim_workers,
                     threads_per_sim = args.threads_per_sim,
                     hopsy_thin = args.hopsy_thin
                     )
            agent.load_agent(load_path=args.load)
            agent.search(iterations=args.iters, max_samples=args.samples)
        elif args.load == "" and args.load_samples != "":
            agent = MCTS(
                     lb = np.zeros(args.dims),      # the lower bound of each problem dimensions
                     ub = ub,       # the upper bound of each problem dimensions
                     dims = args.dims,              # the problem dimensions
                     ninits = 0,           # the number of random samples used in initializations 
                     A_ineq = A_ineq,
                     b_ineq = b_ineq,
                     A_eq = None, b_eq = None,
                     func = v,               # function object to be optimized
                     Cp = args.cp,              # Cp for MCTS
                     leaf_size = args.leaf_size, # tree leaf size
                     kernel_type = 'linear', #SVM configruation
                     gamma_type = "auto",    #SVM configruation
                     solver_type = 'turbo',
                     num_threads = args.threads,
                     sim_workers = args.sim_workers,
                     threads_per_sim = args.threads_per_sim,
                     hopsy_thin = args.hopsy_thin
                     )
            samples_path, best_trace_path = args.load_samples.split(",")
            agent.load_samples_from_file(samples=samples_path, best_trace=best_trace_path)
            agent.dynamic_treeify()
            agent.search(iterations=args.iters, max_samples=args.samples)        
        elif args.load != "" and args.load_samples != "":
            print("loading both mcts agent, then new samples")
            agent = MCTS(
                     lb = np.zeros(args.dims),      # the lower bound of each problem dimensions
                     ub = ub,       # the upper bound of each problem dimensions
                     dims = args.dims,              # the problem dimensions
                     ninits = 0,           # the number of random samples used in initializations 
                     A_ineq = A_ineq,
                     b_ineq = b_ineq,
                     A_eq = None, b_eq = None,
                     func = v,               # function object to be optimized
                     Cp = args.cp,              # Cp for MCTS
                     leaf_size = args.leaf_size, # tree leaf size
                     kernel_type = 'linear', #SVM configruation
                     gamma_type = "auto",    #SVM configruation
                     solver_type = 'turbo',
                     num_threads = args.threads,
                     sim_workers = args.sim_workers,
                     threads_per_sim = args.threads_per_sim,
                     hopsy_thin = args.hopsy_thin
                     )
            agent.load_agent(load_path=args.load) 
            agent.load_samples_from_file(samples=samples_path, best_trace=best_trace_path)
            agent.dynamic_treeify()
            agent.search(iterations=args.iters, max_samples=args.samples) 
        else:
            print("else branch")
            agent = MCTS(
                     lb = np.zeros(args.dims),      # the lower bound of each problem dimensions
                     ub = ub,       # the upper bound of each problem dimensions
                     dims = args.dims,              # the problem dimensions
                     ninits = args.n_init_pts,           # the number of random samples used in initializations 
                     A_ineq = A_ineq,
                     b_ineq = b_ineq,
                     A_eq = None, b_eq = None,
                     func = v,               # function object to be optimized
                     Cp = args.cp,              # Cp for MCTS
                     leaf_size = args.leaf_size, # tree leaf size
                     kernel_type = 'linear', #SVM configruation
                     gamma_type = "auto",    #SVM configruation
                     solver_type = 'turbo',
                     num_threads = args.threads,
                     sim_workers = args.sim_workers,
                     threads_per_sim = args.threads_per_sim,
                     hopsy_thin = args.hopsy_thin
                     )
            agent.search(iterations = args.iters, max_samples=args.samples)
        agent.dump_agent(name=args.name+"mcts_agent", out_dir=args.out_dir)
    elif args.method == "bo":
        agent = MCTS(
                     lb = np.zeros(args.dims),      # the lower bound of each problem dimensions
                     ub = np.ones(args.dims),       # the upper bound of each problem dimensions
                     dims = args.dims,              # the problem dimensions
                     ninits = args.n_init_pts,           # the number of random samples used in initializations 
                     A_ineq = A_ineq,
                     b_ineq = b_ineq,
                     A_eq = None, b_eq = None,
                     func = v,               # function object to be optimized
                     Cp = 10,              # Cp for MCTS
                     leaf_size = np.inf, # tree leaf size
                     kernel_type = 'linear', #SVM configruation
                     gamma_type = "auto",    #SVM configruation
                     solver_type = 'bo',
                     num_threads = args.num_threads,
                     sim_workers = args.sim_workers,
                     threads_per_sim = args.threads_per_sim,
                     hopsy_thin = args.hopsy_thin
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
