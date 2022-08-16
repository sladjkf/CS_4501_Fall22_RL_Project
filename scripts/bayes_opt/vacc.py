import numpy as np
import warnings
import dill
import pandas as pd
# TODO Need to import spatial tSIR module. is there a better way to do this??
import sys
project_path = "/scratch/nrw5cq/measles_metapop/{}"
sys.path.append(project_path.format("scripts/"))
from spatial_tsir import *
"""
Write a wrapper for the stochastic tSIR that takes in different representations
of the vaccination vector and runs the simulation.
"""
class VaccRateOptEngine:
    """
    Wrapper class that should make it easier to 
    query the spatial tSIR model in an optimization problem.
    It serves as the "oracle" that you can query during the optimization routine.
    """
    # def __init__(self,
    #         V_0,inf_seed,
    #         sim_config, pop, distances,
    #         obj,V_repr,
    #         aux_param):
    def __init__(self,
            opt_config, V_0, seed,
            sim_config, pop, distances):
        # TODO: fill out the rest of the docstring
        """
        Create a new optimization oracle, initializing with certain fixed parameters.

        Note (as is the case with the other routines in this project) that the labeling
        of 'V_0', 'seed','pop', and 'distances' are implicitly assumed to be the same
        (e.g row N of pop refers to the same locality as does entry N of V_0 and so on).

        Args:
            opt_config (dict):
                Dictionary containing the parameters governing the behavior 
                of the objective function.
                Mandatory keys (must be specified):
                - 'obj': str
                    Specifies which summary statistic ('objective') will be computed from
                    the simulation and be used as the target of the objective function.
                    Possible values: (TODO)
                - 'V_repr': str
                    Specifies the format in which the vaccination vector V will be
                    represented.
                    Possible values:
                - 'constraint_bnd': float
                    Specifies the constraint that gets used to constrain the decision
                    space of the optimization problem. Broadly speaking, the constraint
                    will be of the form |V_0 - V'| < c where c is the constant specified
                    by 'constraint_bnd.'
            V_0 (numpy.ndarray):
                Vector of length (# of patches).
                The initial vaccination vector. Is used in the computation of the constraint.
            seed (numpy.ndarray):
                Vector of length (# of patches).
                The initial seeding strategy vector.
            sim_config (dict):
                Used to specify the parameters of the tSIR simulation.
            pop (Pandas.DataFrame):
                DataFrame. Has (# of patches) rows, and at least the column
                'patch_id' - names of the locations
                'pop' - population size of that location
            distances (numpy.ndarray):
                2D matrix of dimension (# of patches) x (# of patches).
                It specifies the distances between patches.
        """
        # store constants for simulation runs
        # store values relevant to optimization routine
        # that stay fixed over the duration of the optimization routine
        # TODO: validate parameters (e.g S+I+R = pop)?
        # TODO: objective with auxiliary parameters (i.e attacksize prob needs to specify the cutoff)?
        # input validation - these are essential
        # TODO: type checking too?
        assert 'obj' in opt_config.keys()
        assert 'V_repr' in opt_config.keys()
        assert 'constraint_bnd' in opt_config.keys()
        if opt_config['obj'] in ['attacksize_prob']: 
            # if you've specified the CDF based objective
            # you need to specify the cutoff quantity
            assert 'attacksize_cutoff' in opt_config.keys()

        assert opt_config['V_repr'] in ["ratio","max_ratio","raw"], "invalid V_repr (vaccine vector representation) argument passed"
        assert opt_config['obj'] in ["attacksize","peak","attacksize_prob"], "invalid objective function name passed"

        # save arguments as attributes of the object
        self.opt_config = opt_config
        self.sim_params = sim_config
        self.V_0 = V_0
        self.seed = seed
        self.pop_df = pop
        self.distances = distances

        # precompute vector forms of population, max to avoid recomputing it later
        self._pop_vec = np.array(self.pop_df['pop']) 
        self._max_pop = max(self._pop_vec)
        # store (input vector, objective) pairs
        self.eval_history = {'input':None,'output':None}
    def check_constraint(self,V_prime):
        # derivations for these constraints were in overleaf doc 'measles_optimization'
        #P = np.array(self.sim_params['pop']['pop']) # index 'pop' column from dataframe
        P = self._pop_vec
        V_0 = self.V_0
        V_repr = self.opt_config['V_repr']
        constraint_bnd = self.opt_config['constraint_bnd']
        if V_repr == "ratio":
            # check the basic domain constraint
            in_domain = (0 < V_prime).all() and (V_prime < 1).all()
            result_num = ((V_0 - V_prime) @ P)/np.linalg.norm(P,ord=1)
        elif V_repr == "max_ratio":
            in_domain = (0 < V_prime).all() and (V_prime < 1).all()
            result_num = (np.linalg.norm(P,ord=np.inf)/np.linalg.norm(P,ord=1))*\
                    np.abs(np.linalg.norm(V_0,ord=1) - np.linalg.norm(V_prime,ord=1))
        elif V_repr == "raw":
            in_domain = (0 < V_prime).all()
            result_num = np.linalg.norm(V_0-V_prime,ord=1) < constraint_bnd
        else:
            print("No constraint function exists for the V_repr type you've passed")
            result = -np.inf
        print(result_num) # print the constraint value
        result = (result_num < constraint_bnd) and in_domain
        return result
    def query(self,V_prime=None,seed_prime=None,
            multithread=True,pool=None,n_sim=None):
        if (type(V_prime) == type(None) and type(seed_prime) is type(None)) or\
                (type(V_prime) != type(None) and type(seed_prime) != type(None)):
            print("Please provide either V_prime or seed_prime")

        if type(seed_prime) != type(None):
            V_prime = self.V_0
            eval_mode = "seed"
        elif type(V_prime) != type(None):
            # default to computation with the passed V_0
            seed_prime = self.seed
            eval_mode = "V"
        else:
            print("bad argumetns") # TODO:
            return

        passed = self.check_constraint(V_prime)
        if not passed:
            print("Constraint violated")
            return np.array([-1]) # TODO: probably bad behavior, but ok for placeholding?

        # SETUP INITIAL STATE #
        if self.opt_config['V_repr'] == "max_ratio":
            V_unscaled = self._max_pop*V_prime
        elif self.opt_config['V_repr'] == "ratio":
            V_unscaled = self._pop_vec*V_prime # element by element
        elif self.opt_config['V_repr'] == "raw":
            pass
        else:
            raise ValueError("Invalid string for V_repr")
        V_unscaled = np.round(V_unscaled)
        initial_state = np.zeros((len(self.pop_df.index),2))
        #initial_state[:,0] = self.seed
        initial_state[:,0] = seed_prime
        initial_state[:,1] = V_unscaled

        if not multithread:
            # TODO singlethread evaluation
            pass
        assert type(pool) != type(None) and type(n_sim) != type(None),\
                "You must pass a multithread.Pool object and n_sim when in multithreading mode"
        sim_pool = spatial_tSIR_pool(
                    config = self.sim_params,
                    patch_pop = self.pop_df,
                    initial_state = initial_state,
                    n_sim = n_sim,
                    distances = self.distances
                )
        sim_pool.run_simulation(pool=pool)

        # return a sample of the statistic
        if self.opt_config['obj']=="attacksize":
            result = sim_pool.get_attack_size_samples()
        elif self.opt_config['obj']=="peak":
            result = np.max(sim_pool.get_samples(),axis=1)
        elif self.opt_config['obj']=="attacksize_prob":
            # It seems more natural to formulate it as
            # 1 if exceeded, 0 if below the bound
            # so the probability is probability of attack size above this cutoff?
            result = np.int64(sim_pool.get_attack_size_samples() > self.opt_config['attacksize_cutoff'])
        print(eval_mode)
        # keep a record of the evaluation results
        if type(self.eval_history['input']) == type(None) and type(self.eval_history['output']) == type(None):
            if eval_mode == "V":
                self.eval_history['input'] = np.array([V_prime])
            elif eval_mode == "seed":
                self.eval_history['input'] = np.array([seed_prime])
            self.eval_history['output'] = np.array([result])
        else:
            if eval_mode == "V":
                self.eval_history['input']= np.concatenate([self.eval_history['input'],np.array([V_prime])],axis=0)
            elif eval_mode == "seed":
                self.eval_history['input']= np.concatenate([self.eval_history['input'],np.array([seed_prime])],axis=0)
            self.eval_history['output']= np.concatenate([self.eval_history['output'],np.array([result])],axis=0)
        return result
    def save_eval_history(self,
            path=None,
            as_csv=False,as_serial=True):
        # handle exceptional cases
        if path is None:
            raise ValueError("A path must be supplied.")
        if (self.eval_history['input'] is None) and (self.eval_history['output'] is None):
            print("No data to write.")
            return
        if as_serial:
            dill.dump(self.eval_history,file=out_file)
        elif as_csv:
            input_shape = np.shape(self.eval_history['input'])
            joined = np.concatenate([self.eval_history['input'],self.eval_history['output']],axis=1)
            joined = pd.DataFrame(joined)
            joined.rename(columns={0:'input',input_shape[1]:'output'})
            joined.to_csv(file=path,index=False)
        else:
            print("no valid save format specified")
