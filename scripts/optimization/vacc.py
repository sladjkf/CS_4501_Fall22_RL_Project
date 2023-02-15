import numpy as np
import warnings
import dill
import pandas as pd
# TODO Need to import spatial tSIR module. is there a better way to do this??
import sys
from scripts.spatial_tsir import *
import datetime
import os
import multiprocess
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
            sim_config, pop, distances, 
            agg_vector=None, agg_size=None):
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
                    will be of the form |V_0 - V'| = c where c is the constant specified
                    by 'constraint_bnd.'
                - 'aggregate': boolean
                    Since solving a high dimensional optimization problem can be challenging,
                    one can simplify the objective by simultaneously decreasing vaccination
                    across the some regions at once and treat it as one region for the sake of 
                    the optimization.
                    Setting aggregate to true enables this behavior in the function, but this
                    requires the optional argument 'agg_vector' to be passed.
                    How this works: if [z_1, ..., z_n] in R^n is the finer spatial level, we aggregate
                    to a coarser spatial level in R^m with m <= n. agg_vector represents the function
                    mapping localities in the finer spatial level to the localities in the coarser
                    spatial level as a vector in R^n, with entries in {1, ... ,m} which correspond to
                    the entries of the vector representation at the coarse spatial level.
                    This requires you to pass the following optional arguments:
                        - agg_vector: the mapping between spatial resolutions
                        - agg_size: the number of localities in the coarser spatial level
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
        assert 'constraint_type' in opt_config.keys()
        assert opt_config['constraint_type'] in ['ineq','eq']
        if opt_config['obj'] in ['attacksize_prob']: 
            # if you've specified the CDF based objective
            # you need to specify the cutoff quantity
            assert 'attacksize_cutoff' in opt_config.keys()

        assert opt_config['V_repr'] in ["ratio","max_ratio","raw"], "invalid V_repr (vaccine vector representation) argument passed"
        assert opt_config['obj'] in ["attacksize","peak","attacksize_prob"], "invalid objective function name passed"

        assert len(V_0) == len(seed) == len(pop.index) == distances.shape[0] == distances.shape[1]
        # save arguments as attributes of the object
        self.opt_config = opt_config
        self.sim_params = sim_config
        self.V_0 = V_0
        self.seed = seed
        self.pop_df = pop
        self.distances = distances
        self.dims = len(V_0)

        # precompute vector forms of population, max to avoid recomputing it later
        self._pop_vec = np.array(self.pop_df['pop']) 
        self._max_pop = max(self._pop_vec)
        self._pop_norm = np.sum(self._pop_vec)
        # store (input vector, objective) pairs
        self.eval_history = {'input':None,'output':None}

        self.aggregate = False
        self.agg_vector = None
        if 'aggregate' in opt_config.keys():
            assert agg_vector is not None
            assert agg_size is not None
            assert len(agg_vector)==len(V_0)
            self.aggregate = True
            self.agg_vector = agg_vector
            self.agg_size = agg_size
        else:
            self.aggregate = False
            self.agg_vector = None
            self.agg_size = None

    def check_constraint(self,V_delta):
        vacc_not_decreased_past_zero = all(self.V_0 - V_delta >= 0)
        if self.opt_config['constraint_type'] == 'eq':
            budget_satisfied = np.isclose(
                    V_delta @ self._pop_vec,
                    self.opt_config['constraint_bnd']*self._pop_norm)
        if self.opt_config['constraint_type'] == 'ineq':
            budget_satisfied = np.all(V_delta @ self._pop_vec <= self.opt_config['constraint_bnd']*self._pop_norm)
        return vacc_not_decreased_past_zero and budget_satisfied

    def query(self,
            V_delta=None,
            multithread=True,pool=None,n_sim=None,
            return_sim_pool=False):

        if self.aggregate:
            assert len(V_delta)==self.agg_size
            V_delta_new = np.zeros(self.dims)
            for i in range(self.dims):
                V_delta_new.put(i, V_delta[self.agg_vector[i]])
            V_delta = V_delta_new
            
        passed = self.check_constraint(V_delta)
        if not passed:
            print("Constraint violated")
            return np.array([-1]) # TODO: probably bad behavior, but ok for placeholding?

        # SETUP INITIAL STATE #
        # TODO: modify computation of V_unscaled such that V_prime
        # represents a change in vaccination rates rather than a new state
        if self.opt_config['V_repr'] == "max_ratio":
            V_unscaled = self._max_pop*(self.V_0 - V_delta)
        elif self.opt_config['V_repr'] == "ratio":
            V_unscaled = self._pop_vec*(self.V_0 - V_delta)  # element by element
        elif self.opt_config['V_repr'] == "raw":
            pass
        else:
            raise ValueError("Invalid string for V_repr")
        V_unscaled = np.round(V_unscaled)
        initial_state = np.zeros((len(self.pop_df.index),2))
        initial_state[:, 0] = self.seed
        #initial_state[:,0] = seed_prime
        initial_state[:, 1] = V_unscaled

        if not multithread:
            # TODO singlethread evaluation, but i'm almost always going to multithread
            print("singlethread not implemented")
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
        # keep a record of the evaluation results
        if type(self.eval_history['input']) == type(None) and type(self.eval_history['output']) == type(None):
            # just store as a list to enable ragged inputs
            self.eval_history['input'] = [np.array(V_delta)]
            self.eval_history['output'] = [np.array(result)]
        else:
            self.eval_history['input'].append(np.array(V_delta))
            self.eval_history['output'].append(np.array(result))
        if return_sim_pool:
            return result, sim_pool
        else:
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
            with open(path+".save","wb") as out_file:
                dill.dump(self.eval_history, file=out_file)
        elif as_csv: #TODO: how to deal with ragged input arrays?
            input_shape = np.shape(self.eval_history['input'])
            #joined = np.concatenate([self.eval_history['input'], self.eval_history['output']], axis=1)
            #joined = pd.DataFrame(joined)
            num_evals = len(self.eval_history['input'])
            with open(path+".csv","w") as out_file:
                print("input"+","*(input_shape[1]-1)+","+"output",file=out_file)
                for i in range(num_evals):
                    this_input = self.eval_history['input'][i]
                    this_output = self.eval_history['output'][i]
                    input_str = np.array2string(this_input,
                                                separator=",",
                                                max_line_width=np.inf,
                                                precision=5).strip("[]")
                    output_str = np.array2string(this_output,
                                                 separator=",",
                                                 max_line_width=np.inf,
                                                 precision=5).strip("[]")
                    print(input_str, file=out_file, end="")
                    print(output_str, file=out_file, end="")
                    print(file=out_file)
        else:
            print("no valid save format specified")

class VaccProblemLAMCTSWrapper:
    def __init__(self,
            opt_config, V_0, seed,
            sim_config, pop, distances,
            cores, n_sim,
            negate, scale,
            output_dir, name,
            agg_vector=None, agg_size=None):
        self.engine = VaccRateOptEngine(
            opt_config=opt_config,
            V_0=V_0, seed=seed,
            sim_config=sim_config,
            pop=pop,
            distances=distances,
            agg_vector=agg_vector, agg_size=agg_size
        )
        self.pool = multiprocess.Pool(cores)
        self.best_x = None
        self.best_y = None
        self.last_x = None
        self.last_y = None

        self.n_sim = n_sim
        self.cores = cores
        assert os.path.exists(output_dir), "invalid directory"

        date = datetime.datetime.now()
        self.output_file_trace = "{}vacc_sim_{}_{}_best_trace.csv".format(output_dir,name,date.isoformat())
        self.output_file_samples = "{}vacc_sim_{}_{}_samples.csv".format(output_dir,name,date.isoformat())

        if negate:
            self.sign = -1
        else:
            self.sign = 1
        assert len(V_0) == len(seed) == len(pop) == distances.shape[0] == distances.shape[1]

        if scale:
            self.scale = np.sum(pop['pop'])
        else:
            self.scale = 1

        if agg_size is not None:
            self.dims = len(V_0)
        else:
            self.dims = agg_size
        self.lb = np.zeros(self.dims)
        self.ub = np.ones(self.dims)

    def __call__(self, x):
        result = self.engine.query(V_delta=x, pool=self.pool, n_sim=self.n_sim)
        result = np.mean(result)
        if self.best_x is None and self.best_y is None:
            self.best_x = x
            self.best_y = result
        else:
            if self.best_y <= result:
                self.best_x = x
                self.best_y = result
        self.last_x = x
        self.last_y = result
        self.track()
        return self.sign*result/self.scale
        
    def track(self):
        if os.path.exists(self.output_file_trace):
            with open(self.output_file_trace, "a+") as log:
                np.savetxt(log,self.best_x,delimiter=",", newline=",")
                print(self.best_y,file=log)
        else:
            with open(self.output_file_trace, "w+") as log:
                np.savetxt(log,self.best_x,delimiter=",", newline=",")
                print(self.best_y,file=log)

        if os.path.exists(self.output_file_samples):
            with open(self.output_file_samples, "a+") as log:
                np.savetxt(log,self.last_x,delimiter=",", newline=",")
                print(self.last_y,file=log)
        else:
            with open(self.output_file_samples, "w+") as log:
                np.savetxt(log,self.last_x,delimiter=",", newline=",")
                print(self.last_y,file=log)

    def close_pool(self):
        self.pool.close()
