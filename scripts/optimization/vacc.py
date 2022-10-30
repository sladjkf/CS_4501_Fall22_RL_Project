import numpy as np
import warnings
import dill
import pandas as pd
# TODO Need to import spatial tSIR module. is there a better way to do this??
import sys
from scripts.spatial_tsir import *
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

    def __init__(self, opt_config, V_0, seed, sim_config, pop, distances):
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
            V_0 (numpy.ndarray):
                Vector of length (# of patches).
                The initial vaccination vector. Is used in the computation of the constraint.
            seed (numpy.ndarray):
                Vector of length (# of patches).
                The initial seeding strategy vector.
            sim_config (dict):
                Used to specify the parameters of the tSIR simulation.
            pop (numpy.ndarray):
                Vector of length (# of patches).
                The population vector. Contains the number of people in each region.
            distances (numpy.ndarray):
                2D matrix of dimension (# of patches) x (# of patches).
                It specifies the distances between patches.
        """
        # TODO: validate parameters (e.g S+I+R = pop)?
        # TODO: objective with auxiliary parameters (i.e attacksize prob needs to specify the cutoff)?
        # TODO: type checking too?
        assert 'obj' in opt_config.keys()
        assert 'V_repr' in opt_config.keys()
        assert 'constraint_bnd' in opt_config.keys()
        if opt_config['obj'] in ['attacksize_prob']:
            # if you've specified the CDF based objective
            # you need to specify the cutoff quantity
            assert 'attacksize_cutoff' in opt_config.keys()

        assert opt_config['V_repr'] in ["ratio", "max_ratio",
                                        "raw"], "invalid V_repr (vaccine vector representation) argument passed"
        assert opt_config['obj'] in ["attacksize", "peak",
                                     "attacksize_prob"], "invalid objective function name passed"

        # save arguments as attributes of the object
        self.opt_config = opt_config
        self.sim_params = sim_config
        self.V_0 = V_0
        self.seed = seed
        self.pop_vec = pop
        self.distances = distances

        # precompute vector forms of population, max to avoid recomputing it later
        self._max_pop = max(self.pop_vec)
        self._pop_norm = np.sum(self.pop_vec)

        # store (input vector, objective) pairs
        self.eval_history = {'input': [], 'output': []}

    def passed_constraint(self, V_delta):
        vacc_not_decreased_past_zero = all(self.V_0 - V_delta >= 0)
        budget_satisfied = np.isclose(V_delta @ self.pop_vec, self.opt_config['constraint_bnd']*self._pop_norm)
        
        return vacc_not_decreased_past_zero and budget_satisfied

    def query(self, V_delta=None, pool=None, n_sim=None, return_sim_pool=False):
        if not self.passed_constraint(V_delta):
            print("Constraint violated")
            return None

        # SETUP INITIAL STATE #
        # TODO: modify computation of V_unscaled such that V_prime
        # represents a change in vaccination rates rather than a new state
        if self.opt_config['V_repr'] == "max_ratio":
            V_unscaled = self._max_pop*(self.V_0 - V_delta)
        
        elif self.opt_config['V_repr'] == "ratio":
            V_unscaled = self.pop_vec * (self.V_0 - V_delta)  # element by element
        
        elif self.opt_config['V_repr'] == "raw":
            pass
        
        else:
            raise ValueError("Invalid string for V_repr")
        
        V_unscaled = np.round(V_unscaled)
        initial_state = np.zeros((self.pop_vec.shape[0], 2))
        initial_state[:, 0] = self.seed
        initial_state[:, 1] = V_unscaled

        assert type(pool) != type(None) and type(n_sim) != type(None),\
            "You must pass a multithread.Pool object and n_sim when in multithreading mode"

        sim_pool = spatial_tSIR_pool(
            config=self.sim_params,
            patch_pop=self.pop_vec,
            initial_state=initial_state,
            n_sim=n_sim,
            distances=self.distances)

        sim_pool.run_simulation(pool=pool)

        # return a sample of the statistic
        if self.opt_config['obj'] == "attacksize":
            result = sim_pool.get_attack_size_samples()

        elif self.opt_config['obj'] == "peak":
            result = np.max(sim_pool.get_samples(), axis=1)
        
        elif self.opt_config['obj'] == "attacksize_prob":
            # It seems more natural to formulate it as
            # 1 if exceeded, 0 if below the bound
            # so the probability is probability of attack size above this cutoff?
            result = np.int64(sim_pool.get_attack_size_samples()
                              > self.opt_config['attacksize_cutoff'])
        
        # keep a record of the evaluation results
        self.eval_history['input'].append(V_delta)
        self.eval_history['output'].append([result])
        
        if return_sim_pool:
            return result, sim_pool
        else:
            return result

    def save_eval_history(self,
                          path=None,
                          as_csv=False, as_serial=True):
        # handle exceptional cases
        if path is None:
            raise ValueError("A path must be supplied.")
        if (self.eval_history['input'] is None) and (self.eval_history['output'] is None):
            print("No data to write.")
            return
        if as_serial:
            with open(path+".save", "wb") as out_file:
                dill.dump(self.eval_history, file=out_file)
        elif as_csv:  # TODO: how to deal with ragged input arrays?
            input_shape = np.shape(self.eval_history['input'])
            #joined = np.concatenate([self.eval_history['input'], self.eval_history['output']], axis=1)
            #joined = pd.DataFrame(joined)
            num_evals = len(self.eval_history['input'])
            with open(path+".csv", "w") as out_file:
                print("input"+"," *
                      (input_shape[1]-1)+","+"output", file=out_file)
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
