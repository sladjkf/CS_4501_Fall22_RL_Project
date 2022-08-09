import numpy as np
import warnings

# Need to import spatial tSIR module. is there a better way to do this??
import sys
project_path = "/home/nick/Documents/4tb_sync/UVA GDrive/Summer 2022 (C4GC with BII)/measles_metapop/{}"
sys.path.append(project_path.format("scripts/"))
from spatial_tsir import *
"""
Write a wrapper for the stochastic tSIR that takes in different representations
of the vaccination vector and runs the simulation.
"""
class VaccRateOptEngine:
    """
    Wrapper class that should make it easier to 
    query the spatial tSIR model in an optimization problem
    """
    # def __init__(self,
    #         V_0,inf_seed,
    #         sim_config, pop, distances,
    #         obj,V_repr,
    #         aux_param):
    def __init__(self,
            opt_params, V_0, seed, # TODO: consistent terminology.. 'params','config'
            sim_config, pop, distances):
        # TODO: fill out the rest of the docstring
        """
        Args:
            V_0 (numpy.ndarray):
                Vector of length (# of patches).
                The initial vaccination vector.
                Is used in the computation of the constraint.
            inf_seed (numpy.ndarray):
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
            obj (str):
                A string specifying which objective function gets computed.
            V_repr (str):
                A string specifying how the vaccination vectors are represented.
        """
        # store constants for simulation runs
        self.sim_params = sim_config
        # store values relevant to optimization routine
        # that stay fixed over the duration of the optimization routine
        # TODO: validate parameters (e.g S+I+R = pop)?
        # TODO: objective with auxiliary parameters (i.e attacksize prob needs to specify the cutoff)?
        # input validation - these are essential
        # TODO: type checking too?
        #assert 'V_0' in opt_params.keys()
        #assert 'seed' in opt_params.keys()
        assert 'obj' in opt_params.keys()
        assert 'V_repr' in opt_params.keys()
        assert 'constraint_bnd' in opt_params.keys() # bound
        if opt_params['obj'] in ['attacksize_prob']: 
            # if you've specified the CDF based objective
            # you need to specify the cutoff quantity
            assert 'attacksize_cutoff' in opt_params.keys()

        assert opt_params['V_repr'] in ["ratio","max_ratio","raw"], "invalid V_repr (vaccine vector representation) argument passed"
        assert opt_params['obj'] in ["attacksize","peak","attacksize_prob"], "invalid objective function name passed"

        self.opt_params = opt_params
        self.V_0 = V_0
        self.seed = seed
        self.pop_df = pop
        self._pop_vec = np.array(self.pop_df['pop'])
        self._max_pop = max(self._pop_vec)
        self.distances = distances
        self.evaluation_history = [] # store (input vector, objective) pairs
    def check_constraint(self,V_prime):
        # derivations for these constraints were in overleaf doc 'measles_optimization'
        #P = np.array(self.sim_params['pop']['pop']) # index 'pop' column from dataframe
        P = self._pop_vec
        V_0 = self.V_0
        V_repr = self.opt_params['V_repr']
        constraint_bnd = self.opt_params['constraint_bnd']
        if V_repr == "ratio":
            result = ((V_0 - V_prime) @ P)/np.linalg.norm(P,ord=1) < constraint_bnd
        elif V_repr == "max_ratio":
            result = (np.linalg.norm(P,ord=np.inf)/np.linalg.norm(P,ord=1))*\
                    np.abs(np.linalg.norm(V_0,ord=1) - np.linalg.norm(V_prime,ord=1)) < constraint_bnd
        elif V_repr == "raw":
            result = np.linalg.norm(V_0-V_prime,ord=1) < constraint_bnd
        else:
            print("No constraint function exists for the V_repr type you've passed")
            result = False
        return result
    def query(self,V_prime,
            multithread=True,pool=None,n_sim=None):
        passed = self.check_constraint(V_prime)
        if not passed:
            # TODO check constraint
            print("Constraint violated")

        # SETUP INITIAL STATE #
        # need to "untransform" back to raw counts
        # I guess it needs to be rounded? not sure if my code works like that
        # TODO: get and store max_pop and pop_vec to avoid recomputation?
        # but best if you can avoid recomputing max_pop to avoid repeated O(n) op
        if self.opt_params['V_repr'] == "max_ratio":
            V_unscaled = self._max_pop*V_prime
        elif self.opt_params['V_repr'] == "ratio":
            V_unscaled = self._pop_vec*V_prime # element by element
        elif self.opt_params['V_repr'] == "raw":
            pass
        else:
            raise ValueError("Invalid string for V_repr")
        V_unscaled = np.round(V_unscaled)
        initial_state = np.zeros((len(self.pop_df.index),2))
        initial_state[:,0] = self.seed
        initial_state[:,1] = V_unscaled

        if not multithread:
            # TODO singlethread evaluation
            pass
        assert type(pool) != None and type(n_sim) != None,\
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
        if self.opt_params['obj']=="attacksize":
            result = sim_pool.get_attack_size_samples()
        elif self.opt_params['obj']=="peak":
            result = np.max(sim_pool.get_samples(),axis=1)
        elif self.opt_params['obj']=="attacksize_prob":
            result = sim_pool.get_attack_size_samples() < c

        return result

    def save_eval_history(self,path):
        # save initial config 
        # (input vector, objective) pairs?
        pass

    # implement some objective functions here?

