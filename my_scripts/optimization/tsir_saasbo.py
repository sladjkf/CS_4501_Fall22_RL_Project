"""
Setup and run the tSIR simulation using SAASBO optimization technique.
"""
import argparse
import numpy as np
import scipy
import pandas as pd
import numpyro
from numpyro.util import enable_x64
from functools import partial
import sys
import multiprocess
import dill
with open("project_dir.txt") as f:
    project_path = f.read().strip() + "{}"

sys.path.append(project_path.format("saasbo/"))
#sys.path.append(project_path.format("scripts/bayes_opt/"))

from saasbo import run_saasbo
import scripts.optimization.vacc as vacc

def attacksize_mean(engine,pool, n_sim, q):
    """
    Small wrapper for optimization oracle
    needed for minimization routine
    Use functools partial to get a single function out
    """
    return -np.mean(engine.query(V_prime=q, pool=pool, n_sim=n_sim))

def inv_tr_lin_solve(point,l,u, dim=None):
    """
    Constraint transformation function.
    """
    if dim is None:
        dim = len(point)
    coeff_matrix = np.eye(dim)
    # add bottom row of ones
    # coeff_matrix[dim-1,:] = np.ones(dim)
    # add coefficients to off-diagonal
    for i in range(0,dim-1):
        coeff_matrix[i,i+1] = (l-point[i])/(u-l)
    b = np.zeros(dim)
    b[dim-1] = point[dim-1]
    L = np.tril(np.ones((dim,dim)))
    x = scipy.linalg.solve(coeff_matrix@L,b)
    return x

def attacksize_mean_tr_input(engine, pool, n_sim, 
        constr_low, constr_up, pop_vec,
        q_prime, dim=None):
    # undo the transformation (in reverse order)
    # if it was change of variable -> transformation
    # to undo it, transformation -> change of variable

    # transform it back
    q_prime_tr = inv_tr_lin_solve(q_prime, constr_low, constr_up, dim)
    # then undo the change of variable
    q = q_prime_tr*pop_vec
    return attacksize_mean(engine, pool, n_sim, q)

def main(args):
    #### load data ###
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
    # lb = np.array(V_0-0.08)
    # ub = np.array(V_0)
    ub = V_0 @ np.array(vacc_df['pop'])
    lb = ub - opt_config['constraint_bnd']*np.linalg.norm(vacc_df['pop'],ord=np.inf)
    ub_vec, lb_vec = np.repeat(ub,len(I)), np.repeat(lb,len(I))
    with multiprocess.Pool(args.cores) as p:
        #objective=partial(attacksize_mean,engine=engine,pool=p,n_sim=args.n_sim)
        objective = lambda x: attacksize_mean_tr_input(q_prime=x, engine=engine, pool=p, n_sim=args.n_sim, constr_up=ub, constr_low=lb, pop_vec=np.array(vacc_df['pop']), dim = len(I))
        run_saasbo(
            objective,
            lb_vec,
            ub_vec,
            args.max_evals,
            num_init_evals=15,
            seed=args.seed,
            alpha=0.01,
            num_warmup=256,
            num_samples=256,
            thinning=32,
            device=args.device,
        )
    with open(args.save_path,"wb") as save_file:
        dill.dump(engine.eval_history,save_file)

    


if __name__ == "__main__":
    #assert numpyro.__version__.startswith("0.7")
    parser = argparse.ArgumentParser(description="Run SAASBO on the tSIR vaccination problem.")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--max-evals", default=25, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument("--cores", default=12, type=int, help='how many cores to use')
    parser.add_argument("--n_sim", default=50, type=int, help='number of samples to draw in evaluation of objective')
    parser.add_argument("--save_path",default="saasbo_run.save",type=str,help="where to save the output of the run?")
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    enable_x64()
    numpyro.set_host_device_count(1)

    main(args)
