"""
load configuration file and initialize the function
"""
import configparser
import pandas as pd
import numpy as np
from scripts.optimization import vacc

config_dir = "config/5_by_5/{}"

sim_param_config = configparser.ConfigParser()
sim_param_config.optionxform = str  # retain case sensitivity

# load simulation files
with open(config_dir.format("params.cfg")) as config_file:
    sim_param_config.read_file(config_file)
    tsir_config_str = dict(sim_param_config['tsir_config'])
    tsir_config = {}
    for k,v in tsir_config_str.items():
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
        cores=5, n_sim=1000,
        output_dir = "/home/nick/Documents/4tb_sync/UVA GDrive/Semesters/Semester 7/CS 4501/project/CS_4501_Fall22_RL_Project/output/",
        name="5by5"
    )