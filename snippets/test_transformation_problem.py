import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open("project_dir.txt") as f:
    project_path = f.read().strip() + "{}"

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

constraint_bnd = 0.5
V_0 = np.array((vacc_df['pop']-vacc_df['nVaccCount'])/vacc_df['pop'])

ub = V_0 @ vacc_df['pop']
lb = ub - constraint_bnd*np.linalg.norm(vacc_df['pop'],ord=np.inf)
eq_constraint = constraint_bnd*np.linalg.norm(vacc_df['pop'],ord=np.inf)

def generate_random_sum(ub,lb,length=705):
    target_sum = np.random.random()*(ub-lb) + lb
    # generate sum between 0 and 1
    x = list(np.random.random(length-1))
    x.insert(0,0)
    x.append(1)
    x.sort()
    return np.diff(x)*target_sum

def generate_random_sum_exact(c,length=705):
    # generate sum between 0 and 1
    x = list(np.random.random(length-1))
    x.insert(0,0)
    x.append(1)
    x.sort()
    return np.diff(x)*c

def generate_bounded(ub,lb,length=705):
    return np.random.random(length)*(ub-lb)+lb

def inv_tr_lin_solve(point,l,u,c,dim=None):
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
    #b[dim-1] = point[dim-1]
    b[dim-1] = c
    L = np.tril(np.ones((dim,dim)))
    x = scipy.linalg.solve(coeff_matrix@L,b)
    return x

point = generate_bounded(100,200,length=20)
inv_tr_lin_solve(point,100,200,eq_constraint)

