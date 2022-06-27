import numpy as np
import pandas as pd
from numpy import random

def random_network(seed=1234,
                   n=100,
                   x_lims=[-500,500],
                   y_lims=[-500,500],
                  population=1000,
                  mode="unif"):
    random.seed(seed)
    if mode == "unif":
        x=random.default_rng().uniform(x_lims[0],x_lims[1],n)
        y=random.default_rng().uniform(y_lims[0],y_lims[1],n)
    elif mode == "gaussian":
        x=random.default_rng().normal(loc=(x_lims[1]-x_lims[0])/2,scale=(x_lims[1]-x_lims[0])/4,size=n)
        y=random.default_rng().normal(loc=(y_lims[1]-y_lims[0])/2,scale=(x_lims[1]-x_lims[0])/4,size=n)
    if type(population) == int:
        return pd.DataFrame({"x":x,"y":y,"pop":np.repeat(population,n)})
    

def gravity(network,infected,
           params):
    '''
    Generate gravity weights for a given network.
    Output:
        - adjacency matrix of weights
        - vector of m parameters
    network: 
        a data frame of locations
        each row is a location, specified by tuple (x,y,population)
    infected:
        a vector of infection counts
        rows should match those in 'network'
    '''
    tau1,tau2,rho,theta = params["tau1"], params["tau2"], params["rho"], params["theta"]
    num_locations = network.shape[0]
    adj_matrix = np.zeros((num_locations,num_locations))
    # compute matrix
    for i in range(0,num_locations):
        for j in range(0,num_locations):
            if i == j: continue
            else:
                # compute edge weight from i to j
                loc_i = network.iloc[i,:]
                loc_j = network.iloc[j,:]
                #distance=np.sqrt(sum((np.array([loc_i["x"],loc_i["y"]])-np.array([loc_i["x"],loc_i["y"]]))**2))
                distance=np.linalg.norm(np.array(loc_i[["x","y"]])-np.array(loc_j[["x","y"]]),ord=2)
                #print(loc_i,loc_j,distance)
                adj_matrix[i][j] = (infected[i]**tau1) * (loc_j["pop"]**tau2) / (distance**rho)
    # compute the parameters for the gamma distr
    m = np.array([theta*sum([adj_matrix[j][k] for j in range(0,num_locations) if j != k]) for k in range(0,num_locations)])
    return {"matrix":adj_matrix,"influx":m}
            
            
    
    