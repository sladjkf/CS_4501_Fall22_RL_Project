import numpy as np
import pandas as pd
from numpy import random
from scipy.stats import nbinom, gamma

from itertools import chain
from functools import partial

import warnings
#import multiprocess

##### NETWORK MODEL: GRAVITY MODEL #####


def random_network(seed=1234,
                   n=100,
                   x_lims=[-500,500],
                   y_lims=[-500,500],
                  population=1000,
                  mode="unif"):
    """
    Generate a random network to use with the gravity model.
    Parameters
    ----------
    seed : int, optional
        The default is 1234.
    n : TYPE, optional
        DESCRIPTION. The default is 100.
    x_lims : TYPE, optional
        DESCRIPTION. The default is [-500,500].
    y_lims : TYPE, optional
        DESCRIPTION. The default is [-500,500].
    population : TYPE, optional
        DESCRIPTION. The default is 1000.
    mode : TYPE, optional
        DESCRIPTION. The default is "unif".

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    random.seed(seed)
    if mode == "unif":
        x=random.default_rng().uniform(x_lims[0],x_lims[1],n)
        y=random.default_rng().uniform(y_lims[0],y_lims[1],n)
    elif mode == "gaussian":
        # TODO: note that the SD is just fixed... that's fine.
        x=random.default_rng().normal(loc=(x_lims[1]-x_lims[0])/2,scale=(x_lims[1]-x_lims[0])/4,size=n)
        y=random.default_rng().normal(loc=(y_lims[1]-y_lims[0])/2,scale=(x_lims[1]-x_lims[0])/4,size=n)
    if type(population) == int:
        return pd.DataFrame({"x":x,"y":y,"pop":np.repeat(population,n)})

def grenfell_network(x_lims=(-500,500),
                     y_lims=(-500,500),
                     core_pop=500000,
                     sat_pop=100000):
    x = np.zeros(250)
    y = np.zeros(250)
    pop = np.repeat(sat_pop,250)
    
    # add the core city
    x[0] = 0
    y[0] = 0
    pop[0] = core_pop
    
    core = (x[0],y[0])
    
    # add the "satellites"
    angle_increment=0.025
    segment_increment=50
    
    last_x = core[0]
    last_y = core[1]
    arm_length =10
    for i in range(1,arm_length*25):
        if i%arm_length == 0:
            last_x=core[0]
            last_y=core[1]
        this_x = last_x + segment_increment*np.cos(angle_increment*i)
        this_y = last_y + segment_increment*np.sin(angle_increment*i)
        x[i],y[i] = this_x, this_y
        last_x, last_y = this_x, this_y
        
    return pd.DataFrame({"x":x,"y":y,"pop":pop})


def get_distances(network):
    """
    Generate the network distances for use in the gravity model.
    Pretty naive, but it works.
    Parameters
    ----------
    network : DataFrame 
        column specification: (x,y)
    Returns
    -------
    numpy matrix. (# of locations) x (# of locations).
    """
    num_locations = len(network.index)
    coord_pairs = np.array(network[["x","y"]])
    distance_matrix = np.zeros((num_locations,num_locations))
    def inner_loop(i,j):
        distance = np.linalg.norm(coord_pairs[i,:]-coord_pairs[j,:],ord=2)
        if distance <= 1e-6:
            print("Found distance of less than 1e-6. Maybe there are duplicate rows?")
        distance_matrix[i][j] = distance
        
    for i in range(0,num_locations):
        for j in list(range(0,i)) + list(range(i+1,num_locations)):
            distance = np.linalg.norm(coord_pairs[i,:]-coord_pairs[j,:],ord=2)
            #if distance <= 1e-6:
            #    print("Found distance of less than 1e-6. Maybe there are duplicate rows?")
            distance_matrix[i][j] = distance

    return distance_matrix
            
def gravity(network,distances,infected,params,
        parallel=False,cores=4):
    '''
    Generate gravity weights for a given network.
    Output:
        - adjacency matrix of weights
        - vector of m parameters
    network: 
        a data frame of locations
        each row is a location, specified by tuple (x,y,pop)
    distances:
        adjacency matrix of distances (save computation time)
        (# of locations) x (# of locations)
        It's symmetric.
    infected:
        a vector of infection counts
        rows should match those in 'network'
    '''
    tau1,tau2,rho,theta = params["tau1"], params["tau2"], params["rho"], params["theta"]
    num_locations = network.shape[0]
    adj_matrix = np.zeros((num_locations,num_locations))
    # compute matrix
    # TODO: can definitely be more efficient (vectorization?)
    # i.e don't check condition on inner loop but just do the checking before hand
    
    pop_vec = np.array(network["pop"])
    #if parallel == True:
    #    # generate list of iterables?
    #    i_arr = np.repeat(np.arange(0,num_locations),num_locations)
    #    j_arr = np.tile(np.arange(0,num_locations),num_locations)
    #    #j_arr = np.concatenate([np.delete(np.arange(0,num_locations),i) for i in range(0,num_locations)])
    #    pairs = zip(i_arr,j_arr)

    #    def compute_influx(ij,infected,distances,tau1,tau2,rho):
    #        i = ij[0]
    #        j = ij[1]
    #        if i==j:
    #            return 0
    #        return (infected[i]**tau1) * (pop_vec[j]**tau2) / (distances[i][j]**rho)

    #    compute_influx_partial = partial(compute_influx, 
    #            infected=infected,distances=distances,tau1=tau1,tau2=tau2,rho=rho)
    #    p = multiprocess.Pool(cores)
    #    result = p.map_async(compute_influx_partial,pairs).get()
    #    p.terminate()

    #    adj_matrix = np.array(result).reshape((num_locations,num_locations))

    #    print(adj_matrix)

    #    # compute m (axis shouldn't really matter since symmetric?)
    #    # subtract off diagonal entries
    #    #m = np.sum(adj_matrix,axis=1)
    #    #m = theta*(m - np.array([adj_matrix[k,k] for k in range(0,adj_matrix.shape[0])]))
    #    m = np.array([
    #            theta*sum([adj_matrix[j][k] for j in chain(range(0,k),range(k+1,num_locations))]) 
    #                  for k in range(0,num_locations)
    #            ])

    #else:
    if False:
        # ---- old serial method with for loop ---#
        for i in range(0,num_locations):
            for j in chain(range(0,i),range(i+1,num_locations)):
                adj_matrix[i][j] = (infected[i]**tau1) * (pop_vec[j]**tau2) / (distances[i][j]**rho)
                
        # TODO: same here, don't check condition on inner loop
        # compute the parameters for the gamma distr
        m = np.array([
                theta*sum([adj_matrix[j][k] for j in chain(range(0,k),range(k+1,num_locations))]) 
                      for k in range(0,num_locations)
                ])
    if True:
        # ---- attempt to vectorize ---- #
        # repeat vectors akin to i,j in for loop
        inf_i = np.repeat(infected,num_locations)
        pop_j = np.tile(pop_vec,num_locations)

        # add 1 to diagonal to avoid numerical error
        distances_ij = distances + np.eye(distances.shape[0])
        distances_ij = np.reshape(distances_ij,num_locations**2)

        # compute gravity weights and reshape
        try:
            adj_mat_flat = (inf_i**tau1) * (pop_j**tau2) / (distances_ij**rho)
        except RuntimeWarning:
            print("num_locations",num_locations)
            print(np.argwhere(np.isnan(inf_i)))
            print(np.argwhere(np.isnan(pop_j)))
            print(np.argwhere(np.isnan(distances_ij)))
            print(np.argwhere(distances_ij==0))
        adj_matrix = adj_mat_flat.reshape(num_locations,num_locations)

        # remove diagonal entries by using a "hollow" matrix (zeros on diagonal, 1 elsewhere)
        adj_matrix = adj_matrix*(1-np.eye(num_locations))
        
        # if any nan still remains... remove it?
        adj_matrix = np.nan_to_num(adj_matrix)
        # freely sum; since diagonal is 0 don't need to add conditional
        m = theta*np.sum(adj_matrix,axis=0)
        #print(m)

    return {"matrix":adj_matrix,"influx":m}
            
            
#### DISEASE MODEL: Spatial tSIR MODEL ####
    
class spatial_tSIR:
    def __init__(self, 
            config, 
            patch_pop, initial_state,
            distances=None
            ):
        """
        Initialize the disease simulation with some parameters.

        Parameters
        ----------
        config : dict
            Dictionary containing parameters for the disease simulation and the model.
            Parameters include
            - iters: Number of iterations to run the simulation for
            - tau1, tau2, rho, theta: Parameters for the gravity model.
            - alpha: Mixing exponent
            - beta: Contact rate
        patch_pop : DataFrame, dimension N x 4
            DataFrame with the column specification
            (patch_id, pop, x, y)
        distances : numpy matrix, dimension N x N (optional)
            precomputed distance matrix from locations in patch_pop
            labeling needs to correspond to the ordering in patch_pop
        initial_state: numpy matrix. dimension N x 2
            Describe the initial state of the simulation.
            Column specification: (I, R)
            Rows correspond to the ones in patch_df.        
        Returns
        -------
        None.

        """
        # num_patches and initial_state length must match
        self.num_patches = len(patch_pop.index)
        assert self.num_patches == initial_state.shape[0], "number of rows in patch_df, initial_state didn't match"
        
        # initialize the first state matrix
        # to check: make sure I+R <= S
        self.state_matrix_series = []
        S_0 = patch_pop['pop'] - initial_state[:,0] - initial_state[:,1]
        self.state_matrix_series.append(
            np.stack([S_0,initial_state[:,0],initial_state[:,1]],axis=1)
        )
        
        self.config = config
        self.patch_pop = patch_pop

        if type(distances) != type(None):
            self.distances = distances
        else:
            self.distances = get_distances(network=self.patch_pop)
    def run_simulation(self,verbose=False):
        
        # setup parameters
        # beta = self.config['beta']
        alpha = self.config['alpha']
        pop_vec = self.patch_pop['pop']
        
        if 'birth_t' in self.config.keys():
            birth_rate_t = self.config['birth_t']
        elif 'birth' in self.config.keys():
            birth_rate_t = [self.config['birth']]*26
        else:
            birth_rate_t = [0]*26
            
        if 'beta_t' in self.config.keys():
            beta_t = self.config['beta_t']
        else:
            beta_t = [self.config['beta']]*26 # or however long the epidemic period is..

        # compute the distances
        #distances = get_distances(network=self.patch_pop)
        for iter_num in range(self.config['iters']):
            # get last S,I,R counts
            last_matrix = self.state_matrix_series[-1]
            S_t, I_t, R_t = last_matrix[:,0], last_matrix[:,1], last_matrix[:,2]
            pop_t = S_t + I_t + R_t
            
            # get current beta
            # currently assuming mod 26 (transmission rate over a year)
            beta = beta_t[iter_num % 26]
            try:
                birth_rate = birth_rate_t[iter_num]
            except IndexError as e:
                warnings.warn("Not enough birth rate data supplied, just using last value available")
                birth_rate = birth_rate_t[-1]
            
            # compute new infections
            delta_I_t = np.zeros(self.num_patches)
            # TODO: allow alternate network weight parameterizations?
            grav_model_out = gravity(network=self.patch_pop, 
                    distances=self.distances, infected=I_t, params=self.config)
            infection_influx = grav_model_out["influx"]
            iota_t = np.array([gamma.rvs(scale=1,a=a) if a > 0 else 0 for a in infection_influx])
            #print(infection_influx)
            for patch_k in range(0,self.num_patches):
                if I_t[patch_k] or iota_t[patch_k] > 0:
                    if iota_t[patch_k] < 1:
                        iota_tk = round(iota_t[patch_k])
                    else:
                        iota_tk = iota_t[patch_k]
                    # TODO: to normalize or not?
                    # well... shouldn't, right? expected number of infections
                    # in next time step is not a proportion.
                    # But must divide by population at time t if a birth rate incorporated.
                    lambda_k_t = ( beta*S_t[patch_k] * (I_t[patch_k]+iota_tk)**(alpha) ) / pop_t[patch_k]
                    #lambda_k_t = ( beta*S_t[patch_k] * (I_t[patch_k]+iota_t[patch_k])**(alpha) )
                    # TODO: parameterizing the negative binomial correctly???
                    n = I_t[patch_k]+iota_t[patch_k]
                    p = n/(n+lambda_k_t)
                    
                    # take min, since you can't infect more individuals than there are susceptibles
                    try:
                        delta_I_t[patch_k] = max(min(S_t[patch_k],nbinom.rvs(n,p)),0)
                    except ValueError:
                        warnings.warn("distribution parameterized incorrectly.")
                        print(n,p,lambda_k_t,flush=True)
                        return
                else:
                    delta_I_t[patch_k] = 0
    
            next_matrix = np.stack(
                [
                    S_t-delta_I_t+np.int64(birth_rate*pop_vec),
                    delta_I_t,
                    R_t+I_t
                ],
                axis=1
            )
            self.state_matrix_series.append(next_matrix)
            
            if verbose:
                print("---------------ITERATION {}--------".format(iter_num))
                print("current beta",beta)
                print("infection vector", I_t)
                print("influx",iota_t)
                print("change in I",delta_I_t,flush=True)
    def plot_epicurve(self, normalize=False, total=False,
                      select=None, time_range=None, alpha=0.25):
        if normalize and not total:
            ts_data = self.get_ts_matrix()/np.array(self.get_ts_matrix().max())
        if not normalize and total:
            ts_data = np.sum(self.get_ts_matrix(),axis=1)
        else:
            ts_data = self.get_ts_matrix()
        
        if select != None:
            ts_data = ts_data.iloc[:,select]
            if total:
                print("Total enabled - ignoring select")
        if time_range != None:
            if total:
                ts_data = ts_data.iloc[time_range[0]:time_range[1]]
            else:
                ts_data = ts_data.iloc[time_range[0]:time_range[1],:]
        
        if total:
            return ts_data.plot(legend=False)
        else:
            return ts_data.plot(legend=False,alpha=alpha)

    def get_ts_matrix(self):
        return pd.DataFrame(np.array([x[:,1] for x in self.state_matrix_series]))
    
    def correlation_matrix(self,normalize=True):
        if normalize:
            ts_matrix = self.get_ts_matrix()/np.array(self.get_ts_matrix().max())
        else:
            ts_matrix = self.get_ts_matrix()
        return np.corrcoef(ts_matrix.T)
