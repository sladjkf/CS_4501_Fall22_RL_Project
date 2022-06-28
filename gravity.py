import numpy as np
import pandas as pd
from numpy import random
from scipy.stats import nbinom, gamma

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
                     y_lims=(-500,500)):
    x = np.zeros(250)
    y = np.zeros(250)
    pop = np.repeat(100000,250)
    
    # add the core city
    x[0] = 0
    y[0] = 0
    pop[0] = 5000000
    
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
    Generate the network distances for use in the gravity model
    Parameters
    ----------
    network : DataFrame 
        column specification: (x,y)
    Returns
    -------
    numpy matrix. (# of locations) x (# of locations).
    """
    num_locations = len(network.index)
    distance_matrix = np.zeros((num_locations,num_locations))
    def inner_loop(i,j):
        distance = np.linalg.norm(coord_pairs[i,:]-coord_pairs[j,:],ord=2)
        if distance <= 1e-6:
            print("Found distance of less than 1e-6. Maybe there are duplicate rows?")
        distance_matrix[i][j] = distance
        
    for i in range(0,num_locations):
        for j in range(0,i):
            inner_loop(i,j)
        for j in range(i+1,num_locations):
            inner_loop(i,j)
    return distance_matrix
            
def gravity(network=None, distances=None
           infected,
           params):
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
    
    #x_vector = np.array(network["x"])
    #y_vector = np.array(network["y"])
    coord_pairs = np.array(network[["x","y"]])
    pop_vec = network["pop"]
    def inner_loop(i,j):
        distance = np.linalg.norm(coord_pairs[i,:]-coord_pairs[j,:],ord=2)
        if distance <= 1e-6:
            print(distance)
            print(coord_pairs[i,:],coord_pairs[j,:])
        adj_matrix[i][j] = (infected[i]**tau1) * (pop_vec[j]**tau2) / (distance**rho)
    
    # unrolled loops
    #for k in range(1,num_locations):
    #    inner_loop(0,k)
    #for j in range(0,num_locations-1):
    #    inner_loop(num_locations-1,j)
    for i in range(0,num_locations):
        for j in range(0,i):
            inner_loop(i,j)
        for j in range(i+1,num_locations):
            inner_loop(i,j)
    # TODO: same here, don't check condition on inner loop
    # compute the parameters for the gamma distr
    m = np.array([theta*sum([adj_matrix[j][k] for j in range(0,num_locations) if j != k]) for k in range(0,num_locations)])
    return {"matrix":adj_matrix,"influx":m}
            
            
#### DISEASE MODEL: Spatial tSIR MODEL ####
    
class spatial_tSIR:
    def __init__(self, config, patch_pop, initial_state):
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
            np.stack([S_0,initial_state[:,0],initial_state[:,0]],axis=1)
        )
        
        self.config = config
        self.patch_pop = patch_pop
    def run_simulation(self,verbose=False):
        
        beta = self.config['beta']
        alpha = self.config['alpha']
        pop_vec = self.patch_pop['pop']
        if 'birth' in self.config.keys():
            birth_rate = self.config['birth']
        else:
            birth_rate = 0

        for iter_num in range(self.config['iters']):
            last_matrix = self.state_matrix_series[-1]
            S_t, I_t, R_t = last_matrix[:,0], last_matrix[:,1], last_matrix[:,2]
            # compute new infections
            delta_I_t = np.zeros(self.num_patches)
            # TODO: allow alternate network weight parameterizations?
            grav_model_out = gravity(network=self.patch_pop, infected=I_t, params=self.config)
            infection_influx = grav_model_out["influx"]
            iota_t = np.array([gamma.rvs(scale=1,a=a) if a > 0 else 0 for a in infection_influx])
            #print(infection_influx)
            for patch_k in range(0,self.num_patches):
                if I_t[patch_k] or iota_t[patch_k] > 0:
                    lambda_k_t = ( beta*S_t[patch_k] * (I_t[patch_k]+iota_t[patch_k])**(alpha) ) / pop_vec[patch_k]
                    n = I_t[patch_k]+iota_t[patch_k]
                    p = 1/(1+(lambda_k_t/n))
                    
                    # take min, since you can't infect more individuals than there are susceptibles
                    try:
                        delta_I_t[patch_k] = min(S_t[patch_k],nbinom.rvs(n,p))
                    except ValueError:
                        print(n,p,flush=True)
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
                print("infection vector", I_t)
                print("influx",iota_t)
                print("change in I",delta_I_t,flush=True)
    def plot_epicurve(self):
        ts_data = pd.DataFrame(np.array([x[:,1] for x in self.state_matrix_series]))
        return ts_data.plot(legend=False)

    def get_ts_matrix(self):
        return pd.DataFrame(np.array([x[:,1] for x in self.state_matrix_series]))
    
    def correlation_matrix(self):
        ts_matrix = self.get_ts_matrix()
        return np.corrcoef(sim.get_ts_matrix().T)
