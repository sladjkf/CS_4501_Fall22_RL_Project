import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from numpy import random

from scipy.stats import nbinom, gamma
import matplotlib.pyplot as plt
from itertools import chain
from functools import partial

import warnings
from multiprocess import Pool

import scipy.stats
import time

import dill


##### MOBILITY MODEL: GRAVITY MODEL #####

def random_network(seed=1234,
                   n=100,
                   x_lims=[-500, 500],
                   y_lims=[-500, 500],
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
        x = random.default_rng().uniform(x_lims[0], x_lims[1], n)
        y = random.default_rng().uniform(y_lims[0], y_lims[1], n)
    elif mode == "gaussian":
        # TODO: note that the SD is just fixed... that's fine.
        x = random.default_rng().normal(
            loc=(x_lims[1]-x_lims[0])/2, scale=(x_lims[1]-x_lims[0])/4, size=n)
        y = random.default_rng().normal(
            loc=(y_lims[1]-y_lims[0])/2, scale=(x_lims[1]-x_lims[0])/4, size=n)
    if type(population) == int:
        return pd.DataFrame({"x": x, "y": y, "pop": np.repeat(population, n)})


def grenfell_network(x_lims=(-500, 500),
                     y_lims=(-500, 500),
                     core_pop=500000,
                     sat_pop=100000):
    """
    Generate a synthetic network that's similar to the one used in the paper
    "Measles Metapopulation Dynamics: A Gravity Model for Epidemiological Coupling and Dynamics" (Xia et al.)
    in the section "Theoretical Dynamics"

    x_lims,y_lims: 
        Tuple of length 2. Float elements.
        Used to specify the limits of the grid to generate the network on.
    core_pop:
        Population of the core city. Positive integer.
    sat_pop:
        Population of the satellite cities. Positive integer.

    Return:
        A DataFrame with columns [x,y,pop].
        Each row specifies a city or subpopulation
        x and y are the coordinates
        pop is the population of a city.
    """

    x = np.zeros(250)
    y = np.zeros(250)
    pop = np.repeat(sat_pop, 250)

    # add the core city
    x[0] = 0
    y[0] = 0
    pop[0] = core_pop

    core = (x[0], y[0])

    # add the "satellites"
    angle_increment = 0.025
    segment_increment = 50

    last_x = core[0]
    last_y = core[1]
    arm_length = 10
    for i in range(1, arm_length*25):
        if i % arm_length == 0:
            last_x = core[0]
            last_y = core[1]
        this_x = last_x + segment_increment*np.cos(angle_increment*i)
        this_y = last_y + segment_increment*np.sin(angle_increment*i)
        x[i], y[i] = this_x, this_y
        last_x, last_y = this_x, this_y

    return pd.DataFrame({"x": x, "y": y, "pop": pop})


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
    The ordering of the rows and the columns is the same as in the argument 'network.'
    """
    num_locations = len(network.index)
    coord_pairs = np.array(network[["x", "y"]])
    distance_matrix = np.zeros((num_locations, num_locations))

    def inner_loop(i, j):
        distance = np.linalg.norm(coord_pairs[i, :]-coord_pairs[j, :], ord=2)
        if distance <= 1e-6:
            print("Found distance of less than 1e-6. Maybe there are duplicate rows?")
        distance_matrix[i][j] = distance

    for i in range(0, num_locations):
        for j in list(range(0, i)) + list(range(i+1, num_locations)):
            distance = np.linalg.norm(
                coord_pairs[i, :]-coord_pairs[j, :], ord=2)
            # if distance <= 1e-6:
            #    print("Found distance of less than 1e-6. Maybe there are duplicate rows?")
            distance_matrix[i][j] = distance

    return distance_matrix


def gravity(network, distances, infected, params, parallel=False, cores=4, variant="xia"):
    '''
    Generate gravity weights for a given network. 
    May throw warnings if there are 0 entries off-diagonal in the 'distances' matrix

    network: np.ndarray
        vector with population of each region
    distances: numpy array
        Dimensions: (# of locations) x (# of locations)
        Matrix of distances. It's symmetric.
        The labeling should match that in network.
        (e.g row i in distances and row i in network must refer to the same patch)
    infected:
        a vector of infection counts
        The labeling should match that in 'network'

    variant (optional):
        There are different versions of the gravity model.
        "xia": version in "measles metapopulation dynamics"
        "orig": classical formulation of the gravity model.

    Returns: dictionary
        - 'matrix':  numpy array. 
            Dimensions: (# of locations) x (# of locations)
            Matrix of (unscaled) gravity weights.
        - 'influx': 
            Dimensions: (# of locations)
            vector of (scaled) travel volume into each location in the network. 

        The labeling/ordering of 'matrix','influx' should match that of the 
        distance matrix that was passed in.
    '''
    tau1, tau2, rho, theta = params["tau1"], params["tau2"], params["rho"], params["theta"]
    num_locations = network.shape[0]
    adj_matrix = np.zeros((num_locations, num_locations))
    pop_vec = network

    # ---- old serial method with for loop ---#
    # if False:
    #    for i in range(0,num_locations):
    #        for j in chain(range(0,i),range(i+1,num_locations)):
    #            adj_matrix[i][j] = (infected[i]**tau1) * (pop_vec[j]**tau2) / (distances[i][j]**rho)
    #
    #    # TODO: same here, don't check condition on inner loop
    #    # compute the parameters for the gamma distr
    #    m = np.array([
    #            theta*sum([adj_matrix[j][k] for j in chain(range(0,k),range(k+1,num_locations))])
    #                  for k in range(0,num_locations)
    #            ])

    # ---- vectorized version ---- #
    # repeat vectors akin to i,j in for loop
    #inf_i = np.repeat(infected,num_locations)
    pop_i = np.repeat(pop_vec, num_locations)
    pop_j = np.tile(pop_vec, num_locations)
    inf_j = np.tile(infected, num_locations)

    # add 1 to diagonal to avoid numerical error
    distances_ij = distances + np.eye(distances.shape[0])
    distances_ij = np.reshape(distances_ij, num_locations**2)

    # compute gravity weights and reshape
    try:
        if variant == "xia":
            #adj_mat_flat =  (pop_j**tau1) * (inf_i**tau2) / (distances_ij**rho)
            adj_mat_flat = (pop_i**tau1) * (inf_j**tau2) / (distances_ij**rho)
        elif variant == "orig":
            adj_mat_flat = (pop_i**tau1) * (pop_j**tau2) * \
                (inf_j/pop_j) / (distances_ij**rho)
    except RuntimeWarning:
        print("num_locations", num_locations)
        print(np.argwhere(np.isnan(inf_i)))
        print(np.argwhere(np.isnan(pop_j)))
        print(np.argwhere(np.isnan(distances_ij)))
        print(np.argwhere(distances_ij == 0))
    adj_matrix = adj_mat_flat.reshape(num_locations, num_locations)

    # remove diagonal entries by using a "hollow" matrix (zeros on diagonal, 1 elsewhere)
    adj_matrix = adj_matrix*(1-np.eye(num_locations))

    # if any nan still remains... remove it?
    adj_matrix = np.nan_to_num(adj_matrix)
    # freely sum; since diagonal is 0 don't need to add conditional
    m = theta*np.sum(adj_matrix, axis=0)
    # print(m)

    return {"matrix": adj_matrix, "influx": m}


#### MOBILITY MODEL: Radiation model ####

def get_radiation_flows(pop, distances, mobile):
    """
    Get the flow matrix described by the radiation model (incomplete)
    pop: DataFrame
    distances: NumPy matrix mobile: float in [0,1] or numpy vector
    """
    num_locations = len(pop.index)
    if np.isscalar(mobile):
        mobile_vec = np.repeat(mobile, num_locations)
    else:
        assert len(
            mobile) == num_locations, "mobility vector length doesn't match number of locations in pop"
        mobile_vec = mobile

    flow_matrix = np.zeros((num_locations, num_locations))

    # start with a naive double-for loop
    # could be vectorized or stored

    for i in range(0, num_locations):
        for j in range(0, num_locations):
            pop_i = pop['pop'].iloc[i]
            pop_j = pop['pop'].iloc[j]
            # we can only use the information at the given spatial resolution
            # to compute the total population in a particular radius
            # zip code is pretty fine so maybe ok?
            dist_ij = distances[i, j]

            # mask to remove population sizes of location i and location j
            mask = np.ones(num_locations)
            mask[i], mask[j] = 0, 0
            to_iterate_over = zip(pop['pop']*mask, distances[i, :])
            pop_in_radius = sum(x[0] for x in zip(
                pop['pop'], distances[i, :]) if x[1] < dist_ij)

            with warnings.catch_warnings():  # in case places have 0 population or something?
                warnings.filterwarnings('error')
                try:
                    flow_matrix[i][j] = mobile_vec[i] * (pop_i*pop_j) / (
                        (pop_i + pop_in_radius)*(pop_i + pop_j + pop_in_radius))
                except:
                    flow_matrix[i][j] = 0

    # vectorized version

    return flow_matrix


def get_radiation_influx(flows, infected, theta):
    """
    TODO: Incomplete.
    Turn radiation flows into raw volume flows (suitable for use in Xia's model.)
    Parameters
    ----------
    flows : numpy array (NxN)
        DESCRIPTION.
    infected : numpy array (N)
        DESCRIPTION.
    theta : float
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # can do dimension checks as usual
    N = len(infected)
    # mask diagonal entries
    flows_masked = flows * (np.ones((N, N)) - np.eye(N))
    # left multiply to do the sum
    # theta*sum(flow[i,k]*infected[i] for all i != k) over all k
    return theta*(infected[np.newaxis, :] @ flows_masked)


#### DISEASE MODEL: Spatial tSIR MODEL ####

class spatial_tSIR:
    '''
    Implementation of the spatial tSIR model as described in 
    'Measles Metapopulation Dynamics: A Gravity Model for Epidemiological Coupling and Dynamics.'
    '''

    def __init__(self, config, patch_pop, initial_state, distances=None, seed=None):
        """
        Initialize and parameterize the disease simulation.
        Will warn you if invalid 'initial_state' passed in.

        Parameters
        ----------
        config : dict
            Dictionary containing parameters for the disease simulation and the model.
            Parameters include
            - iters: Number of iterations to run the simulation for
            - tau1, tau2, rho, theta: Parameters for the gravity model.
            - alpha: Mixing exponent
            - beta: Contact rate. Constant over time
            - beta_t: Time-varying contact rate. Requires a list of 26 floats for each biweek of measles simulation.
            - birth: Per-capita birth rate. Constant over time.
            - birth_t: Time-varying per-capita birth rate. Requires a list of 26 floats for each biweek of measles simulation.
            - grav_variant: "xia" or "orig". If not specified, then "xia" is used
        patch_pop : np.ndarray, dimension N
            vector containing the population of each region.
        distances : numpy matrix, dimension N x N (optional)
            precomputed distance matrix from locations in patch_pop
            labeling needs to correspond to the ordering in patch_pop
        initial_state: numpy matrix. dimension N x 2
            Describe the initial state of the simulation.
            Column specification: (I, R)
            Rows correspond to the ones in patch_df.        
        seed: int
            Set a random seed for the simulation.
            By default, will just use the current time as an int if not passed.
        Returns
        -------
        None.

        """
        # num_patches and initial_state length must match
        self.num_patches = patch_pop.shape[0]
        assert self.num_patches == initial_state.shape[0], "number of rows in patch_df, initial_state didn't match"

        # initialize the first state matrix
        # to check: make sure I+R <= S
        self.state_matrix_series = []
        S_0 = patch_pop - initial_state[:, 0] - initial_state[:, 1]
        assert all(S_0 >= 0), "some entries of susceptible state are <0 (or maybe float comparison?) check initial state"
        assert not any(np.isnan(S_0)), "some entries of susceptible state are nan. Check initial state again"
        self.state_matrix_series.append(np.stack([S_0, initial_state[:, 0], initial_state[:, 1]], axis=1))

        self.config = config
        self.patch_pop = patch_pop

        if type(distances) != type(None):
            self.distances = distances
        else:
            self.distances = get_distances(network=self.patch_pop)

        # initialize random seed
        if type(seed) == type(None):
            self.random_state = np.random.RandomState(int(time.time()))
        else:
            self.random_state = np.random.RandomState(seed)

        # check gravity variant mode
        if "grav_variant" not in self.config.keys():
            self.config['grav_variant'] = 'xia'

    def run_simulation(self, verbose=False):
        """
        Run the simulation. 
        The results can be analyzed through other methods of the class, 
        or by accessing the state_matrix_series directly.
        """
        # setup parameters
        # beta = self.config['beta']
        alpha = self.config['alpha']
        pop_vec = self.patch_pop

        if 'birth_t' in self.config.keys():
            birth_rate_t = self.config['birth_t']
        elif 'birth' in self.config.keys():
            birth_rate_t = [self.config['birth']]*26
        else:
            birth_rate_t = [0]*26

        if 'beta_t' in self.config.keys():
            beta_t = self.config['beta_t']
        else:
            # or however long the epidemic period is..
            beta_t = [self.config['beta']]*26

        # compute the distances
        #distances = get_distances(network=self.patch_pop)
        for iter_num in range(self.config['iters']):
            # get last S,I,R counts
            last_matrix = self.state_matrix_series[-1]
            S_t, I_t, R_t = last_matrix[:,0], last_matrix[:, 1], last_matrix[:, 2]
            pop_t = S_t + I_t + R_t

            # stop if I_t = 0 in order to save on simulation
            # TODO: good idea, but you get "ragged nested sequences."
            # how do you fix that?
            # if np.sum(I_t) == 0:
            #    break

            # get current beta
            # currently assuming mod 26 (transmission rate over a year)
            beta = beta_t[iter_num % 26]
            try:
                birth_rate = birth_rate_t[iter_num]
            except IndexError as e:
                #warnings.warn("Not enough birth rate data supplied, just using last value available")
                birth_rate = birth_rate_t[-1]

            # compute new infections
            delta_I_t = np.zeros(self.num_patches)
            # TODO: allow alternate network weight parameterizations?
            grav_model_out = gravity(network=self.patch_pop,
                                     distances=self.distances, infected=I_t, params=self.config, variant=self.config['grav_variant'])
            infection_influx = grav_model_out["influx"]
            
            iota_t = np.zeros(len(infection_influx))

            for index, a in enumerate(infection_influx):
                if a > 0:
                    iota_t[index] = gamma.rvs(scale=1, a=a, random_state=self.random_state)

            for patch_k in range(self.num_patches):
                if I_t[patch_k] or iota_t[patch_k] > 0:
                    if iota_t[patch_k] < 1:
                        iota_tk = round(iota_t[patch_k])
                    else:
                        iota_tk = iota_t[patch_k]
                    # TODO: to normalize or not?
                    # well... shouldn't, right? expected number of infections
                    # in next time step is not a proportion.
                    # But must divide by population at time t if a birth rate incorporated.
                    lambda_k_t = (beta*S_t[patch_k] * (I_t[patch_k]+iota_tk)**(alpha)) / pop_t[patch_k]
                    #lambda_k_t = ( beta*S_t[patch_k] * (I_t[patch_k]+iota_t[patch_k])**(alpha) )
                    # TODO: parameterizing the negative binomial correctly???
                    n = I_t[patch_k]+iota_t[patch_k]
                    p = n/(n+lambda_k_t)

                    # take min, since you can't infect more individuals than there are susceptibles
                    try:
                        delta_I_t[patch_k] = max(min(S_t[patch_k], nbinom.rvs(
                            n, p, random_state=self.random_state)), 0)
                    except ValueError:
                        warnings.warn(
                            "distribution parameterized incorrectly.")
                        print(n, p, lambda_k_t, flush=True)
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
                print("current beta", beta)
                print("infection vector", I_t)
                print("influx", iota_t)
                print("change in I", delta_I_t, flush=True)

    def plot_epicurve(self, normalize=False, total=False,
                      select=None, time_range=None, alpha=0.25):
        """
        Plot the infection counts from the epidemic simulation.

        Choose either one of normalize or total, but not both.
        If both selected nothing will be changed.
        Arguments:
            normalize: bool
                If true, normalize infection counts of each patch to the max of that patch.
                i.e something like i_t = I_t/max(I)
            total: bool
                If true, display the total epicurve summed across all patches.
            select: list of ints or int
                Select which patches to display the time series for.
                The numbering of the patches corresponds to the index of the patch 
                in the 'patch_pop', 'distances' dataframes passed at the beginning of the simulation.
            time_range: list of int, length 2
                Specify the start and end of the time series to display.
            alpha: float
                Transparency parameter of the drawn curves.

        Return: Matplotlib plot.
        """
        if normalize and not total:
            ts_data = self.get_ts_matrix()/np.array(self.get_ts_matrix().max())
        if not normalize and total:
            ts_data = np.sum(self.get_ts_matrix(), axis=1)
        else:
            ts_data = self.get_ts_matrix()

        if select != None:
            ts_data = ts_data.iloc[:, select]
            if total:
                print("Total enabled - ignoring select")
        if time_range != None:
            if total:
                ts_data = ts_data.iloc[time_range[0]:time_range[1]]
            else:
                ts_data = ts_data.iloc[time_range[0]:time_range[1], :]

        if total:
            return ts_data.plot(legend=False)
        else:
            return ts_data.plot(legend=False, alpha=alpha)

    def get_ts_matrix(self):
        return pd.DataFrame(np.array([x[:, 1] for x in self.state_matrix_series]))

    def correlation_matrix(self, normalize=True):
        if normalize:
            ts_matrix = self.get_ts_matrix()/np.array(self.get_ts_matrix().max())
        else:
            ts_matrix = self.get_ts_matrix()
        return np.corrcoef(ts_matrix.T)


class spatial_tSIR_pool:
    '''
    Helper class for running and analyzing many tSIR simulations at once.
    This is useful for obtaining the distribution of the process at any particular time 
    via Monte-Carlo methods.
    '''

    def __init__(self, config=None, patch_pop=None, initial_state=None, n_sim=None, distances=None, load=None):
        '''
        Initialize the the simulation pool.

        If running from scratch:
            It is initialized in a similar way as the 'spatial_tSIR' object.
            Check the documentation for that class.
            n_sim: positive integer
                Controls how many simulations will be run in the pool.

        If loading: (i.e load != None):
            load: path to load from.
            Should be a matrix that was saved from a previous simulation (.npy file).

        Return: None.
        '''
        load_supplied = type(load) != type(None)
        sim_params_supplied = all([type(x) != type(None)
                                  for x in (config, patch_pop, initial_state, n_sim)])

        if load_supplied:
            with open(load, 'rb') as save_file:
                print("loading", load)
                save = dill.load(save_file)
                self.sim_state_mats = save['sim_state_mats']
                self.config = save['config']
                self.n_sim = save['n_sim']

        elif sim_params_supplied:
            self.n_sim = n_sim
            self.config = config
            # seed the simulation to avoid having simulations that get run in the same batch having
            # the same exact result
            # factor of 10 chosen somewhat arbitrarily but needed to be sufficiently large
            seeds = [int(time.time() + i*10) for i in range(0, n_sim)]
            self.sim_list = [spatial_tSIR(config, patch_pop, initial_state, distances=distances, seed=seed) for seed in seeds]
            self.sim_state_mats = None
        else:
            raise ValueError("parameters specified incorrectly - either provide a path in 'load' or provide the tSIR simulation parameters.")

    def save(self, path):
        """
        how to save config?
        """
        with open(path, 'wb') as out_file:
            dill.dump({'config': self.config, 'sim_state_mats': self.sim_state_mats,
                      'n_sim': self.n_sim}, file=out_file)

    def save_csv(self, path, names=None):
        """
        write the results of the simulation to a csv
        """
        time_range = self.config['iters']+1  # +1 for 0th state
        n_sim = self.n_sim
        sim_indices = np.repeat(np.arange(n_sim), time_range)
        time_indices = np.tile(np.arange(time_range), n_sim)
        to_save = np.concatenate(self.sim_state_mats)
        to_save = pd.DataFrame(to_save)
        if type(names) != type(None):
            to_save.columns = names
        to_save.insert(0, "sim_num", sim_indices)
        to_save.insert(1, "time", time_indices)
        to_save.to_csv(path, index=False)

    def run_simulation(self, multi=True, pool=None):
        """
        Run the pool of simulations.
        multi: bool
            If true, the simulations will be run using multiple threads/cores
            via the 'multiprocess' module.
        pool: multiprocess.Pool object

        Return: None.
        """
        if multi and type(Pool) != type(None):
            self.sim_list = pool.map(lambda sim: (sim.run_simulation(), sim)[-1], self.sim_list)
        elif not multi:
            for sim in self.sim_list:
                sim.run_simulation()
        else:
            print("Invalid arguments - either multi=False, or multi=True and a pool object supplied")
        
        # retrieve the state matrices of each simulation and make life slightly easier
        self.sim_state_mats = np.array([np.array(sim.get_ts_matrix()) for sim in self.sim_list])

    def plot_interval(self,
                      select=None, time_range=None,
                      quantiles=[0.025, 0.975], int_type="pred"):
        """
        Plot the curve of infected individuals with a interval of a specified width about the mean path.
        Empirical quantiles are used to get the prediction interval.
        Arguments:
            select: int
                Describes which patch to select.
                Indices correspond to the labeling of the 'patch_pop' or 'distances' matrix
                passed when initializing the simulation.
                If None passed, then the total will be plotted.
            time_range: int list of two elements
            quantiles: float list of two elements
            int_type: str
        """
        samples = self.get_samples(select, time_range)
        mean = np.mean(samples, axis=0).T
        quantiles = self.get_quantiles(select, time_range, quantiles=quantiles)
        fig, ax = plt.subplots()
        ax.plot(mean)
        ax.plot(quantiles, color="red", linestyle="dashed")
        return ax

    def plot_paths(self, select=None, time_range=None, alpha=0.25):
        samples = self.get_samples(select, time_range)
        return plt.plot(samples.T, alpha=alpha)

    def plot_survival(self):
        return plt.plot(self.survival())

    def survival(self):
        """
        get a vector indexed by time
        % of simulations drawn that are nonzero at that time
        can use binomial confidence interval (probably not that necessary; just draw more samples??)
        """
        n_sim = self.n_sim
        time_len = self.config['iters']
        return 1-np.array([sum(self.get_samples()[:, i] < 1e-16)/n_sim for i in range(time_len)])

    def get_mean_sd(self):
        """
        Get time-indexed mean and time-indexed standard deviation for each patch.
        The mean and standard deviation are taken across the simulations.

        Return: dictionary 
            - 'mean': numpy array
                Dimensions: (# of simulation timesteps) x (# of patches)
            - 'sd': numpy array
                Dimensions: (# of simulation timesteps) x (# of patches)
            Row specifies timestep, column specifies which patch.
        """
        ts_matrices = self.sim_state_mats
        # is this right? think it is
        mean_matrix = np.mean(ts_matrices, axis=0)
        sd_matrix = np.std(ts_matrices, axis=0)
        return {"mean": mean_matrix, "sd": sd_matrix}

    def get_samples(self, select=None, time_range=None):
        """
        Get the sample matrix. 
        To make life easier, select will be a scalar. otherwise you get 3d matrices
        select: 
            scalar (which location to get the time series for?)
            If not specfied, get the sample path for the entire simulation
            (aggregate infection counts across all patches)
        time_range: list of length 2: (start, stop) index or scalar
        """
        if type(select) == type(None):
            # aggregate each matrix in the list of state matrices to get the total
            to_return = [np.sum(mat, axis=1) for mat in self.sim_state_mats]
            if np.isscalar(time_range):
                to_return = [ts_matrix[time_range, :]
                             for ts_matrix in to_return]
            elif type(time_range) == list or type(time_range) == tuple:
                to_return = [
                    ts_matrix[range(time_range[0], time_range[1]), :] for ts_matrix in to_return]
        else:
            if type(time_range) == type(None):
                to_return = [ts_matrix[:, select]
                             for ts_matrix in self.sim_state_mats]
            elif np.isscalar(time_range):
                to_return = [ts_matrix[time_range, select]
                             for ts_matrix in self.sim_state_mats]
            else:
                to_return = [ts_matrix[range(
                    time_range[0], time_range[1]), select] for ts_matrix in self.sim_state_mats]
                to_return = np.stack(to_return, axis=1)
        return np.array(to_return)

    def get_quantiles(self, select=None, time_range=None, quantiles=[0.025, 0.975]):
        """
        Get empirical quantiles for 
        """
        samples = self.get_samples(select, time_range)
        return np.quantile(samples, q=quantiles, axis=0).T

    def get_CIs(self, select, time_range):
        if select == -1:
            pass
        else:
            samples = self.get_samples(select, time_range)
            intervals = np.zeros((samples.shape[0], 2))
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                for t_index in range(samples.shape[0]):
                    sample_t = samples[t_index, :]
                    try:
                        result = bootstrap(
                            (sample_t,), np.mean).confidence_interval
                        intervals[t_index, :] = np.array(
                            [result.low, result.high])
                    except:
                        # if degenerate, should be all same
                        intervals[t_index, :] = np.array(
                            [sample_t[0], sample_t[0]])
        return intervals

    def get_attack_size_samples(self):
        return np.sum(self.get_samples(), axis=1)

    def attack_size_cdf(self, x):
        """
        only gets one pt rn... how to vectorize?
        """
        AS_samples = self.get_attack_size_samples()
        if np.isscalar(x):
            return np.sum(AS_samples < x)/len(AS_samples)
        else:
            return np.array([np.sum(AS_samples < x_i)/len(AS_samples) for x_i in x])
