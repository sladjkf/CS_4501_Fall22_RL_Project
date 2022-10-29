import numpy as np

# 3 by 3 test case
def generate(id):
    if (id == 3):
        pop_vec = np.array([100, 10, 10])
        vacc = np.array([.1, .1, .1])
        seed = np.array([1, 1, 1])
        dist = np.full((3, 3), 1) - np.identity(3)  # with some scaling factor
        return pop_vec, vacc, seed, dist
